import torch
import pickle
import os
import time
import numpy as np
from typing import Optional, Dict, Any

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_geometric.nn import SAGEConv, GATConv

# ==============================================================================
# UTILITY FUNCTIONS (OPTIMIZED)
# ==============================================================================

def edges_to_sorted_lists_fast(edge_index: torch.Tensor, edge_times: torch.Tensor) -> Dict[int, np.ndarray]:
    """Convert edge list to per-source sorted event arrays using numpy vectorization."""
    srcs = edge_index[0].numpy()
    dsts = edge_index[1].numpy()
    times = edge_times.numpy()
    
    # Sort by (source, time) using numpy
    sort_idx = np.lexsort((times, srcs))
    srcs_sorted = srcs[sort_idx]
    dsts_sorted = dsts[sort_idx]
    times_sorted = times[sort_idx]
    
    # Find boundaries between different sources
    unique_srcs, first_idx, counts = np.unique(srcs_sorted, return_index=True, return_counts=True)
    
    per_src = {}
    for i, s in enumerate(unique_srcs):
        start = first_idx[i]
        end = start + counts[i]
        # Store as structured array for efficiency
        per_src[int(s)] = np.column_stack([
            times_sorted[start:end],
            dsts_sorted[start:end]
        ])
    
    return per_src


def build_next_link_labels_fast(edge_index: torch.Tensor, edge_times: torch.Tensor) -> torch.Tensor:
    """Build next-link labels using vectorized numpy operations."""
    per_src = edges_to_sorted_lists_fast(edge_index, edge_times)
    
    # Pre-allocate based on total edges minus unique sources
    total_pairs = sum(len(events) - 1 for events in per_src.values() if len(events) > 1)
    
    if total_pairs == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    pos_src = np.empty(total_pairs, dtype=np.int64)
    pos_dst = np.empty(total_pairs, dtype=np.int64)
    
    idx = 0
    for s, events in per_src.items():
        n = len(events) - 1
        if n > 0:
            pos_src[idx:idx + n] = s
            pos_dst[idx:idx + n] = events[1:, 1].astype(np.int64)  # Next destinations
            idx += n
    
    return torch.from_numpy(np.stack([pos_src, pos_dst]))


def build_all_future_labels_fast(edge_index: torch.Tensor, edge_times: torch.Tensor) -> torch.Tensor:
    """Build all-future labels using vectorized operations."""
    per_src = edges_to_sorted_lists_fast(edge_index, edge_times)
    
    # Calculate total pairs: sum of k*(k-1)/2 for each source with k events
    total_pairs = sum(len(e) * (len(e) - 1) // 2 for e in per_src.values())
    
    if total_pairs == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    pos_src = np.empty(total_pairs, dtype=np.int64)
    pos_dst = np.empty(total_pairs, dtype=np.int64)
    
    idx = 0
    for s, events in per_src.items():
        dsts = events[:, 1].astype(np.int64)
        n = len(dsts)
        for i in range(n - 1):
            count = n - i - 1
            pos_src[idx:idx + count] = s
            pos_dst[idx:idx + count] = dsts[i + 1:]
            idx += count
    
    return torch.from_numpy(np.stack([pos_src, pos_dst]))


def build_next_labels_from_splits(train_edge_index, train_edge_times, val_data, test_data):
    """Construct the next-edge per node after train/val for evaluation."""
    train_end = train_edge_times.max().item()
    val_end = val_data.edge_attr[:, 2].max().item()

    all_edge_index = torch.cat([train_edge_index, val_data.edge_index, test_data.edge_index], dim=1)
    all_times = torch.cat([train_edge_times, val_data.edge_attr[:, 2], test_data.edge_attr[:, 2]], dim=0)
    per_src = edges_to_sorted_lists_fast(all_edge_index, all_times)

    val_src, val_dst = [], []
    test_src, test_dst = [], []

    for s, events in per_src.items():
        times = events[:, 0]
        dsts = events[:, 1].astype(np.int64)
        
        # Find first edge after train_end
        after_train_mask = times > train_end
        if after_train_mask.any():
            first_after_train_idx = np.argmax(after_train_mask)
            next_after_train = dsts[first_after_train_idx]
            # Check if any edge is in val period
            if (times[after_train_mask] <= val_end).any():
                val_src.append(s)
                val_dst.append(next_after_train)
        
        # Find first edge after val_end
        after_val_mask = times > val_end
        if after_val_mask.any():
            first_after_val_idx = np.argmax(after_val_mask)
            test_src.append(s)
            test_dst.append(dsts[first_after_val_idx])

    val_labels = torch.tensor([val_src, val_dst], dtype=torch.long) if val_src else torch.empty((2, 0), dtype=torch.long)
    test_labels = torch.tensor([test_src, test_dst], dtype=torch.long) if test_src else torch.empty((2, 0), dtype=torch.long)
    return val_labels, test_labels


def build_adjacency_tensors(edge_index: torch.Tensor, num_nodes: int):
    """
    Build adjacency data structures optimized for fast negative sampling.
    Returns:
        adj_offsets: [num_nodes + 1] cumulative neighbor counts
        adj_neighbors: [num_edges * 2] flattened neighbor list
    """
    # Make undirected
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    src, dst = edge_index_undirected[0], edge_index_undirected[1]
    
    # Sort by source
    sort_idx = torch.argsort(src)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]
    
    # Count neighbors per node
    neighbor_counts = torch.zeros(num_nodes, dtype=torch.long)
    neighbor_counts.scatter_add_(0, src_sorted, torch.ones_like(src_sorted))
    
    # Build CSR-like structure
    adj_offsets = torch.zeros(num_nodes + 1, dtype=torch.long)
    adj_offsets[1:] = neighbor_counts.cumsum(0)
    
    return adj_offsets, dst_sorted


class NegativeSampler:
    """Efficient negative sampler with precomputed adjacency."""
    
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device: str = 'cuda'):
        self.num_nodes = num_nodes
        self.device = device
        
        # Build adjacency on CPU (memory efficient)
        adj_offsets, adj_neighbors = build_adjacency_tensors(edge_index, num_nodes)
        self.adj_offsets = adj_offsets
        self.adj_neighbors = adj_neighbors
        
        # Build neighbor sets for fast lookup (Python sets are fast for membership)
        self.neighbor_sets = [set() for _ in range(num_nodes)]
        for i in range(num_nodes):
            start, end = adj_offsets[i].item(), adj_offsets[i + 1].item()
            self.neighbor_sets[i] = set(adj_neighbors[start:end].tolist())
    
    def sample(self, batch_src: torch.Tensor, num_neg: int = 5) -> torch.Tensor:
        """Sample hard negatives (2-hop neighbors that aren't 1-hop)."""
        batch_size = batch_src.size(0)
        src_list = batch_src.cpu().tolist()
        
        # Pre-allocate output
        neg_dst = torch.empty(batch_size, num_neg, dtype=torch.long)
        
        for i, s in enumerate(src_list):
            neighbors = self.neighbor_sets[s]
            
            # Collect 2-hop neighbors
            candidates = set()
            for nbr in neighbors:
                candidates.update(self.neighbor_sets[nbr])
            candidates.discard(s)
            candidates -= neighbors
            
            if not candidates:
                # Fallback to random
                neg_dst[i] = torch.randint(0, self.num_nodes, (num_neg,))
            else:
                cand_list = list(candidates)
                if len(cand_list) >= num_neg:
                    indices = torch.randperm(len(cand_list))[:num_neg]
                    neg_dst[i] = torch.tensor([cand_list[j] for j in indices])
                else:
                    # Sample with replacement
                    indices = torch.randint(0, len(cand_list), (num_neg,))
                    neg_dst[i] = torch.tensor([cand_list[j] for j in indices])
        
        neg_src = batch_src.unsqueeze(1).expand(-1, num_neg).cpu()
        return torch.stack([neg_src.reshape(-1), neg_dst.reshape(-1)]).to(self.device)


# ==============================================================================
# MODEL CLASSES (OPTIMIZED)
# ==============================================================================

class LightGCN(nn.Module):
    """LightGCN with precomputed normalization for efficiency."""
    
    def __init__(self, num_nodes: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))
        
        # Cache for normalized edge weights
        self._cached_norm = None
        self._cached_edge_index = None

    def _compute_norm(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute and cache normalization coefficients."""
        row, col = edge_index
        deg = degree(col, self.num_nodes).clamp(min=1)
        norm = 1.0 / torch.sqrt(deg[row] * deg[col])
        return norm

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        # Check if we can use cached normalization
        if self._cached_edge_index is not None and torch.equal(edge_index, self._cached_edge_index):
            norm = self._cached_norm
        else:
            norm = self._compute_norm(edge_index)
            self._cached_norm = norm
            self._cached_edge_index = edge_index
        
        row, col = edge_index
        x = self.embedding.weight
        out = x
        
        for _ in range(self.num_layers):
            messages = norm.unsqueeze(1) * x[col]
            x = scatter_add(messages, row, dim=0, dim_size=self.num_nodes)
            x = x + out  # Residual
            x = F.normalize(x, p=2, dim=1)
            out = out + x
        
        return out / (self.num_layers + 1)


class GraphSAGE(nn.Module):
    """GraphSAGE with optimizations."""
    
    def __init__(self, num_nodes: int, embedding_dim: int = 64, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.normalize(x, p=2, dim=1)


class GAT(nn.Module):
    """GAT with optimizations."""
    
    def __init__(self, num_nodes: int, embedding_dim: int = 64, hidden_dim: int = 64, 
                 num_layers: int = 2, heads: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(embedding_dim, hidden_dim // heads, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return F.normalize(x, p=2, dim=1)


class BPRLoss(nn.Module):
    """BPR Loss - optimized."""
    
    def __init__(self, lambda_reg: float = 1e-5):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, 
                emb_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        if emb_params is not None and self.lambda_reg > 0:
            loss = loss + self.lambda_reg * emb_params.norm(p=2).pow(2)
        return loss


# ==============================================================================
# TRAINING AND EVALUATION (OPTIMIZED)
# ==============================================================================

def train_one_epoch_bpr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_labels: torch.Tensor,
    mp_edge_index: torch.Tensor,
    neg_sampler: NegativeSampler,
    num_neg: int = 5,
    batch_size: int = 4096,
    device: str = 'cuda',
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
    verbose: bool = True
) -> float:
    """Train one epoch with BPR loss and AMP support."""
    model.train()
    train_labels = train_labels.to(device)
    mp_edge_index = mp_edge_index.to(device)
    num_pos = train_labels.size(1)
    total_loss = 0.0

    loss_fn = BPRLoss(lambda_reg=1e-5)
    num_batches = (num_pos + batch_size - 1) // batch_size
    
    # Shuffle training labels each epoch
    perm = torch.randperm(num_pos, device=device)
    train_labels = train_labels[:, perm]
    
    for batch_idx, start in enumerate(range(0, num_pos, batch_size)):
        end = min(start + batch_size, num_pos)
        batch_src = train_labels[0, start:end]
        batch_dst = train_labels[1, start:end]
        actual_batch_size = end - start

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        with autocast(enabled=use_amp):
            emb = model(mp_edge_index)
            
            # Sample negatives
            neg_edge_index = neg_sampler.sample(batch_src, num_neg)
            neg_src = neg_edge_index[0].view(actual_batch_size, num_neg)
            neg_dst = neg_edge_index[1].view(actual_batch_size, num_neg)

            # Compute scores
            pos_score = (emb[batch_src] * emb[batch_dst]).sum(dim=1, keepdim=True)
            neg_score = (emb[neg_src] * emb[neg_dst]).sum(dim=2)
            
            loss = loss_fn(pos_score, neg_score, emb_params=model.embedding.weight)
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * actual_batch_size
        
        if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}", end='\r')

    return total_loss / num_pos


@torch.no_grad()
def evaluate(
    model: nn.Module,
    mp_edge_index: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    batch_size: int = 2048,
    device: str = 'cuda'
) -> tuple:
    """Evaluate with Hit@k and Mean Rank - optimized."""
    model.eval()
    labels = labels.to(device)
    mp_edge_index = mp_edge_index.to(device)
    
    with autocast(enabled=True):
        emb = model(mp_edge_index)
    
    num_pos = labels.size(1)
    total_hits = 0.0
    total_rank = 0.0

    for start in range(0, num_pos, batch_size):
        end = min(start + batch_size, num_pos)
        batch_src = labels[0, start:end]
        batch_dst = labels[1, start:end]
        actual_batch_size = end - start

        # Compute scores efficiently
        batch_emb = emb[batch_src]
        scores = batch_emb @ emb.t()  # [B, num_nodes]

        true_scores = scores[torch.arange(actual_batch_size, device=device), batch_dst]
        rank = (scores >= true_scores.unsqueeze(1)).sum(dim=1)
        
        total_hits += (rank <= k).float().sum().item()
        total_rank += rank.float().sum().item()

    return total_hits / num_pos, total_rank / num_pos


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(data_dir: str):
    """Load train/val/test data and node mapping from directory."""
    print("Loading datasets...")
    start_time = time.time()
    
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "val_data.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), weights_only=False)
    
    print(f"Datasets loaded in {time.time() - start_time:.2f}s")

    with open(os.path.join(data_dir, "node_mapping.pkl"), "rb") as f:
        node_mapping = pickle.load(f)
    
    print(f"Train edges: {train_data.edge_index.size(1):,}, "
          f"Val edges: {val_data.edge_index.size(1):,}, "
          f"Test edges: {test_data.edge_index.size(1):,}")
    
    return train_data, val_data, test_data, node_mapping


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train_pipeline(
    data_dir: str = None,
    model_type: str = "lightgcn",
    training_mode: str = "next_link",
    use_subset: bool = True,
    subset_frac: float = 0.1,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    num_epochs: int = None,
    batch_size: int = 4096,
    learning_rate: float = 5e-3,
    num_neg: int = 5,
    eval_freq: int = None,
    device: str = None,
    use_amp: bool = True,
    compile_model: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete GNN training pipeline for link prediction.
    
    Args:
        data_dir: Path to data directory. If None, uses 'Data' folder next to this script.
        model_type: GNN architecture - "lightgcn", "graphsage", or "gat"
        training_mode: Label generation mode - "next_link" or "all_future"
        use_subset: Whether to use a subset of data for quick iteration
        subset_frac: Fraction of training edges to use if use_subset=True
        embedding_dim: Dimension of input node embeddings
        hidden_dim: Hidden dimension for GraphSAGE/GAT (ignored for LightGCN)
        num_layers: Number of GNN layers
        num_heads: Number of attention heads for GAT (ignored for others)
        num_epochs: Number of training epochs. If None, uses 100 for subset, 10 for full.
        batch_size: Training batch size
        learning_rate: Learning rate for Adam optimizer
        num_neg: Number of negative samples per positive
        eval_freq: Evaluate every N epochs. If None, uses 1 for subset, 5 for full.
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        use_amp: Use automatic mixed precision (faster on GPU)
        compile_model: Use torch.compile for PyTorch 2.0+ (can be slower first epoch)
        verbose: Whether to print training progress
        
    Returns:
        dict with 'model', 'test_hit10', 'test_mr', 'val_hit10', 'val_mr'
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Disable AMP on CPU
    if device == 'cpu':
        use_amp = False
    
    # Setup data directory
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "Data")
    
    # Load data
    train_data, val_data, test_data, node_mapping = load_data(data_dir)
    
    # ==== SETUP TRAINING DATA ====
    print("\n" + "=" * 60)
    print("SETTING UP TRAINING DATA")
    print("=" * 60)
    
    num_nodes = train_data.x.size(0)
    mp_edge_index = train_data.edge_index
    train_edge_times = train_data.edge_attr[:, 2]
    
    if use_subset:
        print(f"QUICK ITERATION MODE: Using {subset_frac * 100:.0f}% of training data")
        num_edges_to_use = int(mp_edge_index.size(1) * subset_frac)
        mp_edge_index = mp_edge_index[:, :num_edges_to_use]
        train_edge_times = train_edge_times[:num_edges_to_use]
    
    print(f"Number of nodes: {num_nodes:,}")
    print(f"Training edges: {mp_edge_index.size(1):,}")
    
    # Build training labels (optimized)
    print(f"Building training labels (mode: {training_mode})...")
    start_time = time.time()
    if training_mode == "next_link":
        train_labels = build_next_link_labels_fast(mp_edge_index, train_edge_times)
    elif training_mode == "all_future":
        train_labels = build_all_future_labels_fast(mp_edge_index, train_edge_times)
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}. Use 'next_link' or 'all_future'.")
    print(f"Training labels built in {time.time() - start_time:.2f}s")
    print(f"Number of positive pairs: {train_labels.size(1):,}")
    
    # Subset labels if too many
    max_labels_threshold = 50000 if training_mode == "next_link" else 10000
    if use_subset and train_labels.size(1) > max_labels_threshold:
        print(f"Subsetting training labels to {max_labels_threshold:,} for quick iteration")
        perm = torch.randperm(train_labels.size(1))[:max_labels_threshold]
        train_labels = train_labels[:, perm]
        print(f"Using {train_labels.size(1):,} training pairs")
    
    # Build val/test labels
    print("Building validation/test labels...")
    start_time = time.time()
    val_labels, test_labels = build_next_labels_from_splits(
        mp_edge_index, train_edge_times, val_data, test_data
    )
    print(f"Val/Test labels built in {time.time() - start_time:.2f}s")
    print(f"Validation labels: {val_labels.size(1):,}, Test labels: {test_labels.size(1):,}")
    
    # Build negative sampler (optimized)
    print("Building negative sampler...")
    start_time = time.time()
    neg_sampler = NegativeSampler(train_data.edge_index, num_nodes, device)
    print(f"Negative sampler built in {time.time() - start_time:.2f}s")
    
    # ==== INITIALIZE MODEL ====
    print("\n" + "=" * 60)
    print(f"INITIALIZING MODEL: {model_type.upper()}")
    print("=" * 60)
    
    if model_type == "lightgcn":
        model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)
    elif model_type == "graphsage":
        model = GraphSAGE(num_nodes=num_nodes, embedding_dim=embedding_dim, 
                         hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == "gat":
        model = GAT(num_nodes=num_nodes, embedding_dim=embedding_dim,
                   hidden_dim=hidden_dim, num_layers=num_layers, heads=num_heads)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lightgcn', 'graphsage', or 'gat'.")
    
    model = model.to(device)
    
    # Optionally compile model (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler() if use_amp else None
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    
    # ==== TRAINING ====
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    if num_epochs is None:
        num_epochs = 100 if use_subset else 10
    if eval_freq is None:
        eval_freq = 1 if use_subset else 5
    
    best_val_hit10 = 0.0
    mp_edge_index = mp_edge_index.to(device)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        loss = train_one_epoch_bpr(
            model, optimizer, train_labels, mp_edge_index, neg_sampler,
            num_neg=num_neg, batch_size=batch_size, device=device,
            use_amp=use_amp, scaler=scaler, verbose=verbose
        )
        
        epoch_time = time.time() - epoch_start
        if verbose:
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s | Loss: {loss:.4f}")
        
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:
            if verbose:
                print("Running validation...")
            eval_start = time.time()
            hit10, mr = evaluate(model, mp_edge_index, val_labels, k=10, batch_size=2048, device=device)
            eval_time = time.time() - eval_start
            if verbose:
                print(f"Epoch {epoch} | Loss {loss:.4f} | Val Hit@10 {hit10:.4f} | MR {mr:.1f} | Eval time: {eval_time:.2f}s")
            best_val_hit10 = max(best_val_hit10, hit10)
    
    # ==== FINAL EVALUATION ====
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    val_hit10, val_mr = evaluate(model, mp_edge_index, val_labels, k=10, batch_size=2048, device=device)
    test_hit10, test_mr = evaluate(model, mp_edge_index, test_labels, k=10, batch_size=2048, device=device)
    
    print(f"Val Hit@10: {val_hit10:.4f}, Val MeanRank: {val_mr:.1f}")
    print(f"Test Hit@10: {test_hit10:.4f}, Test MeanRank: {test_mr:.1f}")
    
    return {
        'model': model,
        'val_hit10': val_hit10,
        'val_mr': val_mr,
        'test_hit10': test_hit10,
        'test_mr': test_mr,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    results = train_pipeline(
        model_type="lightgcn",      # "lightgcn", "graphsage", or "gat"
        training_mode="next_link",  # "next_link" or "all_future"
        use_subset=True,            # Set to False for full training
        subset_frac=0.1,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        num_epochs=100,
        batch_size=4096,
        learning_rate=5e-3,
        num_neg=5,
        use_amp=True,               # Mixed precision (faster on GPU)
        compile_model=False,        # torch.compile (PyTorch 2.0+)
    )
