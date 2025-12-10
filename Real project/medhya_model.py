import torch
import pickle
import os
import time

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_geometric.nn import SAGEConv, GATConv

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def edges_to_sorted_lists(edge_index, edge_attr_dates):
    """Convert edge list to per-source sorted event lists."""
    E = edge_index.size(1)
    srcs = edge_index[0].tolist()
    dsts = edge_index[1].tolist()
    times = edge_attr_dates.tolist()
    per_src = {}
    for i in range(E):
        s = srcs[i]; d = dsts[i]; t = times[i]
        per_src.setdefault(s, []).append((t, d, i))
    for s in per_src:
        per_src[s].sort(key=lambda x: x[0])
    return per_src


def build_next_link_labels(edge_index, edge_times):
    """Build positive pairs (s -> d_next) for immediate next edge only.
    
    For each source node's sequence of edges (sorted by time), creates pairs
    where each edge predicts the immediate next edge from that source.
    This aligns training with next-link prediction evaluation.
    """
    per_src = edges_to_sorted_lists(edge_index, edge_times)
    pos_src = []
    pos_dst = []
    for s, events in per_src.items():
        for i in range(len(events) - 1):
            pos_src.append(s)
            pos_dst.append(events[i + 1][1])
    if len(pos_src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([pos_src, pos_dst], dtype=torch.long)


def build_all_future_labels(edge_index, edge_times):
    """Build all positive pairs (s -> d_later) for any later edge within window.
    
    For each source node's sequence of edges (sorted by time), creates pairs
    where each edge predicts ALL future edges from that source (not just next).
    This creates O(k^2) pairs per source with k edges, providing more training signal.
    """
    per_src = edges_to_sorted_lists(edge_index, edge_times)
    pos_src = []
    pos_dst = []
    for s, events in per_src.items():
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                pos_src.append(s)
                pos_dst.append(events[j][1])
    if len(pos_src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([pos_src, pos_dst], dtype=torch.long)


def build_next_labels_from_splits(train_edge_index, train_edge_times, val_data, test_data):
    """Construct the next-edge per node after train/val for evaluation."""
    train_end = train_edge_times.max().item()
    val_end = val_data.edge_attr[:, 2].max().item()

    all_edge_index = torch.cat([train_edge_index, val_data.edge_index, test_data.edge_index], dim=1)
    all_times = torch.cat([train_edge_times, val_data.edge_attr[:, 2], test_data.edge_attr[:, 2]], dim=0)
    per_src = edges_to_sorted_lists(all_edge_index, all_times)

    val_src, val_dst = [], []
    test_src, test_dst = [], []

    for s, events in per_src.items():
        next_after_train = next((d for t, d, _ in events if t > train_end), None)
        if next_after_train is not None and any(t > train_end and t <= val_end for t, _, _ in events):
            val_src.append(s)
            val_dst.append(next_after_train)
        next_after_val = next((d for t, d, _ in events if t > val_end), None)
        if next_after_val is not None:
            test_src.append(s)
            test_dst.append(next_after_val)

    val_labels = torch.tensor([val_src, val_dst], dtype=torch.long) if len(val_src) > 0 else torch.empty((2, 0), dtype=torch.long)
    test_labels = torch.tensor([test_src, test_dst], dtype=torch.long) if len(test_src) > 0 else torch.empty((2, 0), dtype=torch.long)
    return val_labels, test_labels


def build_adjacency_list(edge_index, num_nodes):
    """Build adjacency list from edge index."""
    adj = [set() for _ in range(num_nodes)]
    edge_pairs = edge_index.t()
    for u, v in edge_pairs:
        adj[u.item()].add(v.item())
        adj[v.item()].add(u.item())
    return [list(adj_set) for adj_set in adj]


def sample_hard_negative_edges_grouped(batch_src, adj, num_nodes, num_neg=5, device='cuda'):
    """Sample hard negative edges (neighbors of neighbors that aren't direct neighbors)."""
    neg_src_list, neg_dst_list = [], []
    src_list = batch_src.cpu().tolist()
    
    for s in src_list:
        neighbors = adj[s]
        
        # Find hard negatives: neighbors of neighbors
        candidates = set()
        for nbr in neighbors:
            candidates.update(adj[nbr])
        
        # Remove true neighbors and self
        candidates.discard(s)
        candidates.difference_update(neighbors)
        
        if len(candidates) == 0:
            sampled = torch.randint(0, num_nodes, (num_neg,), device=device, dtype=torch.long)
        else:
            cand_list = list(candidates)
            if len(cand_list) >= num_neg:
                indices = torch.randperm(len(cand_list), device=device)[:num_neg]
                sampled = torch.tensor([cand_list[i] for i in indices.cpu()], device=device, dtype=torch.long)
            else:
                cand_tensor = torch.tensor(cand_list, device=device, dtype=torch.long)
                repeats = torch.randint(0, len(cand_list), (num_neg - len(cand_list),), device=device)
                sampled = torch.cat([cand_tensor, cand_tensor[repeats]])

        neg_src_list.append(torch.full((num_neg,), s, dtype=torch.long, device=device))
        neg_dst_list.append(sampled)

    neg_src = torch.cat(neg_src_list)
    neg_dst = torch.cat(neg_dst_list)
    return torch.stack([neg_src, neg_dst], dim=0)


# ==============================================================================
# MODEL CLASSES
# ==============================================================================

class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
    
    Key characteristics:
    - No learnable weight matrices (only embeddings)
    - Simple normalized message passing
    - Averages embeddings across all layers
    """
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))

    def propagate(self, edge_index, x):
        row, col = edge_index
        deg = degree(col, x.size(0)).clamp(min=1)
        norm = 1.0 / torch.sqrt(deg[row] * deg[col])
        messages = norm.unsqueeze(1) * x[col]
        out = scatter_add(messages, row, dim=0, dim_size=x.size(0))
        return out

    def forward(self, edge_index):
        x = self.embedding.weight
        out = [x]
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x)
            x = x + out[-1] if len(out) > 0 else x
            x = F.normalize(x, p=2, dim=1)
            out.append(x)
        return torch.stack(out, dim=0).mean(dim=0)

    def score(self, emb, edge_index):
        src = emb[edge_index[0]]
        dst = emb[edge_index[1]]
        return (src * dst).sum(dim=1)


class GraphSAGE(nn.Module):
    """
    GraphSAGE: Inductive Representation Learning on Large Graphs.
    
    Key characteristics:
    - Learnable weight matrices at each layer
    - Mean aggregation of neighbor features
    - ReLU activation between layers
    """
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, edge_index):
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
        return F.normalize(x, p=2, dim=1)

    def score(self, emb, edge_index):
        src = emb[edge_index[0]]
        dst = emb[edge_index[1]]
        return (src * dst).sum(dim=1)


class GAT(nn.Module):
    """
    GAT: Graph Attention Networks.
    
    Key characteristics:
    - Learnable attention weights between nodes
    - Multi-head attention for stability
    - ELU activation between layers
    """
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=64, num_layers=2, heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))
        
        self.convs = nn.ModuleList()
        # First layer: embedding_dim -> hidden_dim (with multi-head)
        self.convs.append(GATConv(embedding_dim, hidden_dim // heads, heads=heads, concat=True))
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        # Last layer: single head for final output
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))

    def forward(self, edge_index):
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.elu(x)
        return F.normalize(x, p=2, dim=1)

    def score(self, emb, edge_index):
        src = emb[edge_index[0]]
        dst = emb[edge_index[1]]
        return (src * dst).sum(dim=1)


class BPRLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1e-5):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, pos_scores, neg_scores, emb_params=None):
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        if emb_params is not None and self.lambda_reg > 0:
            loss += self.lambda_reg * emb_params.norm(p=2).pow(2)
        return loss


# ==============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ==============================================================================

def train_one_epoch_bpr(model, optimizer, train_labels, mp_edge_index, adj,
                        num_neg=5, batch_size=4096, device='cuda', verbose=True):
    """Train one epoch with BPR loss."""
    model.train()
    train_labels = train_labels.to(device)
    mp_edge_index = mp_edge_index.to(device)
    num_pos = train_labels.size(1)
    total_loss = 0.0

    loss_fn = BPRLoss(lambda_reg=1e-5)
    num_batches = (num_pos + batch_size - 1) // batch_size
    first_batch = True
    
    for batch_idx, start in enumerate(range(0, num_pos, batch_size)):
        end = min(start + batch_size, num_pos)
        batch_src = train_labels[0, start:end]
        batch_dst = train_labels[1, start:end]

        optimizer.zero_grad()
        emb = model(mp_edge_index)

        neg_edge_index = sample_hard_negative_edges_grouped(
            batch_src=batch_src,
            adj=adj,
            num_nodes=model.num_nodes,
            num_neg=num_neg,
            device=device
        )
        neg_src = neg_edge_index[0].view(end - start, num_neg)
        neg_dst = neg_edge_index[1].view(end - start, num_neg)

        pos_score = (emb[batch_src] * emb[batch_dst]).sum(dim=1, keepdim=True)
        neg_score = (emb[neg_src] * emb[neg_dst]).sum(dim=2)

        loss = loss_fn(pos_score, neg_score, emb_params=model.embedding.weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if first_batch and verbose:
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            print(f"  First batch gradient norm: {grad_norm:.6f}")
            print(f"  First batch pos_score mean: {pos_score.mean().item():.4f}, neg_score mean: {neg_score.mean().item():.4f}")
            first_batch = False
        
        optimizer.step()
        total_loss += loss.item() * (end - start)
        
        if verbose and (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}", end='\r')

    return total_loss / num_pos


def evaluate(model, mp_edge_index, labels, k=10, batch_size=1024, device='cuda'):
    """Evaluate model with Hit@k and Mean Rank metrics."""
    model.eval()
    with torch.no_grad():
        labels = labels.to(device)
        mp_edge_index = mp_edge_index.to(device)
        emb = model(mp_edge_index)
        num_pos = labels.size(1)
        hits, ranks = [], []

        for start in range(0, num_pos, batch_size):
            end = min(start + batch_size, num_pos)
            batch_src = labels[0, start:end]
            batch_dst = labels[1, start:end]

            batch_emb = emb[batch_src]
            scores = batch_emb @ emb.t()

            true_scores = scores[torch.arange(end - start, device=device), batch_dst]
            rank = (scores >= true_scores.unsqueeze(1)).sum(dim=1)
            ranks.append(rank)
            hits.append((rank <= k).float())

        ranks = torch.cat(ranks)
        hits = torch.cat(hits)

    return hits.mean().item(), ranks.float().mean().item()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(data_dir):
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
    verbose: bool = True,
):
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
        verbose: Whether to print training progress
        
    Returns:
        dict with 'model', 'test_hit10', 'test_mr', 'val_hit10', 'val_mr'
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    # Build training labels based on mode
    print(f"Building training labels (mode: {training_mode})...")
    start_time = time.time()
    if training_mode == "next_link":
        train_labels = build_next_link_labels(mp_edge_index, train_edge_times)
    elif training_mode == "all_future":
        train_labels = build_all_future_labels(mp_edge_index, train_edge_times)
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
    
    # Build adjacency list
    print("Building global adjacency list...")
    start_time = time.time()
    adj = build_adjacency_list(train_data.edge_index, num_nodes)
    print(f"Global adjacency list built in {time.time() - start_time:.2f}s")
    avg_degree = sum(len(a) for a in adj) / num_nodes
    print(f"Average node degree: {avg_degree:.2f}")
    
    # ==== INITIALIZE MODEL ====
    print("\n" + "=" * 60)
    print(f"INITIALIZING MODEL: {model_type.upper()}")
    print("=" * 60)
    
    if model_type == "lightgcn":
        model = LightGCN(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        )
    elif model_type == "graphsage":
        model = GraphSAGE(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    elif model_type == "gat":
        model = GAT(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=num_heads
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lightgcn', 'graphsage', or 'gat'.")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {next(model.parameters()).device}")
    
    # ==== TRAINING ====
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    if num_epochs is None:
        num_epochs = 100 if use_subset else 10
    if eval_freq is None:
        eval_freq = 1 if use_subset else 5
    
    best_val_hit10 = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        if epoch == 0:
            emb_before = model.embedding.weight.data.clone()
        
        loss = train_one_epoch_bpr(
            model, optimizer, train_labels, mp_edge_index, adj,
            num_neg=num_neg, batch_size=batch_size, device=device, verbose=verbose
        )
        
        if epoch == 0 and verbose:
            emb_after = model.embedding.weight.data
            emb_change = (emb_after - emb_before).abs().mean().item()
            print(f"  Embedding change after epoch 0: {emb_change:.6f}")
        
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
    # Example usage - modify these parameters as needed
    results = train_pipeline(
        model_type="lightgcn",      # "lightgcn", "graphsage", or "gat"
        training_mode="next_link",  # "next_link" or "all_future"
        use_subset=True,            # Set to False for full training
        subset_frac=0.1,            # Fraction of data when use_subset=True
        embedding_dim=64,
        hidden_dim=64,              # For GraphSAGE/GAT
        num_layers=3,
        num_heads=4,                # For GAT only
        num_epochs=100,             # Will auto-adjust if None
        batch_size=4096,
        learning_rate=5e-3,
        num_neg=5,
    )
