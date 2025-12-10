import torch
import pickle
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any

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
    """Build positive pairs (s -> d_next) for immediate next edge only."""
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
    """Build all positive pairs (s -> d_later) for any later edge within window."""
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
        candidates = set()
        for nbr in neighbors:
            candidates.update(adj[nbr])
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
# METRICS COMPUTATION
# ==============================================================================

class MetricsComputer:
    """
    Compute link prediction metrics: Hit@K, Recall@K, MRR, NDCG@K, AUC, Mean Rank.
    
    For single-relevant-item evaluation (next-link prediction):
    - Hit@K = Recall@K (binary: did we rank the true item in top K?)
    - MRR = Mean Reciprocal Rank = mean(1/rank)
    - NDCG@K = Normalized DCG = 1/log2(rank+1) if rank <= K, else 0
    - AUC approximated via pos vs neg score comparison
    """
    
    def __init__(self, ks: List[int] = [10, 20, 50]):
        self.ks = sorted(ks)
    
    @torch.no_grad()
    def compute_ranking_metrics(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1024,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Compute all ranking metrics from embeddings and labels.
        
        Args:
            emb: Node embeddings [num_nodes, dim]
            labels: [2, num_labels] source-destination pairs
            batch_size: Batch size for computation
            device: Device for computation
            
        Returns:
            Dict with metrics: hit@k, mrr, ndcg@k, mean_rank for each k
        """
        labels = labels.to(device)
        emb = emb.to(device)
        num_pos = labels.size(1)
        
        if num_pos == 0:
            return {f'hit@{k}': 0.0 for k in self.ks} | {'mrr': 0.0, 'mean_rank': 0.0} | {f'ndcg@{k}': 0.0 for k in self.ks}
        
        all_ranks = []
        
        for start in range(0, num_pos, batch_size):
            end = min(start + batch_size, num_pos)
            batch_src = labels[0, start:end]
            batch_dst = labels[1, start:end]
            actual_batch_size = end - start
            
            # Compute scores for all candidates
            batch_emb = emb[batch_src]  # [B, dim]
            scores = batch_emb @ emb.t()  # [B, num_nodes]
            
            # Get true scores and compute ranks
            true_scores = scores[torch.arange(actual_batch_size, device=device), batch_dst]
            # Rank = number of items with score >= true score (1-indexed)
            ranks = (scores >= true_scores.unsqueeze(1)).sum(dim=1)
            all_ranks.append(ranks)
        
        ranks = torch.cat(all_ranks).float()
        
        # Compute metrics
        metrics = {}
        
        # Hit@K (same as Recall@K for single relevant item)
        for k in self.ks:
            metrics[f'hit@{k}'] = (ranks <= k).float().mean().item()
        
        # MRR - Mean Reciprocal Rank
        metrics['mrr'] = (1.0 / ranks).mean().item()
        
        # NDCG@K - for single relevant item: 1/log2(rank+1) if rank <= K
        for k in self.ks:
            # NDCG = DCG / IDCG, where IDCG = 1/log2(2) = 1 for single item at rank 1
            dcg = torch.where(ranks <= k, 1.0 / torch.log2(ranks + 1), torch.zeros_like(ranks))
            metrics[f'ndcg@{k}'] = dcg.mean().item()
        
        # Mean Rank
        metrics['mean_rank'] = ranks.mean().item()
        
        return metrics
    
    @torch.no_grad()
    def compute_auc(
        self,
        model: nn.Module,
        mp_edge_index: torch.Tensor,
        pos_labels: torch.Tensor,
        adj: List[List[int]],
        num_neg_samples: int = 100,
        batch_size: int = 1024,
        device: str = 'cuda'
    ) -> float:
        """
        Compute AUC by comparing positive scores vs negative scores.
        
        AUC = P(pos_score > neg_score) approximated via sampling.
        """
        model.eval()
        pos_labels = pos_labels.to(device)
        mp_edge_index = mp_edge_index.to(device)
        emb = model(mp_edge_index)
        
        num_pos = pos_labels.size(1)
        if num_pos == 0:
            return 0.5
        
        total_correct = 0
        total_comparisons = 0
        
        for start in range(0, num_pos, batch_size):
            end = min(start + batch_size, num_pos)
            batch_src = pos_labels[0, start:end]
            batch_dst = pos_labels[1, start:end]
            actual_batch_size = end - start
            
            # Positive scores
            pos_scores = (emb[batch_src] * emb[batch_dst]).sum(dim=1)  # [B]
            
            # Sample negatives and compute scores
            neg_edge_index = sample_hard_negative_edges_grouped(
                batch_src, adj, model.num_nodes, num_neg=num_neg_samples, device=device
            )
            neg_src = neg_edge_index[0].view(actual_batch_size, num_neg_samples)
            neg_dst = neg_edge_index[1].view(actual_batch_size, num_neg_samples)
            neg_scores = (emb[neg_src] * emb[neg_dst]).sum(dim=2)  # [B, num_neg]
            
            # AUC: count how often pos > neg
            # pos_scores: [B], neg_scores: [B, num_neg]
            comparisons = (pos_scores.unsqueeze(1) > neg_scores).float()
            total_correct += comparisons.sum().item()
            total_comparisons += comparisons.numel()
        
        return total_correct / max(total_comparisons, 1)


# ==============================================================================
# METRICS HISTORY TRACKER
# ==============================================================================

class MetricsHistory:
    """Track metrics over training for plotting."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.epochs = []
    
    def log(self, epoch: int, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics for an epoch."""
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        
        for key, value in metrics.items():
            full_key = f'{prefix}{key}' if prefix else key
            self.history[full_key].append(value)
    
    def get_history(self) -> Dict[str, Any]:
        """Get full history dict for plotting."""
        return {
            'epochs': self.epochs.copy(),
            **{k: v.copy() for k, v in self.history.items()}
        }
    
    def print_summary(self, epoch: int, prefix: str = ''):
        """Print a summary of metrics for the epoch."""
        print(f"\n{prefix}Metrics at epoch {epoch}:")
        for key in sorted(self.history.keys()):
            if self.history[key]:
                print(f"  {key}: {self.history[key][-1]:.4f}")


# ==============================================================================
# MODEL CLASSES
# ==============================================================================

class LightGCN(nn.Module):
    """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation."""
    
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


class GraphSAGE(nn.Module):
    """GraphSAGE: Inductive Representation Learning on Large Graphs."""
    
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
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.normalize(x, p=2, dim=1)


class GAT(nn.Module):
    """GAT: Graph Attention Networks."""
    
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=64, num_layers=2, heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (embedding_dim ** 0.5))
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(embedding_dim, hidden_dim // heads, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))

    def forward(self, edge_index):
        x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return F.normalize(x, p=2, dim=1)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss."""
    
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

def train_one_epoch_bpr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_labels: torch.Tensor,
    mp_edge_index: torch.Tensor,
    adj: List[List[int]],
    num_neg: int = 5,
    batch_size: int = 4096,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Train one epoch with BPR loss.
    
    Returns:
        Dict with 'bpr_loss' and 'auc_train' (approximate AUC on training batch)
    """
    model.train()
    train_labels = train_labels.to(device)
    mp_edge_index = mp_edge_index.to(device)
    num_pos = train_labels.size(1)
    
    total_loss = 0.0
    total_auc_correct = 0
    total_auc_comparisons = 0

    loss_fn = BPRLoss(lambda_reg=1e-5)
    num_batches = (num_pos + batch_size - 1) // batch_size
    
    for batch_idx, start in enumerate(range(0, num_pos, batch_size)):
        end = min(start + batch_size, num_pos)
        batch_src = train_labels[0, start:end]
        batch_dst = train_labels[1, start:end]
        actual_batch_size = end - start

        optimizer.zero_grad()
        emb = model(mp_edge_index)

        neg_edge_index = sample_hard_negative_edges_grouped(
            batch_src=batch_src,
            adj=adj,
            num_nodes=model.num_nodes,
            num_neg=num_neg,
            device=device
        )
        neg_src = neg_edge_index[0].view(actual_batch_size, num_neg)
        neg_dst = neg_edge_index[1].view(actual_batch_size, num_neg)

        pos_score = (emb[batch_src] * emb[batch_dst]).sum(dim=1, keepdim=True)
        neg_score = (emb[neg_src] * emb[neg_dst]).sum(dim=2)

        loss = loss_fn(pos_score, neg_score, emb_params=model.embedding.weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * actual_batch_size
        
        # Track AUC during training (pos > neg comparisons)
        with torch.no_grad():
            comparisons = (pos_score > neg_score).float()
            total_auc_correct += comparisons.sum().item()
            total_auc_comparisons += comparisons.numel()
        
        if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}", end='\r')

    return {
        'bpr_loss': total_loss / num_pos,
        'auc_train': total_auc_correct / max(total_auc_comparisons, 1)
    }


@torch.no_grad()
def compute_bpr_loss(
    model: nn.Module,
    mp_edge_index: torch.Tensor,
    labels: torch.Tensor,
    adj: List[List[int]],
    num_neg: int = 5,
    batch_size: int = 4096,
    device: str = 'cuda'
) -> float:
    """
    Compute BPR loss on a dataset without backpropagation.
    Used for validation loss tracking.
    """
    model.eval()
    labels = labels.to(device)
    mp_edge_index = mp_edge_index.to(device)
    num_pos = labels.size(1)
    
    if num_pos == 0:
        return 0.0
    
    total_loss = 0.0
    loss_fn = BPRLoss(lambda_reg=0.0)  # No regularization for eval
    
    emb = model(mp_edge_index)
    
    for start in range(0, num_pos, batch_size):
        end = min(start + batch_size, num_pos)
        batch_src = labels[0, start:end]
        batch_dst = labels[1, start:end]
        actual_batch_size = end - start
        
        neg_edge_index = sample_hard_negative_edges_grouped(
            batch_src=batch_src,
            adj=adj,
            num_nodes=model.num_nodes,
            num_neg=num_neg,
            device=device
        )
        neg_src = neg_edge_index[0].view(actual_batch_size, num_neg)
        neg_dst = neg_edge_index[1].view(actual_batch_size, num_neg)
        
        pos_score = (emb[batch_src] * emb[batch_dst]).sum(dim=1, keepdim=True)
        neg_score = (emb[neg_src] * emb[neg_dst]).sum(dim=2)
        
        loss = loss_fn(pos_score, neg_score)
        total_loss += loss.item() * actual_batch_size
    
    return total_loss / num_pos


@torch.no_grad()
def evaluate(
    model: nn.Module,
    mp_edge_index: torch.Tensor,
    labels: torch.Tensor,
    adj: List[List[int]],
    metrics_computer: MetricsComputer,
    num_neg: int = 5,
    batch_size: int = 1024,
    device: str = 'cuda',
    compute_auc: bool = True,
    compute_bpr: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation with all metrics.
    
    Returns:
        Dict with: hit@k, mrr, ndcg@k, mean_rank, auc, bpr_loss for each k in metrics_computer.ks
    """
    model.eval()
    mp_edge_index = mp_edge_index.to(device)
    emb = model(mp_edge_index)
    
    # Compute ranking metrics
    metrics = metrics_computer.compute_ranking_metrics(
        emb=emb,
        labels=labels,
        batch_size=batch_size,
        device=device
    )
    
    # Compute AUC
    if compute_auc:
        auc = metrics_computer.compute_auc(
            model=model,
            mp_edge_index=mp_edge_index,
            pos_labels=labels,
            adj=adj,
            num_neg_samples=100,
            batch_size=batch_size,
            device=device
        )
        metrics['auc'] = auc
    
    # Compute BPR loss on validation set
    if compute_bpr:
        bpr_loss = compute_bpr_loss(
            model=model,
            mp_edge_index=mp_edge_index,
            labels=labels,
            adj=adj,
            num_neg=num_neg,
            batch_size=batch_size,
            device=device
        )
        metrics['bpr_loss'] = bpr_loss
    
    return metrics


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
    eval_ks: List[int] = [10, 20, 50],
    device: str = None,
    verbose: bool = True,
    save_results_to: str = None,
    run_name: str = None,
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
        eval_ks: List of K values for Hit@K, NDCG@K metrics
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        verbose: Whether to print training progress
        save_results_to: Directory to save results. If None, results are not saved.
        run_name: Name for this run. If None, auto-generated from model_type and training_mode.
        
    Returns:
        dict with:
        - 'model': trained model
        - 'history': MetricsHistory object with all metrics over training
        - 'final_val_metrics': final validation metrics dict
        - 'final_test_metrics': final test metrics dict
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup data directory
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "Data")
    
    # Initialize metrics tracking
    metrics_computer = MetricsComputer(ks=eval_ks)
    history = MetricsHistory()
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Evaluation K values: {eval_ks}")
    
    # ==== TRAINING ====
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    if num_epochs is None:
        num_epochs = 100 if use_subset else 10
    if eval_freq is None:
        eval_freq = 1 if use_subset else 5
    
    mp_edge_index = mp_edge_index.to(device)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        if verbose:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print('='*60)
        
        # Train
        train_metrics = train_one_epoch_bpr(
            model, optimizer, train_labels, mp_edge_index, adj,
            num_neg=num_neg, batch_size=batch_size, device=device, verbose=verbose
        )
        
        # Log training metrics
        history.log(epoch, train_metrics, prefix='train_')
        
        epoch_time = time.time() - epoch_start
        if verbose:
            print(f"\n  Train Loss: {train_metrics['bpr_loss']:.4f} | "
                  f"Train AUC: {train_metrics['auc_train']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Evaluate
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:
            if verbose:
                print("  Running validation...")
            eval_start = time.time()
            
            val_metrics = evaluate(
                model, mp_edge_index, val_labels, adj, metrics_computer,
                num_neg=num_neg, batch_size=2048, device=device, 
                compute_auc=True, compute_bpr=True
            )
            history.log(epoch, val_metrics, prefix='val_')
            
            eval_time = time.time() - eval_start
            if verbose:
                k_main = eval_ks[0]
                print(f"  Val Loss: {val_metrics['bpr_loss']:.4f} | "
                      f"Val Hit@{k_main}: {val_metrics[f'hit@{k_main}']:.4f} | "
                      f"MRR: {val_metrics['mrr']:.4f} | "
                      f"AUC: {val_metrics['auc']:.4f} | "
                      f"MR: {val_metrics['mean_rank']:.1f} | "
                      f"Time: {eval_time:.2f}s")
    
    # ==== FINAL EVALUATION ====
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    final_val_metrics = evaluate(
        model, mp_edge_index, val_labels, adj, metrics_computer,
        num_neg=num_neg, batch_size=2048, device=device, 
        compute_auc=True, compute_bpr=True
    )
    
    final_test_metrics = evaluate(
        model, mp_edge_index, test_labels, adj, metrics_computer,
        num_neg=num_neg, batch_size=2048, device=device, 
        compute_auc=True, compute_bpr=True
    )
    
    print("\nValidation Metrics:")
    for k, v in sorted(final_val_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\nTest Metrics:")
    for k, v in sorted(final_test_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    results = {
        'model': model,
        'history': history.get_history(),
        'final_val_metrics': final_val_metrics,
        'final_test_metrics': final_test_metrics,
    }
    
    # Auto-save results if requested
    if save_results_to is not None:
        if run_name is None:
            run_name = f"{model_type}_{training_mode}"
        
        config = {
            'model_type': model_type,
            'training_mode': training_mode,
            'use_subset': use_subset,
            'subset_frac': subset_frac,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_neg': num_neg,
            'eval_ks': eval_ks,
        }
        save_results(results, run_name, save_results_to, config)
    
    return results


# ==============================================================================
# PLOTTING UTILITIES
# ==============================================================================

def plot_training_curves(history: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot training curves from history dict.
    
    Args:
        history: Dict from MetricsHistory.get_history()
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    epochs = history['epochs']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=14)
    
    # Plot 1: BPR Loss (Train vs Val)
    ax = axes[0, 0]
    if 'train_bpr_loss' in history:
        ax.plot(epochs, history['train_bpr_loss'], 'b-', label='Train Loss')
    if 'val_bpr_loss' in history:
        val_epochs = [e for i, e in enumerate(epochs) if i < len(history['val_bpr_loss'])]
        ax.plot(val_epochs, history['val_bpr_loss'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BPR Loss')
    ax.set_title('BPR Loss (Train vs Val)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: AUC
    ax = axes[0, 1]
    if 'train_auc_train' in history:
        ax.plot(epochs, history['train_auc_train'], 'b-', label='Train AUC')
    if 'val_auc' in history:
        val_epochs = [e for i, e in enumerate(epochs) if i < len(history['val_auc'])]
        ax.plot(val_epochs, history['val_auc'], 'r-', label='Val AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Hit@K
    ax = axes[0, 2]
    for key in sorted(history.keys()):
        if key.startswith('val_hit@'):
            k = key.split('@')[1]
            val_epochs = [e for i, e in enumerate(epochs) if i < len(history[key])]
            ax.plot(val_epochs, history[key], label=f'Hit@{k}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hit@K')
    ax.set_title('Hit@K (Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: MRR
    ax = axes[1, 0]
    if 'val_mrr' in history:
        val_epochs = [e for i, e in enumerate(epochs) if i < len(history['val_mrr'])]
        ax.plot(val_epochs, history['val_mrr'], 'g-', label='Val MRR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MRR')
    ax.set_title('Mean Reciprocal Rank')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: NDCG@K
    ax = axes[1, 1]
    for key in sorted(history.keys()):
        if key.startswith('val_ndcg@'):
            k = key.split('@')[1]
            val_epochs = [e for i, e in enumerate(epochs) if i < len(history[key])]
            ax.plot(val_epochs, history[key], label=f'NDCG@{k}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NDCG@K')
    ax.set_title('NDCG@K (Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Mean Rank
    ax = axes[1, 2]
    if 'val_mean_rank' in history:
        val_epochs = [e for i, e in enumerate(epochs) if i < len(history['val_mean_rank'])]
        ax.plot(val_epochs, history['val_mean_rank'], 'm-', label='Val Mean Rank')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Rank')
    ax.set_title('Mean Rank (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# ==============================================================================
# RESULTS SAVING AND LOADING
# ==============================================================================

def save_results(
    results: Dict[str, Any],
    run_name: str,
    save_dir: str = "results",
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save training results to disk for later comparison.
    
    Args:
        results: Dict returned from train_pipeline()
        run_name: Unique name for this run (e.g., "lightgcn_next_link")
        save_dir: Directory to save results
        config: Optional config dict to save alongside results
        
    Returns:
        Path to saved file
    """
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create saveable dict (exclude model, which isn't JSON serializable)
    save_data = {
        'run_name': run_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'history': results['history'],
        'final_val_metrics': results['final_val_metrics'],
        'final_test_metrics': results['final_test_metrics'],
    }
    
    if config is not None:
        save_data['config'] = config
    
    filepath = os.path.join(save_dir, f"{run_name}.json")
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved to {filepath}")
    
    # Also save model separately
    model_path = os.path.join(save_dir, f"{run_name}_model.pt")
    torch.save(results['model'].state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return filepath


def load_results(run_name: str, save_dir: str = "results") -> Dict[str, Any]:
    """
    Load saved results from disk.
    
    Args:
        run_name: Name of the run to load
        save_dir: Directory where results are saved
        
    Returns:
        Dict with history, final_val_metrics, final_test_metrics, config
    """
    import json
    
    filepath = os.path.join(save_dir, f"{run_name}.json")
    with open(filepath, 'r') as f:
        return json.load(f)


def list_saved_runs(save_dir: str = "results") -> List[str]:
    """List all saved run names in a directory."""
    if not os.path.exists(save_dir):
        return []
    return [f.replace('.json', '') for f in os.listdir(save_dir) if f.endswith('.json')]


def compare_runs(
    run_names: List[str],
    save_dir: str = "results",
    metrics_to_compare: List[str] = ['hit@10', 'hit@20', 'mrr', 'ndcg@10', 'auc', 'bpr_loss'],
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison plots and table for multiple runs.
    
    Args:
        run_names: List of run names to compare
        save_dir: Directory where results are saved
        metrics_to_compare: Which metrics to show in comparison
        save_path: Optional path to save comparison figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    # Load all runs
    runs = {}
    for name in run_names:
        try:
            runs[name] = load_results(name, save_dir)
        except FileNotFoundError:
            print(f"Warning: Could not find results for {name}")
    
    if not runs:
        print("No runs found to compare!")
        return
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("FINAL METRICS COMPARISON")
    print("=" * 80)
    
    # Header
    header = f"{'Metric':<15}"
    for name in runs:
        header += f"{name:<20}"
    print(header)
    print("-" * 80)
    
    # Rows for each metric (validation)
    print("\nValidation:")
    for metric in metrics_to_compare:
        row = f"  {metric:<13}"
        for name, data in runs.items():
            val = data['final_val_metrics'].get(metric, float('nan'))
            row += f"{val:<20.4f}"
        print(row)
    
    # Rows for each metric (test)
    print("\nTest:")
    for metric in metrics_to_compare:
        row = f"  {metric:<13}"
        for name, data in runs.items():
            val = data['final_test_metrics'].get(metric, float('nan'))
            row += f"{val:<20.4f}"
        print(row)
    
    # Create comparison plots
    num_metrics = len(metrics_to_compare)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=14)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(range(len(runs)))
    
    # Plot training curves for key metrics
    plot_metrics = ['val_bpr_loss', 'val_auc', 'val_hit@10', 'val_mrr', 'val_ndcg@10', 'val_mean_rank']
    titles = ['BPR Loss', 'AUC', 'Hit@10', 'MRR', 'NDCG@10', 'Mean Rank']
    
    for idx, (metric, title) in enumerate(zip(plot_metrics, titles)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        for (name, data), color in zip(runs.items(), colors):
            history = data['history']
            if metric in history:
                epochs = history['epochs'][:len(history[metric])]
                ax.plot(epochs, history[metric], '-', color=color, label=name, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison figure saved to {save_path}")
    
    plt.show()


def create_final_metrics_table(
    run_names: List[str],
    save_dir: str = "results",
    metrics: List[str] = ['hit@10', 'hit@20', 'hit@50', 'mrr', 'ndcg@10', 'ndcg@20', 'auc', 'bpr_loss', 'mean_rank'],
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create a pandas DataFrame comparing final metrics across runs.
    
    Args:
        run_names: List of run names to compare
        save_dir: Directory where results are saved
        metrics: Metrics to include
        save_path: Optional CSV path to save table
        
    Returns:
        pandas DataFrame if pandas is installed, else None
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")
        return None
    
    rows = []
    for name in run_names:
        try:
            data = load_results(name, save_dir)
            row = {'run': name, 'split': 'val'}
            row.update({m: data['final_val_metrics'].get(m, float('nan')) for m in metrics})
            rows.append(row)
            
            row = {'run': name, 'split': 'test'}
            row.update({m: data['final_test_metrics'].get(m, float('nan')) for m in metrics})
            rows.append(row)
        except FileNotFoundError:
            print(f"Warning: Could not find {name}")
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Table saved to {save_path}")
    
    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # ===========================================================================
    # EXAMPLE 1: Train a single model and save results
    # ===========================================================================
    results = train_pipeline(
        model_type="lightgcn",      # "lightgcn", "graphsage", or "gat"
        training_mode="next_link",  # "next_link" or "all_future"
        use_subset=True,            # Set to False for full training
        subset_frac=0.1,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        num_epochs=50,
        batch_size=4096,
        learning_rate=5e-3,
        num_neg=5,
        eval_ks=[10, 20, 50],       # K values for Hit@K, NDCG@K
        save_results_to="results",  # Save to results/ directory
        run_name="lightgcn_next_link",  # Name for this run
    )
    
    # Plot training curves for this run
    plot_training_curves(results['history'], save_path='results/lightgcn_next_link_curves.png')
    
    # ===========================================================================
    # EXAMPLE 2: Train multiple models and compare them
    # ===========================================================================
    # Uncomment to run all models:
    # 
    # for model_type in ["lightgcn", "graphsage", "gat"]:
    #     for training_mode in ["next_link", "all_future"]:
    #         train_pipeline(
    #             model_type=model_type,
    #             training_mode=training_mode,
    #             use_subset=True,
    #             num_epochs=50,
    #             save_results_to="results",
    #             run_name=f"{model_type}_{training_mode}",
    #         )
    # 
    # # Compare all runs
    # compare_runs(
    #     run_names=["lightgcn_next_link", "graphsage_next_link", "gat_next_link"],
    #     save_dir="results",
    #     save_path="results/model_comparison.png"
    # )
    # 
    # # Create comparison table
    # df = create_final_metrics_table(
    #     run_names=list_saved_runs("results"),
    #     save_dir="results",
    #     save_path="results/metrics_table.csv"
    # )
    # print(df)
    
    # ===========================================================================
    # EXAMPLE 3: Load and compare previously saved runs
    # ===========================================================================
    # saved_runs = list_saved_runs("results")
    # print(f"Found saved runs: {saved_runs}")
    # 
    # if len(saved_runs) >= 2:
    #     compare_runs(saved_runs, save_dir="results")
