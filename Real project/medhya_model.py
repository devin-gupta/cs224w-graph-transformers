import torch
import pickle
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import degree
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Data")

# Load PyTorch datasets
# weights_only=False is needed for PyTorch Geometric data objects
print("Loading datasets...")
start_time = time.time()
train_data = torch.load(os.path.join(data_dir, "train_data.pt"), weights_only=False)
val_data = torch.load(os.path.join(data_dir, "val_data.pt"), weights_only=False)
test_data = torch.load(os.path.join(data_dir, "test_data.pt"), weights_only=False)
print(f"Datasets loaded in {time.time() - start_time:.2f}s")

# Load the node_mapping
with open(os.path.join(data_dir, "node_mapping.pkl"), "rb") as f:
    node_mapping = pickle.load(f)
print(f"Train edges: {train_data.edge_index.size(1)}, Val edges: {val_data.edge_index.size(1)}, Test edges: {test_data.edge_index.size(1)}")

def edges_to_sorted_lists(edge_index, edge_attr_dates):
    E = edge_index.size(1)
    srcs = edge_index[0].tolist()
    dsts = edge_index[1].tolist()
    times = edge_attr_dates.tolist()
    per_src = {}
    for i in range(E):
        s = srcs[i]; d = dsts[i]; t = times[i]
        per_src.setdefault(s, []).append((t,d,i))
    for s in per_src:
        per_src[s].sort(key=lambda x: x[0])
    return per_src

def build_any_future_labels_within_window(edge_index, edge_times):
    """Build all positive pairs (s -> d_later) for any later edge within window."""
    per_src = edges_to_sorted_lists(edge_index, edge_times)
    pos_src = []
    pos_dst = []
    for s, events in per_src.items():
        for i in range(len(events)):
            for j in range(i+1, len(events)):
                pos_src.append(s)
                pos_dst.append(events[j][1])
    if len(pos_src) == 0:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor([pos_src, pos_dst], dtype=torch.long)

def sample_hard_negative_edges_grouped(batch_src, adj, num_nodes, num_neg=5, device='cuda'):
    """
    Args:
        batch_src: Tensor of source nodes in the current batch (e.g., shape [4096])
        adj: List of lists (GLOBAL adjacency list)
    """
    neg_src_list, neg_dst_list = [], []

    # Move batch to CPU list for fast iteration, but keep tensors on device
    src_list = batch_src.cpu().tolist()
    
    # Optimize: Pre-allocate tensors on device
    batch_size = len(src_list)
    
    for idx, s in enumerate(src_list):
        # Use the GLOBAL adjacency list (adj) to find neighbors
        neighbors = adj[s]
        
        # 1. Find Hard Negatives (Neighbors of Neighbors) - optimized with set operations
        candidates = set()
        for nbr in neighbors:
            candidates.update(adj[nbr])
        
        # Remove true neighbors and self
        candidates.discard(s)
        candidates.difference_update(neighbors)
        
        # 2. Selection Logic - optimized
        if len(candidates) == 0:
            # Fallback: Random sampling if no hard negatives exist
            sampled = torch.randint(0, num_nodes, (num_neg,), device=device, dtype=torch.long)
        else:
            cand_list = list(candidates)
            if len(cand_list) >= num_neg:
                # Pick num_neg random unique ones
                indices = torch.randperm(len(cand_list), device=device)[:num_neg]
                sampled = torch.tensor([cand_list[i] for i in indices.cpu()], device=device, dtype=torch.long)
            else:
                # Not enough candidates? Repeat them to fill the spots
                cand_tensor = torch.tensor(cand_list, device=device, dtype=torch.long)
                repeats = torch.randint(0, len(cand_list), (num_neg - len(cand_list),), device=device)
                sampled = torch.cat([cand_tensor, cand_tensor[repeats]])

        neg_src_list.append(torch.full((num_neg,), s, dtype=torch.long, device=device))
        neg_dst_list.append(sampled)

    neg_src = torch.cat(neg_src_list)
    neg_dst = torch.cat(neg_dst_list)
    
    # Return edges [2, batch_size * num_neg]
    return torch.stack([neg_src, neg_dst], dim=0)

class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

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
            out.append(x)
        return torch.stack(out, dim=0).mean(dim=0)

    def score(self, emb, edge_index):
        src = emb[edge_index[0]]
        dst = emb[edge_index[1]]
        return (src * dst).sum(dim=1)

class BPRLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, pos_scores, neg_scores, emb_params=None):
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        if emb_params is not None and self.lambda_reg > 0:
            loss += self.lambda_reg * emb_params.norm(p=2).pow(2)
        return loss

def train_one_epoch_bpr(
    model,
    optimizer,
    train_labels,
    mp_edge_index,  # ADDED: Need the full graph for message passing
    num_neg=5,
    batch_size=4096,
    device='cuda'
):
    model.train()
    train_labels = train_labels.to(device)
    mp_edge_index = mp_edge_index.to(device)  # Ensure edge_index is on device
    num_pos = train_labels.size(1)
    total_loss = 0.0

    loss_fn = BPRLoss(lambda_reg=1e-4)
    
    # Generate embeddings ONCE for the entire epoch (much faster!)
    with torch.no_grad():  # Temporarily no grad for embedding generation
        emb = model(mp_edge_index)
    
    num_batches = (num_pos + batch_size - 1) // batch_size
    
    for batch_idx, start in enumerate(range(0, num_pos, batch_size)):
        end = min(start + batch_size, num_pos)
        batch_src = train_labels[0, start:end]
        batch_dst = train_labels[1, start:end]

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Sample negatives (using pre-computed embeddings)
        neg_edge_index = sample_hard_negative_edges_grouped(
            batch_src=batch_src,
            adj=global_adj, 
            num_nodes=model.num_nodes,
            num_neg=num_neg,
            device=device
        )
        neg_src = neg_edge_index[0].view(end - start, num_neg)
        neg_dst = neg_edge_index[1].view(end - start, num_neg)

        # 3. Calculate scores using pre-computed embeddings
        pos_score = (emb[batch_src] * emb[batch_dst]).sum(dim=1, keepdim=True)
        neg_score = (emb[neg_src] * emb[neg_dst]).sum(dim=2)

        # 4. Compute loss and step
        loss = loss_fn(pos_score, neg_score, emb_params=model.embedding.weight)
        loss.backward()
        optimizer.step()
        
        # Update embeddings after gradient step (for next batch)
        with torch.no_grad():
            emb = model(mp_edge_index)

        total_loss += loss.item() * (end - start)
        
        # Progress indicator
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}", end='\r')

    avg_loss = total_loss / num_pos
    return avg_loss

def build_next_labels_from_splits(train_data, val_data, test_data):
    """Construct the next-edge per node after train/val for evaluation."""
    train_end = train_data.edge_attr[:,2].max().item()
    val_end = val_data.edge_attr[:,2].max().item()

    all_edge_index = torch.cat([train_data.edge_index, val_data.edge_index, test_data.edge_index], dim=1)
    all_times = torch.cat([train_data.edge_attr[:,2], val_data.edge_attr[:,2], test_data.edge_attr[:,2]], dim=0)
    per_src = edges_to_sorted_lists(all_edge_index, all_times)

    val_src, val_dst = [], []
    test_src, test_dst = [], []

    for s, events in per_src.items():
        next_after_train = next((d for t,d,_ in events if t>train_end), None)
        if next_after_train is not None and any(t>train_end and t<=val_end for t,_,_ in events):
            val_src.append(s); val_dst.append(next_after_train)
        next_after_val = next((d for t,d,_ in events if t>val_end), None)
        if next_after_val is not None:
            test_src.append(s); test_dst.append(next_after_val)

    val_labels = torch.tensor([val_src,val_dst], dtype=torch.long) if len(val_src)>0 else torch.empty((2,0), dtype=torch.long)
    test_labels = torch.tensor([test_src,test_dst], dtype=torch.long) if len(test_src)>0 else torch.empty((2,0), dtype=torch.long)
    return val_labels, test_labels

def evaluate(model, mp_edge_index, labels, k=10, batch_size=1024, device='cuda'):
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

            batch_emb = emb[batch_src]          # [B, D]
            scores = batch_emb @ emb.t()        # [B, num_nodes]

            true_scores = scores[torch.arange(end - start), batch_dst]
            rank = (scores >= true_scores.unsqueeze(1)).sum(dim=1)
            ranks.append(rank)
            hits.append((rank <= k).float())

        ranks = torch.cat(ranks)
        hits = torch.cat(hits)

    return hits.mean().item(), ranks.float().mean().item()

print("\n" + "="*60)
print("SETTING UP TRAINING DATA")
print("="*60)

num_nodes = train_data.x.size(0)
mp_edge_index = train_data.edge_index
print(f"Number of nodes: {num_nodes:,}")
print(f"Training edges: {mp_edge_index.size(1):,}")

print("Building training labels...")
start_time = time.time()
train_labels = build_any_future_labels_within_window(
    train_data.edge_index, train_data.edge_attr[:, 2]
)
print(f"Training labels built in {time.time() - start_time:.2f}s")
print(f"Number of positive pairs: {train_labels.size(1):,}")

print("Building validation/test labels...")
start_time = time.time()
val_labels, test_labels = build_next_labels_from_splits(train_data, val_data, test_data)
print(f"Val/Test labels built in {time.time() - start_time:.2f}s")
print(f"Validation labels: {val_labels.size(1):,}, Test labels: {test_labels.size(1):,}")

# Create a global adjacency list from the FULL training set
# Optimized: Use sets for faster lookups and convert to lists at the end
print("Building global adjacency list...")
start_time = time.time()
global_adj = [set() for _ in range(num_nodes)]
edge_pairs = train_data.edge_index.t()
for u, v in edge_pairs:
    global_adj[u.item()].add(v.item())
    global_adj[v.item()].add(u.item())
# Convert sets to lists for compatibility
global_adj = [list(adj_set) for adj_set in global_adj]
print(f"Global adjacency list built in {time.time() - start_time:.2f}s")
avg_degree = sum(len(adj) for adj in global_adj) / num_nodes
print(f"Average node degree: {avg_degree:.2f}")

print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)
model = LightGCN(num_nodes=num_nodes, embedding_dim=64, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {num_params:,} parameters")
print(f"Model moved to: {next(model.parameters()).device}")

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(1):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/1")
    
    # FIXED: Pass mp_edge_index for message passing
    loss = train_one_epoch_bpr(
        model,
        optimizer,
        train_labels,
        mp_edge_index,  # CRITICAL FIX: Pass the full graph for embeddings
        num_neg=5,
        batch_size=4096,
        device=device
    )
    
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s | Loss: {loss:.4f}")

    if epoch % 5 == 0 or epoch == 49:
        print("Running validation...")
        eval_start = time.time()
        hit10, mr = evaluate(
            model,
            mp_edge_index,
            val_labels,
            k=10,
            batch_size=2048,
            device=device
        )
        eval_time = time.time() - eval_start
        print(f"Epoch {epoch} | Loss {loss:.4f} | Val Hit@10 {hit10:.4f} | MR {mr:.1f} | Eval time: {eval_time:.2f}s")

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
hit10, mr = evaluate(
    model,
    mp_edge_index,
    test_labels,
    k=10,
    batch_size=2048,
    device=device
)
print(f"Test Hit@10: {hit10:.4f}")
print(f"Test MeanRank: {mr:.1f}")
