#!/usr/bin/env python3
"""GIN-based Logic Equivalence Checking (LEC)"""
import os, sys, csv, glob, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import degree
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import PolarGate
from load_dataset import OrderedData, calculate_forward_level
from tqdm import tqdm

GATE_TO_INDEX = {'PI': 0, 'AND': 1, 'NOT': 2}
CACHE_DIR = "output/pyg_cache"


def _read_delta(f):
    result, shift = 0, 0
    while True:
        byte = f.read(1)
        if not byte:
            return 0
        val = byte[0]
        result |= (val & 0x7F) << shift
        if (val & 0x80) == 0:
            break
        shift += 7
    return result

def topological_sort(graph):
    "按照拓扑排序重新排列，******注意：只重新映射了要用到的属性，其他属性不变******"
    reorder_id = []     # new_id2old_id
    reorder_dist = {}   # old_id2new_id
    queue = []

    edge_index = graph.edge_index
    indegree = degree(edge_index[1], num_nodes=graph.num_nodes)
    src_dst_dist = {} # 空间换时间
    for i in range(edge_index.shape[1]):
        src = edge_index[0][i].item()
        dst = edge_index[1][i].item()
        if src not in src_dst_dist:
            src_dst_dist[src] = []
        src_dst_dist[src].append(dst)
    queue.extend(torch.where(indegree == 0)[0].tolist())

    with tqdm(total=graph.num_nodes, desc="Reordering graph", leave=False) as pbar:
        while queue:
            node = queue.pop(0)
            reorder_id.append(node)
            reorder_dist[node] = len(reorder_id) - 1

            target = src_dst_dist.get(node, [])
            for t in target:
                indegree[t] -= 1
                if indegree[t] == 0:
                    queue.append(t)
            pbar.update(1)
    assert len(reorder_id) == graph.num_nodes
    reorder_id = torch.tensor(reorder_id, dtype=torch.long)
    return reorder_id, reorder_dist

def Data2OrderedData(data):

    # print("  Converting to OrderedData...")
    # 重新拓扑排序
    reorder_id, reorder_dist = topological_sort(data)
    x_torch = data.x[reorder_id]
    edge_index_torch = torch.tensor([[reorder_dist[data.edge_index[0][i].item()], reorder_dist[data.edge_index[1][i].item()]] for i in range(data.edge_index.shape[1])], dtype=torch.long).t().contiguous()
    
    # 转化成OrderedData
    edge_sign_and = x_torch[:,1]
    edge_sign_not = x_torch[:,2]
    edge_sign_node = edge_sign_and * 1 + edge_sign_not * -1
    edge_sign = edge_sign_node[edge_index_torch[1]]  # source node's sign determines the edge sign

    assert edge_index_torch.shape[0] == 2  # edge_index_torch: [2, num_edges]
    edge_sign_torch = edge_sign.to(torch.long)

    forward_level, forward_index = calculate_forward_level(edge_index_torch, x_torch.shape[0])

    graph = OrderedData(edge_index=edge_index_torch, edge_sign=edge_sign_torch, x=x_torch,
                        forward_level=forward_level, forward_index=forward_index)

    # get left/right neibor ID seq
    ID_seq = torch.zeros((2, graph.num_nodes), dtype=torch.long)
    ID_seq_mask = torch.zeros((2, graph.num_nodes), dtype=torch.bool)
    for i in tqdm(range(graph.edge_index.shape[1]), desc="Get ID_seq", leave=False):
        target = graph.edge_index[1][i]
        source = graph.edge_index[0][i]

        if ID_seq_mask[0][target] == False:
            ID_seq[0][target] = source
            ID_seq_mask[0][target] = True
        elif ID_seq_mask[1][target] == False:
            ID_seq[1][target] = source
            ID_seq_mask[1][target] = True
        else:
            assert False, f"Node {target} has more than 2 inputs!"
    graph.ID_seq = ID_seq
    graph.ID_seq_mask = ID_seq_mask

    return graph

def parse_aig_to_pyg(aig_path):
    try:
        with open(aig_path, 'rb') as f:
            header_line = f.readline().decode('ascii').strip()
            parts = header_line.split()
            if parts[0] != 'aig' or len(parts) < 6:
                return None
            n_vars, n_inputs = int(parts[1]), int(parts[2])
            n_latch, n_outputs, n_ands = int(parts[3]), int(parts[4]), int(parts[5])
            if n_latch != 0:
                return None
            x_data = [GATE_TO_INDEX['PI']] * n_inputs + [GATE_TO_INDEX['AND']] * n_ands
            edge_index = []
            for _ in range(n_outputs):
                f.readline()
            has_not = [-1] * (n_inputs + n_ands)
            for i in range(n_ands):
                d0, d1 = _read_delta(f), _read_delta(f)
                and_lit = 2 * (i + 1 + n_inputs)
                f1_lit = and_lit - d0
                f2_lit = and_lit - d0 - d1
                f1_idx, f1_neg = f1_lit // 2 - 1, f1_lit % 2
                f2_idx, f2_neg = f2_lit // 2 - 1, f2_lit % 2
                and_idx = i + n_inputs
                for fidx, fneg in [(f1_idx, f1_neg), (f2_idx, f2_neg)]:
                    src = fidx
                    if fneg and src >= 0:
                        if has_not[src] == -1:
                            not_idx = len(x_data)
                            x_data.append(GATE_TO_INDEX['NOT'])
                            edge_index.append([src, not_idx])
                            has_not[src] = not_idx
                        src = has_not[src]
                    if src >= 0:
                        edge_index.append([src, and_idx])
        if not x_data:
            return None
        num_types = len(GATE_TO_INDEX)
        x_tensor = torch.zeros(len(x_data), num_types, dtype=torch.float)
        for i, t in enumerate(x_data):
            x_tensor[i, t] = 1.0
        ei = (torch.tensor(edge_index, dtype=torch.long).t().contiguous()
              if edge_index else torch.zeros(2, 0, dtype=torch.long))
        return Data(x=x_tensor, edge_index=ei)
    except Exception:
        return None

def _pairs_txt_path(split_dir):
    return os.path.join(split_dir, "pairs.txt")


def _generate_pairs_txt(split_dir, orig_dir):
    designs = sorted([d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))])
    design_variants = {}
    for name in designs:
        orig_path = os.path.join(orig_dir, f"{name}.aig")
        if not os.path.exists(orig_path):
            print(f"  Warning: original AIG not found for {name}, skipping")
            continue
        variants = sorted(glob.glob(os.path.join(split_dir, name, "variant_*.aig")))
        if variants:
            design_variants[name] = (orig_path, variants)
    valid_names = sorted(design_variants.keys())
    # Collect all variant paths (not originals) for negative sampling
    all_variant_pools = {}
    for name in valid_names:
        _, variants = design_variants[name]
        all_variant_pools[name] = variants
    lines = []
    for name in valid_names:
        orig_path, variants = design_variants[name]
        # Positive: orig vs each variant (same design)
        for vp in variants:
            lines.append("1 " + name + " " + orig_path + " " + vp)
        # Negative: orig vs random variant from a different design
        other_names = [n for n in valid_names if n != name]
        n_pos = len(variants)
        neg_count = 0
        while neg_count < n_pos:
            other = random.choice(other_names)
            vp_other = random.choice(all_variant_pools[other])
            lines.append("0 " + name + " " + orig_path + " " + vp_other)
            neg_count += 1
    random.shuffle(lines)
    txt_path = _pairs_txt_path(split_dir)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Created {txt_path}({len(lines)} pairs)")
    return txt_path

def load_split_data(model_name,split_dir, orig_dir):
    txt_path = _pairs_txt_path(split_dir)
    if not os.path.exists(txt_path):
        print(f"  pairs.txt not found, generating...")
        _generate_pairs_txt(split_dir, orig_dir)
    design_pairs = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_s, circuit, path_a, path_b = line.split(None, 3)
            g_a = load_or_parse(model_name,path_a)
            g_b = load_or_parse(model_name,path_b)
            if g_a is None or g_b is None:
                continue
            design_pairs.setdefault(circuit, []).append((g_a, g_b, int(label_s)))
    total = sum(len(v) for v in design_pairs.values())
    print(f"  Loaded {total} pairs ({len(design_pairs)}designs) from {txt_path}")
    return design_pairs

def load_or_parse(model_name, aig_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    # rel = os.path.relpath(aig_path).replace(os.sep, "__").replace("/", "__")
    rel = os.path.basename(aig_path)
    if "variant" in rel:
        parent_dir = os.path.basename(os.path.dirname(aig_path))
        rel = f"{parent_dir}__{rel}"
    base_path = os.path.join(CACHE_DIR, "PolarGate"if "PolarGate" in model_name else "")
    cache_path = os.path.join(base_path, rel.replace(".aig", ".pt"))
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    else:
        os.makedirs(base_path, exist_ok=True)

    g = parse_aig_to_pyg(aig_path)
    if "PolarGate" in model_name:
        g = Data2OrderedData(g)

    if g is not None:
        torch.save(g, cache_path)
    return g



def make_gin_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
        nn.ReLU(), nn.Linear(out_dim, out_dim))


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.convs.append(GINConv(make_gin_mlp(in_d, hidden_dim), train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.proj = nn.Linear(hidden_dim * num_layers, hidden_dim * 2)

    def forward(self, x, edge_index, batch):
        layer_outs = []
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            layer_outs.append(h)
        readouts = []
        for h_l in layer_outs:
            readouts.append(global_mean_pool(h_l, batch))
        return self.proj(torch.cat(readouts, dim=1))


class SiameseLEC(nn.Module):
    def __init__(self, args, pretrained=True, input_dim=3, hidden_dim=64, num_layers=3):
        super().__init__()
        self.encoder_name = args.model
        if "PolarGate" in self.encoder_name:
            input_dim = 3
            hidden_dim = 256
            num_layers = 9
            self.encoder = PolarGate(args, args.device, input_dim, hidden_dim, num_layers)
            print(f"Using {self.encoder_name} encoder")
        else:
            raise NotImplementedError(f"Unsupported model: {args.model}")
        
        if pretrained and args.load_pretrain_path:
            print(f"Loading pretrained weights from: {args.load_pretrain_path}")
            state_dict = torch.load(args.load_pretrain_path, weights_only=True)
            self.encoder.load_state_dict(state_dict['model_state_dict'], strict=False)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)

        emb_dim = hidden_dim * 2
        
        self.classifier = nn.Sequential( 
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, batch_a, batch_b):
        if "PolarGate" in self.encoder_name:
            a_edge_index_s = torch.cat((batch_a.edge_index.t(), batch_a.edge_sign.unsqueeze(1)), dim=1).long()
            emb_a,_ = self.encoder(batch_a.x, a_edge_index_s, batch_a)
            if "MetaLA" in self.encoder_name: # second modified
                emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_a_mean = global_mean_pool(emb_a, batch_a.batch)
            
            emb_a = self.proj(emb_a_mean)
            if "MetaLA" in self.encoder_name: # first modified
                emb_a = F.normalize(emb_a, p=2, dim=1)

            b_edge_index_s = torch.cat((batch_b.edge_index.t(), batch_b.edge_sign.unsqueeze(1)), dim=1).long()
            emb_b,_ = self.encoder(batch_b.x, b_edge_index_s, batch_b)
            if "MetaLA" in self.encoder_name:
                emb_b = F.normalize(emb_b, p=2, dim=1)
            emb_b_mean = global_mean_pool(emb_b, batch_b.batch)
            
            emb_b = self.proj(emb_b_mean)
            if "MetaLA" in self.encoder_name:
                emb_b = F.normalize(emb_b, p=2, dim=1)
        else:
            raise NotImplementedError(f"Unsupported model: {self.encoder_name}")
        
        combined = torch.cat([emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=1)
        return self.classifier(combined).squeeze(-1)


def collate_pairs(pair_list):
    graphs_a, graphs_b, labels = [], [], []
    for ga, gb, y in pair_list:
        graphs_a.append(ga)
        graphs_b.append(gb)
        labels.append(y)
    return (Batch.from_data_list(graphs_a),
            Batch.from_data_list(graphs_b),
            torch.tensor(labels, dtype=torch.float))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_a, batch_b, labels in loader:
        batch_a, batch_b = batch_a.to(device), batch_b.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_a, batch_b), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_pairs(model, pairs, device, batch_size=32):
    model.eval()
    all_labels, all_preds = [], []
    bs = batch_size
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    while bs >= 1:
        try:
            loader = DataLoader(pairs, batch_size=bs, shuffle=False,
                                collate_fn=collate_pairs)
            all_labels, all_preds = [], []
            for batch_a, batch_b, labels in loader:
                batch_a, batch_b = batch_a.to(device), batch_b.to(device)
                logits = model(batch_a, batch_b)
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = bs // 2
            if bs < 1:
                raise RuntimeError("OOM even with batch_size=1")
            print(f"  OOM, reducing eval batch_size to {bs}")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    n = max(len(all_labels), 1)
    return np.array(all_labels, dtype=int), np.array(all_preds, dtype=int), total_ms / n

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
    }


def write_results_csv(path, rows):
    """写入所有结果到单个CSV: 20个电路 + train_avg + test_avg"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ['Circuit', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Infer_Time_ms']
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r['Circuit'],
                             f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
                             f"{r['Recall']:.4f}", f"{r['F1-Score']:.4f}",
                             f"{r['Infer_Time_ms']:.2f}"])


def main():
    parser = argparse.ArgumentParser(description="GIN-based LEC")
    parser.add_argument("--train-dir", default="/home/hx/polargate_extended/gnn_v4/gnn/dataset/train_easy")
    parser.add_argument("--test-dir", default="/home/hx/polargate_extended/gnn_v4/gnn/dataset/test")
    parser.add_argument("--orig-dir", default="/home/hx/polargate_extended/gnn_v2/gnn/dataset/openlsd")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--model", type=str, choices=["PolarGate","PolarGate_MetaLA"])
    parser.add_argument("--load_pretrain_path", type=str)
    parser.add_argument("--save-dir", default="output/gnn_lec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--froze_pretrain", action="store_true",default=False)
    args = parser.parse_args()

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading training data...")
    train_design_pairs = load_split_data(args.model, args.train_dir, args.orig_dir)
    print("Loading test data...")
    test_design_pairs = load_split_data(args.model,args.test_dir, args.orig_dir)

    all_train_pairs = []
    for pairs in train_design_pairs.values():
        all_train_pairs.extend(pairs)
    all_test_pairs = []
    for pairs in test_design_pairs.values():
        all_test_pairs.extend(pairs)
    print(f"Train pairs: {len(all_train_pairs)}, Test pairs: {len(all_test_pairs)}")

    input_dim = len(GATE_TO_INDEX)
    model = SiameseLEC(args).to(device)

    if args.froze_pretrain:
        # 1. 固定 encoder 参数
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")

        # 2. 仅将需要梯度的参数传给优化器
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    print(f"Model: {args.model}, hidden={args.hidden_dim}, device={device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(all_train_pairs, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_pairs)

    # ---- Training ----
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        yt, yp, _ = evaluate_pairs(model, all_train_pairs, device, args.batch_size)
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)
        tag = ""
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            tag = " *"
        print(f"Epoch {epoch:3d}/{args.epochs}| Loss={loss:.4f} | TrainAcc={acc:.4f} F1={f1:.4f}{tag}")

    # Load best model
    best_path = os.path.join(args.save_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print("Loaded best model for evaluation.")

    # ---- Evaluation ----
    csv_rows = []
    train_names = sorted(train_design_pairs.keys())
    test_names = sorted(test_design_pairs.keys())

    def eval_and_print(title, design_pairs, names):
        rows = []
        print(f"\n{'=' * 60}")
        print(f"Per-Circuit Results ({title})")
        print('=' * 60)
        for name in names:
            pairs = design_pairs[name]
            yt, yp, t_ms = evaluate_pairs(model, pairs, device, 1)
            m = compute_metrics(yt, yp)
            m['Infer_Time_ms'] = t_ms
            m['Circuit'] = name
            rows.append(m)
            print(f"  {name:30s}| Acc={m['Accuracy']:.4f} Prec={m['Precision']:.4f}"
                  f"Rec={m['Recall']:.4f} F1={m['F1-Score']:.4f}| {t_ms:.2f}ms")
        return rows

    train_rows = eval_and_print("Train Set", train_design_pairs, train_names)
    test_rows = eval_and_print("Test Set", test_design_pairs, test_names)
    csv_rows = train_rows + test_rows

    def avg_row(rows, label):
        n = len(rows)
        if n == 0:
            return {'Circuit': label, 'Accuracy': 0, 'Precision': 0,
                    'Recall': 0, 'F1-Score': 0, 'Infer_Time_ms': 0}
        return {
            'Circuit': label,
            'Accuracy':      sum(r['Accuracy'] for r in rows) / n,
            'Precision':     sum(r['Precision'] for r in rows) / n,
            'Recall':        sum(r['Recall'] for r in rows) / n,
            'F1-Score':      sum(r['F1-Score'] for r in rows) / n,
            'Infer_Time_ms': sum(r['Infer_Time_ms'] for r in rows) / n,
        }

    tr_avg = avg_row(train_rows, 'train_avg')
    te_avg = avg_row(test_rows, 'test_avg')
    csv_rows.append(tr_avg)
    csv_rows.append(te_avg)

    for label, avg in [("Train Set", tr_avg), ("Test Set", te_avg)]:
        print(f"\n{'=' * 60}")
        print(f"{label} Average")
        print('=' * 60)
        print(f"  Accuracy: {avg['Accuracy']:.4f}")
        print(f"  Precision: {avg['Precision']:.4f}")
        print(f"  Recall: {avg['Recall']:.4f}")
        print(f"  F1-Score: {avg['F1-Score']:.4f}")
        print(f"  Infer_Time_ms: {avg['Infer_Time_ms']:.2f}")

    csv_path = os.path.join(args.save_dir, "results.csv")
    write_results_csv(csv_path, csv_rows)
    print(f"\nCSV results saved to: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())