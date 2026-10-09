"""Step 4+5+6+7: a two-stream ST-GCN autoencoder on Alpha-per-node data.

ST-GCN gets Alpha-per-node (13 joints x [x,y,conf]), never Beta (pairwise
features aren't a single-node property), with the same Dense window as the
TCN AE for a direct comparison.

Simplified 2s-AGCN style: a joints stream (spatial graph conv + temporal
conv) and a bones stream (joint-to-joint vectors, channel-mixing + temporal
conv, no spatial graph of its own). Bone latents get mapped back into joint
space and fused with the joints latents before a shared bottleneck; the
decoder mirrors the encoder to reconstruct the joints input.

Reuses the precomputed alpha_features_v2 (T,39), reshaped back to (T,13,3).
Same train/early-stop/threshold protocol as the other AE runs."""
import os
import glob
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support

FEAT_DIR = "/home/fei/mik0542/gamma_pernode_v2"
SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
V = 13
NC = 7
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
         (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
E = len(EDGES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_adjacency(V, edges, self_loop=True):
    A = np.zeros((V, V), dtype=np.float32)
    for p, c in edges:
        A[p, c] = 1.0
        A[c, p] = 1.0
    if self_loop:
        A += np.eye(V, dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-6)))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def make_node_windows(split, label_filter=None):
    labels_path = os.path.join(FEAT_DIR, split, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    files = sorted(glob.glob(os.path.join(FEAT_DIR, split, "*.npy")))
    windows, win_labels = [], []
    for path in files:
        clip_name = os.path.splitext(os.path.basename(path))[0]
        label = labels[clip_name]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(path).reshape(-1, V, NC)
        T = feat.shape[0]
        span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, WINDOW_STEP):
            idx = np.arange(start, start + span, FRAME_STRIDE)
            windows.append(feat[idx])
            win_labels.append(1 if label == "anomaly" else 0)
    if not windows:
        return np.zeros((0, SEQ_LEN, V, NC), dtype=np.float32), np.array([])
    return np.stack(windows).astype(np.float32), np.array(win_labels)


class NodeWindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        return torch.from_numpy(w.transpose(2, 0, 1))


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.gcn(x)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.tcn(x)
        return self.act(self.bn(x))


class TemporalBlock2D(nn.Module):
    """Bones stream - jen channel mixing + temporal conv, bez spatial grafu."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tcn(x)
        return self.act(self.bn(x))


class TwoStreamSTGCNAutoencoder(nn.Module):
    def __init__(self, V, edges, A_joints):
        super().__init__()
        parent_idx = [e[0] for e in edges]
        child_idx = [e[1] for e in edges]
        self.register_buffer("parent_idx", torch.tensor(parent_idx, dtype=torch.long))
        self.register_buffer("child_idx", torch.tensor(child_idx, dtype=torch.long))

        self.joint_encoder = nn.Sequential(
            STGCNBlock(NC, 32, A_joints),
            STGCNBlock(32, 16, A_joints),
        )
        self.bone_encoder = nn.Sequential(
            TemporalBlock2D(3, 32),
            TemporalBlock2D(32, 16),
        )

        incidence = np.zeros((V, len(edges)), dtype=np.float32)
        for e_idx, c in enumerate(child_idx):
            incidence[c, e_idx] = 1.0
        self.register_buffer("incidence", torch.tensor(incidence))

        self.fuse = nn.Conv2d(32, 16, kernel_size=1)
        self.bottleneck = STGCNBlock(16, 8, A_joints)

        self.decoder = nn.Sequential(
            STGCNBlock(8, 16, A_joints),
            STGCNBlock(16, 32, A_joints),
        )
        self.out_proj = nn.Conv2d(32, NC, kernel_size=1)

    def compute_bones(self, joints):
        x = joints[:, 0]
        y = joints[:, 1]
        dx = x[..., self.child_idx] - x[..., self.parent_idx]
        dy = y[..., self.child_idx] - y[..., self.parent_idx]
        length = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        return torch.stack([dx, dy, length], dim=1)

    def forward(self, joints):
        bones = self.compute_bones(joints)

        j_lat = self.joint_encoder(joints)
        b_lat = self.bone_encoder(bones)
        b_on_j = torch.einsum("bcte,ve->bctv", b_lat, self.incidence)

        fused = torch.cat([j_lat, b_on_j], dim=1)
        fused = self.fuse(fused)
        z = self.bottleneck(fused)

        out = self.decoder(z)
        return self.out_proj(out)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_n = 0.0, 0
    criterion = nn.MSELoss()
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_n += batch.shape[0]
    return total_loss / max(total_n, 1)


def per_window_mse(model, windows, batch_size=256):
    model.eval()
    mses = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i + batch_size].transpose(0, 3, 1, 2)).to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2, 3))
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses) if mses else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt_path = "/home/fei/mik0542/stgcn_ae_gammapernode_v2_seed{}_best.pt".format(args.seed)
    results_path = "/home/fei/mik0542/krok7_results_stgcn_gammapernode_v2_seed{}.json".format(args.seed)

    print("Device:", device, "| seed:", args.seed)

    train_windows, _ = make_node_windows("train")
    val_safe_windows, _ = make_node_windows("val", label_filter="safe")
    val_all_windows, val_all_labels = make_node_windows("val")
    test_all_windows, test_all_labels = make_node_windows("test")

    print("Train windows:", train_windows.shape)
    print("Val safe windows (early stopping):", val_safe_windows.shape)
    print("Val all windows (thresholding):", val_all_windows.shape, "anomaly frac:", val_all_labels.mean())
    print("Test all windows (final eval):", test_all_windows.shape, "anomaly frac:", test_all_labels.mean())

    train_loader = DataLoader(NodeWindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
    val_safe_loader = DataLoader(NodeWindowDataset(val_safe_windows), batch_size=BATCH_SIZE, shuffle=False)

    A_joints = build_adjacency(V, EDGES)
    model = TwoStreamSTGCNAutoencoder(V, EDGES, A_joints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_safe_loader, optimizer=None)
        print("Epoch {:3d} | train MSE {:.6f} | val(safe) MSE {:.6f}".format(epoch, train_loss, val_loss))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("Early stopping at epoch {} (best val(safe) MSE {:.6f})".format(epoch, best_val))
                break

    model.load_state_dict(torch.load(ckpt_path))

    val_mse = per_window_mse(model, val_all_windows)
    test_mse = per_window_mse(model, test_all_windows)

    val_roc_auc = roc_auc_score(val_all_labels, val_mse)
    val_pr_auc = average_precision_score(val_all_labels, val_mse)

    fpr, tpr, thresholds = roc_curve(val_all_labels, val_mse)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    frozen_threshold = float(thresholds[best_idx])

    test_roc_auc = roc_auc_score(test_all_labels, test_mse)
    test_pr_auc = average_precision_score(test_all_labels, test_mse)
    test_preds = (test_mse >= frozen_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_all_labels, test_preds, average="binary", zero_division=0)

    results = {
        "seed": args.seed,
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
        "frozen_threshold": frozen_threshold,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "best_val_mse_early_stopping": best_val,
    }

    print("\n=== Krok 7 results (Gamma-per-node, YOLO26-pose, two-stream ST-GCN AE, Dense window, seed={}) ===".format(args.seed))
    for k, v in results.items():
        print("  {}: {}".format(k, v))

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", results_path)


if __name__ == "__main__":
    main()
