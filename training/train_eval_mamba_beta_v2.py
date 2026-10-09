"""Step 4+5+6+7 (Beta, v2 data): a "Mamba"-style state-space autoencoder.

Important: the server only has a GTX 1070 (Pascal, sm_61), which the
official mamba-ssm CUDA kernels don't support well. Uses a simplified
stand-in instead: a diagonal linear state-space block (h_t = a*h_{t-1} +
(1-a)*x_t, learnable per-channel a), which is NOT input-selective like real
Mamba/S6 -- a fair comparison of "does a recurrent/stateful approach help
over TCN convolution," not a genuine Mamba result.

Uses a Long-term window (SEQ_LEN=30, FRAME_STRIDE=3, ~3.0s) instead of the
planned stride=6, since stride=6 would drop over 60% of anomaly clips. Rest
of the protocol matches the TCN AE: train on safe-drive, early stop on
val-safe MSE, threshold from Youden's J on val, frozen for test."""
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

FEAT_DIR = "/home/fei/mik0542/beta_features_v2"
SEQ_LEN = 30
FRAME_STRIDE = 3
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(split, label_filter=None):
    labels_path = os.path.join(FEAT_DIR, split, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    files = sorted(glob.glob(os.path.join(FEAT_DIR, split, "*.npy")))
    windows, win_labels, win_clips = [], [], []
    for path in files:
        clip_name = os.path.splitext(os.path.basename(path))[0]
        label = labels[clip_name]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(path)
        T = feat.shape[0]
        span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, WINDOW_STEP):
            idx = np.arange(start, start + span, FRAME_STRIDE)
            windows.append(feat[idx])
            win_labels.append(1 if label == "anomaly" else 0)
            win_clips.append(clip_name)
    if not windows:
        return np.zeros((0, SEQ_LEN, 52), dtype=np.float32), np.array([]), []
    return np.stack(windows).astype(np.float32), np.array(win_labels), win_clips


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i].T)


class SimpleSSMBlock(nn.Module):
    """Diagonalni linearni rekurence per-channel: h_t = a*h_{t-1} + (1-a)*x_t.
    Zjednodusena, NE-selektivni (a nezavisi na vstupu) nahrada za Mamba/S6."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, out_ch)
        self.a_logit = nn.Parameter(torch.zeros(out_ch))
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        xt = x.transpose(1, 2)
        u = self.in_proj(xt)
        a = torch.sigmoid(self.a_logit)
        B_, T, C = u.shape
        h = torch.zeros(B_, C, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = a * h + (1 - a) * u[:, t, :]
            outs.append(h)
        y = torch.stack(outs, dim=1).transpose(1, 2)
        return self.act(self.bn(y))


class MambaLikeAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            SimpleSSMBlock(n_features, 64),
            SimpleSSMBlock(64, 32),
            SimpleSSMBlock(32, 16),
        )
        self.decoder = nn.Sequential(
            SimpleSSMBlock(16, 32),
            SimpleSSMBlock(32, 64),
            nn.Conv1d(64, n_features, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
            batch = torch.from_numpy(windows[i:i + batch_size].transpose(0, 2, 1)).to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses) if mses else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt_path = "/home/fei/mik0542/mamba_beta_v2_seed{}_best.pt".format(args.seed)
    results_path = "/home/fei/mik0542/krok7_results_mamba_beta_v2_seed{}.json".format(args.seed)

    print("Device:", device, "| seed:", args.seed)

    train_windows, _, _ = make_windows("train")
    val_safe_windows, _, _ = make_windows("val", label_filter="safe")
    val_all_windows, val_all_labels, _ = make_windows("val")
    test_all_windows, test_all_labels, _ = make_windows("test")

    print("Train windows:", train_windows.shape)
    print("Val safe windows (early stopping):", val_safe_windows.shape)
    print("Val all windows (thresholding):", val_all_windows.shape, "anomaly frac:", val_all_labels.mean())
    print("Test all windows (final eval):", test_all_windows.shape, "anomaly frac:", test_all_labels.mean())

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
    val_safe_loader = DataLoader(WindowDataset(val_safe_windows), batch_size=BATCH_SIZE, shuffle=False)

    n_features = train_windows.shape[2]
    model = MambaLikeAutoencoder(n_features).to(device)
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

    print("\n=== Krok 7 results (Beta, YOLO26-pose, Mamba-like AE, Long-term(30/3) window, seed={}) ===".format(args.seed))
    for k, v in results.items():
        print("  {}: {}".format(k, v))

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", results_path)


if __name__ == "__main__":
    main()
