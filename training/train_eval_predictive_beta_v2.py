"""Step 5 (optional, plan-recommended): a predictive model instead of a
reconstruction autoencoder.

Motivation: reconstruction AEs can learn a "shortcut" and reconstruct
anomalies too well. Predicting future pose from past pose avoids this since
the model never sees the future directly.

Same Dense window as the TCN AE (60 frames), split into 45 frames of context
(input) and 15 frames of horizon (target). TCN encoder over the context,
bottleneck, TCN decoder unrolls the horizon. Loss and anomaly score are both
prediction MSE. Rest of the protocol matches the AE runs."""
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
SEQ_LEN = 60
CONTEXT = 45
HORIZON = SEQ_LEN - CONTEXT
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
N_FEATURES = 52

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(split, label_filter=None):
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
        feat = np.load(path)
        T = feat.shape[0]
        span = (SEQ_LEN - 1) * FRAME_STRIDE + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, WINDOW_STEP):
            idx = np.arange(start, start + span, FRAME_STRIDE)
            windows.append(feat[idx])
            win_labels.append(1 if label == "anomaly" else 0)
    if not windows:
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.array([])
    return np.stack(windows).astype(np.float32), np.array(win_labels)


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        past = w[:CONTEXT].T
        future = w[CONTEXT:].T
        return torch.from_numpy(past), torch.from_numpy(future)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        pad = dilation * 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.shape[2]] if out.shape[2] > x.shape[2] else out
        return self.act(self.bn(out))


class PredictiveTCN(nn.Module):
    def __init__(self, n_features, horizon):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            TCNBlock(n_features, 64, dilation=1),
            TCNBlock(64, 32, dilation=2),
            TCNBlock(32, 16, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            TCNBlock(16, 32, dilation=1),
            TCNBlock(32, 64, dilation=1),
            nn.Conv1d(64, n_features, kernel_size=1),
        )

    def forward(self, past):
        z = self.encoder(past)
        z = self.pool(z)
        z = z.repeat(1, 1, self.horizon)
        return self.decoder(z)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_n = 0.0, 0
    criterion = nn.MSELoss()
    with torch.set_grad_enabled(is_train):
        for past, future in loader:
            past, future = past.to(device), future.to(device)
            pred = model(past)
            loss = criterion(pred, future)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * past.shape[0]
            total_n += past.shape[0]
    return total_loss / max(total_n, 1)


def per_window_pred_mse(model, windows, batch_size=256):
    model.eval()
    mses = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            past = torch.from_numpy(batch[:, :CONTEXT].transpose(0, 2, 1)).to(device)
            future = torch.from_numpy(batch[:, CONTEXT:].transpose(0, 2, 1)).to(device)
            pred = model(past)
            mse = ((pred - future) ** 2).mean(dim=(1, 2))
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses) if mses else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt_path = "/home/fei/mik0542/predictive_beta_v2_seed{}_best.pt".format(args.seed)
    results_path = "/home/fei/mik0542/krok7_results_predictive_beta_v2_seed{}.json".format(args.seed)

    print("Device:", device, "| seed:", args.seed, "| context:", CONTEXT, "horizon:", HORIZON)

    train_windows, _ = make_windows("train")
    val_safe_windows, _ = make_windows("val", label_filter="safe")
    val_all_windows, val_all_labels = make_windows("val")
    test_all_windows, test_all_labels = make_windows("test")

    print("Train windows:", train_windows.shape)
    print("Val safe windows (early stopping):", val_safe_windows.shape)
    print("Val all windows (thresholding):", val_all_windows.shape, "anomaly frac:", val_all_labels.mean())
    print("Test all windows (final eval):", test_all_windows.shape, "anomaly frac:", test_all_labels.mean())

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
    val_safe_loader = DataLoader(WindowDataset(val_safe_windows), batch_size=BATCH_SIZE, shuffle=False)

    model = PredictiveTCN(N_FEATURES, HORIZON).to(device)
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

    val_mse = per_window_pred_mse(model, val_all_windows)
    test_mse = per_window_pred_mse(model, test_all_windows)

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

    print("\n=== Krok 7 results (Beta, YOLO26-pose, Predictive TCN, context45/horizon15, seed={}) ===".format(args.seed))
    for k, v in results.items():
        print("  {}: {}".format(k, v))

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", results_path)


if __name__ == "__main__":
    main()
