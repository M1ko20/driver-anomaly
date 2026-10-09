"""Split-based (train/val/test dirs with labels.json) train+eval harness for
the TCN autoencoder, shared by every train_eval_tcn_ae_*.py script. They
differ only in feature directory, feature count, windowing config, and the
strings used for checkpoint/result paths and the printed title.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader

from common.tcn_ae import TCNAutoencoder, WindowDataset, per_window_mse, train_with_early_stopping


def make_windows(feat_dir, split, seq_len, frame_stride, window_step, n_features, label_filter=None):
    """Returns (windows, window_labels, window_clip_names)."""
    labels_path = os.path.join(feat_dir, split, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    files = sorted(glob.glob(os.path.join(feat_dir, split, "*.npy")))
    windows, win_labels, win_clips = [], [], []
    for path in files:
        clip_name = os.path.splitext(os.path.basename(path))[0]
        label = labels[clip_name]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(path)
        T = feat.shape[0]
        span = (seq_len - 1) * frame_stride + 1
        if T < span:
            continue
        for start in range(0, T - span + 1, window_step):
            idx = np.arange(start, start + span, frame_stride)
            windows.append(feat[idx])
            win_labels.append(1 if label == "anomaly" else 0)
            win_clips.append(clip_name)
    if not windows:
        return np.zeros((0, seq_len, n_features), dtype=np.float32), np.array([]), []
    return np.stack(windows).astype(np.float32), np.array(win_labels), win_clips


def run(*, feat_dir, n_features, seq_len=60, frame_stride=1, window_step=1,
        batch_size=64, max_epochs=100, patience=10, lr=1e-3,
        seed, ckpt_path, results_path, title):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Device:", device, "| seed:", seed)

    train_windows, _, _ = make_windows(feat_dir, "train", seq_len, frame_stride, window_step, n_features)
    val_safe_windows, _, _ = make_windows(
        feat_dir, "val", seq_len, frame_stride, window_step, n_features, label_filter="safe")
    val_all_windows, val_all_labels, _ = make_windows(feat_dir, "val", seq_len, frame_stride, window_step, n_features)
    test_all_windows, test_all_labels, _ = make_windows(
        feat_dir, "test", seq_len, frame_stride, window_step, n_features)

    print("Train windows:", train_windows.shape)
    print("Val safe windows (early stopping):", val_safe_windows.shape)
    print("Val all windows (thresholding):", val_all_windows.shape, "anomaly frac:", val_all_labels.mean())
    print("Test all windows (final eval):", test_all_windows.shape, "anomaly frac:", test_all_labels.mean())

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=batch_size, shuffle=True)
    val_safe_loader = DataLoader(WindowDataset(val_safe_windows), batch_size=batch_size, shuffle=False)

    model = TCNAutoencoder(train_windows.shape[2]).to(device)
    best_val, _ = train_with_early_stopping(
        model, train_loader, val_safe_loader, device, lr, max_epochs, patience,
        ckpt_path=ckpt_path, val_label="val(safe)")

    model.load_state_dict(torch.load(ckpt_path))

    val_mse = per_window_mse(model, val_all_windows, device)
    test_mse = per_window_mse(model, test_all_windows, device)

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
        "seed": seed,
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

    print("\n=== Krok 7 results ({}, seed={}) ===".format(title, seed))
    for k, v in results.items():
        print("  {}: {}".format(k, v))

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", results_path)
    return results
