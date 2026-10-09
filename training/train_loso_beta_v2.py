"""Leave-One-Subject-Out (LOSO) cross-validation for a TCN autoencoder on Beta
features (YOLO26-pose, our best feature set so far). Based on the
methodology of train_LeaveOneOut.py from an earlier attempt, same features
and architecture.

Each of the 8 drivers gets one LOSO fold: that driver is test (all clips),
one of the remaining 7 is val (safe clips only, for early stopping), and the
rest are train (safe clips only). Standardization is fit separately per
fold on train-safe frames only.

Outputs: AUC per fold, a pooled ROC curve, and a plot of all per-fold ROC
curves plus the mean curve with a +-1 std band."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.tcn_ae import TCNAutoencoder, WindowDataset, per_window_mse, run_epoch

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
N_FEATURES = 52
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_meta():
    with open(os.path.join(RAW_DIR, "meta.json")) as f:
        return json.load(f)


def make_windows(clip_names, meta_by_clip, label_filter=None):
    windows, win_labels = [], []
    for clip_name in clip_names:
        label = meta_by_clip[clip_name]["label"]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(os.path.join(RAW_DIR, clip_name + ".npy"))
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


def main():
    meta = load_meta()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Drivers ({}): {}".format(len(drivers), drivers))

    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}

    fold_results = []
    all_fold_fprs, all_fold_tprs = [], []
    pooled_labels, pooled_scores = [], []

    for fold_idx, test_driver in enumerate(drivers):
        remaining = [d for d in drivers if d != test_driver]
        val_driver = remaining[fold_idx % len(remaining)]
        train_drivers = [d for d in remaining if d != val_driver]

        print("\n{}".format("-" * 60))
        print("FOLD {}/{}: test={} val={} train={}".format(
            fold_idx + 1, len(drivers), test_driver, val_driver, train_drivers))

        train_clips = [c for d in train_drivers for c in clips_by_driver[d]]
        val_clips = clips_by_driver[val_driver]
        test_clips = clips_by_driver[test_driver]

        train_windows_raw, _ = make_windows(train_clips, meta_by_clip, label_filter="safe")
        val_windows_raw, _ = make_windows(val_clips, meta_by_clip, label_filter="safe")
        test_windows_raw, test_labels = make_windows(test_clips, meta_by_clip)

        if len(train_windows_raw) == 0 or len(val_windows_raw) == 0 or len(test_windows_raw) == 0:
            print("  PRESKAKUJI - chybi data (train={} val={} test={})".format(
                len(train_windows_raw), len(val_windows_raw), len(test_windows_raw)))
            continue

        flat_train = train_windows_raw.reshape(-1, N_FEATURES)
        mean = flat_train.mean(axis=0)
        std = flat_train.std(axis=0)
        std[std < 1e-6] = 1e-6

        def standardize(w):
            return (w - mean) / std

        train_windows = standardize(train_windows_raw)
        val_windows = standardize(val_windows_raw)
        test_windows = standardize(test_windows_raw)

        print("  Train windows: {} | Val(safe) windows: {} | Test windows: {} (anomaly frac {:.3f})".format(
            train_windows.shape[0], val_windows.shape[0], test_windows.shape[0], test_labels.mean()))

        train_loader = DataLoader(WindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(WindowDataset(val_windows), batch_size=BATCH_SIZE, shuffle=False)

        model = TCNAutoencoder(N_FEATURES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val = float("inf")
        patience_ctr = 0
        best_state = None

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device, optimizer=None)
            if epoch == 1 or epoch % 10 == 0:
                print("    Epoch {:3d} | train MSE {:.6f} | val MSE {:.6f}".format(epoch, train_loss, val_loss))

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print("    Early stopping at epoch {} (best val MSE {:.6f})".format(epoch, best_val))
                    break

        model.load_state_dict(best_state)

        test_mse = per_window_mse(model, test_windows, device)
        fpr, tpr, _ = roc_curve(test_labels, test_mse)
        fold_auc = auc(fpr, tpr)
        print("  >> FOLD {} test_driver={} AUC={:.4f}".format(fold_idx + 1, test_driver, fold_auc))

        fold_results.append((test_driver, fold_auc))
        all_fold_fprs.append(fpr)
        all_fold_tprs.append(tpr)
        pooled_labels.extend(test_labels.tolist())
        pooled_scores.extend(test_mse.tolist())

    print("\n{}".format("=" * 60))
    print("VYSLEDKY LOSO CROSS-VALIDATION (Beta, TCN AE, Dense okno)")
    print("=" * 60)
    auc_values = [r[1] for r in fold_results]
    mean_auc = float(np.mean(auc_values))
    std_auc = float(np.std(auc_values))
    print("\n{:<8} {:<10} {:<10}".format("Fold", "Ridic", "AUC"))
    for i, (drv, auc_val) in enumerate(fold_results):
        print("{:<8} {:<10} {:.4f}".format(i + 1, drv, auc_val))
    print("PRUMER: {:.4f} +/- {:.4f}".format(mean_auc, std_auc))

    pooled_labels = np.array(pooled_labels)
    pooled_scores = np.array(pooled_scores)
    pooled_fpr, pooled_tpr, _ = roc_curve(pooled_labels, pooled_scores)
    pooled_auc = auc(pooled_fpr, pooled_tpr)
    print("\nPoolovana AUC: {:.4f}".format(pooled_auc))

    results_summary = {
        "fold_results": [{"driver": d, "auc": a} for d, a in fold_results],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_auc": float(pooled_auc),
    }
    with open("/home/fei/mik0542/loso_beta_v2_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    plt.figure(figsize=(8, 7))
    plt.plot(pooled_fpr, pooled_tpr, color="darkorange", lw=2.5,
              label="Pooled ROC (AUC = {:.4f})".format(pooled_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LOSO Pooled ROC - Beta + TCN AE (YOLO26-pose)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_pooled_beta_tcn_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs_interp = []
    for i, (fpr_i, tpr_i) in enumerate(zip(all_fold_fprs, all_fold_tprs)):
        drv, auc_i = fold_results[i]
        plt.plot(fpr_i, tpr_i, lw=1, alpha=0.4, label="Fold {} {} (AUC={:.2f})".format(i + 1, drv, auc_i))
        interp_tpr = np.interp(mean_fpr, fpr_i, tpr_i)
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color="darkorange", lw=2.5,
              label="Mean ROC (AUC = {:.4f} +/- {:.4f})".format(mean_auc, std_auc))

    std_tpr = np.std(tprs_interp, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color="darkorange", alpha=0.15, label="+/- 1 std")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LOSO Per-Fold ROC - Beta + TCN AE (YOLO26-pose)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_folds_beta_tcn_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nGrafy ulozeny: roc_LOSO_pooled_beta_tcn_v2.png, roc_LOSO_folds_beta_tcn_v2.png")
    print("Hotovo.")


if __name__ == "__main__":
    main()
