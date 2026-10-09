"""LOSO cross-validation for a TCN autoencoder on Epsilon features (Beta 52 +
hand geometry 8 from easy_ViTPose wholebody = 60 dims). Same protocol as
train_loso_beta_v2.py, plus a per-action AUC breakdown to see the effect of
hand features on texting/radio specifically."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.tcn_ae import TCNAutoencoder, WindowDataset, per_window_mse, run_epoch

RAW_DIR = "/home/fei/mik0542/epsilon_raw_v2"
SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
N_FEATURES = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta():
    with open(os.path.join(RAW_DIR, "meta.json")) as f:
        return json.load(f)


def make_windows(clip_names, meta_by_clip, label_filter=None, track=False):
    windows, win_labels, win_clip, win_action = [], [], [], []
    for clip_name in clip_names:
        m = meta_by_clip[clip_name]
        label = m["label"]
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
            if track:
                win_clip.append(clip_name)
                win_action.append(m["action"])
    if not windows:
        empty = np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32)
        return (empty, np.array([]), [], []) if track else (empty, np.array([]))
    windows = np.stack(windows).astype(np.float32)
    win_labels = np.array(win_labels)
    if track:
        return windows, win_labels, win_clip, win_action
    return windows, win_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    meta = load_meta()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Seed {} | Drivers ({}): {}".format(seed, len(drivers), drivers))

    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}

    fold_results = []
    all_fold_fprs, all_fold_tprs = [], []
    pooled_labels, pooled_scores = [], []
    diag_by_fold = {}

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
        test_windows_raw, test_labels, test_clip_names, test_actions = make_windows(
            test_clips, meta_by_clip, track=True)

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

        safe_mask = test_labels == 0
        safe_scores = test_mse[safe_mask]
        test_actions_arr = np.array(test_actions)
        actions_here = sorted(set(a for a, l in zip(test_actions, test_labels) if l == 1))
        action_breakdown = {}
        for act in actions_here:
            act_mask = (test_actions_arr == act) & (test_labels == 1)
            act_scores = test_mse[act_mask]
            if len(act_scores) == 0:
                continue
            y = np.concatenate([np.zeros(len(safe_scores)), np.ones(len(act_scores))])
            s = np.concatenate([safe_scores, act_scores])
            fpr_a, tpr_a, _ = roc_curve(y, s)
            auc_a = auc(fpr_a, tpr_a)
            action_breakdown[act] = {"auc": float(auc_a), "n_windows": int(act_mask.sum())}

        diag_by_fold[test_driver] = {
            "fold_auc": float(fold_auc),
            "n_test_windows": int(len(test_labels)),
            "anomaly_frac": float(test_labels.mean()),
            "action_breakdown": action_breakdown,
        }

    print("\n{}".format("=" * 60))
    print("VYSLEDKY LOSO CROSS-VALIDATION (Epsilon = Beta+hand, TCN AE, Dense okno, seed={})".format(seed))
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

    print("\n--- Per-action breakdown ---")
    for drv, d in diag_by_fold.items():
        print("\nDriver {} | fold AUC {:.4f} | n_windows {} | anomaly_frac {:.3f}".format(
            drv, d["fold_auc"], d["n_test_windows"], d["anomaly_frac"]))
        for act, ab in sorted(d["action_breakdown"].items(), key=lambda kv: -kv[1]["auc"]):
            print("    {:<22} AUC={:.4f}  n_windows={}".format(act, ab["auc"], ab["n_windows"]))

    results_summary = {
        "seed": seed,
        "fold_results": [{"driver": d, "auc": a} for d, a in fold_results],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_auc": float(pooled_auc),
        "per_driver_diagnostic": diag_by_fold,
    }
    with open("/home/fei/mik0542/loso_epsilon_v2_seed{}_results.json".format(seed), "w") as f:
        json.dump(results_summary, f, indent=2)

    plt.figure(figsize=(8, 7))
    plt.plot(pooled_fpr, pooled_tpr, color="darkorange", lw=2.5,
              label="Pooled ROC (AUC = {:.4f})".format(pooled_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LOSO Pooled ROC - Epsilon (Beta+hand) + TCN AE (YOLO26 + easy_ViTPose)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_pooled_epsilon_v2_seed{}.png".format(seed), dpi=150, bbox_inches="tight")
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
    plt.title("LOSO Per-Fold ROC - Epsilon (Beta+hand) + TCN AE (YOLO26 + easy_ViTPose)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_folds_epsilon_v2_seed{}.png".format(seed), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nGrafy ulozeny: roc_LOSO_pooled_epsilon_v2_seed{0}.png, roc_LOSO_folds_epsilon_v2_seed{0}.png".format(seed))
    print("Hotovo.")


if __name__ == "__main__":
    main()
