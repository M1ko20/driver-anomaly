"""Long-term-window LOSO: same as train_loso_beta_diag_v2.py but with the
Long-term window (SEQ_LEN=30, FRAME_STRIDE=3, ~3.0s) instead of Dense
(60/1, ~2.0s). Checks whether the Long-term window's single-split win
(0.935->0.951 test ROC-AUC) holds up under LOSO."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

from common.tcn_ae import TCNAutoencoder, WindowDataset, per_window_mse, run_epoch

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
SEQ_LEN = 30
FRAME_STRIDE = 3
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
N_FEATURES = 52
SEED = 0
CKPT_DIR = "/home/fei/mik0542/loso_checkpoints_longterm"
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


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
    meta = load_meta()
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Drivers ({}): {}".format(len(drivers), drivers))

    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d] for d in drivers}

    fold_results = []
    diag_by_fold = {}
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
        test_windows_raw, test_labels, test_clip_names, test_actions = make_windows(
            test_clips, meta_by_clip, track=True)

        if len(train_windows_raw) == 0 or len(val_windows_raw) == 0 or len(test_windows_raw) == 0:
            print("  PRESKAKUJI - chybi data")
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
        pooled_labels.extend(test_labels.tolist())
        pooled_scores.extend(test_mse.tolist())

        torch.save({
            "model_state": best_state,
            "mean": mean, "std": std,
            "test_driver": test_driver, "val_driver": val_driver,
            "train_drivers": train_drivers, "fold_auc": fold_auc,
            "seq_len": SEQ_LEN, "frame_stride": FRAME_STRIDE, "n_features": N_FEATURES,
        }, os.path.join(CKPT_DIR, "fold_{}_{}.pt".format(fold_idx, test_driver)))

        safe_mask = test_labels == 0
        safe_scores = test_mse[safe_mask]
        actions_here = sorted(set(a for a, l in zip(test_actions, test_labels) if l == 1))
        action_breakdown = {}
        test_actions_arr = np.array(test_actions)
        for act in actions_here:
            act_mask = (test_actions_arr == act) & (test_labels == 1)
            act_scores = test_mse[act_mask]
            if len(act_scores) == 0:
                continue
            y = np.concatenate([np.zeros(len(safe_scores)), np.ones(len(act_scores))])
            s = np.concatenate([safe_scores, act_scores])
            fpr_a, tpr_a, _ = roc_curve(y, s)
            auc_a = auc(fpr_a, tpr_a)
            action_breakdown[act] = {"auc": float(auc_a), "n_windows": int(act_mask.sum()),
                                      "n_clips": len(set(c for c, a, l in zip(test_clip_names, test_actions, test_labels) if a == act and l == 1))}

        diag_by_fold[test_driver] = {
            "fold_auc": float(fold_auc),
            "n_test_windows": int(len(test_labels)),
            "anomaly_frac": float(test_labels.mean()),
            "action_breakdown": action_breakdown,
        }

    auc_values = [a for _, a in fold_results]
    mean_auc = float(np.mean(auc_values))
    std_auc = float(np.std(auc_values))
    pooled_fpr, pooled_tpr, _ = roc_curve(np.array(pooled_labels), np.array(pooled_scores))
    pooled_auc = float(auc(pooled_fpr, pooled_tpr))

    with open("/home/fei/mik0542/loso_longterm_diagnostic.json", "w") as f:
        json.dump({
            "mean_auc": mean_auc, "std_auc": std_auc, "pooled_auc": pooled_auc,
            "per_driver": diag_by_fold,
        }, f, indent=2)

    print("\n{}".format("=" * 60))
    for drv, d in diag_by_fold.items():
        print("\nDriver {} | fold AUC {:.4f} | n_windows {} | anomaly_frac {:.3f}".format(
            drv, d["fold_auc"], d["n_test_windows"], d["anomaly_frac"]))
        for act, ab in sorted(d["action_breakdown"].items(), key=lambda kv: -kv[1]["auc"]):
            print("    {:<22} AUC={:.4f}  n_windows={:<6} n_clips={}".format(
                act, ab["auc"], ab["n_windows"], ab["n_clips"]))

    print("\nMEAN AUC {:.4f} +/- {:.4f} | POOLED AUC {:.4f}".format(mean_auc, std_auc, pooled_auc))
    print("Saved: loso_longterm_diagnostic.json, checkpoints in {}".format(CKPT_DIR))


if __name__ == "__main__":
    main()
