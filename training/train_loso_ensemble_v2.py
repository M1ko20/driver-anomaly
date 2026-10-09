"""Score-level ensembling under LOSO: instead of concatenating Beta+hand+gaze
into one 65-dim vector (Zeta, which washes out to about the same as Beta
alone), trains three separate TCN autoencoders (Beta, Epsilon, Zeta) and
combines their anomaly scores after training.

Idea: each feature set captures a different kind of signal, and forcing them
through one shared bottleneck (Zeta) can drown out the weaker ones --
combining scores instead avoids that.

Uses the exact same fold assignment as train_loso_beta_v2.py so the three
feature sets line up per fold. Each sub-model gets its own per-fold z-score
scaler. Each sub-model's test MSE is normalized against its own val-safe
distribution before combining (mean and max)."""
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

FEATURE_SETS = [
    {"name": "beta", "raw_dir": "/home/fei/mik0542/beta_raw_v2", "n_features": 52},
    {"name": "epsilon", "raw_dir": "/home/fei/mik0542/epsilon_raw_v2", "n_features": 60},
    {"name": "zeta", "raw_dir": "/home/fei/mik0542/zeta_raw_v2", "n_features": 65},
]

SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta(raw_dir):
    with open(os.path.join(raw_dir, "meta.json")) as f:
        return json.load(f)


def make_windows(raw_dir, n_features, clip_names, meta_by_clip, label_filter=None, track=False):
    windows, win_labels, win_clip, win_action = [], [], [], []
    for clip_name in clip_names:
        m = meta_by_clip[clip_name]
        label = m["label"]
        if label_filter is not None and label != label_filter:
            continue
        feat = np.load(os.path.join(raw_dir, clip_name + ".npy"))
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
        empty = np.zeros((0, SEQ_LEN, n_features), dtype=np.float32)
        return (empty, np.array([]), [], []) if track else (empty, np.array([]))
    windows = np.stack(windows).astype(np.float32)
    win_labels = np.array(win_labels)
    if track:
        return windows, win_labels, win_clip, win_action
    return windows, win_labels


def train_one_feature_set(fs, train_clips, val_clips, test_clips, meta_by_clip):
    name, raw_dir, n_features = fs["name"], fs["raw_dir"], fs["n_features"]

    train_windows_raw, _ = make_windows(raw_dir, n_features, train_clips, meta_by_clip, label_filter="safe")
    val_windows_raw, _ = make_windows(raw_dir, n_features, val_clips, meta_by_clip, label_filter="safe")
    test_windows_raw, test_labels, test_clip_names, test_actions = make_windows(
        raw_dir, n_features, test_clips, meta_by_clip, track=True)

    flat_train = train_windows_raw.reshape(-1, n_features)
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

    model = TCNAutoencoder(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)

    val_mse = per_window_mse(model, val_windows, device)
    test_mse = per_window_mse(model, test_windows, device)

    v_mean, v_std = val_mse.mean(), max(val_mse.std(), 1e-8)
    test_mse_z = (test_mse - v_mean) / v_std

    fpr, tpr, _ = roc_curve(test_labels, test_mse)
    fold_auc = auc(fpr, tpr)

    print("    [{}] best val MSE {:.6f} | fold AUC {:.4f}".format(name, best_val, fold_auc))

    return {
        "name": name,
        "test_labels": test_labels,
        "test_clip_names": test_clip_names,
        "test_actions": test_actions,
        "test_mse_z": test_mse_z,
        "fold_auc": float(fold_auc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    meta = load_meta(FEATURE_SETS[0]["raw_dir"])
    meta_by_clip = {m["clip"]: m for m in meta}
    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Seed {} | Drivers ({}): {}".format(seed, len(drivers), drivers))

    available = set(meta_by_clip.keys())
    for fs in FEATURE_SETS:
        have = {os.path.splitext(f)[0] for f in os.listdir(fs["raw_dir"]) if f.endswith(".npy")}
        available &= have
    n_dropped = len(meta_by_clip) - len(available)
    if n_dropped:
        print("Klipy chybejici v aspon jedne feature sade (vyrazeny ze VSECH foldu): {}".format(n_dropped))

    clips_by_driver = {d: [m["clip"] for m in meta if m["driver_id"] == d and m["clip"] in available] for d in drivers}

    per_model_fold_results = {fs["name"]: [] for fs in FEATURE_SETS}
    ensemble_mean_fold_results, ensemble_max_fold_results = [], []
    pooled_labels_mean, pooled_scores_mean = [], []
    pooled_labels_max, pooled_scores_max = [], []
    all_fold_fprs_mean, all_fold_tprs_mean = [], []
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

        sub_results = []
        for fs in FEATURE_SETS:
            r = train_one_feature_set(fs, train_clips, val_clips, test_clips, meta_by_clip)
            sub_results.append(r)
            per_model_fold_results[fs["name"]].append((test_driver, r["fold_auc"]))

        ref_labels = sub_results[0]["test_labels"]
        for r in sub_results[1:]:
            assert np.array_equal(ref_labels, r["test_labels"]), \
                "Label mismatch across feature sets in fold {} - windowing misaligned!".format(fold_idx)

        stacked_z = np.stack([r["test_mse_z"] for r in sub_results], axis=0)
        ensemble_mean = stacked_z.mean(axis=0)
        ensemble_max = stacked_z.max(axis=0)

        fpr_m, tpr_m, _ = roc_curve(ref_labels, ensemble_mean)
        auc_mean = auc(fpr_m, tpr_m)
        fpr_x, tpr_x, _ = roc_curve(ref_labels, ensemble_max)
        auc_max = auc(fpr_x, tpr_x)

        print("  >> FOLD {} ENSEMBLE_MEAN AUC={:.4f} | ENSEMBLE_MAX AUC={:.4f}".format(
            fold_idx + 1, auc_mean, auc_max))

        ensemble_mean_fold_results.append((test_driver, float(auc_mean)))
        ensemble_max_fold_results.append((test_driver, float(auc_max)))
        pooled_labels_mean.extend(ref_labels.tolist())
        pooled_scores_mean.extend(ensemble_mean.tolist())
        pooled_labels_max.extend(ref_labels.tolist())
        pooled_scores_max.extend(ensemble_max.tolist())
        all_fold_fprs_mean.append(fpr_m)
        all_fold_tprs_mean.append(tpr_m)

        test_actions_arr = np.array(sub_results[0]["test_actions"])
        safe_mask = ref_labels == 0
        safe_scores = ensemble_mean[safe_mask]
        actions_here = sorted(set(a for a, l in zip(sub_results[0]["test_actions"], ref_labels) if l == 1))
        action_breakdown = {}
        for act in actions_here:
            act_mask = (test_actions_arr == act) & (ref_labels == 1)
            act_scores = ensemble_mean[act_mask]
            if len(act_scores) == 0:
                continue
            y = np.concatenate([np.zeros(len(safe_scores)), np.ones(len(act_scores))])
            s = np.concatenate([safe_scores, act_scores])
            fpr_a, tpr_a, _ = roc_curve(y, s)
            auc_a = auc(fpr_a, tpr_a)
            action_breakdown[act] = {"auc": float(auc_a), "n_windows": int(act_mask.sum())}

        diag_by_fold[test_driver] = {
            "ensemble_mean_auc": float(auc_mean),
            "ensemble_max_auc": float(auc_max),
            "per_model_auc": {r["name"]: r["fold_auc"] for r in sub_results},
            "n_test_windows": int(len(ref_labels)),
            "anomaly_frac": float(ref_labels.mean()),
            "action_breakdown": action_breakdown,
        }

    print("\n{}".format("=" * 60))
    print("VYSLEDKY LOSO ENSEMBLE (Beta+Epsilon+Zeta score fusion, seed={})".format(seed))
    print("=" * 60)

    def summarize(fold_results, label):
        vals = [a for _, a in fold_results]
        m, s = float(np.mean(vals)), float(np.std(vals))
        print("{}: mean AUC {:.4f} +/- {:.4f}".format(label, m, s))
        return m, s

    for fs in FEATURE_SETS:
        summarize(per_model_fold_results[fs["name"]], "  {} (solo, this run)".format(fs["name"]))
    mean_auc, std_auc = summarize(ensemble_mean_fold_results, "ENSEMBLE_MEAN")
    max_mean_auc, max_std_auc = summarize(ensemble_max_fold_results, "ENSEMBLE_MAX")

    pooled_fpr, pooled_tpr, _ = roc_curve(np.array(pooled_labels_mean), np.array(pooled_scores_mean))
    pooled_auc_mean = float(auc(pooled_fpr, pooled_tpr))
    pooled_fpr_x, pooled_tpr_x, _ = roc_curve(np.array(pooled_labels_max), np.array(pooled_scores_max))
    pooled_auc_max = float(auc(pooled_fpr_x, pooled_tpr_x))
    print("Pooled AUC ensemble_mean: {:.4f} | ensemble_max: {:.4f}".format(pooled_auc_mean, pooled_auc_max))

    print("\n--- Per-action breakdown (ensemble_mean) ---")
    for drv, d in diag_by_fold.items():
        print("\nDriver {} | ensemble_mean AUC {:.4f} | per_model_auc {}".format(
            drv, d["ensemble_mean_auc"], {k: round(v, 3) for k, v in d["per_model_auc"].items()}))
        for act, ab in sorted(d["action_breakdown"].items(), key=lambda kv: -kv[1]["auc"]):
            print("    {:<22} AUC={:.4f}  n_windows={}".format(act, ab["auc"], ab["n_windows"]))

    results_summary = {
        "seed": seed,
        "per_model_fold_results": {
            fs["name"]: [{"driver": d, "auc": a} for d, a in per_model_fold_results[fs["name"]]]
            for fs in FEATURE_SETS
        },
        "ensemble_mean_fold_results": [{"driver": d, "auc": a} for d, a in ensemble_mean_fold_results],
        "ensemble_max_fold_results": [{"driver": d, "auc": a} for d, a in ensemble_max_fold_results],
        "ensemble_mean_auc": mean_auc, "ensemble_mean_std": std_auc,
        "ensemble_max_auc": max_mean_auc, "ensemble_max_std": max_std_auc,
        "pooled_auc_mean": pooled_auc_mean, "pooled_auc_max": pooled_auc_max,
        "per_driver_diagnostic": diag_by_fold,
    }
    with open("/home/fei/mik0542/loso_ensemble_v2_seed{}_results.json".format(seed), "w") as f:
        json.dump(results_summary, f, indent=2)

    plt.figure(figsize=(8, 7))
    plt.plot(pooled_fpr, pooled_tpr, color="darkorange", lw=2.5,
              label="Pooled ROC ensemble_mean (AUC = {:.4f})".format(pooled_auc_mean))
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("LOSO Pooled ROC - Score Ensemble (Beta+Epsilon+Zeta)")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_pooled_ensemble_v2_seed{}.png".format(seed), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs_interp = []
    for i, (fpr_i, tpr_i) in enumerate(zip(all_fold_fprs_mean, all_fold_tprs_mean)):
        drv, auc_i = ensemble_mean_fold_results[i]
        plt.plot(fpr_i, tpr_i, lw=1, alpha=0.4, label="Fold {} {} (AUC={:.2f})".format(i + 1, drv, auc_i))
        interp_tpr = np.interp(mean_fpr, fpr_i, tpr_i)
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color="darkorange", lw=2.5,
              label="Mean ROC (AUC = {:.4f} +/- {:.4f})".format(mean_auc, std_auc))
    std_tpr = np.std(tprs_interp, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                      color="darkorange", alpha=0.15, label="+/- 1 std")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("LOSO Per-Fold ROC - Score Ensemble (Beta+Epsilon+Zeta)")
    plt.legend(loc="lower right", fontsize=8); plt.grid(True, alpha=0.3)
    plt.savefig("/home/fei/mik0542/roc_LOSO_folds_ensemble_v2_seed{}.png".format(seed), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nHotovo. Grafy: roc_LOSO_pooled_ensemble_v2_seed{0}.png, roc_LOSO_folds_ensemble_v2_seed{0}.png".format(seed))


if __name__ == "__main__":
    main()
