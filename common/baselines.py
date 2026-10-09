"""Shared PCA/OCSVM/IsolationForest baseline harness, used by every
baselines_*.py script. They differ only in feature dir/dims, whether a
--seed CLI flag is exposed, and the results path.

Trains on safe-drive only, picks the threshold on val (Youden's J), evaluates
on test (ROC-AUC, PR-AUC, frozen-threshold P/R/F1). OCSVM/IsolationForest run
on a train subsample for speed; PCA uses the full train set.
"""
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.svm import OneClassSVM

from common.tcn_ae_eval import make_windows as _make_split_windows

SEQ_LEN = 60
FRAME_STRIDE = 1
WINDOW_STEP = 1


def make_flat_windows(feat_dir, split, n_features, label_filter=None):
    windows, labels, _ = _make_split_windows(
        feat_dir, split, SEQ_LEN, FRAME_STRIDE, WINDOW_STEP, n_features, label_filter)
    return windows.reshape(windows.shape[0], -1), labels


def freeze_threshold_and_eval(name, val_scores, val_labels, test_scores, test_labels, results):
    val_roc_auc = roc_auc_score(val_labels, val_scores)
    val_pr_auc = average_precision_score(val_labels, val_scores)

    fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    frozen_threshold = float(thresholds[best_idx])

    test_roc_auc = roc_auc_score(test_labels, test_scores)
    test_pr_auc = average_precision_score(test_labels, test_scores)
    test_preds = (test_scores >= frozen_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0)

    print("\n=== {} ===".format(name))
    print("  val_roc_auc: {:.4f}  val_pr_auc: {:.4f}".format(val_roc_auc, val_pr_auc))
    print("  frozen_threshold: {:.6f}".format(frozen_threshold))
    print("  test_roc_auc: {:.4f}  test_pr_auc: {:.4f}".format(test_roc_auc, test_pr_auc))
    print("  test_precision: {:.4f}  test_recall: {:.4f}  test_f1: {:.4f}".format(precision, recall, f1))

    results[name] = {
        "val_roc_auc": val_roc_auc, "val_pr_auc": val_pr_auc,
        "frozen_threshold": frozen_threshold,
        "test_roc_auc": test_roc_auc, "test_pr_auc": test_pr_auc,
        "test_precision": precision, "test_recall": recall, "test_f1": f1,
    }


def run(*, feat_dir, n_features, results_path, seed=42, subsample_train=8000,
        pca_n_components=20, include_seed_in_results=False):
    rng = np.random.RandomState(seed)

    train_windows, _ = make_flat_windows(feat_dir, "train", n_features)
    val_windows, val_labels = make_flat_windows(feat_dir, "val", n_features)
    test_windows, test_labels = make_flat_windows(feat_dir, "test", n_features)

    print("Train windows:", train_windows.shape)
    print("Val windows:", val_windows.shape, "anomaly frac:", val_labels.mean())
    print("Test windows:", test_windows.shape, "anomaly frac:", test_labels.mean())
    if include_seed_in_results:
        print("Seed:", seed)

    if len(train_windows) > subsample_train:
        sub_idx = rng.choice(len(train_windows), subsample_train, replace=False)
        train_sub = train_windows[sub_idx]
    else:
        train_sub = train_windows
    print("Train subsample for OCSVM/IF:", train_sub.shape)

    results = {"seed": seed} if include_seed_in_results else {}

    pca = PCA(n_components=pca_n_components, random_state=seed)
    pca.fit(train_windows)

    def pca_mse(X):
        Z = pca.transform(X)
        Xr = pca.inverse_transform(Z)
        return ((X - Xr) ** 2).mean(axis=1)

    val_scores = pca_mse(val_windows)
    test_scores = pca_mse(test_windows)
    freeze_threshold_and_eval("PCA_reconstruction", val_scores, val_labels, test_scores, test_labels, results)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(train_sub)
    val_scores = -ocsvm.decision_function(val_windows)
    test_scores = -ocsvm.decision_function(test_windows)
    freeze_threshold_and_eval("OneClassSVM", val_scores, val_labels, test_scores, test_labels, results)

    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    iso.fit(train_sub)
    val_scores = -iso.score_samples(val_windows)
    test_scores = -iso.score_samples(test_windows)
    freeze_threshold_and_eval("IsolationForest", val_scores, val_labels, test_scores, test_labels, results)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", results_path)
    return results
