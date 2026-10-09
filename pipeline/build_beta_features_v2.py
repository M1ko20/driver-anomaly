"""Step 3 (Beta) on v2 data: Alpha (39) + geometric features (13) = 52 dims.

Geometric features use the fact that Step 2 (normalize_skeleton_v2.py)
centers every frame on the neck, so the neck is always (0,0) and doesn't
need storing separately.

8 distances (hand-to-face, arm extension, hand-to-hip, hand-to-hand, head
tilt) and 5 angles (elbows, shoulders, neck tilt), chosen to separate
distraction actions like phone use, texting, and reaching. Missing frames
forward-filled, standardization fit on train safe-drive frames."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from common.paths import p
from common.pose_features import load_clip_keypoints as load_clip_raw

NORM_DIR = p("pose_norm_v2")
MANIFEST_PATH = p("clips_v2_manifest.json")
OUT_DIR = p("beta_features_v2")

os.makedirs(OUT_DIR, exist_ok=True)


def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))


def angle_at(vertex, p1, p2, eps=1e-8):
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos_a = (v1 * v2).sum(axis=-1) / (n1 * n2 + eps)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.arccos(cos_a)


def build_beta(arr, kp_names):
    idx = {name: i for i, name in enumerate(kp_names)}
    xy = arr[:, :, :2]
    T = arr.shape[0]

    nose = xy[:, idx["nose"]]
    l_sh = xy[:, idx["left_shoulder"]]
    r_sh = xy[:, idx["right_shoulder"]]
    l_el = xy[:, idx["left_elbow"]]
    r_el = xy[:, idx["right_elbow"]]
    l_wr = xy[:, idx["left_wrist"]]
    r_wr = xy[:, idx["right_wrist"]]
    l_hip = xy[:, idx["left_hip"]]
    r_hip = xy[:, idx["right_hip"]]
    neck = np.zeros_like(nose)
    vertical_up = np.tile(np.array([0.0, -1.0]), (T, 1))

    distances = np.stack([
        dist(nose, l_wr), dist(nose, r_wr),
        dist(l_sh, l_wr), dist(r_sh, r_wr),
        dist(l_hip, l_wr), dist(r_hip, r_wr),
        dist(l_wr, r_wr),
        dist(nose, neck),
    ], axis=1)

    angles = np.stack([
        angle_at(l_el, l_sh, l_wr),
        angle_at(r_el, r_sh, r_wr),
        angle_at(l_sh, l_el, l_hip),
        angle_at(r_sh, r_el, r_hip),
        angle_at(neck, nose, neck + vertical_up),
    ], axis=1)

    alpha_flat = arr.reshape(T, -1)
    return np.concatenate([alpha_flat, distances, angles], axis=1).astype(np.float32)


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print("Manifest entries:", len(manifest))

    split_data = {"train": [], "val": [], "test": []}
    for entry in manifest:
        clip_name = entry["clip"]
        norm_path = os.path.join(NORM_DIR, clip_name + ".json")
        if not os.path.exists(norm_path):
            continue
        arr, kp_names = load_clip_raw(norm_path)
        if arr is None:
            continue
        feat = build_beta(arr, kp_names)
        split_data[entry["split"]].append((clip_name, feat, entry["label"]))

    for split in ["train", "val", "test"]:
        n_safe = sum(1 for _, _, l in split_data[split] if l == "safe")
        n_anom = sum(1 for _, _, l in split_data[split] if l == "anomaly")
        print("{}: {} clips (safe={}, anomaly={})".format(split, len(split_data[split]), n_safe, n_anom))

    train_safe_stack = np.concatenate(
        [feat for _, feat, label in split_data["train"] if label == "safe"], axis=0)
    mean = train_safe_stack.mean(axis=0)
    std = train_safe_stack.std(axis=0)
    std[std < 1e-6] = 1e-6

    np.savez(os.path.join(OUT_DIR, "scaler.npz"), mean=mean, std=std)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(OUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        labels = {}
        for clip_name, feat, label in split_data[split]:
            feat_std = (feat - mean) / std
            np.save(os.path.join(split_dir, clip_name + ".npy"), feat_std.astype(np.float32))
            labels[clip_name] = label
        with open(os.path.join(split_dir, "labels.json"), "w") as f:
            json.dump(labels, f, indent=2)
        print("Saved {} clips to {}".format(len(split_data[split]), split_dir))

    print("Feature dim:", train_safe_stack.shape[1], "(39 Alpha + 8 distances + 5 angles)")
    print("Done.")


if __name__ == "__main__":
    main()
