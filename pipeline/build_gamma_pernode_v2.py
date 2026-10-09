"""Step 3 (Gamma-per-node) on v2 data, for ST-GCN rather than TCN/Mamba.

ST-GCN needs per-node channels, not Beta's pairwise features. Builds a
(T, 13, 7) shape: (x, y, conf, vx, vy, ax, ay) per joint, with the same
Savitzky-Golay smoothing before differentiation as the flat Gamma
features."""
import json
import os
import numpy as np
from scipy.signal import savgol_filter

NORM_DIR = "/home/fei/mik0542/pose_norm_v2"
MANIFEST_PATH = "/home/fei/mik0542/clips_v2_manifest.json"
OUT_DIR = "/home/fei/mik0542/gamma_pernode_v2"

SG_WINDOW = 5
SG_POLYORDER = 2
V = 13

os.makedirs(OUT_DIR, exist_ok=True)


def load_clip_raw(path):
    with open(path) as f:
        data = json.load(f)

    kp_names = data["keypoint_names"]
    n_kp = len(kp_names)
    n_frames = len(data["frames"])

    arr = np.full((n_frames, n_kp, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(n_frames, dtype=bool)

    for fr in data["frames"]:
        idx = fr["frame_id"]
        if fr["detected"]:
            for ki, kp in enumerate(fr["keypoints"]):
                arr[idx, ki, 0] = kp["x"]
                arr[idx, ki, 1] = kp["y"]
                arr[idx, ki, 2] = kp["visibility"]
            valid_mask[idx] = True

    last_valid = None
    for i in range(n_frames):
        if valid_mask[i]:
            last_valid = arr[i].copy()
        elif last_valid is not None:
            arr[i] = last_valid
    if not valid_mask[0]:
        first_valid_idx = np.argmax(valid_mask) if valid_mask.any() else None
        if first_valid_idx is not None:
            for i in range(first_valid_idx):
                arr[i] = arr[first_valid_idx]

    if not valid_mask.any():
        return None

    return arr


def build_gamma_pernode(arr):
    T = arr.shape[0]
    xy = arr[:, :, :2]
    conf = arr[:, :, 2:3]

    if T >= SG_WINDOW:
        smoothed = savgol_filter(xy, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
                                  axis=0, mode="nearest")
    else:
        smoothed = xy.copy()
    velocity = np.gradient(smoothed, axis=0)
    acceleration = np.gradient(velocity, axis=0)

    return np.concatenate([xy, conf, velocity, acceleration], axis=2).astype(np.float32)


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
        arr = load_clip_raw(norm_path)
        if arr is None:
            continue
        feat = build_gamma_pernode(arr)
        split_data[entry["split"]].append((clip_name, feat, entry["label"]))

    for split in ["train", "val", "test"]:
        n_safe = sum(1 for _, _, l in split_data[split] if l == "safe")
        n_anom = sum(1 for _, _, l in split_data[split] if l == "anomaly")
        print("{}: {} clips (safe={}, anomaly={})".format(split, len(split_data[split]), n_safe, n_anom))

    train_safe_stack = np.concatenate(
        [feat.reshape(feat.shape[0], -1) for _, feat, label in split_data["train"] if label == "safe"], axis=0)
    mean = train_safe_stack.mean(axis=0)
    std = train_safe_stack.std(axis=0)
    std[std < 1e-6] = 1e-6

    np.savez(os.path.join(OUT_DIR, "scaler.npz"), mean=mean, std=std)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(OUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        labels = {}
        for clip_name, feat, label in split_data[split]:
            feat_flat = feat.reshape(feat.shape[0], -1)
            feat_std = (feat_flat - mean) / std
            np.save(os.path.join(split_dir, clip_name + ".npy"), feat_std.astype(np.float32))
            labels[clip_name] = label
        with open(os.path.join(split_dir, "labels.json"), "w") as f:
            json.dump(labels, f, indent=2)
        print("Saved {} clips to {}".format(len(split_data[split]), split_dir))

    print("Feature dim:", train_safe_stack.shape[1], "(13 nodes x [x,y,conf,vx,vy,ax,ay] = 91)")
    print("Done.")


if __name__ == "__main__":
    main()
