"""Step 3 (Alpha) on v2 data, using the real per-driver split from the clip
manifest instead of the random split in the proof-of-concept version.

Reads the split (already decided at extraction time) from the manifest and
buckets the prepared pose_norm_v2 clips accordingly. Same 39-dim Alpha
feature vector, forward-fill for missing frames, standardization fit on
train safe-drive only.

Saves each clip as a .npy (T, 39) plus a label.json so evaluation can compute
ROC/PR/F1 later."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from common.paths import p
from common.pose_features import load_clip_keypoints

NORM_DIR = p("pose_norm_v2")
MANIFEST_PATH = p("clips_v2_manifest.json")
OUT_DIR = p("alpha_features_v2")

os.makedirs(OUT_DIR, exist_ok=True)


def load_clip_alpha(path):
    arr, _ = load_clip_keypoints(path)
    if arr is None:
        return None
    return arr.reshape(arr.shape[0], -1)


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print("Manifest entries:", len(manifest))

    split_data = {"train": [], "val": [], "test": []}
    for entry in manifest:
        clip_name = entry["clip"]
        norm_path = os.path.join(NORM_DIR, clip_name + ".json")
        if not os.path.exists(norm_path):
            print("  SKIP (no normalized pose):", clip_name)
            continue
        feat = load_clip_alpha(norm_path)
        if feat is None:
            print("  SKIP (no valid frames):", clip_name)
            continue
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

    print("Feature dim:", train_safe_stack.shape[1], "(13 keypoints x [x,y,conf])")
    print("Scaler fit on train safe_drive frames only:", train_safe_stack.shape[0], "frames")
    print("Done.")


if __name__ == "__main__":
    main()
