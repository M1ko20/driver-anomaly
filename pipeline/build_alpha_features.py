"""Step 3 (Alpha only) pipeline validation.

Not the final Step 0 split -- clips have no driver ID or anomalies yet, so
this just does a random 70/15/15 clip-level split to validate the pipeline
end to end (load -> Alpha features -> standardize -> window -> train AE ->
reconstruct). Will be replaced by the per-driver split once re-extraction
adds driver ID and anomalies.

Alpha feature vector per frame: 13 keypoints x (x, y, confidence) = 39
values. Missing frames are forward-filled; standardization is fit on train
only."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import random
import numpy as np

from common.paths import p
from common.pose_features import load_clip_alpha

IN_DIR = p("pose_norm")
OUT_DIR = p("alpha_features")
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    random.seed(SEED)
    files = sorted(glob.glob(os.path.join(IN_DIR, "clip_*.json")))
    print("Clips found:", len(files))

    clips = []
    for path in files:
        result = load_clip_alpha(path)
        if result is None:
            print("  SKIP (no valid frames):", path)
            continue
        clips.append((result[1], result[0]))

    random.shuffle(clips)
    n = len(clips)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_clips = clips[:n_train]
    val_clips = clips[n_train:n_train + n_val]
    test_clips = clips[n_train + n_val:]
    print("Split (placeholder, clip-level random): train={} val={} test={}".format(
        len(train_clips), len(val_clips), len(test_clips)))

    train_stack = np.concatenate([c[1] for c in train_clips], axis=0)
    mean = train_stack.mean(axis=0)
    std = train_stack.std(axis=0)
    std[std < 1e-6] = 1e-6

    np.savez(os.path.join(OUT_DIR, "scaler.npz"), mean=mean, std=std)

    for split_name, split_clips in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
        split_dir = os.path.join(OUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for clip_name, feat in split_clips:
            feat_std = (feat - mean) / std
            np.save(os.path.join(split_dir, clip_name + ".npy"), feat_std.astype(np.float32))
        print("Saved {} clips to {}".format(len(split_clips), split_dir))

    print("Feature dim:", train_stack.shape[1], "(13 keypoints x [x,y,conf])")
    print("Done.")


if __name__ == "__main__":
    main()
