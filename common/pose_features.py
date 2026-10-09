"""Shared loader for normalized pose clips (pose_norm*/clip_XXXX.json).

Forward-fills missing/undetected frames from the last valid frame (and
backfills the leading gap from the first valid frame), matching the
original build_alpha_features*/build_beta_features_v2 behavior.
"""
import json
import numpy as np


def load_clip_keypoints(path, data=None):
    """Returns (arr, kp_names) with arr shaped (n_frames, n_keypoints, 3) as
    (x, y, visibility), or (None, None) if the clip has no valid frames."""
    if data is None:
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
        return None, None

    return arr, kp_names


def load_clip_alpha(path):
    """Returns (flat, clip_name) with flat shaped (n_frames, n_keypoints*3),
    or None if the clip has no valid frames. Matches build_alpha_features*.py."""
    with open(path) as f:
        data = json.load(f)
    arr, _ = load_clip_keypoints(path, data=data)
    if arr is None:
        return None
    return arr.reshape(arr.shape[0], -1), data["clip"]
