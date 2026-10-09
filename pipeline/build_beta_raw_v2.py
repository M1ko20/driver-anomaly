"""Beta features without standardization (raw), for LOSO cross-validation.

LOSO needs the scaler refit per fold (train/test drivers change each fold),
so this just saves raw Beta features plus a driver_id/label manifest and
lets the LOSO script standardize at runtime. Same geometry as
build_beta_features_v2.py (52 dims), just without the final z-score step."""
import json
import os
import numpy as np

NORM_DIR = "/home/fei/mik0542/pose_norm_v2"
MANIFEST_PATH = "/home/fei/mik0542/clips_v2_manifest.json"
OUT_DIR = "/home/fei/mik0542/beta_raw_v2"

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
        return None, None

    return arr, kp_names


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
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = []
    for entry in manifest:
        clip_name = entry["clip"]
        norm_path = os.path.join(NORM_DIR, clip_name + ".json")
        if not os.path.exists(norm_path):
            print("  SKIP (no normalized pose):", clip_name)
            continue
        arr, kp_names = load_clip_raw(norm_path)
        if arr is None:
            print("  SKIP (no valid frames):", clip_name)
            continue
        feat = build_beta(arr, kp_names)
        np.save(os.path.join(OUT_DIR, clip_name + ".npy"), feat)
        meta.append({
            "clip": clip_name,
            "driver_id": entry["driver_id"],
            "label": entry["label"],
            "action": entry["action"],
        })

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    drivers = sorted(set(m["driver_id"] for m in meta))
    print("Drivers found:", drivers)
    for d in drivers:
        n_safe = sum(1 for m in meta if m["driver_id"] == d and m["label"] == "safe")
        n_anom = sum(1 for m in meta if m["driver_id"] == d and m["label"] == "anomaly")
        print("  {}: safe={} anomaly={}".format(d, n_safe, n_anom))

    print("Total clips with features:", len(meta))
    print("Done.")


if __name__ == "__main__":
    main()
