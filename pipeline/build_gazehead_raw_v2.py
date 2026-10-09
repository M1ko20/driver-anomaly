"""Head-pose (pitch/yaw/roll) + gaze (yaw/pitch) features, raw (unstandardized
-- LOSO fits the scaler per fold).

Reads pose_gazehead_v2/*.npy (T, 5) in degrees, forward-fills frames with no
detected face, and normalizes to a stable range (degrees/90) so the scale
roughly matches the other feature sets before standardization."""
import json
import os
import glob
import numpy as np

GAZEHEAD_DIR = "/home/fei/mik0542/pose_gazehead_v2"
MANIFEST_PATH = "/home/fei/mik0542/clips_v2_manifest.json"
OUT_DIR = "/home/fei/mik0542/gazehead_raw_v2"

os.makedirs(OUT_DIR, exist_ok=True)


def forward_fill(arr):
    """arr: (T, 5), NaN radky (nedetekovany oblicej) forward-fill; pripadne
    chybejici zacatecni framy zpetne vypln prvnim platnym radkem."""
    T = arr.shape[0]
    valid = ~np.isnan(arr).any(axis=1)
    filled = arr.copy()

    last_valid = None
    for t in range(T):
        if valid[t]:
            last_valid = filled[t].copy()
        elif last_valid is not None:
            filled[t] = last_valid

    if not valid[0]:
        first_idx = np.argmax(valid) if valid.any() else None
        if first_idx is not None and first_idx > 0:
            filled[:first_idx] = filled[first_idx]

    return filled, valid


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    manifest_by_clip = {m["clip"]: m for m in manifest}

    files = sorted(glob.glob(os.path.join(GAZEHEAD_DIR, "clip_*.npy")))
    print("Gazehead pose files found:", len(files))

    meta = []
    n_ok, n_skip, n_never_detected = 0, 0, 0
    total_frames, total_valid = 0, 0

    for path in files:
        clip_name = os.path.splitext(os.path.basename(path))[0]
        if clip_name not in manifest_by_clip:
            print("  SKIP (not in manifest):", clip_name)
            n_skip += 1
            continue

        raw = np.load(path)
        filled, valid = forward_fill(raw)

        if not valid.any():
            print("  SKIP (face never detected):", clip_name)
            n_never_detected += 1
            continue

        feat = (filled / 90.0).astype(np.float32)
        np.save(os.path.join(OUT_DIR, clip_name + ".npy"), feat)

        entry = manifest_by_clip[clip_name]
        meta.append({
            "clip": clip_name,
            "driver_id": entry["driver_id"],
            "label": entry["label"],
            "action": entry["action"],
        })
        n_ok += 1
        total_frames += len(valid)
        total_valid += valid.sum()

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("OK: {}, skipped(not-in-manifest): {}, skipped(never-detected): {}".format(
        n_ok, n_skip, n_never_detected))
    print("Overall frame-level hit rate: {:.4f} ({}/{})".format(
        total_valid / total_frames, total_valid, total_frames))
    print("Done.")


if __name__ == "__main__":
    main()
