"""Hand-geometry features from wholebody (easy_ViTPose) hand keypoints, meant
to attach to Beta (see build_epsilon_raw_v2.py). Raw (unstandardized) --
LOSO fits the scaler per fold.

Input: pose_hands_v2/*.npy (T, 133, 3), COCO-WholeBody layout. Normalized
independently of the YOLO26 skeleton (own center/scale from shoulder
keypoints), since the two detectors' pixel frames don't necessarily line up
exactly.

8 features per frame targeted at texting/phonecall/radio: pinch distance,
hand openness, hand-to-ear distance, and hand-to-nose distance, each
computed for both hands."""
import json
import os
import glob
import numpy as np

HANDS_DIR = "/home/fei/mik0542/pose_hands_v2"
MANIFEST_PATH = "/home/fei/mik0542/clips_v2_manifest.json"
OUT_DIR = "/home/fei/mik0542/hand_raw_v2"

os.makedirs(OUT_DIR, exist_ok=True)

CONF_THRESH = 0.15

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_EAR, R_EAR = 3, 4
L_HAND_ROOT, R_HAND_ROOT = 91, 112
L_TIPS = [95, 99, 103, 107, 111]
R_TIPS = [116, 120, 124, 128, 132]
L_THUMB_TIP, L_INDEX_TIP = 95, 99
R_THUMB_TIP, R_INDEX_TIP = 116, 120


def load_hand_xy(path):
    """Nacte (T,133,3) v (y,x,score), vrati (T,133,2) xy + (T,133) conf,
    s forward-fill pro framy/klouby pod prahem confidence."""
    raw = np.load(path)
    T = raw.shape[0]
    xy = raw[:, :, [1, 0]].astype(np.float32)
    conf = raw[:, :, 2].astype(np.float32)

    valid = conf >= CONF_THRESH
    xy_filled = xy.copy()

    for k in range(133):
        last_valid = None
        for t in range(T):
            if valid[t, k]:
                last_valid = xy_filled[t, k].copy()
            elif last_valid is not None:
                xy_filled[t, k] = last_valid
        if not valid[0, k]:
            first_idx = np.argmax(valid[:, k]) if valid[:, k].any() else None
            if first_idx is not None and first_idx > 0:
                xy_filled[:first_idx, k] = xy_filled[first_idx, k]

    frame_has_body = valid[:, L_SHOULDER] & valid[:, R_SHOULDER]
    return xy_filled, frame_has_body


def dist(p1, p2):
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))


def build_hand_features(xy_filled, frame_has_body):
    center = (xy_filled[:, L_SHOULDER] + xy_filled[:, R_SHOULDER]) / 2.0
    scale = dist(xy_filled[:, L_SHOULDER], xy_filled[:, R_SHOULDER])
    scale = np.maximum(scale, 1e-3)

    norm = (xy_filled - center[:, None, :]) / scale[:, None, None]

    l_pinch = dist(norm[:, L_THUMB_TIP], norm[:, L_INDEX_TIP])
    r_pinch = dist(norm[:, R_THUMB_TIP], norm[:, R_INDEX_TIP])

    l_open = np.mean([dist(norm[:, L_HAND_ROOT], norm[:, tip]) for tip in L_TIPS], axis=0)
    r_open = np.mean([dist(norm[:, R_HAND_ROOT], norm[:, tip]) for tip in R_TIPS], axis=0)

    l_to_ear = dist(norm[:, L_HAND_ROOT], norm[:, L_EAR])
    r_to_ear = dist(norm[:, R_HAND_ROOT], norm[:, R_EAR])

    l_to_nose = dist(norm[:, L_HAND_ROOT], norm[:, NOSE])
    r_to_nose = dist(norm[:, R_HAND_ROOT], norm[:, NOSE])

    feat = np.stack([l_pinch, r_pinch, l_open, r_open,
                      l_to_ear, r_to_ear, l_to_nose, r_to_nose], axis=1).astype(np.float32)

    return feat


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    manifest_by_clip = {m["clip"]: m for m in manifest}

    hand_files = sorted(glob.glob(os.path.join(HANDS_DIR, "clip_*.npy")))
    print("Hand pose files found:", len(hand_files))

    meta = []
    n_ok, n_skip = 0, 0
    for path in hand_files:
        clip_name = os.path.splitext(os.path.basename(path))[0]
        if clip_name not in manifest_by_clip:
            print("  SKIP (not in manifest):", clip_name)
            n_skip += 1
            continue

        xy_filled, frame_has_body = load_hand_xy(path)
        if not frame_has_body.any():
            print("  SKIP (no body ever detected):", clip_name)
            n_skip += 1
            continue

        feat = build_hand_features(xy_filled, frame_has_body)
        np.save(os.path.join(OUT_DIR, clip_name + ".npy"), feat)

        entry = manifest_by_clip[clip_name]
        meta.append({
            "clip": clip_name,
            "driver_id": entry["driver_id"],
            "label": entry["label"],
            "action": entry["action"],
        })
        n_ok += 1

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("OK: {}, skipped: {}".format(n_ok, n_skip))
    print("Done.")


if __name__ == "__main__":
    main()
