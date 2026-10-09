"""Delta features = Beta (52, raw) + 4 features targeted at the weakest LOSO
categories (texting, radio): left/right wrist speed, head yaw proxy, and
gaze-down proxy.

Computed directly from the Alpha portion (first 39 columns) already in
beta_raw_v2, no new pose extraction needed.

Keypoint order (from normalize_skeleton_v2.py): 0=nose 1=l_eye 2=r_eye
3=l_ear 4=r_ear 5=l_shoulder 6=r_shoulder 7=l_elbow 8=r_elbow 9=l_wrist
10=r_wrist 11=l_hip 12=r_hip."""
import os
import json
import glob
import numpy as np
from scipy.signal import savgol_filter

RAW_DIR = "/home/fei/mik0542/beta_raw_v2"
OUT_DIR = "/home/fei/mik0542/delta_raw_v2"
os.makedirs(OUT_DIR, exist_ok=True)

KP = {"nose": 0, "l_ear": 3, "r_ear": 4, "l_shoulder": 5, "r_shoulder": 6,
      "l_wrist": 9, "r_wrist": 10}


def xy(feat, name):
    i = KP[name]
    return feat[:, i * 3], feat[:, i * 3 + 1]


def smooth_speed(x, y):
    T = len(x)
    if T >= 5:
        x = savgol_filter(x, window_length=5, polyorder=2)
        y = savgol_filter(y, window_length=5, polyorder=2)
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)


def build_one(feat):
    lwx, lwy = xy(feat, "l_wrist")
    rwx, rwy = xy(feat, "r_wrist")
    lex, _ = xy(feat, "l_ear")
    rex, _ = xy(feat, "r_ear")
    nx, ny = xy(feat, "nose")
    lsx, lsy = xy(feat, "l_shoulder")
    rsx, rsy = xy(feat, "r_shoulder")

    left_wrist_speed = smooth_speed(lwx, lwy)
    right_wrist_speed = smooth_speed(rwx, rwy)
    head_yaw_proxy = (lex - rex).astype(np.float32)
    gaze_down_proxy = (ny - (lsy + rsy) / 2.0).astype(np.float32)

    extra = np.stack([left_wrist_speed, right_wrist_speed,
                       head_yaw_proxy, gaze_down_proxy], axis=1)
    return np.concatenate([feat, extra], axis=1).astype(np.float32)


def main():
    meta = json.load(open(os.path.join(RAW_DIR, "meta.json")))
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.npy")))
    print("Zpracovavam {} klipu...".format(len(files)))
    for i, fp in enumerate(files):
        clip = os.path.basename(fp)[:-4]
        out_fp = os.path.join(OUT_DIR, clip + ".npy")
        if os.path.exists(out_fp):
            continue
        feat = np.load(fp)
        delta = build_one(feat)
        assert delta.shape[1] == 56, delta.shape
        np.save(out_fp, delta)
        if (i + 1) % 100 == 0:
            print("  {}/{}".format(i + 1, len(files)))

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Hotovo. Vystup: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
