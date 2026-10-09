"""Zeta = Epsilon (60) + gaze/head-pose (5) = 65 dims, raw (unstandardized --
LOSO fits the scaler per fold).

Concatenates epsilon_raw_v2 and gazehead_raw_v2 along the feature axis,
trimming to the shorter frame count as a safety margin."""
import json
import os
import numpy as np

EPSILON_DIR = "/home/fei/mik0542/epsilon_raw_v2"
GAZEHEAD_DIR = "/home/fei/mik0542/gazehead_raw_v2"
OUT_DIR = "/home/fei/mik0542/zeta_raw_v2"

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(os.path.join(GAZEHEAD_DIR, "meta.json")) as f:
        gazehead_meta = json.load(f)

    meta = []
    n_ok, n_skip, n_trunc = 0, 0, 0
    for m in gazehead_meta:
        clip = m["clip"]
        eps_path = os.path.join(EPSILON_DIR, clip + ".npy")
        gh_path = os.path.join(GAZEHEAD_DIR, clip + ".npy")
        if not os.path.exists(eps_path):
            print("  SKIP (no epsilon features):", clip)
            n_skip += 1
            continue

        eps = np.load(eps_path)
        gh = np.load(gh_path)

        T = min(eps.shape[0], gh.shape[0])
        if eps.shape[0] != gh.shape[0]:
            n_trunc += 1

        feat = np.concatenate([eps[:T], gh[:T]], axis=1).astype(np.float32)
        np.save(os.path.join(OUT_DIR, clip + ".npy"), feat)
        meta.append(m)
        n_ok += 1

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("OK: {}, skipped: {}, truncated-for-mismatch: {}".format(n_ok, n_skip, n_trunc))
    sample = np.load(os.path.join(OUT_DIR, meta[0]["clip"] + ".npy"))
    print("Feature dim check (expect 65):", sample.shape[1])
    print("Done.")


if __name__ == "__main__":
    main()
