"""Epsilon = Beta (52) + hand geometry (8) = 60 dims, raw (unstandardized --
LOSO fits the scaler per fold).

Concatenates beta_raw_v2 and hand_raw_v2 along the feature axis, trimming to
the shorter of the two frame counts as a safety margin."""
import json
import os
import numpy as np

BETA_DIR = "/home/fei/mik0542/beta_raw_v2"
HAND_DIR = "/home/fei/mik0542/hand_raw_v2"
OUT_DIR = "/home/fei/mik0542/epsilon_raw_v2"

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(os.path.join(HAND_DIR, "meta.json")) as f:
        hand_meta = json.load(f)

    meta = []
    n_ok, n_skip, n_trunc = 0, 0, 0
    for m in hand_meta:
        clip = m["clip"]
        beta_path = os.path.join(BETA_DIR, clip + ".npy")
        hand_path = os.path.join(HAND_DIR, clip + ".npy")
        if not os.path.exists(beta_path):
            print("  SKIP (no beta features):", clip)
            n_skip += 1
            continue

        beta = np.load(beta_path)
        hand = np.load(hand_path)

        T = min(beta.shape[0], hand.shape[0])
        if beta.shape[0] != hand.shape[0]:
            n_trunc += 1

        feat = np.concatenate([beta[:T], hand[:T]], axis=1).astype(np.float32)
        np.save(os.path.join(OUT_DIR, clip + ".npy"), feat)
        meta.append(m)
        n_ok += 1

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("OK: {}, skipped: {}, truncated-for-mismatch: {}".format(n_ok, n_skip, n_trunc))
    sample = np.load(os.path.join(OUT_DIR, meta[0]["clip"] + ".npy"))
    print("Feature dim check (expect 60):", sample.shape[1])
    print("Done.")


if __name__ == "__main__":
    main()
