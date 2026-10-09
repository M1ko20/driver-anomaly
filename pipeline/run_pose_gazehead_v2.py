"""Head-pose (pitch/yaw/roll) + gaze (yaw/pitch) extraction over clips_v2,
using `uniface` (same author/models as the gaze-estimation and
head-pose-estimation repos, cleaner API).

Detects the face once per frame (SCRFD-500M) and reuses the same crop for
both models. Runs on CPU (ONNX models, ~43ms/frame), sharded across the
server's 40 cores via --shard/--num-shards.

Frames with no detected face get a NaN row. Also logs hit-rate, since
driver-cam footage is often off-angle compared to the frontal datasets these
models were trained on."""
import os
import glob
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

CLIPS_DIR = "/home/fei/mik0542/clips_v2"
OUT_DIR = "/home/fei/mik0542/pose_gazehead_v2"

N_FEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"

    os.makedirs(OUT_DIR, exist_ok=True)

    import uniface
    from uniface.constants import GazeWeights, SCRFDWeights

    detector = uniface.SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)
    headpose = uniface.HeadPose()
    gaze = uniface.MobileGaze(model_name=GazeWeights.MOBILEONE_S0)

    all_clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "clip_*.mp4")))
    clips = [c for i, c in enumerate(all_clips) if i % args.num_shards == args.shard]
    print("[shard{}] {}/{} clips".format(args.shard, len(clips), len(all_clips)))

    total_frames, total_hits = 0, 0

    for clip_idx, clip_path in enumerate(clips):
        clip_name = Path(clip_path).stem
        out_path = os.path.join(OUT_DIR, clip_name + ".npy")

        if os.path.exists(out_path):
            print("[shard{}] [{}/{}] SKIP (exists): {}".format(args.shard, clip_idx + 1, len(clips), clip_name))
            continue

        cap = cv2.VideoCapture(clip_path)
        rows = []
        n_hits = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect(frame)
            if not faces:
                rows.append(np.full(N_FEATS, np.nan, dtype=np.float32))
                continue

            face = max(faces, key=lambda f: f.confidence)
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                rows.append(np.full(N_FEATS, np.nan, dtype=np.float32))
                continue

            hp = headpose.estimate(crop)
            gz = gaze.estimate(crop)
            rows.append(np.array([
                hp.pitch, hp.yaw, hp.roll,
                np.degrees(gz.yaw), np.degrees(gz.pitch),
            ], dtype=np.float32))
            n_hits += 1
        cap.release()

        if not rows:
            print("[shard{}] [{}/{}] SKIP (no frames read): {}".format(
                args.shard, clip_idx + 1, len(clips), clip_name))
            continue

        arr = np.stack(rows, axis=0)
        np.save(out_path, arr)

        total_frames += len(rows)
        total_hits += n_hits
        hit_rate = n_hits / len(rows) if rows else 0.0

        print("[shard{}] [{}/{}] {} -> {} frames, hit_rate={:.2f}, {:.1f}s, saved {}".format(
            args.shard, clip_idx + 1, len(clips), clip_name, arr.shape[0], hit_rate,
            time.time() - t0, out_path))

    if total_frames > 0:
        print("[shard{}] DONE. Overall hit rate this shard: {:.4f} ({}/{})".format(
            args.shard, total_hits / total_frames, total_hits, total_frames))
    print("[shard{}] shard done.".format(args.shard))


if __name__ == "__main__":
    main()
