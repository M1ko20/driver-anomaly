"""YOLO26-pose over the whole body-camera videos of the sessions (not over the pre-cut clips),
for the evaluation on uninterrupted recordings. Same detector settings as run_pose_v2.py
(conf 0.25, iou 0.7, imgsz 640), same selection of the person (highest box confidence), same
normalization (common/normalize_skeleton_core.py) and the same Geometry features
(build_beta_raw_v2.py), including the filling of the frames without a detection from the last
valid frame.

Runs on a CPU (the GPUs are shared); --shard/--num-shards split the videos between processes.
Output: sessions_v2/<video stem>.npz with
  beta      (T, 52) float32   Geometry features of every frame of the video
  detected  (T,) bool         False where the frame was filled from a neighbouring frame
  raw       (T, 17, 3) float32  keypoints of the selected person in pixels (nan if none)
  box_conf  (T,) float32
"""
import os
import sys
import json
import argparse
import numpy as np

HOME = "/home/fei/mik0542"
sys.path.insert(0, HOME)
sys.path.insert(0, os.path.join(HOME, "pipeline"))
from common.normalize_skeleton_core import normalize_frame, UPPER_BODY_KEYPOINTS  # noqa: E402
from build_beta_raw_v2 import build_beta  # noqa: E402

MODEL_PATH = os.path.join(HOME, "yolo26l-pose.pt")
OUT_DIR = os.path.join(HOME, "sessions_v2")
KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)   # for a quick test
    ap.add_argument("--only", default="")                  # substring of the video name
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from ultralytics import YOLO
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join(HOME, "clips_v2_manifest.json")))
    videos = sorted(set(m["source_video"] for m in manifest))
    # longest first inside a shard does not matter; round-robin over the sorted sizes balances the shards
    videos = sorted(videos, key=lambda v: -os.path.getsize(v))
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard and args.only in v]
    model = YOLO(MODEL_PATH)
    for vi, video in enumerate(videos):
        stem = os.path.splitext(os.path.basename(video))[0]
        out_path = os.path.join(OUT_DIR, stem + ("_test" if args.max_frames else "") + ".npz")
        if os.path.exists(out_path):
            print("SKIP", stem, flush=True)
            continue
        raw, conf, frames = [], [], []
        for fid, result in enumerate(model.predict(source=video, stream=True, device="cpu", verbose=False,
                                                   conf=0.25, iou=0.7, imgsz=640)):
            if args.max_frames and fid >= args.max_frames:
                break
            persons = []
            if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
                kd = result.keypoints.data.cpu().numpy()
                bc = result.boxes.conf.cpu().numpy()
                for pi in range(kd.shape[0]):
                    # the same rounding as in the JSON files written by run_pose_v2.py
                    persons.append({"person_id": pi, "bbox_conf": round(float(bc[pi]), 4), "kd": kd[pi],
                                    "keypoints": [{"name": KEYPOINT_NAMES[k], "x": round(float(kd[pi][k][0]), 2),
                                                   "y": round(float(kd[pi][k][1]), 2),
                                                   "visibility": round(float(kd[pi][k][2]), 4)} for k in range(17)]})
            fr = normalize_frame({"frame_id": fid, "persons": persons})
            frames.append(fr)
            if persons:
                best = max(persons, key=lambda p: p["bbox_conf"])
                raw.append(best["kd"].astype(np.float32))
                conf.append(best["bbox_conf"])
            else:
                raw.append(np.full((17, 3), np.nan, np.float32))
                conf.append(np.nan)
            if fid % 2000 == 0:
                print("  {} frame {}".format(stem[:30], fid), flush=True)
        T = len(frames)
        arr = np.full((T, len(UPPER_BODY_KEYPOINTS), 3), np.nan, dtype=np.float32)
        valid = np.zeros(T, dtype=bool)
        for i, fr in enumerate(frames):
            if fr["detected"]:
                arr[i] = [[k["x"], k["y"], k["visibility"]] for k in fr["keypoints"]]
                valid[i] = True
        # the same filling as load_clip_raw of build_beta_raw_v2.py
        last = None
        for i in range(T):
            if valid[i]:
                last = arr[i].copy()
            elif last is not None:
                arr[i] = last
        if valid.any() and not valid[0]:
            first = int(np.argmax(valid))
            arr[:first] = arr[first]
        beta = build_beta(arr, UPPER_BODY_KEYPOINTS) if valid.any() else np.zeros((T, 52), np.float32)
        np.savez_compressed(out_path, beta=beta, detected=valid, raw=np.stack(raw), box_conf=np.array(conf, np.float32))
        print("[{}/{}] {} -> {} frames, {} without detection".format(vi + 1, len(videos), stem, T, int((~valid).sum())), flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
