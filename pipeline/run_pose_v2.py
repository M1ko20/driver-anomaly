"""Step 1: YOLO26-pose extraction over clips_v2 (safe + anomaly), parallelized
across multiple GPUs via --gpu and --shard/--num-shards (round-robin split
so shard runtimes stay balanced).

GPU 0 on sumo2 is shared with other users -- default usage is GPU 1,2,3."""
import json
import os
import glob
import argparse
from pathlib import Path

MODEL_PATH = "/home/fei/mik0542/yolo26l-pose.pt"
CLIPS_DIR = "/home/fei/mik0542/clips_v2"
OUT_DIR = "/home/fei/mik0542/pose_json_v2"

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.makedirs(OUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print("[gpu{}] Model loaded".format(args.gpu))

    all_clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "clip_*.mp4")))
    clips = [c for i, c in enumerate(all_clips) if i % args.num_shards == args.shard]
    print("[gpu{}] shard {}/{}: {} clips".format(args.gpu, args.shard, args.num_shards, len(clips)))

    for clip_idx, clip_path in enumerate(clips):
        clip_name = Path(clip_path).stem
        out_path = os.path.join(OUT_DIR, clip_name + ".json")

        if os.path.exists(out_path):
            print("[gpu{}] [{}/{}] SKIP (exists): {}".format(args.gpu, clip_idx + 1, len(clips), clip_name))
            continue

        frames_data = []
        frame_id = 0

        for result in model.predict(source=clip_path, stream=True, device=0,
                                    verbose=False, conf=0.25, iou=0.7, imgsz=640):
            frame_entry = {"frame_id": frame_id, "persons": []}

            if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
                kps_data = result.keypoints.data
                boxes_xyxy = result.boxes.xyxy if result.boxes is not None else None
                boxes_conf = result.boxes.conf if result.boxes is not None else None

                for pi in range(kps_data.shape[0]):
                    kps = kps_data[pi]
                    person_entry = {
                        "person_id": pi,
                        "bbox_conf": round(float(boxes_conf[pi]), 4) if boxes_conf is not None else None,
                        "bbox_xyxy": [round(float(v), 1) for v in boxes_xyxy[pi]] if boxes_xyxy is not None else None,
                        "keypoints": [
                            {
                                "name": KEYPOINT_NAMES[ki],
                                "x": round(float(kps[ki][0]), 2),
                                "y": round(float(kps[ki][1]), 2),
                                "visibility": round(float(kps[ki][2]), 4)
                            }
                            for ki in range(17)
                        ]
                    }
                    frame_entry["persons"].append(person_entry)

            frames_data.append(frame_entry)
            frame_id += 1

        output = {
            "clip": clip_name,
            "source_video": clip_path,
            "total_frames": frame_id,
            "keypoint_names": KEYPOINT_NAMES,
            "frames": frames_data
        }

        with open(out_path, "w") as f:
            json.dump(output, f)

        print("[gpu{}] [{}/{}] {} -> {} frames, saved {}".format(
            args.gpu, clip_idx + 1, len(clips), clip_name, frame_id, out_path))

    print("[gpu{}] shard done.".format(args.gpu))


if __name__ == "__main__":
    main()
