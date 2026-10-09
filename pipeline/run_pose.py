import json
import os
import glob
from pathlib import Path

MODEL_PATH = "/home/fei/mik0542/yolo26l-pose.pt"
CLIPS_DIR = "/home/fei/mik0542/safe_drive_clips"
OUT_DIR = "/home/fei/mik0542/pose_json"

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

os.makedirs(OUT_DIR, exist_ok=True)

from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "clip_*.mp4")))
print("Clips to process:", len(clips))

for clip_idx, clip_path in enumerate(clips):
    clip_name = Path(clip_path).stem
    out_path = os.path.join(OUT_DIR, clip_name + ".json")

    if os.path.exists(out_path):
        print("[{}/{}] SKIP (exists): {}".format(clip_idx+1, len(clips), clip_name))
        continue

    frames_data = []
    frame_id = 0
    fps_val = None

    for result in model.predict(source=clip_path, stream=True, device=0,
                                verbose=False, conf=0.25, iou=0.7, imgsz=640):
        if fps_val is None and hasattr(result, 'speed'):
            pass

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

    print("[{}/{}] {} -> {} frames, saved {}".format(
        clip_idx+1, len(clips), clip_name, frame_id, out_path))

print("All done.")
