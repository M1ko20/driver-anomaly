"""Step 2: skeleton normalization core, shared by normalize_skeleton.py and
normalize_skeleton_v2.py (they differ only in IN_DIR/OUT_DIR).

Input: pose_json*/clip_XXXX.json (17 COCO keypoints from YOLO26-pose).
Output: pose_norm*/clip_XXXX.json (13 upper-body keypoints, centered +
scale-normalized).

Per frame: pick the highest-confidence person, drop the leg keypoints,
center on the neck (shoulder midpoint), and scale by shoulder width
(falling back to torso length if the shoulders aren't trustworthy).
Confidence is kept as a feature, not normalized away.
"""
import json
import os
import glob
import math

UPPER_BODY_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
]

MIN_VIS_FOR_SCALE = 0.1
MIN_SHOULDER_WIDTH_PX = 5.0


def kp_dict(person):
    return {k["name"]: k for k in person["keypoints"]}


def pick_person(persons):
    if not persons:
        return None
    return max(persons, key=lambda p: p["bbox_conf"] if p["bbox_conf"] is not None else -1.0)


def compute_scale_ref(kps):
    ls, rs = kps["left_shoulder"], kps["right_shoulder"]
    if ls["visibility"] >= MIN_VIS_FOR_SCALE and rs["visibility"] >= MIN_VIS_FOR_SCALE:
        width = math.hypot(ls["x"] - rs["x"], ls["y"] - rs["y"])
        if width >= MIN_SHOULDER_WIDTH_PX:
            return width, "shoulder_width"

    lh, rh = kps.get("left_hip"), kps.get("right_hip")
    if lh and rh and ls and rs:
        neck_x, neck_y = (ls["x"] + rs["x"]) / 2.0, (ls["y"] + rs["y"]) / 2.0
        mid_hip_x, mid_hip_y = (lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0
        length = math.hypot(neck_x - mid_hip_x, neck_y - mid_hip_y)
        if length >= MIN_SHOULDER_WIDTH_PX:
            return length, "torso_length"

    return None, None


def normalize_frame(frame):
    person = pick_person(frame["persons"])
    if person is None:
        return {"frame_id": frame["frame_id"], "detected": False, "keypoints": None}

    kps = kp_dict(person)
    ls, rs = kps["left_shoulder"], kps["right_shoulder"]
    neck_x, neck_y = (ls["x"] + rs["x"]) / 2.0, (ls["y"] + rs["y"]) / 2.0

    scale_ref, scale_source = compute_scale_ref(kps)
    if scale_ref is None:
        return {"frame_id": frame["frame_id"], "detected": False, "keypoints": None}

    norm_kps = []
    for name in UPPER_BODY_KEYPOINTS:
        k = kps[name]
        norm_kps.append({
            "name": name,
            "x": round((k["x"] - neck_x) / scale_ref, 5),
            "y": round((k["y"] - neck_y) / scale_ref, 5),
            "visibility": k["visibility"],
        })

    return {
        "frame_id": frame["frame_id"],
        "detected": True,
        "scale_ref": round(scale_ref, 3),
        "scale_source": scale_source,
        "keypoints": norm_kps,
    }


def run(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    in_files = sorted(glob.glob(os.path.join(in_dir, "clip_*.json")))
    print("Clips to normalize:", len(in_files))

    total_frames = 0
    total_missing = 0

    for idx, path in enumerate(in_files):
        clip_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, clip_name + ".json")
        if os.path.exists(out_path):
            print("[{}/{}] SKIP (exists): {}".format(idx + 1, len(in_files), clip_name))
            continue

        with open(path) as f:
            data = json.load(f)

        norm_frames = [normalize_frame(fr) for fr in data["frames"]]
        n_missing = sum(1 for fr in norm_frames if not fr["detected"])
        total_frames += len(norm_frames)
        total_missing += n_missing

        out = {
            "clip": data["clip"],
            "source_video": data["source_video"],
            "total_frames": data["total_frames"],
            "keypoint_names": UPPER_BODY_KEYPOINTS,
            "centering": "neck (mid-shoulder) -> (0,0)",
            "scale": "shoulder_width, fallback torso_length",
            "frames": norm_frames,
        }

        with open(out_path, "w") as f:
            json.dump(out, f)

        print("[{}/{}] {} -> {} frames ({} missing/undetected), saved {}".format(
            idx + 1, len(in_files), clip_name, len(norm_frames), n_missing, out_path))

    print("\nDone. Total frames: {}, missing/undetected: {} ({:.2f}%)".format(
        total_frames, total_missing, 100.0 * total_missing / max(total_frames, 1)))
