"""Step 0: extracts clips with driver ID and anomaly labels (replaces
extract_safe_drive.py).

Train gets safe-drive clips only; val/test get safe-drive plus distraction
actions (reach, phone, texting, radio, etc.). Writes a manifest.json
(driver_id, group, session, action, label, split) for evaluation and
auditability.

Fixed, subject-disjoint, stratified split across the 8 drivers (train:
gA_1/gB_6/gB_7/gC_13, val: gA_5/gB_9, test: gB_10/gC_14)."""
import json
import os
import subprocess
import glob

BASE_DIR = "/home/fei/mik0542/Drivers/dmd"
OUT_DIR = "/home/fei/mik0542/clips_v2"
MANIFEST_PATH = "/home/fei/mik0542/clips_v2_manifest.json"

DRIVER_SPLIT = {
    "gA_1": "train", "gB_6": "train", "gB_7": "train", "gC_13": "train",
    "gA_5": "val", "gB_9": "val",
    "gB_10": "test", "gC_14": "test",
    "gE_28": "train", "gE_29": "train", "gF_23": "train",
    "gZ_33": "train", "gZ_36": "train", "gZ_37": "train",
}

ANOMALY_TYPES = {
    "reach_side", "reach_backseat", "talking_to_passenger", "hair_and_makeup",
    "phonecall_right", "phonecall_left", "texting_right", "texting_left",
    "radio", "drinking",
}
SAFE_TYPE = "safe_drive"

os.makedirs(OUT_DIR, exist_ok=True)


def get_fps(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    val = result.stdout.strip().decode()
    num, den = val.split("/")
    return float(num) / float(den)


def parse_driver_id(json_path):
    rel = os.path.relpath(json_path, BASE_DIR)
    parts = rel.split(os.sep)
    group, driver, session = parts[0], parts[1], parts[2]
    return "{}_{}".format(group, driver), group, driver, session


def main():
    json_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*_rgb_ann_distraction.json"), recursive=True))
    print("Found {} annotation files".format(len(json_files)))

    manifest = []
    clip_counters = {}

    for json_path in json_files:
        driver_id, group, driver, session = parse_driver_id(json_path)
        split = DRIVER_SPLIT.get(driver_id)
        if split is None:
            print("  SKIP (driver not in split map): {}".format(driver_id))
            continue

        video_path = json_path.replace("_rgb_ann_distraction.json", "_rgb_body.mp4")
        if not os.path.exists(video_path):
            print("  SKIP (no body video): {}".format(json_path))
            continue

        with open(json_path) as f:
            data = json.load(f)
        actions = data["openlabel"]["actions"]

        intervals = []
        for k, v in actions.items():
            t = v.get("type", "")
            if not t.startswith("driver_actions/"):
                continue
            action = t.split("/", 1)[1]
            if action == SAFE_TYPE:
                label = "safe"
            elif action in ANOMALY_TYPES:
                label = "anomaly"
            else:
                continue
            for interval in v.get("frame_intervals", []):
                intervals.append((action, label, interval["frame_start"], interval["frame_end"]))

        if not intervals:
            continue

        fps = get_fps(video_path)
        print("Processing {} | driver={} split={} fps={:.4f} | {} intervals".format(
            video_path, driver_id, split, fps, len(intervals)))

        for action, label, start_frame, end_frame in intervals:
            start_sec = start_frame / fps
            end_sec = (end_frame + 1) / fps
            duration = end_sec - start_sec
            if duration < 0.5:
                continue

            key = (driver_id, session)
            clip_counters[key] = clip_counters.get(key, 0) + 1
            idx = clip_counters[key]
            clip_name = "clip_{}_{}_{}_{:04d}".format(driver_id, session, action, idx)
            clip_path = os.path.join(OUT_DIR, clip_name + ".mp4")

            if not os.path.exists(clip_path):
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_sec),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-an",
                    clip_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    print("  ERROR clipping {}: {}".format(clip_path, result.stderr.decode()[-300:]))
                    continue

            manifest.append({
                "clip": clip_name,
                "driver_id": driver_id,
                "group": group,
                "session": session,
                "action": action,
                "label": label,
                "split": split,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": round(duration, 3),
                "source_video": video_path,
            })

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nTotal clips: {}".format(len(manifest)))
    for split in ["train", "val", "test"]:
        split_entries = [m for m in manifest if m["split"] == split]
        n_safe = sum(1 for m in split_entries if m["label"] == "safe")
        n_anom = sum(1 for m in split_entries if m["label"] == "anomaly")
        print("  {}: {} clips (safe={}, anomaly={})".format(split, len(split_entries), n_safe, n_anom))
    print("Manifest saved to", MANIFEST_PATH)


if __name__ == "__main__":
    main()
