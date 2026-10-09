import json
import os
import subprocess
import glob

BASE_DIR = "/home/fei/mik0542/Drivers/dmd"
OUT_DIR = "/home/fei/mik0542/safe_drive_clips"
FINAL_VIDEO = "/home/fei/mik0542/safe_drive_body_all.mp4"

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

def frames_to_seconds(frame, fps):
    return frame / fps

json_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*_rgb_ann_distraction.json"), recursive=True))
print("Found {} JSON files".format(len(json_files)))

all_clips = []
clip_index = 0

for json_path in json_files:
    folder = os.path.dirname(json_path)
    base = json_path.replace("_rgb_ann_distraction.json", "_rgb_body.mp4")
    if not os.path.exists(base):
        print("  SKIP (no body video): {}".format(json_path))
        continue

    with open(json_path) as f:
        data = json.load(f)

    actions = data["openlabel"]["actions"]

    safe_intervals = []
    for k, v in actions.items():
        if v.get("type") == "driver_actions/safe_drive":
            safe_intervals.extend(v.get("frame_intervals", []))

    if not safe_intervals:
        print("  No safe_drive intervals in: {}".format(json_path))
        continue

    fps = get_fps(base)
    print("Processing {} | fps={:.4f} | {} safe intervals".format(base, fps, len(safe_intervals)))

    for interval in safe_intervals:
        start_frame = interval["frame_start"]
        end_frame = interval["frame_end"]
        start_sec = frames_to_seconds(start_frame, fps)
        end_sec = frames_to_seconds(end_frame + 1, fps)
        duration = end_sec - start_sec

        if duration < 0.5:
            continue

        clip_path = os.path.join(OUT_DIR, "clip_{:04d}.mp4".format(clip_index))
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", base,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            clip_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("  ERROR clipping {}: {}".format(clip_path, result.stderr.decode()[-300:]))
        else:
            all_clips.append(clip_path)
            print("  Clip {:04d}: frames {}-{} ({:.2f}s)".format(clip_index, start_frame, end_frame, duration))
            clip_index += 1

print("\nExtracted {} clips total".format(len(all_clips)))

concat_list = os.path.join(OUT_DIR, "concat_list.txt")
with open(concat_list, "w") as f:
    for clip in all_clips:
        f.write("file '{}'\n".format(clip))

print("Concatenating into {} ...".format(FINAL_VIDEO))
concat_cmd = [
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0",
    "-i", concat_list,
    "-c", "copy",
    FINAL_VIDEO
]
result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result.returncode != 0:
    print("ERROR concatenating: {}".format(result.stderr.decode()[-500:]))
else:
    print("Done! Final video: {}".format(FINAL_VIDEO))
