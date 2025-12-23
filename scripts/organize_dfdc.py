import json
import shutil
from pathlib import Path

# SOURCE
DFDC_ROOT = Path(r"C:\Users\maana\Downloads\DFDC")
TRAIN_ROOT = DFDC_ROOT / "train_sample_videos"
METADATA_PATH = DFDC_ROOT / "metadata.json"

# DESTINATION
REAL_DST = Path(r"C:\Studies\Projects\DeepTrace\data\real\dfdc")
FAKE_DST = Path(r"C:\Studies\Projects\DeepTrace\data\fake\dfdc")

REAL_DST.mkdir(parents=True, exist_ok=True)
FAKE_DST.mkdir(parents=True, exist_ok=True)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

real_count = fake_count = skipped = 0

for video_name, info in metadata.items():
    src_file = TRAIN_ROOT / video_name  # <-- mp4 file

    if not src_file.exists():
        skipped += 1
        continue

    if info["label"] == "REAL":
        dst = REAL_DST / video_name
        real_count += 1
    else:
        dst = FAKE_DST / video_name
        fake_count += 1

    if not dst.exists():
        shutil.copy2(src_file, dst)

print("DFDC organization complete")
print(f"REAL  videos: {real_count}")
print(f"FAKE  videos: {fake_count}")
print(f"Skipped      : {skipped}")
