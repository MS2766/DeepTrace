import cv2
import os
from pathlib import Path
import random


DATA_ROOT = Path("data")
OUT_ROOT = Path("data_frames")
NUM_FRAMES = 8
IMG_SIZE = 224


def extract_from_video(video_path, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    indices = sorted(
        random.sample(
            range(total_frames),
            min(NUM_FRAMES, total_frames)
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(str(out_dir / f"{i:03d}.jpg"), frame)
        saved += 1

    cap.release()
    return saved > 0


def main():
    for label in ["real", "fake"]:
        src_dir = DATA_ROOT / label
        dst_dir = OUT_ROOT / label

        videos = list(src_dir.rglob("*.mp4"))
        print(f"{label}: {len(videos)} videos")

        for i, video in enumerate(videos):
            video_id = video.stem
            out_dir = dst_dir / video_id

            if out_dir.exists():
                continue

            ok = extract_from_video(video, out_dir)
            if not ok:
                print(f"Failed: {video.name}")

            if i % 50 == 0:
                print(f"{label}: processed {i}/{len(videos)}")

    print("Frame extraction complete")


if __name__ == "__main__":
    main()
