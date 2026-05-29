"""
Training — Step 1: Extract frames from videos for annotation.
Usage:
  python training/collect_data.py --source video.mp4 --out data/raw --fps 1
"""
from __future__ import annotations
import argparse
import os
import cv2


def extract_frames(source, out_dir: str, fps: float = 1.0,
                   max_frames: int = 2000) -> int:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / fps)))

    count = saved = 0
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(path, frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames → {out_dir}")
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out",    default="data/raw")
    p.add_argument("--fps",    type=float, default=1.0)
    p.add_argument("--max",    type=int,   default=2000)
    args = p.parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    extract_frames(src, args.out, args.fps, args.max)


if __name__ == "__main__":
    main()
