"""
Convert video files (from teammate) to skeleton .npy format for training.

Usage:
  python scripts/video_to_skeleton.py --input path/to/custom_data --output data/custom

Expects videos named: <exercise>_<quality>_<number>.mp4
  e.g. wall_pushup_good_01.mp4, heel_raise_moderate_03.mp4
  quality: good (→0.85), moderate (→0.55), poor (→0.25)
"""

import os
import sys
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.pose_extraction import PoseExtractor

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")
QUALITY_SCORE = {"good": 0.85, "moderate": 0.55, "poor": 0.25}


def parse_filename(name: str) -> tuple:
    """Return (exercise, quality, score) or (None, None, 0.5)."""
    base = os.path.splitext(name)[0].lower()
    for q, score in QUALITY_SCORE.items():
        m = re.match(rf"^([a-z_]+)_{q}_(\d+)$", base)
        if m:
            return m.group(1), q, score
    # Fallback: exercise_01 → 0.5
    m = re.match(r"^([a-z_]+)_(\d+)$", base)
    if m:
        return m.group(1), "unknown", 0.5
    return None, None, 0.5


def process_video(path: str, extractor: PoseExtractor) -> np.ndarray:
    """Extract skeleton sequence (T, 33, 3) from video."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, landmarks = extractor.extract(frame)
        if landmarks is not None:
            frames.append(landmarks.astype(np.float32))
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else np.zeros((0, 33, 3), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert teammate videos to skeleton .npy for training"
    )
    parser.add_argument("--input", type=str, required=True, help="Folder with exercise subfolders/videos")
    parser.add_argument("--output", type=str, default="data/custom", help="Output path (data/custom)")
    parser.add_argument("--min-frames", type=int, default=30, help="Skip videos shorter than this")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[Error] Input not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    extractor = PoseExtractor()

    total = 0
    for root, _, files in os.walk(args.input):
        for f in sorted(files):
            if not f.lower().endswith(VIDEO_EXT):
                continue
            path = os.path.join(root, f)
            exercise, quality, score = parse_filename(f)
            if exercise is None:
                # Try parent folder as exercise
                rel = os.path.relpath(root, args.input)
                if rel != ".":
                    ex_from_dir = rel.replace("\\", "/").split("/")[0]
                    exercise = ex_from_dir
                else:
                    exercise = "unknown"

            out_dir = os.path.join(args.output, exercise)
            os.makedirs(out_dir, exist_ok=True)

            # Avoid overwrite: seq_001.npy style
            existing = [x for x in os.listdir(out_dir) if x.endswith(".npy")]
            idx = len(existing)
            out_name = f"seq_{idx:03d}.npy"
            out_path = os.path.join(out_dir, out_name)

            print(f"Processing {f} -> {exercise}/{out_name} (score={score:.2f})")
            seq = process_video(path, extractor)

            if len(seq) < args.min_frames:
                print(f"  Skip: too short ({len(seq)} frames)")
                continue

            np.save(out_path, seq)
            total += 1

            # Append to labels.csv
            labels_path = os.path.join(out_dir, "labels.csv")
            exists = os.path.exists(labels_path)
            with open(labels_path, "a", newline="", encoding="utf-8") as lf:
                if not exists:
                    lf.write("file,score\n")
                lf.write(f"{out_name},{score}\n")

    extractor.close()
    print(f"\nDone. Converted {total} videos to skeleton format in {args.output}")
    print("Next: python train.py --dataset custom")


if __name__ == "__main__":
    main()
