"""
Convert images (e.g. from Google) to skeleton .npy format for training.

Usage:
  python scripts/images_to_skeleton.py --input path/to/images --output data/custom

Expects images in subfolders by exercise: input/wall_pushup/img1.jpg, input/heel_raise/img2.png
Or named: wall_pushup_good_01.jpg, heel_raise_02.png
  quality in name: good (→0.85), moderate (→0.55), poor (→0.25), else →0.7

Note: Images provide only static poses. The model expects temporal sequences (motion).
Each image is repeated to form a 64-frame sequence. Training quality will be limited
compared to real video data, but this works for proof-of-concept or augmenting synthetic data.
"""

import os
import sys
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.pose_extraction import PoseExtractor

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
QUALITY_SCORE = {"good": 0.85, "moderate": 0.55, "poor": 0.25}
SEQ_LEN = 64


def parse_filename(name: str) -> tuple:
    """Return (exercise, quality, score) or (None, None, 0.7)."""
    base = os.path.splitext(name)[0].lower()
    for q, score in QUALITY_SCORE.items():
        m = re.match(rf"^([a-z_]+)_{q}_(\d+)$", base)
        if m:
            return m.group(1), q, score
    m = re.match(r"^([a-z_]+)_(\d+)$", base)
    if m:
        return m.group(1), "unknown", 0.7
    return None, None, 0.7


def process_image(path: str, extractor: PoseExtractor) -> np.ndarray:
    """Extract pose from image, return (33, 3) or empty array."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((0, 33, 3), dtype=np.float32)
    _, landmarks = extractor.extract(img)
    if landmarks is None:
        return np.zeros((0, 33, 3), dtype=np.float32)
    return landmarks.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to skeleton .npy for training (static poses)"
    )
    parser.add_argument("--input", type=str, required=True, help="Folder with exercise subfolders/images")
    parser.add_argument("--output", type=str, default="data/custom", help="Output path")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Repeat pose to this many frames")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[Error] Input not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    extractor = PoseExtractor()

    total = 0
    for root, _, files in os.walk(args.input):
        for f in sorted(files):
            if not f.lower().endswith(IMAGE_EXT):
                continue
            path = os.path.join(root, f)
            exercise, _, score = parse_filename(f)
            if exercise is None:
                rel = os.path.relpath(root, args.input).replace("\\", "/")
                exercise = rel.split("/")[0] if rel != "." else "unknown"

            out_dir = os.path.join(args.output, exercise)
            os.makedirs(out_dir, exist_ok=True)

            existing = [x for x in os.listdir(out_dir) if x.endswith(".npy")]
            idx = len(existing)
            out_name = f"seq_{idx:03d}.npy"
            out_path = os.path.join(out_dir, out_name)

            print(f"Processing {f} -> {exercise}/{out_name} (score={score:.2f})")
            pose = process_image(path, extractor)

            if pose.size == 0:
                print(f"  Skip: no pose detected")
                continue

            # Repeat pose to form sequence (T, 33, 3)
            seq = np.tile(pose[np.newaxis, :, :], (args.seq_len, 1, 1)).astype(np.float32)
            np.save(out_path, seq)
            total += 1

            labels_path = os.path.join(out_dir, "labels.csv")
            exists = os.path.exists(labels_path)
            with open(labels_path, "a", newline="", encoding="utf-8") as lf:
                if not exists:
                    lf.write("file,score\n")
                lf.write(f"{out_name},{score}\n")

    extractor.close()
    print(f"\nDone. Converted {total} images to skeleton format in {args.output}")
    print("Note: Static images are repeated to sequences. Video data gives better results.")
    print("Next: python train.py --dataset custom")


if __name__ == "__main__":
    main()
