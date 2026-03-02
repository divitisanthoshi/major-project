"""
Re-encode a video using OpenCV (no ffmpeg required).
Use when a video has H.264/container errors and you want a clean copy.

Usage:
  python scripts/fix_corrupt_video.py data/exercises_videos_import/squat/squat_good_01.mp4
  python scripts/fix_corrupt_video.py path/to/video.mp4 --output path/to/video_fixed.mp4
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2


def main():
    ap = argparse.ArgumentParser(description="Re-encode video with OpenCV (no ffmpeg)")
    ap.add_argument("input", type=str, help="Input video path")
    ap.add_argument("--output", type=str, default=None, help="Output path (default: input_fixed.mp4)")
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        print(f"[Error] Not found: {inp}")
        return 1

    out = args.output
    if not out:
        base, ext = os.path.splitext(inp)
        out = base + "_fixed" + (ext or ".mp4")
    out = os.path.abspath(out)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[Error] Could not open: {inp}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (w, h))

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        n += 1
        if n % 100 == 0:
            print(f"  ... {n} frames", flush=True)

    cap.release()
    writer.release()

    if n == 0:
        print("[Error] No frames read (file may be corrupt).")
        return 1

    print(f"Done. Wrote {n} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
