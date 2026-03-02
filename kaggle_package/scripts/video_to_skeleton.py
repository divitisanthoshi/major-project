"""
Convert video files (from teammate) to skeleton .npy format for training.

Usage:
  python scripts/video_to_skeleton.py --input path/to/custom_data --output data/custom
  python scripts/video_to_skeleton.py --input data/downloaded_videos --output data/custom --skip-frames 2 --workers 4  # faster

Expects videos named: <exercise>_<quality>_<number>.mp4
  e.g. wall_pushup_good_01.mp4, heel_raise_moderate_03.mp4
  quality: good (→0.85), moderate (→0.55), poor (→0.25)
"""
# Reduce log noise from TensorFlow and MediaPipe (set before imports)
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO off, 2=WARNING off, 3=ERROR off
os.environ.setdefault("GLOG_minloglevel", "2")      # MediaPipe / absl C++ logs

import sys
import argparse
import re
from multiprocessing import Pool, cpu_count

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


def process_video(
    path: str,
    extractor: PoseExtractor,
    skip_frames: int = 1,
    max_frames: int = 0,
    progress_interval: int = 0,
    progress_name: str = "",
) -> np.ndarray:
    """Extract skeleton sequence (T, 33, 3) from video. skip_frames=2 uses every 2nd frame (faster). max_frames=0 means no cap.
    If progress_interval > 0, prints progress every N frames (single-worker only)."""
    cap = cv2.VideoCapture(path)
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    n = 0
    last_report = 0
    while True:
        if max_frames > 0 and len(frames) >= max_frames:
            if progress_interval and progress_name:
                print(f"  {progress_name}: capped at {len(frames)} frames", flush=True)
            break
        ret, frame = cap.read()
        if not ret:
            if len(frames) > 0 and progress_name:
                print(f"  [{progress_name}] Video decode stopped early (corrupt/truncated?). Saved {len(frames)} frames.", flush=True)
            break
        if n % skip_frames != 0:
            n += 1
            continue
        _, landmarks = extractor.extract(frame)
        if landmarks is not None:
            frames.append(landmarks.astype(np.float32))
        n += 1
        if progress_interval and progress_name and len(frames) - last_report >= progress_interval:
            last_report = len(frames)
            total_msg = f" / ~{total_raw // max(1, skip_frames)}" if total_raw else ""
            print(f"  ... {len(frames)} frames{total_msg}", flush=True)
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else np.zeros((0, 33, 3), dtype=np.float32)


def _process_one(item):
    """Worker: process one video and save. Returns (out_dir, out_name, score) or None."""
    path, out_dir, out_name, score, min_frames, skip_frames, model_complexity, skip_existing, max_frames = item
    if skip_existing and os.path.exists(os.path.join(out_dir, out_name)):
        return "skipped"
    try:
        extractor = PoseExtractor(model_complexity=model_complexity)
        seq = process_video(path, extractor, skip_frames=skip_frames, max_frames=max_frames)
        extractor.close()
    except Exception:
        return None
    if len(seq) < min_frames:
        return None
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    np.save(out_path, seq)
    return (out_dir, out_name, score)


def main():
    parser = argparse.ArgumentParser(
        description="Convert teammate videos to skeleton .npy for training"
    )
    parser.add_argument("--input", type=str, required=True, help="Folder with exercise subfolders/videos")
    parser.add_argument("--output", type=str, default="data/custom", help="Output path (data/custom)")
    parser.add_argument("--min-frames", type=int, default=30, help="Skip videos shorter than this")
    parser.add_argument("--skip-frames", type=int, default=1, help="Use every Nth frame only (2=2x faster)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (e.g. 4 for faster)")
    parser.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe: 0=fast lite, 1=full, 2=heavy")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos that already have an output .npy (resume interrupted run)")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (e.g. 3000) to avoid hangs on long/corrupt files; 0=no cap")
    parser.add_argument("--progress", type=int, default=200, metavar="N", help="Print progress every N frames (0=off). Default 200.")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[Error] Input not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Build (path, exercise, score) then assign unique seq indices per exercise
    by_exercise = {}
    for root, _, files in os.walk(args.input):
        for f in sorted(files):
            if not f.lower().endswith(VIDEO_EXT):
                continue
            path = os.path.join(root, f)
            exercise, quality, score = parse_filename(f)
            if exercise is None:
                rel = os.path.relpath(root, args.input)
                exercise = rel.replace("\\", "/").split("/")[0] if rel != "." else "unknown"
            by_exercise.setdefault(exercise, []).append((path, score))
    tasks = []
    for exercise, items in sorted(by_exercise.items()):
        out_dir = os.path.join(args.output, exercise)
        for idx, (path, score) in enumerate(items):
            out_name = f"seq_{idx:03d}.npy"
            tasks.append((path, out_dir, out_name, score, args.min_frames, args.skip_frames, args.model_complexity, args.skip_existing, args.max_frames))

    if not tasks:
        print("[Error] No video files found.")
        sys.exit(1)

    if args.workers <= 1:
        extractor = PoseExtractor(model_complexity=args.model_complexity)
        total = 0
        max_frames = getattr(args, "max_frames", 0)
        for path, out_dir, out_name, score, min_frames, skip_frames, _, skip_existing, _ in tasks:
            out_path = os.path.join(out_dir, out_name)
            if skip_existing and os.path.exists(out_path):
                print(f"Skipping (exists) {os.path.basename(path)} -> {os.path.basename(out_dir)}/{out_name}")
                continue
            exercise = os.path.basename(out_dir)
            print(f"Processing {os.path.basename(path)} -> {exercise}/{out_name} (score={score:.2f})", flush=True)
            progress_interval = getattr(args, "progress", 0) or 0
            seq = process_video(
                path,
                extractor,
                skip_frames=skip_frames,
                max_frames=max_frames,
                progress_interval=progress_interval,
                progress_name=os.path.basename(path),
            )
            if len(seq) < min_frames:
                print(f"  Skip: too short ({len(seq)} frames)")
                continue
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, out_name), seq)
            total += 1
            labels_path = os.path.join(out_dir, "labels.csv")
            exists = os.path.exists(labels_path)
            with open(labels_path, "a", newline="", encoding="utf-8") as lf:
                if not exists:
                    lf.write("file,score\n")
                lf.write(f"{out_name},{score}\n")
        extractor.close()
    else:
        n_workers = min(args.workers, cpu_count() or 2, len(tasks))
        print(f"[Info] Using {n_workers} workers, skip_frames={args.skip_frames}, model_complexity={args.model_complexity}")
        with Pool(n_workers) as pool:
            results = pool.map(_process_one, tasks)
        total = 0
        for r in results:
            if r is None or r == "skipped":
                continue
            out_dir, out_name, score = r
            total += 1
            labels_path = os.path.join(out_dir, "labels.csv")
            exists = os.path.exists(labels_path)
            with open(labels_path, "a", newline="", encoding="utf-8") as lf:
                if not exists:
                    lf.write("file,score\n")
                lf.write(f"{out_name},{score}\n")
    print(f"\nDone. Converted {total} videos to skeleton format in {args.output}")
    print("Next: python train.py --dataset custom")


if __name__ == "__main__":
    main()
