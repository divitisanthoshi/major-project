"""
Full training pipeline: download videos (optional) → skeleton extraction → train.

Usage:
  python scripts/run_train_pipeline.py
  python scripts/run_train_pipeline.py --skip-download   # use existing data/custom or data/downloaded_videos
  python scripts/run_train_pipeline.py --download-only   # only download from config/video_sources.yaml
  python scripts/run_train_pipeline.py --skeleton-only   # only run video_to_skeleton (input: data/downloaded_videos)
"""

import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list, cwd: str = None) -> bool:
    cwd = cwd or ROOT
    print(f"[Pipeline] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true", help="Do not download; use existing videos/skeleton data")
    ap.add_argument("--download-only", action="store_true", help="Only download videos from video_sources.yaml")
    ap.add_argument("--skeleton-only", action="store_true", help="Only run video_to_skeleton (input: data/downloaded_videos)")
    ap.add_argument("--video-input", type=str, default="data/downloaded_videos", help="Input folder for video_to_skeleton")
    ap.add_argument("--skeleton-output", type=str, default="data/custom", help="Output folder for skeleton .npy")
    ap.add_argument("--model-output", type=str, default="models/rehab_model.keras", help="Output model path")
    ap.add_argument("--dataset", type=str, default="custom", help="Dataset for train.py: custom, all, synthetic, ...")
    args = ap.parse_args()

    os.chdir(ROOT)

    # 1. Download (optional)
    if not args.skip_download and not args.skeleton_only:
        config_path = os.path.join(ROOT, "config", "video_sources.yaml")
        if os.path.isfile(config_path):
            cmd = [sys.executable, "scripts/download_training_videos.py", "--config", config_path, "--output", "data/downloaded_videos"]
            github_cfg = os.path.join(ROOT, "config", "github_sources.yaml")
            if os.path.isfile(github_cfg):
                cmd.extend(["--also-github", github_cfg])
            ok = run(cmd)
            if not ok:
                print("[Pipeline] Download had errors; continuing anyway.")
        else:
            print("[Pipeline] No config/video_sources.yaml; skipping download.")
        if args.download_only:
            return

    # 2. Videos → skeleton
    if not args.download_only:
        video_input = os.path.abspath(os.path.join(ROOT, args.video_input))
        skeleton_out = os.path.abspath(os.path.join(ROOT, args.skeleton_output))
        if os.path.isdir(video_input):
            ok = run([
                sys.executable, "scripts/video_to_skeleton.py",
                "--input", video_input,
                "--output", skeleton_out,
            ])
            if not ok:
                print("[Pipeline] video_to_skeleton failed.")
                sys.exit(1)
        else:
            print(f"[Pipeline] Video input not found: {video_input}. Use --skip-download if you already have data/custom.")
        if args.skeleton_only:
            return

    # 3. Train
    train_cmd = [
        sys.executable, "train.py",
        "--dataset", args.dataset,
        "--output", args.model_output,
    ]
    if not run(train_cmd):
        sys.exit(1)
    print("[Pipeline] Done. Run: python main.py")


if __name__ == "__main__":
    main()
