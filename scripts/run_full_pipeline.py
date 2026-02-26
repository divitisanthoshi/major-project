"""
Run the full training pipeline with no manual steps: download videos → skeleton extraction → train.

Usage:
  python scripts/run_full_pipeline.py
  python scripts/run_full_pipeline.py --no-download   # skip download, use existing videos
  python scripts/run_full_pipeline.py --skeleton-only # stop after skeleton extraction (no train)
  python scripts/run_full_pipeline.py --train-only   # skip download and skeleton (use existing data/custom)
  python scripts/run_full_pipeline.py --train-background  # run training in background (you can do other work)

Training time (CPU, ~39k sequences): --fast ~1–3 hours (25 epochs, early stop); full 100 epochs ~20–40+ hours.
"""

import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list, step_name: str, background: bool = False) -> bool:
    """Run a command; return True on success, False on failure. If background=True, start and return immediately."""
    print(f"\n{'='*60}\n[Pipeline] {step_name}\n{'='*60}")
    sys.stdout.flush()
    sys.stderr.flush()
    if background:
        log_path = os.path.join(PROJECT_ROOT, "train_log.txt")
        with open(log_path, "w", encoding="utf-8") as logf:
            p = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        print(f"\n[Pipeline] Training running in background (PID {p.pid}).")
        print(f"  Log: {log_path}")
        print("  Model will be saved to models/rehab_model.keras when done.")
        print("  Leave this terminal open so training can finish.")
        return True
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[Pipeline] FAILED: {step_name} (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full training pipeline: download → video_to_skeleton → train (no manual steps)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip video download; use existing data/downloaded_videos",
    )
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help="Stop after skeleton extraction; do not run train",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip download and skeleton; only run train.py on existing data/custom",
    )
    parser.add_argument(
        "--video-config",
        type=str,
        default="config/video_sources.yaml",
        help="Config for download step (default: config/video_sources.yaml)",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default="data/downloaded_videos",
        help="Output dir for downloaded videos (default: data/downloaded_videos)",
    )
    parser.add_argument(
        "--skeleton-output",
        type=str,
        default="data/custom",
        help="Output dir for skeleton .npy (default: data/custom)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast training (25 epochs, batch 64). ~1–3 hrs on CPU with early stopping.",
    )
    parser.add_argument(
        "--train-background",
        action="store_true",
        help="Run training in background; pipeline exits while train continues. Log: train_log.txt",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if not args.train_only:
        if not args.no_download:
            download_cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "download_training_videos.py"),
                "--config", args.video_config,
                "--output", args.video_output,
            ]
            if not run(download_cmd, "Download training videos"):
                sys.exit(1)

        skeleton_cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "video_to_skeleton.py"),
            "--input", args.video_output,
            "--output", args.skeleton_output,
            "--skip-existing",
            "--max-frames", "4000",
            "--progress", "200",
        ]
        if not run(skeleton_cmd, "Video → skeleton (skip existing)"):
            sys.exit(1)

    if args.skeleton_only:
        print("\n[Pipeline] Skeleton-only run finished. Skipping train.")
        return

    train_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "train.py"),
        "--dataset", "custom",
    ]
    if args.fast:
        train_cmd.append("--fast")
        print("[Pipeline] Using --fast (25 epochs, batch 64).")
    if not run(train_cmd, "Train model (dataset=custom)", background=args.train_background):
        sys.exit(1)

    if args.train_background:
        print("\n[Pipeline] All steps done. Training continues in background.")
    else:
        print("\n[Pipeline] All steps completed successfully.")


if __name__ == "__main__":
    main()
