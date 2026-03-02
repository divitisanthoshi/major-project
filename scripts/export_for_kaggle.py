"""
Export training data to a single .npz for uploading to Kaggle.
Run locally, then upload the generated file as a Kaggle dataset for GPU training.

Usage:
  python scripts/export_for_kaggle.py
  python scripts/export_for_kaggle.py --dataset custom --output kaggle_data/data.npz
  python scripts/export_for_kaggle.py --dataset custom --include-all-demo-videos   # convert all demo videos first
"""

import os
import sys
import argparse
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import load_config
from train import get_loader, load_all_datasets

# Folders that may contain demo/training videos to convert to skeleton before export
DEMO_VIDEO_FOLDERS = ["data/downloaded_videos", "data/demos"]


def run_video_to_skeleton(input_dir: str, output_dir: str, skip_existing: bool = True) -> bool:
    """Run video_to_skeleton.py; return True if successful."""
    script = os.path.join(os.path.dirname(__file__), "video_to_skeleton.py")
    cmd = [
        sys.executable,
        script,
        "--input", input_dir,
        "--output", output_dir,
        "--skip-frames", "2",
        "--max-frames", "4000",
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Export training data for Kaggle")
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["custom", "all"],
                        help="Export 'custom' only or 'all' (UI-PRMD + KIMORE + NTU + custom)")
    parser.add_argument("--output", type=str, default="kaggle_data/data.npz",
                        help="Output .npz path (default: kaggle_data/data.npz)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--include-all-demo-videos", action="store_true",
                        help="Convert all demo videos (downloaded_videos, demos) to skeleton in data/custom before export")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    custom_out = os.path.join(project_root, "data", "custom")

    if args.include_all_demo_videos:
        print("[Export] Converting all demo videos to skeleton in data/custom ...")
        for video_dir in DEMO_VIDEO_FOLDERS:
            abs_path = os.path.join(project_root, video_dir)
            if os.path.isdir(abs_path):
                print(f"[Export] Running video_to_skeleton on {video_dir} -> data/custom")
                run_video_to_skeleton(abs_path, custom_out)
            else:
                print(f"[Export] Skip (not found): {video_dir}")
        print("[Export] Done converting demo videos.\n")

    config = load_config(args.config)
    seq_cfg = config.get("sequence", {})
    model_cfg = config.get("model", {})
    seq_len = seq_cfg.get("frame_buffer_size", 64)
    num_joints = model_cfg.get("num_joints", 33)

    if args.dataset == "all":
        print("[Export] Loading all datasets...")
        X, y, metadata = load_all_datasets(config, seq_len=seq_len, num_joints=num_joints, include_custom=True)
    else:
        loader = get_loader(args.dataset, config)
        X, y = loader.load()
        metadata = {args.dataset: len(X)}

    if len(X) == 0:
        print("[Export] No data found. Check dataset path in config.")
        sys.exit(1)

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, X=X.astype(np.float32), y=y.astype(np.float32))
    manifest = {
        "dataset": args.dataset,
        "num_samples": int(len(X)),
        "shape": list(X.shape),
        "seq_len": seq_len,
        "num_joints": num_joints,
        "metadata": metadata,
    }
    manifest_path = out_path.replace(".npz", "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Export] Saved {len(X)} sequences to {out_path}")
    print(f"[Export] Manifest: {manifest_path}")
    print("\nNext: upload kaggle_data/ as a Kaggle dataset, then run train.py with --kaggle-npz on Kaggle (GPU).")


if __name__ == "__main__":
    main()
