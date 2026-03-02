"""
Build a complete package to upload to Kaggle: code + config + pose detection + training.

Includes everything needed to run on Kaggle:
- Pose detection (MediaPipe): src/pose_extraction.py, video_to_skeleton.py
- Training: train.py, config/, src/models/, src/datasets/, etc.
- Scripts: video_to_skeleton, export_for_kaggle, prepare_exercises_videos

Usage:
  python scripts/build_kaggle_package.py
  python scripts/build_kaggle_package.py --include-data   # also copy kaggle_data/data.npz into package
  python scripts/build_kaggle_package.py --output my_kaggle.zip  # create zip directly

Then upload the generated folder (or zip) as a Kaggle dataset. In your notebook,
add this dataset to get full code + pose detection + training in one place.
"""

import os
import shutil
import argparse
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# What to copy into the Kaggle package (relative to project root)
CONFIG_DIR = "config"
SRC_DIR = "src"
TRAIN_PY = "train.py"
SCRIPTS_TO_COPY = [
    "video_to_skeleton.py",
    "export_for_kaggle.py",
    "prepare_exercises_videos_for_training.py",
    "run_train_pipeline.py",
    "images_to_skeleton.py",
]
REQUIREMENTS = "requirements.txt"
DOCS = "docs/KAGGLE_TRAINING_STEPS.md"

# Optional: include pre-exported data so one dataset = code + data
KAGGLE_DATA_NPZ = "kaggle_data/data.npz"
KAGGLE_DATA_MANIFEST = "kaggle_data/data_manifest.json"

DEFAULT_OUTPUT_DIR = "kaggle_package"
README_KAGGLE = """# Rehab Exercise Grading - Kaggle Package

This package contains the full project for training and pose detection.

## Contents
- **config/** – Configuration (sequence length, model, etc.)
- **src/** – Source code: pose extraction (MediaPipe), models (ST-GCN), datasets, preprocessing
- **train.py** – Training script (supports --kaggle-npz)
- **scripts/** – video_to_skeleton.py (pose from video), export_for_kaggle.py, etc.

## On Kaggle
1. Upload this folder as a **Dataset** (e.g. name: `rehab-exercise-code`).
2. If you have pre-extracted data, upload **data.npz** as another dataset (or it may be included here).
3. In a **Notebook**, add both datasets. Set **Accelerator** to **GPU**.
4. Run:

```python
import os
# Path to this code (adjust your dataset slug)
CODE_DIR = "/kaggle/input/rehab-exercise-code"  # or wherever Kaggle mounts it
os.chdir(CODE_DIR)

# If data.npz is in this dataset:
NPZ = CODE_DIR + "/data.npz"
# Or if data is a separate dataset:
# NPZ = "/kaggle/input/rehab-exercise-skeletons/data.npz"

!python train.py --kaggle-npz "$NPZ" --output /kaggle/working/rehab_model.keras --model stgcn
```

## Pose detection on Kaggle
If you uploaded **videos** (e.g. Exercises Videos) as a dataset:

```python
import os
os.chdir("/kaggle/input/rehab-exercise-code")
# Input: folder of videos (exercise subfolders with *_good_*.mp4)
!python scripts/video_to_skeleton.py --input /kaggle/input/your-videos-dataset/videos --output /kaggle/working/data/custom --skip-frames 2
!python scripts/export_for_kaggle.py --dataset custom --output /kaggle/working/kaggle_data/data.npz
# Then train on /kaggle/working/kaggle_data/data.npz
```

See **KAGGLE_TRAINING_STEPS.md** in this package for full steps.
"""


def main():
    ap = argparse.ArgumentParser(description="Build Kaggle package (code + pose + training)")
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output folder (default: kaggle_package)")
    ap.add_argument("--include-data", action="store_true", help="Copy kaggle_data/data.npz (and manifest) into package")
    ap.add_argument("--zip", type=str, default=None, metavar="FILE", help="Create zip file (e.g. kaggle_upload.zip)")
    args = ap.parse_args()

    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def copy_path(rel: str, dest_name: str = None) -> None:
        src = PROJECT_ROOT / rel
        dst = out / (dest_name or rel)
        if not src.exists():
            print(f"[Skip] Missing: {rel}")
            return
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(f"  + {rel}")

    # Config
    copy_path(CONFIG_DIR)
    # Source
    copy_path(SRC_DIR)
    # Train
    copy_path(TRAIN_PY)
    # Scripts
    scripts_dst = out / "scripts"
    scripts_dst.mkdir(exist_ok=True)
    for name in SCRIPTS_TO_COPY:
        src = PROJECT_ROOT / "scripts" / name
        if src.exists():
            shutil.copy2(src, scripts_dst / name)
            print(f"  + scripts/{name}")
    # Requirements
    copy_path(REQUIREMENTS)
    # Docs
    if (PROJECT_ROOT / DOCS).exists():
        (out / "docs").mkdir(exist_ok=True)
        shutil.copy2(PROJECT_ROOT / DOCS, out / DOCS)
        print(f"  + {DOCS}")

    # README for Kaggle
    (out / "README_KAGGLE.md").write_text(README_KAGGLE, encoding="utf-8")
    print("  + README_KAGGLE.md")

    # Optional: include data.npz
    if args.include_data:
        for rel in (KAGGLE_DATA_NPZ, KAGGLE_DATA_MANIFEST):
            src = PROJECT_ROOT / rel
            if src.exists():
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  + {rel}")
            else:
                print(f"[Skip] Not found: {rel}")

    print(f"\nPackage built: {out}")
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_absolute():
            zip_path = PROJECT_ROOT / zip_path
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", out)
        print(f"Zip created: {zip_path}.zip")
    else:
        print("To create a zip: run again with --zip kaggle_upload.zip")
    print("\nNext: Upload this folder (or zip) as a Kaggle dataset. See README_KAGGLE.md inside the package.")
    return 0


if __name__ == "__main__":
    exit(main())
