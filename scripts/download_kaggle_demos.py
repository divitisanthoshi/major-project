#!/usr/bin/env python3
"""
Download an exercise video dataset from Kaggle and copy one video per exercise
into data/demos so the app can use them as reference clips.

Usage:
  pip install kaggle
  # Auth: put kaggle.json in C:\\Users\\<you>\\.kaggle\\kaggle.json
  #   OR set env: $env:KAGGLE_API_TOKEN="your_token"  (PowerShell) / set KAGGLE_API_TOKEN=your_token (cmd)
  python scripts/download_kaggle_demos.py hasyimabdillah/workoutfitness-video --out data/demos

Supported dataset: hasyimabdillah/workoutfitness-video (folders per exercise, videos inside).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

# Exercise keys used by the app (must match main.EXERCISE_NAMES)
EXERCISE_KEYS = [
    "deep_squat", "hurdle_step", "inline_lunge", "side_lunge", "sit_to_stand",
    "standing_leg_raise", "shoulder_abduction", "shoulder_extension", "shoulder_rotation",
    "shoulder_scaption", "hip_abduction", "trunk_rotation", "squat", "leg_raise",
    "reach_and_retrieve", "wall_pushup", "heel_raise", "bird_dog", "glute_bridge",
    "clamshell", "chin_tuck", "marching_in_place", "step_up",
]

DEMO_EXTENSIONS = (".mp4", ".webm", ".avi")

# Map Kaggle folder name (normalized: lowercase, no spaces) -> our exercise key
# Extend this for your dataset's folder names
FOLDER_TO_KEY = {
    "squat": "squat",
    "deepsquat": "deep_squat",
    "side_lunge": "side_lunge",
    "sidelunge": "side_lunge",
    "inline_lunge": "inline_lunge",
    "inlinelunge": "inline_lunge",
    "lunge": "inline_lunge",
    "leg_raise": "leg_raise",
    "legraise": "leg_raise",
    "standing_leg_raise": "standing_leg_raise",
    "standinglegraise": "standing_leg_raise",
    "bird_dog": "bird_dog",
    "birddog": "bird_dog",
    "glute_bridge": "glute_bridge",
    "glutebridge": "glute_bridge",
    "hip_abduction": "hip_abduction",
    "hipabduction": "hip_abduction",
    "clamshell": "clamshell",
    "trunk_rotation": "trunk_rotation",
    "trunkrotation": "trunk_rotation",
    "shoulder_abduction": "shoulder_abduction",
    "shoulderabduction": "shoulder_abduction",
    "shoulder_rotation": "shoulder_rotation",
    "shoulderrotation": "shoulder_rotation",
    "shoulder_extension": "shoulder_extension",
    "shoulderextension": "shoulder_extension",
    "sit_to_stand": "sit_to_stand",
    "sittostand": "sit_to_stand",
    "step_up": "step_up",
    "stepup": "step_up",
    "wall_pushup": "wall_pushup",
    "wallpushup": "wall_pushup",
    "pushup": "wall_pushup",
    "heel_raise": "heel_raise",
    "heelraise": "heel_raise",
    "marching_in_place": "marching_in_place",
    "marchinginplace": "marching_in_place",
    "chin_tuck": "chin_tuck",
    "chintuck": "chin_tuck",
    "hurdle_step": "hurdle_step",
    "hurdlestep": "hurdle_step",
    "reach_and_retrieve": "reach_and_retrieve",
    "reachandretrieve": "reach_and_retrieve",
    "shoulder_scaption": "shoulder_scaption",
    "shoulderscaption": "shoulder_scaption",
    # workoutfitness-video dataset folder
    "barbell_biceps_curl": "shoulder_rotation",
    "barbellbicepscurl": "shoulder_rotation",
}


def normalize_name(s: str) -> str:
    """Lowercase, replace spaces/special with underscore, collapse underscores."""
    s = s.lower().strip()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def folder_name_to_key(folder_name: str) -> str | None:
    normalized = normalize_name(folder_name)
    if normalized in FOLDER_TO_KEY:
        return FOLDER_TO_KEY[normalized]
    if normalized in EXERCISE_KEYS:
        return normalized
    return None


def find_videos_by_folder(root: Path) -> dict[str, list[Path]]:
    """Group video paths by parent folder name (exercise hint)."""
    by_folder: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DEMO_EXTENSIONS:
            continue
        # Use parent folder name as exercise hint (dataset often has one folder per exercise)
        folder = path.parent.name
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(path)
    return by_folder


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Kaggle exercise video dataset into data/demos")
    parser.add_argument("dataset", nargs="?", default="hasyimabdillah/workoutfitness-video", help="Kaggle dataset (default: hasyimabdillah/workoutfitness-video)")
    parser.add_argument("--out", default="data/demos", help="Output directory (default: data/demos)")
    parser.add_argument("--no-download", action="store_true", help="Use existing download; dataset must be in cwd as zip or folder")
    parser.add_argument("--local-only", action="store_true", help="Only fill data/demos from demos_cache and downloaded_videos (no Kaggle)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    cwd = project_root

    # Download (or use existing) unless local-only — use Python API so CLI PATH is not required
    if not args.local_only and not args.no_download:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("Downloading dataset (may take a few minutes)...", flush=True)
            api.dataset_download_files(args.dataset, path=str(cwd), unzip=True)
            print("Download complete.", flush=True)
        except ImportError:
            print("Kaggle package not found. Install: pip install kaggle", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Kaggle download failed: {e}", file=sys.stderr)
            return 1

    copied_kaggle = 0
    if not args.local_only:
        # Locate extracted folder or zip (Kaggle may create dataset name folder or zip)
        slug = args.dataset.split("/")[-1]
        extracted = cwd / slug
        zip_path = cwd / f"{slug}.zip"
        if not extracted.is_dir() and zip_path.is_file():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(cwd)
        if not extracted.is_dir():
            for d in cwd.iterdir():
                if not d.is_dir() or d.name.startswith("."):
                    continue
                if d.name in ("scripts", "data", "src", "kaggle_data", "models", "config"):
                    continue
                if any(d.rglob("*" + ext) for ext in DEMO_EXTENSIONS):
                    extracted = d
                    break
        if not extracted.is_dir():
            print(f"No dataset folder or zip found at {cwd}. Check Kaggle download path.", flush=True)
        else:
            by_folder = find_videos_by_folder(extracted)
            folder_names = sorted(by_folder.keys())
            if folder_names:
                print(f"Dataset folders ({len(folder_names)}): {', '.join(folder_names[:15])}{'...' if len(folder_names) > 15 else ''}", flush=True)
            for folder_name, paths in by_folder.items():
                key = folder_name_to_key(folder_name)
                if not key:
                    continue
                dest = out_dir / f"{key}.mp4"
                if dest.exists():
                    continue
                paths_sorted = sorted(paths, key=lambda p: (0 if p.suffix.lower() == ".mp4" else 1, p.name))
                shutil.copy2(paths_sorted[0], dest)
                print(f"Kaggle: {folder_name} -> {dest}")
                copied_kaggle += 1

    copied_local = 0

    # Fill missing demos from existing local folders so every UI exercise has a demo
    demos_cache = project_root / "data" / "demos_cache"
    downloaded_videos = project_root / "data" / "downloaded_videos"
    for key in EXERCISE_KEYS:
        dest = out_dir / f"{key}.mp4"
        if dest.exists():
            continue
        src_path = None
        # Try demos_cache first, then downloaded_videos/<exercise>/
        cache_mp4 = demos_cache / f"{key}.mp4"
        if cache_mp4.is_file():
            src_path = cache_mp4
        else:
            ex_dir = downloaded_videos / key
            if ex_dir.is_dir():
                videos = sorted(ex_dir.glob("*.mp4"))  # skip .part
                if videos:
                    src_path = videos[0]
        if src_path:
            shutil.copy2(src_path, dest)
            print(f"Local: {src_path.name} -> {dest}")
            copied_local += 1

    total_in_dir = sum(1 for _ in out_dir.glob("*.mp4"))
    missing = [k for k in EXERCISE_KEYS if not (out_dir / f"{k}.mp4").exists()]
    if missing:
        print(f"Missing demos ({len(missing)}): {', '.join(missing)}")
    print(f"Done. {copied_kaggle} from Kaggle, {copied_local} from local. Total in {out_dir}: {total_in_dir} demo(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
