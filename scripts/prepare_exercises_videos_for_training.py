"""
Prepare 'Exercises Videos' folder for training: copy videos into the structure
expected by video_to_skeleton.py (exercise subfolders, names like <exercise>_good_01.mp4).

Usage:
  python scripts/prepare_exercises_videos_for_training.py
  python scripts/prepare_exercises_videos_for_training.py --input "Exercises Videos" --output data/exercises_videos_import

Then run:
  python scripts/video_to_skeleton.py --input data/exercises_videos_import --output data/custom
  python scripts/export_for_kaggle.py --dataset custom --output kaggle_data/data.npz
"""

import os
import re
import shutil
import argparse
from typing import Optional

# Exercise key -> possible filename stems (without extension). First match wins.
# Built from UI exercise names + common variants in "Exercises Videos" folder.
DISPLAY_TO_KEY = {
    "bird dog": "bird_dog",
    "bird_dog": "bird_dog",
    "chin tuck": "chin_tuck",
    "chin_tuck": "chin_tuck",
    "clamshell": "clamshell",
    "clamshells": "clamshell",
    "deep squat": "deep_squat",
    "glute bridge": "glute_bridge",
    "glute_bridge": "glute_bridge",
    "heel raise": "heel_raise",
    "heel raises": "heel_raise",
    "hip abduction": "hip_abduction",
    "hip adduction": "hip_abduction",  # "Hip Adduction + Abduction"
    "hurdle step": "hurdle_step",
    "inline lunge": "inline_lunge",
    "leg raise": "leg_raise",
    "leg raises": "leg_raise",
    "lying leg raises": "leg_raise",
    "marching in place": "marching_in_place",
    "reach and retrieve": "reach_and_retrieve",
    "shoulder abduction": "shoulder_abduction",
    "shoulder extension": "shoulder_extension",
    "shoulder flexion": "shoulder_extension",  # "Shoulder Flexion + extension"
    "shoulder rotation": "shoulder_rotation",
    "shoulder scaption": "shoulder_scaption",
    "side lunge": "side_lunge",
    "sit to stand": "sit_to_stand",
    "squat": "squat",
    "squats": "squat",
    "standing leg raise": "standing_leg_raise",
    "standing leg raises": "standing_leg_raise",
    "step up": "step_up",
    "step_up": "step_up",
    "trunk rotation": "trunk_rotation",
    "trunk rotations": "trunk_rotation",
    "wall push up": "wall_pushup",
    "wall push ups": "wall_pushup",
    "wall_pushup": "wall_pushup",
    "functional movement screen": "deep_squat",  # FMS often used for deep squat
}

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")
DEFAULT_SCORE = 0.85  # treat demo videos as "good" for training


def normalize_stem(name: str) -> str:
    """Lowercase, remove extension, collapse spaces/dashes to single space."""
    base = os.path.splitext(name)[0].strip().lower()
    base = re.sub(r"[-–—]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def stem_to_exercise_key(stem: str) -> Optional[str]:
    """Match filename stem to exercise key. Returns None if no match."""
    stem = normalize_stem(stem)
    # Long YouTube titles: take first part before " - "
    if " - " in stem:
        stem = stem.split(" - ")[0].strip()
    if stem in DISPLAY_TO_KEY:
        return DISPLAY_TO_KEY[stem]
    # Partial match: e.g. "stop doing your squats like this" -> squat
    for display, key in DISPLAY_TO_KEY.items():
        if display in stem or stem in display:
            return key
    # Try exact key
    key_candidate = stem.replace(" ", "_")
    if key_candidate in set(DISPLAY_TO_KEY.values()):
        return key_candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Exercises Videos folder for training (skeleton extraction)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Exercises Videos",
        help="Input folder containing exercise videos (default: Exercises Videos)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/exercises_videos_import",
        help="Output folder for structured copies (default: data/exercises_videos_import)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files (default). Use --symlink to symlink instead.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying (saves disk space).",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(project_root, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.isdir(input_dir):
        print(f"[Error] Input folder not found: {input_dir}")
        return 1

    use_symlink = args.symlink
    by_exercise = {}  # exercise_key -> list of (src_path, stem)

    for f in sorted(os.listdir(input_dir)):
        if not f.lower().endswith(VIDEO_EXT):
            continue
        path = os.path.join(input_dir, f)
        if not os.path.isfile(path):
            continue
        stem = normalize_stem(f)
        key = stem_to_exercise_key(stem)
        if key is None:
            print(f"[Skip] No exercise match for: {f}")
            continue
        by_exercise.setdefault(key, []).append((path, stem))

    if not by_exercise:
        print("[Error] No videos could be matched to exercises. Check filenames.")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    total = 0
    for exercise in sorted(by_exercise.keys()):
        out_sub = os.path.join(output_dir, exercise)
        os.makedirs(out_sub, exist_ok=True)
        for idx, (src, _) in enumerate(by_exercise[exercise], start=1):
            dest_name = f"{exercise}_good_{idx:02d}.mp4"
            dest = os.path.join(out_sub, dest_name)
            if use_symlink:
                if os.path.lexists(dest):
                    os.remove(dest)
                os.symlink(os.path.abspath(src), dest)
            else:
                shutil.copy2(src, dest)
            total += 1
            print(f"  {exercise}/{dest_name} <- {os.path.basename(src)}")

    print(f"\nDone. Prepared {total} videos in {output_dir}")
    print("\nNext steps:")
    print(f"  1. python scripts/video_to_skeleton.py --input {args.output} --output data/custom")
    print("  2. python scripts/export_for_kaggle.py --dataset custom --output kaggle_data/data.npz")
    print("  3. Upload kaggle_data/ to Kaggle and run training (see docs/KAGGLE_TRAINING_STEPS.md)")
    return 0


if __name__ == "__main__":
    exit(main())
