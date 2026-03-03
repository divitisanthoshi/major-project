"""
Create step-by-step instructional images folders for each exercise and
copy the generated PNGs from the Cursor assets directory into
`data/step_images/<exercise_key>/step_01.png`.

Usage (from project root):

    python scripts/setup_step_images.py
"""

import os
import shutil


# Absolute path where Cursor saved the generated illustrations for this project.
ASSETS_DIR = r"C:\Users\santh\.cursor\projects\c-Users-santh-OneDrive-Desktop-MPP-2\assets"


# Mapping: exercise key (as used in EXERCISE_NAMES / DROPDOWN_EXERCISES)
# -> filename of the generated illustration in ASSETS_DIR.
STEP_IMAGE_MAP = {
    "deep_squat": "deep_squat_steps.png",
    "hurdle_step": "hurdle_step_steps.png",
    "inline_lunge": "inline_lunge_steps.png",
    "side_lunge": "side_lunge_steps.png",
    "sit_to_stand": "sit_to_stand_steps.png",
    "standing_leg_raise": "standing_leg_raise_steps.png",
    "shoulder_abduction": "shoulder_abduction_steps.png",
    "shoulder_extension": "shoulder_extension_steps.png",
    "shoulder_rotation": "shoulder_rotation_steps.png",
    "shoulder_scaption": "shoulder_scaption_steps.png",
    "hip_abduction": "hip_abduction_steps.png",
    "trunk_rotation": "trunk_rotation_steps.png",
    "squat": "squat_steps.png",
    "leg_raise": "leg_raise_steps.png",
    "reach_and_retrieve": "reach_and_retrieve_steps.png",
    "wall_pushup": "wall_pushup_steps.png",
    "heel_raise": "heel_raise_steps.png",
    "bird_dog": "bird_dog_steps.png",
    "glute_bridge": "glute_bridge_steps.png",
    "clamshell": "clamshell_steps.png",
    "chin_tuck": "chin_tuck_steps.png",
    "marching_in_place": "marching_in_place_steps.png",
    "step_up": "step_up_steps.png",
}


def get_project_root() -> str:
    """Return absolute path to project root (one level above scripts folder)."""
    here = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(here)
    return os.path.dirname(scripts_dir)


def ensure_step_image(exercise_key: str, filename: str, dest_base: str) -> bool:
    """
    Copy a single step image for one exercise.

    Returns True on success, False if source file is missing.
    """
    src = os.path.join(ASSETS_DIR, filename)
    if not os.path.isfile(src):
        print(f"[WARN] Source image not found for {exercise_key}: {src}")
        return False

    dst_dir = os.path.join(dest_base, exercise_key)
    os.makedirs(dst_dir, exist_ok=True)

    _, ext = os.path.splitext(filename)
    dst = os.path.join(dst_dir, f"step_01{ext.lower()}")

    shutil.copy2(src, dst)
    print(f"[OK] {exercise_key}: {src} -> {dst}")
    return True


def main() -> None:
    project_root = get_project_root()
    dest_base = os.path.join(project_root, "data", "step_images")
    os.makedirs(dest_base, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Assets dir:   {ASSETS_DIR}")
    print(f"Target base:  {dest_base}")
    print()

    ok = 0
    total = len(STEP_IMAGE_MAP)
    for ex_key, fname in STEP_IMAGE_MAP.items():
        if ensure_step_image(ex_key, fname, dest_base):
            ok += 1

    print()
    print(f"Done. Copied {ok}/{total} step images.")
    print("Restart the app and choose an exercise to see the new step image in panel 2.")


if __name__ == "__main__":
    main()

