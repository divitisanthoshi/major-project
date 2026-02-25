"""
Dataset Exercise Reference - Exact exercises in each dataset.

Use these mappings when loading/annotating data from UI-PRMD, KIMORE, or NTU RGB+D.
"""

# UI-PRMD: 10 rehabilitation exercises
UI_PRMD_EXERCISES = [
    "Deep squat",
    "Hurdle step",
    "Inline lunge",
    "Side lunge",
    "Sit to stand",
    "Standing active straight leg raise",
    "Standing shoulder abduction",
    "Standing shoulder extension",
    "Standing shoulder internal-external rotation",
    "Standing shoulder scaption",
]

# KIMORE: 5 rehabilitation exercises (Ex_idx 1-5)
KIMORE_EXERCISES = [
    "Squat",
    "Hip abduction",
    "Sit to stand",
    "Leg raise",
    "Trunk rotation",
]

# NTU RGB+D: 60 actions (A1-A60)
NTU_RGBD_ACTIONS = [
    "Drink water", "Eat meal", "Brush teeth", "Brush hair", "Drop", "Pick up", "Throw",
    "Sit down", "Stand up", "Clapping", "Reading", "Writing", "Tear up paper",
    "Put on jacket", "Take off jacket", "Put on a shoe", "Take off a shoe",
    "Put on glasses", "Take off glasses", "Put on a hat/cap", "Take off a hat/cap",
    "Cheer up", "Hand waving", "Kicking something", "Reach into pocket", "Hopping", "Jump up",
    "Phone call", "Play with phone/tablet", "Type on a keyboard", "Point to something",
    "Taking a selfie", "Check time (from watch)", "Rub two hands", "Nod head/bow",
    "Shake head", "Wipe face", "Salute", "Put palms together", "Cross hands in front",
    "Sneeze/cough", "Staggering", "Falling down", "Headache", "Chest pain",
    "Back pain", "Neck pain", "Nausea/vomiting", "Fan self",
    "Punch/slap", "Kicking", "Pushing", "Pat on back", "Point finger", "Hugging",
    "Giving object", "Touch pocket", "Shaking hands", "Walking towards", "Walking apart",
]

# Our 15 unified exercises (dropdown + training)
# Maps our keys to dataset indices for loading
EXERCISES_15 = [
    "deep_squat", "hurdle_step", "inline_lunge", "side_lunge", "sit_to_stand",
    "standing_leg_raise", "shoulder_abduction", "shoulder_extension",
    "shoulder_rotation", "shoulder_scaption", "hip_abduction", "trunk_rotation",
    "squat", "leg_raise", "reach_and_retrieve",
]

# Novel custom exercises (NOT in UI-PRMD, KIMORE, NTU RGB+D)
# Used for our custom dataset - real-world rehab exercises
CUSTOM_NOVEL_EXERCISES = [
    "wall_pushup", "heel_raise", "bird_dog", "glute_bridge",
    "clamshell", "chin_tuck", "marching_in_place", "step_up",
]

# UI-PRMD ex_01..ex_10 -> our keys (1:1 for first 10)
UI_PRMD_TO_OUR = {
    1: "deep_squat", 2: "hurdle_step", 3: "inline_lunge", 4: "side_lunge",
    5: "sit_to_stand", 6: "standing_leg_raise", 7: "shoulder_abduction",
    8: "shoulder_extension", 9: "shoulder_rotation", 10: "shoulder_scaption",
}

# KIMORE Ex1..Ex5 -> our keys
KIMORE_TO_OUR = {1: "squat", 2: "hip_abduction", 3: "sit_to_stand", 4: "leg_raise", 5: "trunk_rotation"}

# Rehabilitation-relevant NTU actions (indices 0-based)
NTU_REHAB_INDICES = [7, 8, 24, 40, 41, 42, 43, 44, 45, 46, 47, 48]


def get_ui_prmd_exercise_name(index: int) -> str:
    """Index 0-9."""
    if 0 <= index < len(UI_PRMD_EXERCISES):
        return UI_PRMD_EXERCISES[index]
    return f"Exercise_{index}"


def get_kimore_exercise_name(index: int) -> str:
    """Index 0-4 (Ex_idx 1-5)."""
    if 0 <= index < len(KIMORE_EXERCISES):
        return KIMORE_EXERCISES[index]
    return f"Ex_{index + 1}"


def get_ntu_action_name(index: int) -> str:
    """Index 0-59 (A1-A60)."""
    if 0 <= index < len(NTU_RGBD_ACTIONS):
        return NTU_RGBD_ACTIONS[index]
    return f"A{index + 1}"
