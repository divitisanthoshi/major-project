"""
Joint Error Detection - Clinically meaningful posture checks.

Exercise-specific profiles for 15 rehab exercises (UI-PRMD + KIMORE).
"""

import numpy as np
from typing import List, Tuple

# MediaPipe: 11,12 shoulders; 13,14 elbows; 15,16 wrists; 23,24 hips; 25,26 knees; 27,28 ankles


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b formed by vectors ba and bc (degrees)."""
    a, b, c = np.array(a[:3]), np.array(b[:3]), np.array(c[:3])
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-6 or n_bc < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# (j1,j2,j3), min_deg, max_deg, feedback_msg
EXERCISE_PROFILES = {
    "deep_squat": [
        ((23, 25, 27), 70, 130, "Bend your knees more"),
        ((24, 26, 28), 70, 130, "Bend right knee more"),
        ((11, 23, 25), 140, 200, "Keep back straight"),
    ],
    "squat": [
        ((23, 25, 27), 70, 130, "Bend knees more"),
        ((24, 26, 28), 70, 130, "Bend right knee more"),
        ((11, 23, 25), 140, 200, "Keep back straight"),
    ],
    "hurdle_step": [
        ((25, 23, 24), 70, 130, "Step over hurdle properly"),
        ((26, 24, 23), 70, 130, "Step with control"),
        ((11, 23, 25), 140, 200, "Keep torso stable"),
    ],
    "inline_lunge": [
        ((25, 23, 24), 70, 120, "Bend front knee"),
        ((11, 23, 25), 140, 200, "Keep back straight"),
        ((23, 25, 27), 70, 130, "Align knee over ankle"),
    ],
    "side_lunge": [
        ((23, 25, 27), 70, 130, "Bend lunging knee"),
        ((24, 26, 28), 70, 130, "Bend right knee in lunge"),
        ((11, 23, 24), 160, 200, "Keep torso upright"),
    ],
    "sit_to_stand": [
        ((23, 25, 27), 140, 200, "Extend legs to stand"),
        ((11, 23, 25), 140, 200, "Lean forward slightly to rise"),
        ((25, 23, 24), 80, 120, "Feet flat on floor"),
    ],
    "standing_leg_raise": [
        ((23, 25, 27), 150, 200, "Lift leg higher"),
        ((24, 26, 28), 150, 200, "Lift right leg higher"),
        ((11, 23, 24), 170, 200, "Keep torso stable"),
    ],
    "leg_raise": [
        ((23, 25, 27), 150, 200, "Lift leg higher"),
        ((24, 26, 28), 150, 200, "Lift right leg higher"),
        ((11, 23, 24), 170, 200, "Keep torso stable"),
    ],
    "shoulder_abduction": [
        ((11, 13, 15), 150, 200, "Raise arm to side"),
        ((12, 14, 16), 150, 200, "Raise right arm to side"),
        ((23, 11, 12), 170, 200, "Keep shoulders level"),
    ],
    "shoulder_extension": [
        ((11, 13, 15), 100, 180, "Move arm back"),
        ((12, 14, 16), 100, 180, "Move right arm back"),
        ((23, 11, 12), 170, 200, "Keep shoulders level"),
    ],
    "shoulder_rotation": [
        ((11, 13, 15), 80, 150, "Rotate shoulder properly"),
        ((12, 14, 16), 80, 150, "Rotate right shoulder"),
        ((23, 11, 12), 80, 120, "Keep elbow at side"),
    ],
    "shoulder_scaption": [
        ((11, 13, 15), 150, 200, "Raise arm in plane"),
        ((12, 14, 16), 150, 200, "Raise right arm"),
        ((23, 11, 12), 170, 200, "Keep shoulders level"),
    ],
    "hip_abduction": [
        ((23, 25, 27), 160, 200, "Move leg out to side"),
        ((24, 26, 28), 160, 200, "Move right leg out"),
        ((11, 23, 24), 170, 200, "Keep torso upright"),
    ],
    "trunk_rotation": [
        ((11, 23, 24), 80, 120, "Rotate torso more"),
        ((12, 24, 23), 80, 120, "Rotate torso more"),
        ((23, 11, 12), 80, 120, "Keep hips forward"),
    ],
    "reach_and_retrieve": [
        ((11, 13, 15), 100, 180, "Extend arm forward"),
        ((12, 14, 16), 100, 180, "Extend right arm"),
        ((11, 23, 25), 150, 200, "Keep back straight"),
    ],
    # Custom novel exercises
    "wall_pushup": [
        ((11, 13, 15), 70, 180, "Bend arms toward wall"),
        ((12, 14, 16), 70, 180, "Bend right arm toward wall"),
        ((11, 23, 25), 140, 200, "Keep body straight"),
    ],
    "heel_raise": [
        ((23, 25, 27), 160, 200, "Keep legs straight"),
        ((24, 26, 28), 160, 200, "Keep right leg straight"),
        ((11, 23, 24), 170, 200, "Keep torso upright"),
    ],
    "bird_dog": [
        ((11, 13, 15), 150, 200, "Extend arm forward"),
        ((23, 25, 27), 150, 200, "Extend leg back"),
        ((11, 23, 24), 80, 140, "Keep spine neutral"),
    ],
    "glute_bridge": [
        ((23, 25, 27), 70, 120, "Bend knees"),
        ((24, 26, 28), 70, 120, "Bend right knee"),
        ((11, 23, 25), 140, 200, "Lift hips high"),
    ],
    "clamshell": [
        ((23, 25, 27), 60, 120, "Bend knees in clamshell"),
        ((24, 26, 28), 60, 120, "Lift top knee"),
        ((11, 23, 24), 80, 140, "Keep torso stable"),
    ],
    "chin_tuck": [
        ((0, 11, 12), 80, 140, "Keep chin retracted"),
        ((11, 23, 24), 160, 200, "Keep shoulders back"),
        ((23, 11, 12), 80, 140, "Maintain posture"),
    ],
    "marching_in_place": [
        ((23, 25, 27), 70, 130, "Lift knee"),
        ((24, 26, 28), 70, 130, "Lift right knee"),
        ((11, 23, 24), 170, 200, "Keep torso upright"),
    ],
    "step_up": [
        ((25, 23, 24), 70, 130, "Bend knee to step"),
        ((11, 23, 25), 140, 200, "Keep back straight"),
        ((23, 25, 27), 70, 150, "Align knee over ankle"),
    ],
}


def detect_errors(landmarks: np.ndarray, exercise: str) -> Tuple[List[str], List[int]]:
    errors = []
    error_joints = set()
    profile = EXERCISE_PROFILES.get(exercise, EXERCISE_PROFILES["deep_squat"])
    if landmarks is None or len(landmarks) < 28:
        return ["Move into camera view"], list(error_joints)
    for (j1, j2, j3), min_a, max_a, msg in profile:
        if j1 >= len(landmarks) or j2 >= len(landmarks) or j3 >= len(landmarks):
            continue
        a, b, c = landmarks[j1], landmarks[j2], landmarks[j3]
        angle = calculate_angle(a, b, c)
        if angle < min_a:
            errors.append(msg)
            error_joints.update([j1, j2, j3])
        elif angle > max_a:
            m = msg.replace("more", "less").replace("higher", "lower") if "more" in msg or "higher" in msg else msg
            errors.append(m)
            error_joints.update([j1, j2, j3])
    return errors[:3], list(error_joints)


def get_feedback_from_score(score: float) -> str:
    if score >= 0.75:
        return "Good movement"
    elif score >= 0.45:
        return "Almost correct - adjust slightly"
    elif score > 0:
        return "Adjust your posture"
    return "Perform the exercise"
