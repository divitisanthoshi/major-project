"""
Motion-Based Repetition Counter for 15 exercises.
"""

import numpy as np
from collections import deque

# Joint indices: 11,12 shoulders; 15,16 wrists; 23,24 hips; 25,26 knees; 27,28 ankles
MOTION_METRICS = {
    "deep_squat": (23, 24),
    "squat": (23, 24),
    "hurdle_step": (25, 26),
    "inline_lunge": (25, 26),
    "side_lunge": (23, 24),
    "sit_to_stand": (23, 24),
    "standing_leg_raise": (25, 26),
    "leg_raise": (25, 26),
    "shoulder_abduction": (15, 16),
    "shoulder_extension": (15, 16),
    "shoulder_rotation": (15, 16),
    "shoulder_scaption": (15, 16),
    "hip_abduction": (25, 26),
    "trunk_rotation": (11, 12),
    "reach_and_retrieve": (15, 16),
    # Custom novel exercises
    "wall_pushup": (15, 16),
    "heel_raise": (27, 28),
    "bird_dog": (15, 16),
    "glute_bridge": (23, 24),
    "clamshell": (25, 26),
    "chin_tuck": (0, 11),
    "marching_in_place": (25, 26),
    "step_up": (23, 24),
}


class MotionRepCounter:
    def __init__(self, threshold: float = 0.002, min_peak_distance: int = 6, smooth_window: int = 7):
        self.threshold = threshold
        self.min_peak_distance = min_peak_distance
        self.smooth_window = smooth_window
        self.reps = 0
        self.prev_value = None
        self.direction = None
        self.frames_since_rep = 0
        self.current_exercise = "deep_squat"
        self._history = deque(maxlen=smooth_window)

    def set_exercise(self, exercise: str) -> None:
        self.current_exercise = exercise
        self._history.clear()
        self.prev_value = None
        self.direction = None

    def _get_metric(self, landmarks: np.ndarray) -> float:
        indices = MOTION_METRICS.get(self.current_exercise, (23, 24))
        if landmarks is None or len(landmarks) <= max(indices):
            return 0.0
        raw = float(np.mean([landmarks[i][1] for i in indices]))
        self._history.append(raw)
        return float(np.mean(self._history))

    def update(self, landmarks: np.ndarray) -> int:
        val = self._get_metric(landmarks)
        self.frames_since_rep += 1
        if self.prev_value is None:
            self.prev_value = val
            return self.reps
        diff = val - self.prev_value
        if abs(diff) < self.threshold:
            self.prev_value = val
            return self.reps
        new_dir = "down" if diff > 0 else "up"
        # Count one rep on down->up (e.g. squat standing up, sit-to-stand)
        if self.direction == "down" and new_dir == "up" and self.frames_since_rep >= self.min_peak_distance:
            self.reps += 1
            self.frames_since_rep = 0
        self.direction = new_dir
        self.prev_value = val
        return self.reps

    def reset(self) -> None:
        self.reps = 0
        self.prev_value = None
        self.direction = None
        self.frames_since_rep = 0
        self._history.clear()
