"""
Realistic Synthetic Rehabilitation Data

Generates skeleton sequences with exercise-like motion patterns.
Quality score correlates with motion smoothness and consistency.
"""

import numpy as np
from typing import Tuple

# Joint indices for different exercise motion types
# 23,24 = hips, 25,26 = knees, 15,16 = wrists, 11,12 = shoulders
HIP_JOINTS = (23, 24)
KNEE_JOINTS = (25, 26)
WRIST_JOINTS = (15, 16)
SHOULDER_JOINTS = (11, 12)


def _motion_squat(t: np.ndarray, phase: float = 0) -> np.ndarray:
    """Hip Y oscillates (squat down/up)."""
    return 0.4 + 0.3 * np.sin(2 * np.pi * t / 30 + phase)


def _motion_arm_raise(t: np.ndarray, phase: float = 0) -> np.ndarray:
    """Wrist Y oscillates (arm up/down)."""
    return 0.6 - 0.25 * np.sin(2 * np.pi * t / 25 + phase)


def _motion_leg_raise(t: np.ndarray, phase: float = 0) -> np.ndarray:
    """Knee Y oscillates."""
    return 0.5 + 0.2 * np.sin(2 * np.pi * t / 28 + phase)


def _motion_trunk(t: np.ndarray, phase: float = 0) -> np.ndarray:
    """Shoulder X oscillates (trunk rotation)."""
    return 0.5 + 0.2 * np.sin(2 * np.pi * t / 35 + phase)


def generate_exercise_sequence(
    seq_len: int,
    num_joints: int,
    exercise_type: int,
    quality: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one sequence with exercise-like motion.
    quality in [0,1]: higher = smoother, more consistent motion.
    """
    t = np.arange(seq_len, dtype=np.float32)
    base = np.zeros((seq_len, num_joints, 3), dtype=np.float32)
    phase = rng.random() * 2 * np.pi

    motion_types = [_motion_squat, _motion_arm_raise, _motion_leg_raise, _motion_trunk]
    joint_groups = [HIP_JOINTS, WRIST_JOINTS, KNEE_JOINTS, SHOULDER_JOINTS]
    axes = [1, 1, 1, 0]  # Y for most, X for trunk

    ex_idx = exercise_type % len(motion_types)
    motion_fn = motion_types[ex_idx]
    joints = joint_groups[ex_idx]
    axis = axes[ex_idx]

    signal = motion_fn(t, phase)
    noise_scale = 0.02 * (1.0 - quality) + 0.002
    signal = signal + rng.normal(0, noise_scale, seq_len)
    signal = np.clip(signal, 0.1, 0.9)

    for j in joints:
        if j < num_joints:
            base[:, j, axis] = signal

    for j in range(num_joints):
        if j not in joints:
            base[:, j, :] = rng.normal(0, 0.03, (seq_len, 3)) + np.array([0.5, 0.5, 0])

    base = np.clip(base, -0.5, 1.5)
    return base.astype(np.float32)


def generate_realistic_synthetic(
    n_samples: int = 1500,
    seq_len: int = 64,
    num_joints: int = 33,
    n_exercises: int = 15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with exercise-like motion.
    Scores correlate with motion quality (smoothness).
    """
    rng = np.random.default_rng(seed)
    sequences = []
    scores = []

    for _ in range(n_samples):
        quality = rng.beta(2, 2)
        ex_type = rng.integers(0, n_exercises)
        seq = generate_exercise_sequence(seq_len, num_joints, ex_type, quality, rng)
        sequences.append(seq)
        scores.append(quality)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(scores, dtype=np.float32)
    return X, y
