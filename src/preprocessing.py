"""
Pose Preprocessing Module

Handles:
- Normalization of joint coordinates
- Conversion to graph-structured input format
- Temporal frame buffer for sequence modeling
"""

import numpy as np
from typing import Optional
from src.utils.graph_utils import build_adjacency_normalized


class PoseNormalizer:
    """
    Normalizes pose landmarks for stable model input.

    Strategies:
    - Center: subtract hip center (or torso center)
    - Scale: normalize by skeleton scale (e.g., shoulder-hip distance)
    - Optional: z-score normalization
    """

    def __init__(self, use_scale: bool = True):
        self.use_scale = use_scale

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks: center + optional scale.

        Args:
            landmarks: (num_joints, 3) x,y,z

        Returns:
            Normalized landmarks same shape.
        """
        out = landmarks.copy()

        # Center: use midpoint of hips (23, 24) as origin
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        center = (left_hip + right_hip) / 2
        out = out - center

        # Scale: use shoulder-hip distance to normalize size
        if self.use_scale:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale > 1e-6:
                out = out / scale

        return out.astype(np.float32)


class PoseBuffer:
    """
    Temporal buffer for frame sequences.

    Maintains a sliding window of pose frames for sequence-based prediction.
    """

    def __init__(self, maxlen: int = 64):
        """
        Args:
            maxlen: Maximum sequence length (frames).
        """
        self.maxlen = maxlen
        self.buffer: list = []
        self.normalizer = PoseNormalizer()

    def add(self, landmarks: np.ndarray) -> None:
        """Add a frame of landmarks. Normalizes before storing."""
        norm = self.normalizer.normalize(landmarks)
        self.buffer.append(norm)
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)

    def is_ready(self) -> bool:
        """True if buffer has enough frames for prediction."""
        return len(self.buffer) >= self.maxlen

    def get_sequence(self) -> np.ndarray:
        """
        Get current sequence as numpy array.

        Returns:
            Shape (T, V, C) = (maxlen, num_joints, 3).
        """
        seq = np.array(self.buffer[-self.maxlen :], dtype=np.float32)
        return seq

    def clear(self) -> None:
        """Reset buffer."""
        self.buffer.clear()


def sequence_motion_energy(sequence: np.ndarray) -> float:
    """
    Compute how much the pose moves over the sequence (for idle vs active gating).

    Uses mean frame-to-frame L2 change of joint positions across the sequence.
    Returns a non-negative scalar; near 0 = almost no motion, higher = active movement.

    Args:
        sequence: (T, V, C) e.g. (64, 33, 3).

    Returns:
        Motion energy (float).
    """
    if sequence is None or sequence.shape[0] < 2:
        return 0.0
    # (T, V, C) -> frame diffs (T-1, V, C)
    diffs = np.diff(sequence, axis=0)
    # Per-frame magnitude: (T-1, V)
    frame_delta = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Mean over time and joints
    return float(np.mean(frame_delta))


def recent_motion_energy(sequence: np.ndarray, last_n_frames: int = 16) -> float:
    """
    Motion energy over the last N frames only (so we only show quality when moving *now*).

    When the user stops moving, the buffer still has old motion at the start; using only
    the recent tail ensures we stop updating the score as soon as they sit still.
    Aligns with temporal analysis in skeleton-based rehab assessment (e.g. ST-GCN for
    rehab: Deb et al., IEEE TNSRE 2022; https://github.com/fokhruli/STGCN-rehab).

    Args:
        sequence: (T, V, C) e.g. (64, 33, 3).
        last_n_frames: Number of most recent frames to consider.

    Returns:
        Motion energy over the recent window (float).
    """
    if sequence is None or sequence.shape[0] < 2:
        return 0.0
    T = sequence.shape[0]
    n = min(last_n_frames, T - 1)
    if n < 1:
        return 0.0
    tail = sequence[-n - 1 :]  # (n+1, V, C)
    diffs = np.diff(tail, axis=0)
    frame_delta = np.sqrt(np.sum(diffs ** 2, axis=2))
    return float(np.mean(frame_delta))


def prepare_graph_input(
    sequence: np.ndarray,
    adj: Optional[np.ndarray] = None,
    num_joints: int = 33,
) -> tuple:
    """
    Prepare input for ST-GCN: (X, A).

    Args:
        sequence: (T, V, C) - time, vertices (joints), channels.
        adj: Precomputed adjacency. If None, builds from graph_utils.
        num_joints: Number of joints.

    Returns:
        (sequence, adjacency) - both as numpy arrays.
    """
    if adj is None:
        adj = build_adjacency_normalized(num_joints)
    return sequence, adj
