"""
Graph Utilities - Skeleton Connectivity for ST-GCN

Defines the human skeleton as a graph where:
- Nodes = joints (MediaPipe 33 landmarks)
- Edges = body connectivity (bones)

MediaPipe Pose landmark indices: https://google.github.io/mediapipe/solutions/pose.html
"""

import numpy as np
from typing import List, Tuple


# MediaPipe Pose POSE_CONNECTIONS - pairs of joint indices forming edges.
# Format: (start_idx, end_idx) for each bone.
MEDIAPIPE_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),   # Face / nose to ears
    (0, 4), (4, 5), (5, 6), (6, 8),   # Left face
    (9, 10),                            # Mouth
    (11, 12),                           # Shoulders
    (11, 13), (13, 15),                 # Left arm
    (12, 14), (14, 16),                 # Right arm
    (11, 23), (12, 24),                 # Torso from shoulders
    (23, 24),                           # Hip center
    (23, 25), (25, 27),                 # Left leg
    (24, 26), (26, 28),                 # Right leg
    (27, 29), (29, 31),                 # Left foot
    (28, 30), (30, 32),                 # Right foot
]

# Additional edges for symmetric graph (bidirectional) - some papers use this.
# We build adjacency from the connection list.


def build_adjacency_matrix(num_joints: int = 33) -> np.ndarray:
    """
    Build adjacency matrix A for the skeleton graph.

    A[i,j] = 1 if there is an edge between joints i and j.

    Args:
        num_joints: Number of joints (33 for MediaPipe).

    Returns:
        Adjacency matrix of shape (num_joints, num_joints).
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in MEDIAPIPE_SKELETON:
        if i < num_joints and j < num_joints:
            A[i, j] = 1
            A[j, i] = 1  # Undirected
    return A


def build_adjacency_normalized(num_joints: int = 33) -> np.ndarray:
    """
    Build normalized adjacency: D^{-1/2} A D^{-1/2} for symmetric normalization.

    Used in graph convolution for stable training.

    Returns:
        Normalized adjacency of shape (num_joints, num_joints).
    """
    A = build_adjacency_matrix(num_joints)
    D = np.sum(A, axis=1).astype(np.float64)
    D_inv_sqrt = np.zeros_like(D)
    mask = D > 0
    D_inv_sqrt[mask] = 1.0 / np.sqrt(D[mask])
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    return A_norm.astype(np.float32)


def get_skeleton_edges() -> List[Tuple[int, int]]:
    """Return list of (i, j) edge pairs for the skeleton."""
    return MEDIAPIPE_SKELETON.copy()


def get_joint_names() -> List[str]:
    """Human-readable names for MediaPipe joints (for debugging)."""
    return [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]
