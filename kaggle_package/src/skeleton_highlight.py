"""
Skeleton Highlighting - Visual feedback for incorrect joints.

Draws red circles on problematic joints when errors are detected.
"""

import cv2
import numpy as np
from typing import List, Optional


def highlight_joints(frame: np.ndarray, landmarks: np.ndarray, error_joint_indices: List[int]) -> np.ndarray:
    """
    Overlay red circles on joints that need correction.

    Args:
        frame: BGR video frame.
        landmarks: (33, 3) normalized x,y,z.
        error_joint_indices: Joint indices to highlight.

    Returns:
        Frame with highlights.
    """
    if landmarks is None or not error_joint_indices:
        return frame

    h, w = frame.shape[:2]
    for j in error_joint_indices:
        if j >= len(landmarks):
            continue
        x = int(landmarks[j][0] * w)
        y = int(landmarks[j][1] * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 14, (0, 0, 255), 3)
            cv2.circle(frame, (x, y), 12, (0, 100, 255), -1)

    return frame
