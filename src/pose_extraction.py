"""
Pose Extraction Module - MediaPipe BlazePose

Extracts 3D skeletal joint coordinates from video frames using MediaPipe Pose.
BlazePose provides 33 body landmarks in normalized coordinates (x, y, z).
"""

import os
# Reduce MediaPipe C++ log noise (must be before importing mediapipe)
if "GLOG_minloglevel" not in os.environ:
    os.environ["GLOG_minloglevel"] = "2"

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class PoseExtractor:
    """
    Extracts human pose landmarks from RGB frames using MediaPipe BlazePose.

    MediaPipe Pose returns 33 landmarks:
    - 0-10: Face and upper body
    - 11-22: Arms and torso
    - 23-32: Legs
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Args:
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
            model_complexity: 0=lite, 1=full, 2=heavy (trade speed vs accuracy).
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def extract(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract pose landmarks from a BGR frame.

        Args:
            frame: BGR image from OpenCV (H x W x 3).

        Returns:
            Tuple of (frame_with_overlay, landmarks_or_None).
            landmarks: shape (33, 3) with x, y, z per joint (normalized).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        landmarks = None
        if result.pose_landmarks:
            landmarks = []
            for lm in result.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)

            # Draw skeleton on frame (optional - for visualization)
            self.mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        return frame, landmarks

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def extract_pose_from_frame(
    frame: np.ndarray,
    extractor: Optional[PoseExtractor] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to extract pose from a single frame.

    Args:
        frame: BGR image.
        extractor: Optional PoseExtractor. Creates one if None.

    Returns:
        Tuple of (frame, landmarks).
    """
    if extractor is None:
        extractor = PoseExtractor()
        result = extractor.extract(frame)
        extractor.close()
        return result
    return extractor.extract(frame)


def extract_pose_from_static_image(image_path_or_array, model_complexity: int = 1) -> Optional[np.ndarray]:
    """
    Extract pose landmarks from a static image (e.g. step reference image).
    Uses MediaPipe with static_image_mode=True. Returns (33, 3) or None.

    Args:
        image_path_or_array: Path to image file (str) or BGR numpy array (H, W, 3).
        model_complexity: 0, 1, or 2.

    Returns:
        landmarks (33, 3) float32, or None if no pose detected.
    """
    if isinstance(image_path_or_array, str):
        frame = cv2.imread(image_path_or_array)
        if frame is None:
            return None
    else:
        frame = np.asarray(image_path_or_array)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_static = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    result = pose_static.process(rgb)
    pose_static.close()
    if not result.pose_landmarks:
        return None
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark], dtype=np.float32)
    return landmarks
