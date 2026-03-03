"""
Reference-based rep counting: count a rep when the live pose matches a pose reference step.

Reps and quality are independent: quality comes from the model; reps come from
matching the live skeleton to reference poses loaded from step images (pose reference).

Step images: each exercise has one composite image (e.g. step_01.png) containing
steps 1–6 in a single 2x3 grid. We split that image into 6 tiles and load one reference pose per step.
"""

import os
import numpy as np
import cv2
from typing import List, Optional, Tuple

# Step index (0-based) used as "peak" for rep count: when live matches this step, count a rep.
# Step 5 = "Correct peak position" in the 6-step sequence (index 4).
PEAK_STEP_INDEX = 4


def _normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Normalize landmarks: center on hip mid, scale by torso size. Shape (33, 3)."""
    if lm is None or lm.shape[0] < 24:
        return lm
    # Hip center: average of left_hip (23) and right_hip (24)
    hip_center = (lm[23] + lm[24]) / 2
    centered = lm - hip_center
    # Scale by shoulder-hip distance so different body sizes are comparable
    shoulder_center = (lm[11] + lm[12]) / 2
    scale = np.linalg.norm(shoulder_center - hip_center)
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def pose_similarity(live: np.ndarray, ref: np.ndarray) -> float:
    """
    Compare live pose to reference pose. Returns similarity in [0, 1] (1 = same).

    Uses normalized landmark positions and mean distance; converts to similarity.
    """
    if live is None or ref is None or live.shape != ref.shape:
        return 0.0
    ln = _normalize_landmarks(live)
    rn = _normalize_landmarks(ref)
    if ln is None or rn is None:
        return 0.0
    # Mean L2 distance over joints (x, y; z less important for 2D view)
    diff = ln[:, :2] - rn[:, :2]
    mean_dist = np.sqrt(np.mean(diff ** 2))
    # Map distance to similarity: 0 -> 1, 0.3 -> ~0.5, large -> 0
    similarity = float(np.exp(-3.0 * mean_dist))
    return max(0.0, min(1.0, similarity))


# Grid layout: one composite image has steps 1–6 in 2 rows x 3 columns (step 1 top-left → step 6 bottom-right).
COMPOSITE_GRID_ROWS = 2
COMPOSITE_GRID_COLS = 3
COMPOSITE_NUM_STEPS = COMPOSITE_GRID_ROWS * COMPOSITE_GRID_COLS


def load_reference_poses_from_composite(composite_image_path: str) -> List[Optional[np.ndarray]]:
    """
    Load 6 reference poses from a single composite step image (2x3 grid: steps 1–6 in one image).

    Splits the image into 6 tiles, runs pose detection on each tile, returns list of 6 (33,3) or None.
    """
    from src.pose_extraction import extract_pose_from_static_image

    if not composite_image_path or not os.path.isfile(composite_image_path):
        return [None] * COMPOSITE_NUM_STEPS

    img = cv2.imread(composite_image_path)
    if img is None:
        return [None] * COMPOSITE_NUM_STEPS

    h, w = img.shape[:2]
    tile_w = w // COMPOSITE_GRID_COLS
    tile_h = h // COMPOSITE_GRID_ROWS
    refs = []
    for row in range(COMPOSITE_GRID_ROWS):
        for col in range(COMPOSITE_GRID_COLS):
            x1 = col * tile_w
            y1 = row * tile_h
            tile = img[y1 : y1 + tile_h, x1 : x1 + tile_w]
            lm = extract_pose_from_static_image(tile)
            refs.append(lm)
    return refs


def load_reference_poses(step_image_paths: List[str]) -> List[Optional[np.ndarray]]:
    """
    Load reference poses for steps 1–6.

    Each exercise has one composite image (e.g. step_01.png) with all 6 steps in a 2x3 grid.
    We split that image and extract one pose per step. If multiple files exist, we use the first as composite.
    """
    if not step_image_paths:
        return [None] * COMPOSITE_NUM_STEPS
    return load_reference_poses_from_composite(step_image_paths[0])


class ReferenceRepCounter:
    """
    Counts reps when the live pose matches the reference "peak" step from pose reference images.

    Independent of quality score: only pose similarity to the reference step drives rep count.
    """

    def __init__(
        self,
        reference_poses: List[Optional[np.ndarray]],
        peak_step_index: int = PEAK_STEP_INDEX,
        match_threshold: float = 0.55,
        min_frames_between_reps: int = 15,
    ):
        """
        Args:
            reference_poses: List of (33,3) arrays per step; None for steps with no pose.
            peak_step_index: Which step (0-based) counts as "peak" for rep counting.
            match_threshold: Similarity above this = "matching" the reference step.
            min_frames_between_reps: Minimum frames between two rep counts.
        """
        self.reference_poses = reference_poses
        self.peak_step_index = peak_step_index
        self.match_threshold = match_threshold
        self.min_frames_between_reps = min_frames_between_reps
        self.count = 0
        self._in_peak = False
        self._last_rep_frame = -min_frames_between_reps - 1

    def update(self, live_landmarks: Optional[np.ndarray], frame_idx: int) -> int:
        """
        Update with current live pose. Returns current rep count.

        When live pose matches the peak reference step (and we weren't already in peak,
        and enough frames since last rep), count one rep.
        """
        if live_landmarks is None:
            self._in_peak = False
            return self.count

        ref_peak = None
        if 0 <= self.peak_step_index < len(self.reference_poses):
            ref_peak = self.reference_poses[self.peak_step_index]
        if ref_peak is None:
            # No reference for peak step: try best match among any step
            best_sim = 0.0
            for ref in self.reference_poses:
                if ref is not None:
                    sim = pose_similarity(live_landmarks, ref)
                    if sim > best_sim:
                        best_sim = sim
                        ref_peak = ref
            if ref_peak is None:
                return self.count

        sim = pose_similarity(live_landmarks, ref_peak)
        now_matching = sim >= self.match_threshold

        if now_matching and not self._in_peak:
            if frame_idx - self._last_rep_frame >= self.min_frames_between_reps:
                self.count += 1
                self._last_rep_frame = frame_idx
        self._in_peak = now_matching
        return self.count

    def has_references(self) -> bool:
        """True if at least one reference pose was loaded (e.g. from step images)."""
        return any(r is not None for r in self.reference_poses)

    def reset(self) -> None:
        self.count = 0
        self._in_peak = False
        self._last_rep_frame = -self.min_frames_between_reps - 1
