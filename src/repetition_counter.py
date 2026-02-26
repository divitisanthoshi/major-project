"""
Repetition Counting Logic for Rehabilitation Exercises

More robust counting using:
- Peak detection on correctness score over time
- Configurable thresholds
- Motion-based heuristics (optional)
"""

import numpy as np
from collections import deque
from typing import List, Optional


class RepetitionCounter:
    """
    Counts exercise repetitions from a stream of correctness scores.

    Uses peak detection: a "rep" is counted when score rises above threshold
    and then falls below, or vice versa for different exercise types.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        min_peak_distance: int = 8,
        window_size: int = 12,
    ):
        """
        Args:
            threshold: Score threshold for counting a rep (crossing upward).
            min_peak_distance: Minimum frames between consecutive counts.
            window_size: Smoothing window for score (smaller = more responsive).
        """
        self.threshold = threshold
        self.min_peak_distance = min_peak_distance
        self.window_size = window_size
        self.count = 0
        self.score_history: deque = deque(maxlen=min(64, window_size * 3))
        self._last_peak_frame = -min_peak_distance - 1
        self._above_threshold = False

    def update(self, score: float, frame_idx: int = 0) -> int:
        """
        Update with new score. Returns current rep count.

        Counts a rep when: score crosses threshold upward (or downward for "down" exercises).
        """
        self.score_history.append(score)
        # Use recent window only so we react to current rep quality
        recent = list(self.score_history)[-self.window_size:] if self.score_history else [score]
        smoothed = float(np.mean(recent))

        now_above = smoothed >= self.threshold

        if now_above and not self._above_threshold:
            # Crossing upward
            if frame_idx - self._last_peak_frame >= self.min_peak_distance:
                self.count += 1
                self._last_peak_frame = frame_idx
        self._above_threshold = now_above

        return self.count

    def reset(self) -> None:
        """Reset counter and history."""
        self.count = 0
        self.score_history.clear()
        self._last_peak_frame = -self.min_peak_distance - 1
        self._above_threshold = False
