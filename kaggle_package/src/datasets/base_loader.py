"""
Base dataset loader interface.

All dataset loaders should return:
- sequences: (N, T, V, C) - skeleton sequences
- scores: (N,) or (N, 1) - correctness scores in [0, 1]
- Optional: labels for quality categories (Good/Moderate/Poor)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 64,
        stride: int = 8,
        num_joints: int = 33,
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.stride = stride
        self.num_joints = num_joints

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset.

        Returns:
            sequences: (N, T, V, C) float32
            scores: (N,) float32 in [0, 1]
        """
        pass

    def score_to_category(self, score: float) -> str:
        """Map continuous score to quality category."""
        if score >= 0.7:
            return "Good"
        elif score >= 0.4:
            return "Moderate"
        else:
            return "Poor"
