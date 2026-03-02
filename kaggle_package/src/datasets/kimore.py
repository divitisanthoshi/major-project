"""
KIMORE Dataset Loader

KIMORE: Kinematic Assessment of Movement and Clinical Scores for Remote Monitoring
Contains RGB-D skeleton sequences of rehabilitation exercises with quality labels.

Format expected:
- Skeleton sequences (joint positions over time)
- Quality labels or scores per exercise/trial

Structure (typical):
  kimore/
    dataset/
      skel_*.npy or similar
    labels.csv  # trial_id, score, or category
"""

import os
import numpy as np
from typing import Tuple
from src.datasets.base_loader import BaseDatasetLoader


class KimoreLoader(BaseDatasetLoader):
    """Loader for KIMORE dataset."""

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load KIMORE format data."""
        sequences = []
        scores = []

        if not os.path.exists(self.data_path):
            print(f"[KIMORE] Path not found: {self.data_path}. Returning empty dataset.")
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        for root, _, files in os.walk(self.data_path):
            for f in files:
                if not (f.endswith(".npy") or f.endswith(".npz")):
                    continue
                path = os.path.join(root, f)
                try:
                    data = np.load(path)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        arr = data["skel"] if "skel" in data else data[list(data.keys())[0]]
                    else:
                        arr = data
                    arr = np.asarray(arr)
                    if arr.ndim == 2:
                        arr = arr.reshape(-1, self.num_joints, 3)
                    elif arr.ndim == 3:
                        if arr.shape[1] != self.num_joints:
                            arr = arr[:, : self.num_joints, :3]
                    score = self._get_score_from_path(path)
                    for start in range(0, max(1, arr.shape[0] - self.sequence_length + 1), self.stride):
                        chunk = arr[start : start + self.sequence_length]
                        if len(chunk) == self.sequence_length:
                            sequences.append(chunk.astype(np.float32))
                            scores.append(score)
                except Exception as e:
                    print(f"[KIMORE] Skip {path}: {e}")

        if len(sequences) == 0:
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        return np.array(sequences), np.array(scores, dtype=np.float32)

    def _get_score_from_path(self, path: str) -> float:
        """Extract or infer score from filename/path. Override for real labels."""
        return 0.7
