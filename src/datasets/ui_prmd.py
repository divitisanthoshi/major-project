"""
UI-PRMD Dataset Loader

UI-PRMD: University of Idaho - Physical Rehabilitation Movement Dataset
Contains motion capture data of rehabilitation exercises with quality annotations.

Format expected:
- Skeleton data (e.g., .txt, .csv, or .npy) with joint positions per frame
- Score/label files indicating exercise quality

Structure (typical):
  ui_prmd/
    subject_01/
      exercise_01/
        trial_01.txt  # frame lines: j0_x j0_y j0_z j1_x ... or similar
    scores.csv        # subject, exercise, trial, score
"""

import os
import numpy as np
from typing import Tuple, List
from src.datasets.base_loader import BaseDatasetLoader


class UIPRMDLoader(BaseDatasetLoader):
    """Loader for UI-PRMD style rehabilitation dataset."""

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load UI-PRMD format data.

        If data path does not exist, returns empty arrays for simulation.
        Expected file format per trial: CSV/txt with columns for each joint (x,y,z).
        """
        sequences = []
        scores = []

        if not os.path.exists(self.data_path):
            print(f"[UIPRMD] Path not found: {self.data_path}. Returning empty dataset.")
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Walk directories for trial files
        for root, _, files in os.walk(self.data_path):
            for f in files:
                if not (f.endswith(".txt") or f.endswith(".csv") or f.endswith(".npy")):
                    continue
                path = os.path.join(root, f)
                try:
                    seq, score = self._load_trial(path)
                    if seq is not None and len(seq) >= self.sequence_length:
                        # Sliding window
                        for start in range(0, len(seq) - self.sequence_length + 1, self.stride):
                            chunk = seq[start : start + self.sequence_length]
                            sequences.append(chunk)
                            scores.append(score)
                except Exception as e:
                    print(f"[UIPRMD] Skip {path}: {e}")

        if len(sequences) == 0:
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        sequences = np.array(sequences, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        return sequences, scores

    def _load_trial(self, path: str) -> Tuple[np.ndarray, float]:
        """Load single trial file. Returns (sequence, score)."""
        if path.endswith(".npy"):
            data = np.load(path)
        else:
            data = np.loadtxt(path, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(-1, self.num_joints * 3)

        # Reshape to (T, V, 3)
        T = data.shape[0]
        data = data.reshape(T, -1, 3)[:, : self.num_joints, :]
        # Placeholder score (would come from scores.csv in real data)
        score = 0.75
        return data.astype(np.float32), score
