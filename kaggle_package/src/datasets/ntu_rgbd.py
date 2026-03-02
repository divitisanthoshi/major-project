"""
NTU RGB+D Dataset Loader

NTU RGB+D: Large-scale dataset for skeleton-based action recognition.
Can be adapted for rehabilitation by mapping actions to quality scores.

Format: .npy or .skeleton files with joint positions.
Structure: (N, T, V, C) where V=25 (NTU) - we map/align to MediaPipe 33 if needed.
"""

import os
import numpy as np
from typing import Tuple
from src.datasets.base_loader import BaseDatasetLoader


# NTU has 25 joints. Mapping to 33 would require padding or interpolation.
# For compatibility we pad with zeros or replicate nearby joints.
NTU_NUM_JOINTS = 25


class NTURGBDLoader(BaseDatasetLoader):
    """Loader for NTU RGB+D skeleton data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_joints = self.num_joints  # MediaPipe 33

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load NTU RGB+D format. Pad 25->33 joints if needed."""
        sequences = []
        scores = []

        if not os.path.exists(self.data_path):
            print(f"[NTU] Path not found: {self.data_path}. Returning empty dataset.")
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        for root, _, files in os.walk(self.data_path):
            for f in files:
                if not (f.endswith(".npy") or f.endswith(".npz") or "nturgbd" in f.lower()):
                    continue
                path = os.path.join(root, f)
                try:
                    data = np.load(path)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        arr = data[list(data.keys())[0]]
                    else:
                        arr = data
                    arr = np.asarray(arr)
                    # NTU: (N, T, V, C) or (T, V, C)
                    if arr.ndim == 3:
                        arr = arr[np.newaxis, ...]
                    for i in range(arr.shape[0]):
                        seq = arr[i]
                        if seq.shape[0] < self.sequence_length:
                            continue
                        seq = self._pad_joints(seq)
                        for start in range(0, seq.shape[0] - self.sequence_length + 1, self.stride):
                            chunk = seq[start : start + self.sequence_length]
                            sequences.append(chunk.astype(np.float32))
                            scores.append(0.6)  # Placeholder
                except Exception as e:
                    print(f"[NTU] Skip {path}: {e}")

        if len(sequences) == 0:
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        return np.array(sequences), np.array(scores, dtype=np.float32)

    def _pad_joints(self, seq: np.ndarray) -> np.ndarray:
        """Pad from NTU 25 joints to MediaPipe 33 (zeros for extra joints)."""
        T, V, C = seq.shape
        if V >= self.num_joints:
            return seq[:, : self.num_joints, :]
        pad = np.zeros((T, self.num_joints - V, C), dtype=np.float32)
        return np.concatenate([seq, pad], axis=1)
