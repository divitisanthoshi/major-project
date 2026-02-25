"""
Custom Webcam Dataset Loader

Format for recording your own data via webcam:
- Directory of .npy files: each file = (T, 33, 3) skeleton sequence
- Optional: labels.csv with filename, score

Supports both flat and subfolder structure:
  custom_path/
    seq_001.npy
    labels.csv
  OR
  custom_path/
    wall_pushup/
      seq_000.npy
      labels.csv
    heel_raise/
      seq_000.npy
      labels.csv
"""

import os
import csv
import numpy as np
from typing import Tuple, Dict
from src.datasets.base_loader import BaseDatasetLoader


class CustomWebcamDataset(BaseDatasetLoader):
    """Loader for custom webcam-recorded skeleton sequences."""

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load custom .npy skeleton files from data_path (and subdirs).

        Supports:
        - Flat: custom_path/*.npy + custom_path/labels.csv
        - Subfolders: custom_path/<exercise>/*.npy + custom_path/<exercise>/labels.csv
        """
        sequences = []
        scores = []
        labels_map = self._load_all_labels()

        if not os.path.exists(self.data_path):
            print(f"[Custom] Path not found: {self.data_path}. Returning empty dataset.")
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        npy_files = []
        for root, _, files in os.walk(self.data_path):
            for f in sorted(files):
                if not f.endswith(".npy"):
                    continue
                rel = os.path.relpath(os.path.join(root, f), self.data_path).replace("\\", "/")
                npy_files.append((root, f, rel))

        for root, f, rel_key in npy_files:
            path = os.path.join(root, f)
            try:
                arr = np.load(path)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr.reshape(-1, self.num_joints, 3)
                if arr.shape[1] != self.num_joints:
                    arr = arr[:, : self.num_joints, :3]
                score = labels_map.get(rel_key, labels_map.get(f, labels_map.get(os.path.splitext(f)[0], 0.5)))
                for start in range(0, max(1, arr.shape[0] - self.sequence_length + 1), self.stride):
                    chunk = arr[start : start + self.sequence_length]
                    if len(chunk) == self.sequence_length:
                        sequences.append(chunk)
                        scores.append(score)
            except Exception as e:
                print(f"[Custom] Skip {path}: {e}")

        if len(sequences) == 0:
            return np.zeros((0, self.sequence_length, self.num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        return np.array(sequences), np.array(scores, dtype=np.float32)

    def _load_all_labels(self) -> Dict[str, float]:
        """Load labels from root and each subdir. Keys: filename or relpath (normalized)."""
        out = {}
        for root, _, _ in os.walk(self.data_path):
            labels_path = os.path.join(root, "labels.csv")
            if not os.path.exists(labels_path):
                continue
            prefix = os.path.relpath(root, self.data_path).replace("\\", "/")
            if prefix == ".":
                prefix = ""
            with open(labels_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = row.get("file", row.get("filename", ""))
                    try:
                        score = float(row.get("score", 0.5))
                    except ValueError:
                        continue
                    if prefix and "/" not in key and "\\" not in key:
                        full_key = f"{prefix}/{key}"
                        out[full_key] = score
                    out[key] = score
                    base, _ = os.path.splitext(key)
                    out[base] = score
        return out
