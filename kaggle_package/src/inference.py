"""
Real-Time Inference Engine

- Loads trained model
- Runs prediction on buffered pose sequences
- Repetition counting logic
- Quality category mapping
"""

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
from typing import Optional, Tuple
from tensorflow import keras

# Register custom layers so load_model can deserialize them
from src.models.layers import (
    SpatialGraphConv,
    TemporalConv,
    JointAttention,
    ReduceSum,
)

# Keras may look up by short name or by "module.class_name"; provide both
CUSTOM_OBJECTS = {
    "SpatialGraphConv": SpatialGraphConv,
    "TemporalConv": TemporalConv,
    "JointAttention": JointAttention,
    "ReduceSum": ReduceSum,
    "src.models.layers.SpatialGraphConv": SpatialGraphConv,
    "src.models.layers.TemporalConv": TemporalConv,
    "src.models.layers.JointAttention": JointAttention,
    "src.models.layers.ReduceSum": ReduceSum,
}


class InferenceEngine:
    """
    Performs real-time correctness prediction and repetition counting.
    """

    def __init__(self, model_path: Optional[str] = None, model=None):
        """
        Args:
            model_path: Path to saved .keras model. Ignored if model is provided.
            model: Pre-loaded Keras model. Takes precedence over model_path.
        """
        if model is not None:
            self.model = model
        elif model_path:
            # safe_mode=False allows loading Lambda layers (e.g. reduce_sum) from our own saved model
            try:
                self.model = keras.models.load_model(
                    model_path,
                    custom_objects=CUSTOM_OBJECTS,
                    safe_mode=False,
                )
            except TypeError:
                # Older Keras/TF: no safe_mode; try enabling unsafe deserialization
                try:
                    keras.config.enable_unsafe_deserialization()
                except AttributeError:
                    pass
                self.model = keras.models.load_model(
                    model_path, custom_objects=CUSTOM_OBJECTS
                )
        else:
            self.model = None

        self.last_score = 0.0
        self.last_joint_weights = None  # (V,) when model has joint attention
        self.rep_count = 0
        self._rep_state = "low"  # or "high" - for crossing threshold
        self._score_threshold = 0.5
        self._min_frames_between_reps = 15  # avoid double-counting

    def predict(self, sequence: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Predict correctness score (and optional joint attention weights) for a sequence.

        Args:
            sequence: (T, V, C) or (1, T, V, C)

        Returns:
            (score, joint_weights). score in [0, 1]. joint_weights (V,) or None if single-output model.
        """
        if self.model is None:
            return 0.5, None

        if sequence.ndim == 3:
            sequence = np.expand_dims(sequence, axis=0)

        out = self.model.predict(sequence, verbose=0)
        if isinstance(out, list) and len(out) >= 2:
            score = float(np.squeeze(out[0])[()])
            joint_weights = np.squeeze(out[1]).astype(np.float64)  # (V,) or (1, V) -> (V,)
            self.last_score = score
            self.last_joint_weights = joint_weights
            return score, joint_weights
        if isinstance(out, list) and len(out) == 1:
            out = out[0]
        out = np.asarray(out)
        score = float(np.squeeze(out)[()])
        self.last_score = score
        self.last_joint_weights = None
        return score, None

    def update_rep_count(self, score: float) -> int:
        """
        Simple repetition counter: count peaks when score crosses threshold.

        Logic: when score goes from below to above threshold, count +1.
        Debounce with minimum frames between reps.

        Returns:
            Updated rep count.
        """
        # Simplified: crossing 0.5 upward = one rep
        if score >= self._score_threshold:
            if self._rep_state == "low":
                self.rep_count += 1
                self._rep_state = "high"
        else:
            self._rep_state = "low"
        return self.rep_count

    def get_quality_category(self, score: Optional[float] = None) -> str:
        """
        Map score to Good / Moderate / Poor.
        """
        s = score if score is not None else self.last_score
        if s >= 0.7:
            return "Good"
        elif s >= 0.4:
            return "Moderate"
        else:
            return "Poor"

    def reset_reps(self) -> None:
        """Reset repetition counter."""
        self.rep_count = 0
        self._rep_state = "low"
