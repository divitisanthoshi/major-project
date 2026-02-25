"""
ST-GCN: Spatial-Temporal Graph Convolutional Network

Combines:
- Spatial graph convolution: models joint relationships via skeleton connectivity
- Temporal convolution: captures motion dynamics over time
- LSTM: sequence modeling for long-range dependencies
- Attention: highlights influential joints
- Output: continuous correctness score (0-1)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils.graph_utils import build_adjacency_normalized
from src.models.layers import SpatialGraphConv, TemporalConv, ReduceSum


def build_rehab_grading_model(
    num_joints: int = 33,
    in_channels: int = 3,
    sequence_length: int = 64,
    stgcn_channels: list = None,
    lstm_units: int = 128,
    lstm_layers: int = 2,
    attention_units: int = 64,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Build the full rehabilitation exercise quality grading model.

    Architecture:
    1. ST-GCN blocks (spatial graph conv + temporal conv)
    2. LSTM for sequence modeling
    3. Joint attention
    4. Dense layers -> correctness score (sigmoid)

    Args:
        num_joints: Number of skeleton joints (33 for MediaPipe).
        in_channels: Features per joint (3 for x,y,z).
        sequence_length: Number of frames per sequence.
        stgcn_channels: List of output channels per ST-GCN block.
        lstm_units: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        attention_units: Attention hidden size.
        dropout: Dropout rate.

    Returns:
        Compiled Keras Model.
    """
    if stgcn_channels is None:
        stgcn_channels = [64, 64, 128, 256]

    adj = build_adjacency_normalized(num_joints)

    # Input: (B, T, V, C)
    inputs = layers.Input(shape=(sequence_length, num_joints, in_channels))

    x = inputs
    C_in = in_channels

    # ST-GCN blocks
    for i, C_out in enumerate(stgcn_channels):
        # Spatial: graph convolution
        x = SpatialGraphConv(C_out, adj)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

        # Temporal: 1D conv
        x = TemporalConv(C_out, kernel_size=3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

        C_in = C_out

    # Flatten spatial (joint) dimension for LSTM: (B, T, V*C)
    x = layers.Reshape((sequence_length, -1))(x)

    # LSTM layers
    for _ in range(lstm_layers - 1):
        x = layers.LSTM(lstm_units, return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    # Reintroduce joint structure for attention: (B, T, V, C)
    # We need V for attention - use reshape. After LSTM we have (B, T, lstm_units).
    # For joint-level attention we'd need to have kept V. Simplified: use temporal attention instead.
    # Alternative: treat LSTM output as sequence and apply attention over time.
    attention_scores = layers.Dense(1, activation="tanh")(x)
    attention_weights = layers.Softmax(axis=1)(attention_scores)
    x = x * attention_weights
    x = ReduceSum(axis=1)(x)  # (B, lstm_units)

    # Additional dense + output
    x = layers.Dense(attention_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="RehabGradingSTGCN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_simplified_model(
    num_joints: int = 33,
    in_channels: int = 3,
    sequence_length: int = 64,
    lstm_units: int = 128,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Simplified model: LSTM + Attention (no graph conv) for faster prototyping.
    Use when full ST-GCN is too heavy for real-time.
    """
    inputs = layers.Input(shape=(sequence_length, num_joints, in_channels))
    x = layers.Reshape((sequence_length, num_joints * in_channels))(inputs)

    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    # Temporal attention
    attention = layers.Dense(1, activation="tanh")(x)
    attention = layers.Softmax(axis=1)(attention)
    x = layers.Multiply()([x, attention])
    x = ReduceSum(axis=1)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="RehabGradingSimplified")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
