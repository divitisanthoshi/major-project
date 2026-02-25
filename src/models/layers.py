"""
Custom Keras layers for ST-GCN: Graph Convolution and Attention.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional


class SpatialGraphConv(layers.Layer):
    """
    Spatial Graph Convolution over skeleton graph.

    Performs: H' = σ(A @ X @ W)
    where A is adjacency, X is node features, W is weight matrix.
    Uses normalized adjacency for stable training.
    """

    def __init__(self, out_channels: int, adj: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.adj = tf.constant(adj, dtype=tf.float32)
        self.W = None

    def build(self, input_shape):
        # input_shape: (batch, T, V, C_in)
        C_in = input_shape[-1]
        self.W = self.add_weight(
            name="graph_conv_weight",
            shape=(C_in, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (B, T, V, C_in)
        # X @ W -> (B, T, V, C_out)
        x = tf.matmul(inputs, self.W)
        # A @ x: aggregate neighbors. out[i] = sum_j A[i,j] * x[j]
        # A: (V,V), x: (B,T,V,C) -> out: (B,T,V,C)
        x = tf.einsum("ij,btjc->btic", self.adj, x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "adj": self.adj.numpy().tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        adj = config.get("adj")
        if adj is not None:
            adj = np.array(adj, dtype=np.float32)
        else:
            from src.utils.graph_utils import build_adjacency_normalized
            adj = build_adjacency_normalized(33)
        config = dict(config)
        config.pop("adj", None)
        return cls(adj=adj, **config)


class TemporalConv(layers.Layer):
    """
    1D temporal convolution along the time axis.
    Captures local motion patterns.
    """

    def __init__(self, out_channels: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = None

    def build(self, input_shape):
        # input: (B, T, V, C)
        C_in = input_shape[-1]
        self.conv = layers.Conv1D(
            self.out_channels,
            self.kernel_size,
            padding="same",
            data_format="channels_last",
        )
        super().build(input_shape)

    def call(self, inputs):
        # (B, T, V, C) -> reshape to (B*V, T, C)
        B, T, V, C = tf.unstack(tf.shape(inputs))
        x = tf.reshape(inputs, (B * V, T, C))
        x = self.conv(x)
        x = tf.reshape(x, (B, T, V, self.out_channels))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
        })
        return config


class JointAttention(layers.Layer):
    """
    Attention over joint dimension to highlight influential joints.
    Learns which joints matter most for the current prediction.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = None
        self.attention_weights = None

    def build(self, input_shape):
        C = input_shape[-1]
        self.dense = layers.Dense(self.units, activation="tanh")
        self.attention_dense = layers.Dense(1)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (B, T, V, C)
        # Compute attention scores per joint (average over T first or per-frame)
        x = inputs
        # Global average over time -> (B, V, C)
        x_avg = tf.reduce_mean(x, axis=1)
        # Dense -> (B, V, units) -> (B, V, 1)
        h = self.dense(x_avg)
        scores = self.attention_dense(h)  # (B, V, 1)
        weights = tf.nn.softmax(scores, axis=1)  # (B, V, 1)
        self.attention_weights = weights
        # Broadcast: (B, 1, V, 1) * (B, T, V, C)
        out = inputs * tf.reshape(weights, (-1, 1, tf.shape(weights)[1], 1))
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class ReduceSum(layers.Layer):
    """
    Reduce sum along a given axis. Replaces Lambda(tf.reduce_sum) for proper save/load.
    """

    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if self.axis < 0:
            axis = len(shape) + self.axis
        else:
            axis = self.axis
        if 0 <= axis < len(shape):
            shape.pop(axis)
        return tuple(shape)
