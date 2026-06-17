"""Smolgen: learned position-dependent additive attention bias (per head)."""

import tensorflow as tf
from tensorflow.keras import layers


class Smolgen(layers.Layer):
    def __init__(self, n_heads, compress, hidden, gen, shared_out, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.gen = gen
        self.compress = layers.Dense(compress, use_bias=False)
        self.dense1 = layers.Dense(hidden, activation="swish")
        self.ln = layers.LayerNormalization()
        self.dense2 = layers.Dense(n_heads * gen)
        self.shared_out = shared_out  # shared Dense(64*64) instance

    def call(self, x):  # x: [B, 64, d]
        b = tf.shape(x)[0]
        c = self.compress(x)                 # [B, 64, compress]
        c = tf.reshape(c, [b, -1])           # [B, 64*compress]
        h = self.ln(self.dense1(c))          # [B, hidden]
        g = self.dense2(h)                   # [B, n_heads*gen]
        g = tf.reshape(g, [b * self.n_heads, self.gen])
        bias = self.shared_out(g)            # [B*n_heads, 64*64]
        return tf.reshape(bias, [b, self.n_heads, 64, 64])
