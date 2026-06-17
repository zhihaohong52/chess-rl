import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from src.model.smolgen import Smolgen


def test_smolgen_output_shape():
    shared_out = layers.Dense(64 * 64, use_bias=False)
    sm = Smolgen(n_heads=8, compress=32, hidden=128, gen=128, shared_out=shared_out)
    x = tf.random.normal((2, 64, 256))
    bias = sm(x)
    assert tuple(bias.shape) == (2, 8, 64, 64)
    assert np.isfinite(bias.numpy()).all()
