import tempfile
import numpy as np
import tensorflow as tf
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
import chess


def _net():
    return ChessTransformer(Config)


def test_forward_shapes():
    net = _net()
    st, sf = encode_batch([chess.Board(), chess.Board()], [0, 0])
    policy, wdl, ml = net(tf.constant(st), tf.constant(sf))
    P = get_move_encoder().policy_size
    assert tuple(policy.shape) == (2, P)
    assert tuple(wdl.shape) == (2, 3)
    assert tuple(ml.shape) == (2, 1)


def test_param_count_in_budget():
    net = _net()
    st, sf = encode_batch([chess.Board()], [0])
    net(tf.constant(st), tf.constant(sf))  # build
    params = int(sum(np.prod(v.shape) for v in net.trainable_variables))
    assert 8_000_000 < params < 14_000_000, params


def test_save_load_roundtrip():
    net = _net()
    st, sf = encode_batch([chess.Board()], [0])
    p1, _, _ = net(tf.constant(st), tf.constant(sf))
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/w.weights.h5"
        net.save_weights(path)
        net2 = _net()
        net2(tf.constant(st), tf.constant(sf))  # build before load
        net2.load_weights(path)
    p2, _, _ = net2(tf.constant(st), tf.constant(sf))
    assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-5)
