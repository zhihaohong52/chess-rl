"""Policy (attention from->to), WDL value, and moves-left heads."""

import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.game.move_encoder import get_move_encoder

_PROMO_CLASS = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3}


def build_policy_index_map():
    """Return (from_idx[P], to_idx[P], promo_class[P]) from the MoveEncoder."""
    me = get_move_encoder()
    p = me.policy_size
    from_idx = np.zeros(p, dtype=np.int32)
    to_idx = np.zeros(p, dtype=np.int32)
    promo = np.zeros(p, dtype=np.int32)
    for i in range(p):
        mv = me.idx_to_move[i]
        from_idx[i] = mv.from_square
        to_idx[i] = mv.to_square
        promo[i] = _PROMO_CLASS.get(mv.promotion, 0)
    return from_idx, to_idx, promo


class PolicyHead(layers.Layer):
    def __init__(self, d_attn=64, **kwargs):
        super().__init__(**kwargs)
        fi, ti, pr = build_policy_index_map()
        self.flat_ft = tf.constant(fi * 64 + ti, dtype=tf.int32)        # [P]
        up_idx = np.where(pr > 0, fi * 3 + np.maximum(pr - 1, 0), 0)
        self.flat_up = tf.constant(up_idx.astype(np.int32), dtype=tf.int32)  # [P]
        self.up_mask = tf.constant((pr > 0).astype(np.float32), dtype=tf.float32)  # [P]
        self.q = layers.Dense(d_attn, use_bias=False)
        self.k = layers.Dense(d_attn, use_bias=False)
        self.up = layers.Dense(3)  # per-square underpromotion logits
        self.scale = float(d_attn) ** 0.5

    def call(self, sq):  # sq: [B, 64, d]
        b = tf.shape(sq)[0]
        q = self.q(sq)
        k = self.k(sq)
        scores = tf.matmul(q, k, transpose_b=True) / self.scale  # [B, 64, 64]
        base = tf.gather(tf.reshape(scores, [b, 64 * 64]), self.flat_ft, axis=1)  # [B, P]
        up = tf.reshape(self.up(sq), [b, 64 * 3])                                  # [B, 192]
        up_term = tf.gather(up, self.flat_up, axis=1) * self.up_mask               # [B, P]
        return base + up_term


class ValueHead(layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(3)  # raw WDL logits

    def call(self, cls):
        return self.d2(self.d1(cls))


class MovesLeftHead(layers.Layer):
    def __init__(self, hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(1, activation="softplus")

    def call(self, cls):
        return self.d2(self.d1(cls))
