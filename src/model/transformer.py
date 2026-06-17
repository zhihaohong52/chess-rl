"""ChessTransformer: token board -> (policy_logits, wdl_logits, moves_left)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.model.smolgen import Smolgen
from src.model.heads import PolicyHead, ValueHead, MovesLeftHead


class BiasedMHA(layers.Layer):
    """Multi-head self-attention with an additive per-head bias on the logits."""

    def __init__(self, n_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.h = n_heads
        self.d = d_model
        self.dh = d_model // n_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.wo = layers.Dense(d_model)

    def _split(self, t, b, seq):
        return tf.transpose(tf.reshape(t, [b, seq, self.h, self.dh]), [0, 2, 1, 3])

    def call(self, x, bias):  # x: [B,T,d]; bias: [B,h,T,T]
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        q = self._split(self.wq(x), b, t)
        k = self._split(self.wk(x), b, t)
        v = self._split(self.wv(x), b, t)
        scores = tf.matmul(q, k, transpose_b=True) / (float(self.dh) ** 0.5) + bias
        a = tf.nn.softmax(scores, axis=-1)
        o = tf.matmul(a, v)  # [B,h,T,dh]
        o = tf.reshape(tf.transpose(o, [0, 2, 1, 3]), [b, t, self.d])
        return self.wo(o)


class EncoderLayer(layers.Layer):
    def __init__(self, cfg, shared_smolgen_out, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.attn = BiasedMHA(cfg.n_heads, cfg.d_model)
        self.smolgen = Smolgen(cfg.n_heads, cfg.smolgen_compress,
                               cfg.smolgen_hidden, cfg.smolgen_gen, shared_smolgen_out)
        self.ln2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(cfg.d_ff, activation="gelu"),
            layers.Dense(cfg.d_model),
        ])

    def call(self, x):  # x: [B, 65, d]  (index 0 = CLS, 1..64 = squares)
        h = self.ln1(x)
        bias64 = self.smolgen(h[:, 1:, :])                       # [B,h,64,64]
        bias = tf.pad(bias64, [[0, 0], [0, 0], [1, 0], [1, 0]])  # CLS row/col = 0 -> [B,h,65,65]
        x = x + self.attn(h, bias)
        x = x + self.ffn(self.ln2(x))
        return x


class ChessTransformer(keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        d = cfg.d_model
        self.piece_emb = layers.Embedding(13, d)
        self.pos_emb = self.add_weight(name="pos_emb", shape=(1, 64, d),
                                       initializer="random_normal", trainable=True)
        self.cls = self.add_weight(name="cls", shape=(1, 1, d),
                                   initializer="random_normal", trainable=True)
        self.state_mlp = keras.Sequential([
            layers.Dense(d, activation="gelu"),
            layers.Dense(d),
        ])
        shared_smolgen_out = layers.Dense(64 * 64, use_bias=False)
        self.enc = [EncoderLayer(cfg, shared_smolgen_out, name=f"encoder_layer_{i}")
                    for i in range(cfg.n_layers)]
        self.final_ln = layers.LayerNormalization()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        self.moves_left_head = MovesLeftHead()

    def call(self, square_tokens, state_features, training=False):
        pe = self.piece_emb(square_tokens) + self.pos_emb       # [B,64,d]
        cond = self.state_mlp(state_features)[:, None, :]       # [B,1,d]
        pe = pe + cond
        b = tf.shape(pe)[0]
        cls = tf.tile(self.cls, [b, 1, 1]) + cond               # [B,1,d]
        x = tf.concat([cls, pe], axis=1)                        # [B,65,d]
        for layer in self.enc:
            x = layer(x)
        x = self.final_ln(x)
        cls_out = x[:, 0, :]
        sq_out = x[:, 1:, :]
        return self.policy_head(sq_out), self.value_head(cls_out), self.moves_left_head(cls_out)

    @tf.function(reduce_retracing=True)
    def predict_batch(self, square_tokens, state_features):
        return self.call(square_tokens, state_features, training=False)
