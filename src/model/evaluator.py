"""MCTS-facing evaluator wrapping the ChessTransformer."""

import numpy as np
import tensorflow as tf

from src.game.token_encoder import encode_batch
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder


class TransformerEvaluator:
    def __init__(self, net, use_fp16: bool = False):
        self.net = net
        self.me = get_move_encoder()
        self.use_fp16 = use_fp16

    def evaluate(self, board, repetition_count: int = 0):
        return self.evaluate_batch([board], [repetition_count])[0]

    def evaluate_batch(self, boards, reps=None):
        if reps is None:
            reps = [0] * len(boards)
        sq, sf = encode_batch(boards, reps)
        pol_logits, wdl_logits, _ = self.net(tf.constant(sq), tf.constant(sf), training=False)
        pol_logits = np.asarray(pol_logits)
        wdl = tf.nn.softmax(wdl_logits, axis=-1).numpy()

        out = []
        for i, b in enumerate(boards):
            legal = list(b.legal_moves)
            if not legal:
                out.append(({}, float(wdl[i, 0] - wdl[i, 2])))
                continue
            idxs = [self.me.encode(to_canonical_move(mv, b.turn)) for mv in legal]
            logits = pol_logits[i][idxs]
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            policy = {mv: float(p) for mv, p in zip(legal, probs)}
            value = float(wdl[i, 0] - wdl[i, 2])  # P(W) - P(L), side-to-move POV
            out.append((policy, value))
        return out
