"""Pre-encode LabeledPosition -> canonical example dict and TFRecord shards."""

import chess
import numpy as np
import tensorflow as tf

from src.game.token_encoder import encode_position
from src.game.orientation import to_canonical_move
from src.game.move_encoder import get_move_encoder


def encode_example(lp) -> dict:
    """Convert a LabeledPosition into the canonical numpy example dict."""
    board = chess.Board(lp.fen)
    me = get_move_encoder()
    sq, sf = encode_position(board, lp.repetition_count)

    indices, probs = [], []
    for uci, prob in lp.policy:
        cmove = to_canonical_move(chess.Move.from_uci(uci), board.turn)
        indices.append(me.encode(cmove))
        probs.append(float(prob))

    return {
        "square_tokens": sq.astype(np.int8),
        "state_features": sf.astype(np.float32),
        "legal_indices": np.array(indices, dtype=np.int64),
        "legal_probs": np.array(probs, dtype=np.float32),
        "wdl": np.array(lp.wdl, dtype=np.float32),
        "moves_left": np.float32(lp.moves_left),
    }


def _bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def _float_list(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(v)))


def _int64_list(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(v)))


def serialize_example(ex: dict) -> bytes:
    feat = {
        "square_tokens": _bytes_feature(ex["square_tokens"].tobytes()),
        "state_features": _float_list(ex["state_features"]),
        "legal_indices": _int64_list(ex["legal_indices"]),
        "legal_probs": _float_list(ex["legal_probs"]),
        "wdl": _float_list(ex["wdl"]),
        "moves_left": _float_list([float(ex["moves_left"])]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


def write_shard(labeled_positions, path: str) -> int:
    """Write an iterable of LabeledPosition to a TFRecord shard. Returns count."""
    n = 0
    with tf.io.TFRecordWriter(path) as w:
        for lp in labeled_positions:
            w.write(serialize_example(encode_example(lp)))
            n += 1
    return n
