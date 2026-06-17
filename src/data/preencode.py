"""Pre-encode LabeledPosition -> canonical example dict and npz shards."""

import chess
import numpy as np

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


def write_shard(labeled_positions, path: str) -> int:
    """Encode an iterable of LabeledPosition and save as a compressed npz shard.

    Schema:
      square_tokens : int8   [N, 64]
      state_features: float32 [N, 18]
      wdl           : float32 [N, 3]
      moves_left    : float32 [N]
      legal_indices : int32  [total_legal]   — concatenated across all examples
      legal_probs   : float32 [total_legal]  — parallel to legal_indices
      counts        : int32  [N]             — number of legal moves per example

    Returns N (number of examples written).
    """
    sq_list, sf_list, wdl_list, ml_list = [], [], [], []
    idx_list, prob_list, counts = [], [], []

    for lp in labeled_positions:
        ex = encode_example(lp)
        sq_list.append(ex["square_tokens"])          # (64,) int8
        sf_list.append(ex["state_features"])         # (18,) float32
        wdl_list.append(ex["wdl"])                   # (3,) float32
        ml_list.append(ex["moves_left"])             # scalar float32
        idx_list.append(ex["legal_indices"].astype(np.int32))
        prob_list.append(ex["legal_probs"])
        counts.append(len(ex["legal_indices"]))

    n = len(sq_list)
    if n == 0:
        np.savez_compressed(
            path,
            square_tokens=np.empty((0, 64), dtype=np.int8),
            state_features=np.empty((0, 18), dtype=np.float32),
            wdl=np.empty((0, 3), dtype=np.float32),
            moves_left=np.empty((0,), dtype=np.float32),
            legal_indices=np.empty((0,), dtype=np.int32),
            legal_probs=np.empty((0,), dtype=np.float32),
            counts=np.empty((0,), dtype=np.int32),
        )
        return 0

    np.savez_compressed(
        path,
        square_tokens=np.stack(sq_list, axis=0),           # [N, 64] int8
        state_features=np.stack(sf_list, axis=0),          # [N, 18] float32
        wdl=np.stack(wdl_list, axis=0),                    # [N, 3] float32
        moves_left=np.array(ml_list, dtype=np.float32),    # [N] float32
        legal_indices=np.concatenate(idx_list),            # [total] int32
        legal_probs=np.concatenate(prob_list),             # [total] float32
        counts=np.array(counts, dtype=np.int32),           # [N] int32
    )
    return n
