"""Token encoder: board -> 64 square tokens + 18-dim global-state vector (canonical frame)."""

import chess
import numpy as np

from .orientation import canonical_board

STATE_DIM = 18

# On the canonical board the side-to-move is always White.
_PIECE_TO_ID = {
    (chess.WHITE, chess.PAWN): 1, (chess.WHITE, chess.KNIGHT): 2, (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4, (chess.WHITE, chess.QUEEN): 5, (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7, (chess.BLACK, chess.KNIGHT): 8, (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10, (chess.BLACK, chess.QUEEN): 11, (chess.BLACK, chess.KING): 12,
}


def encode_square_tokens(cboard: chess.Board) -> np.ndarray:
    """Return (64,) int32 piece ids for an already-canonicalized board."""
    toks = np.zeros(64, dtype=np.int32)
    for sq, piece in cboard.piece_map().items():
        toks[sq] = _PIECE_TO_ID[(piece.color, piece.piece_type)]
    return toks
