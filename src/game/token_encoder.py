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


def encode_state_features(cboard: chess.Board, repetition_count: int = 0) -> np.ndarray:
    """Return (18,) float32 global-state features for an already-canonicalized board."""
    f = np.zeros(STATE_DIM, dtype=np.float32)
    f[0] = float(cboard.has_kingside_castling_rights(chess.WHITE))
    f[1] = float(cboard.has_queenside_castling_rights(chess.WHITE))
    f[2] = float(cboard.has_kingside_castling_rights(chess.BLACK))
    f[3] = float(cboard.has_queenside_castling_rights(chess.BLACK))
    if cboard.ep_square is not None:
        f[4 + chess.square_file(cboard.ep_square)] = 1.0
        f[12] = 1.0
    f[13] = min(cboard.halfmove_clock / 100.0, 1.0)
    f[14] = min(repetition_count / 3.0, 1.0)
    f[15] = min(cboard.fullmove_number / 200.0, 1.0)
    f[16] = 1.0
    f[17] = 0.0
    return f


def encode_position(board: chess.Board, repetition_count: int = 0):
    """Canonicalize and return (square_tokens[64] int32, state_features[18] float32)."""
    cb = canonical_board(board)
    return encode_square_tokens(cb), encode_state_features(cb, repetition_count)


def encode_batch(boards, reps):
    """Return (int32[B,64], float32[B,18]) for a list of boards and repetition counts."""
    sts = np.zeros((len(boards), 64), dtype=np.int32)
    sfs = np.zeros((len(boards), STATE_DIM), dtype=np.float32)
    for i, (b, r) in enumerate(zip(boards, reps)):
        sts[i], sfs[i] = encode_position(b, r)
    return sts, sfs
