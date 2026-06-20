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


# FEN piece char -> canonical id assuming WHITE is to move (no mirror).
_CHAR_TO_ID = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
# FEN piece char -> canonical id when the board is mirrored (Black to move):
# colors swap, so a White piece becomes the Black id and vice versa.
_CHAR_TO_ID_MIRROR = {c: (i + 6 if i <= 6 else i - 6) for c, i in _CHAR_TO_ID.items()}
_STANDARD_CASTLING = set("KQkq-")


def encode_position_fast(fen: str, repetition_count: int = 0):
    """Parse a FEN string directly into (square_tokens[64] int32, state[18] float32).

    Byte-identical to ``encode_position(chess.Board(fen), repetition_count)`` but
    skips python-chess Board construction, ``Board.mirror()`` and ``piece_map()``.
    The canonical frame puts the side-to-move as White: when Black is to move we
    flip every square vertically (``sq ^ 56``) and swap piece colors, and swap the
    White/Black castling-right slots. Falls back to the legacy path for
    non-standard (e.g. Chess960/Shredder) castling fields.
    """
    parts = fen.split()
    castling = parts[2]
    # Defensive: hand off rare non-standard castling encodings to python-chess.
    if any(ch not in _STANDARD_CASTLING for ch in castling):
        return encode_position(chess.Board(fen), repetition_count)

    board_field = parts[0]
    white_to_move = parts[1] == "w"
    ep = parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    toks = np.zeros(64, dtype=np.int32)
    table = _CHAR_TO_ID if white_to_move else _CHAR_TO_ID_MIRROR
    rank = 7
    for row in board_field.split("/"):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += ord(ch) - 48
            else:
                sq = rank * 8 + file
                toks[sq if white_to_move else sq ^ 56] = table[ch]
                file += 1
        rank -= 1

    f = np.zeros(STATE_DIM, dtype=np.float32)
    K = "K" in castling
    Q = "Q" in castling
    k = "k" in castling
    q = "q" in castling
    if white_to_move:
        f[0], f[1], f[2], f[3] = K, Q, k, q
    else:  # mirror swaps colors: White slots get Black's rights and vice versa
        f[0], f[1], f[2], f[3] = k, q, K, Q
    if ep != "-":
        f[4 + (ord(ep[0]) - 97)] = 1.0  # file is preserved under vertical mirror
        f[12] = 1.0
    f[13] = min(halfmove / 100.0, 1.0)
    f[14] = min(repetition_count / 3.0, 1.0)
    f[15] = min(fullmove / 200.0, 1.0)
    f[16] = 1.0
    f[17] = 0.0
    return toks, f


def encode_batch(boards, reps):
    """Return (int32[B,64], float32[B,18]) for a list of boards and repetition counts."""
    sts = np.zeros((len(boards), 64), dtype=np.int32)
    sfs = np.zeros((len(boards), STATE_DIM), dtype=np.float32)
    for i, (b, r) in enumerate(zip(boards, reps)):
        sts[i], sfs[i] = encode_position(b, r)
    return sts, sfs
