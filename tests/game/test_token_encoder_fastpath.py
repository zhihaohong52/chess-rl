"""`encode_position_fast(fen)` must byte-match the legacy board-based encoder.

The legacy path builds a python-chess Board, canonicalizes via Board.mirror()
(for black to move), and reads piece_map()/castling/ep/clocks. The fast path
parses the FEN string directly to skip Board construction. It must produce the
identical (square_tokens, state_features) for every position, or encoded shards
would change and break baseline reproducibility.
"""
import random

import chess
import numpy as np

from src.game.token_encoder import encode_position, encode_position_fast


def _random_fens(n_games=40, max_plies=80, seed=0):
    rng = random.Random(seed)
    fens = []
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(max_plies):
            fens.append(board.fen())
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
    return fens


CRAFTED = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 10",          # full castling rights
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",             # partial castling rights
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # ep, white to move
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",   # ep, black to move
    "8/P7/8/8/8/8/8/k6K w - - 99 120",                # high clocks, sparse
    "8/8/8/8/8/8/p7/k6K b - - 0 1",                   # black to move, sparse
]


def test_encode_position_fast_matches_legacy():
    fens = CRAFTED + _random_fens()
    checked = 0
    for fen in fens:
        legacy_sq, legacy_sf = encode_position(chess.Board(fen))
        fast_sq, fast_sf = encode_position_fast(fen)
        assert np.array_equal(fast_sq, legacy_sq), f"square_tokens mismatch: {fen}"
        assert fast_sq.dtype == legacy_sq.dtype
        assert np.array_equal(fast_sf, legacy_sf), f"state_features mismatch: {fen}"
        assert fast_sf.dtype == legacy_sf.dtype
        checked += 1
    assert checked > 100


def test_encode_position_fast_respects_repetition_count():
    fen = chess.STARTING_FEN
    for rep in (0, 1, 2, 3):
        legacy = encode_position(chess.Board(fen), rep)[1]
        fast = encode_position_fast(fen, rep)[1]
        assert np.array_equal(fast, legacy)
