"""The fast UCI->canonical-index path must match the legacy from_uci path exactly.

`encode_uci_canonical(uci, turn)` is an optimization that parses the UCI string
arithmetically and looks up a precomputed table, bypassing python-chess
`Move.from_uci` (whose internal `SQUARE_NAMES.index` linear scans dominate
encoding cost). It MUST produce the identical index to the legacy path
`encode(to_canonical_move(Move.from_uci(uci), turn))` for every legal move.
"""
import chess
import pytest

from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move

# Positions chosen to exercise both turns, castling, en passant, and every
# promotion flavour (forward + capture, queen + underpromotions).
SAMPLE_FENS = [
    chess.STARTING_FEN,                                  # white to move, quiet
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",  # black to move
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",            # castling rights, both sides
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant (white)
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",   # en passant (black)
    "8/P7/8/8/8/8/8/k6K w - - 0 1",                     # white forward promotion
    "8/2P5/8/8/8/8/8/k6K w - - 0 1",                     # white forward promotion (c-file)
    "n1n5/1P6/8/8/8/8/8/k6K w - - 0 1",                 # white capture promotions (b7xa8/c8)
    "k6K/8/8/8/8/8/p7/8 b - - 0 1",                     # black forward promotion
    "k6K/8/8/8/8/8/6p1/5N1N b - - 0 1",                 # black capture promotions (g2xf1/h1)
]


def _legal_ucis_with_turn():
    me = get_move_encoder()
    for fen in SAMPLE_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            yield board.turn, move.uci()


def test_fastpath_matches_legacy_for_all_legal_moves():
    me = get_move_encoder()
    checked = 0
    for turn, uci in _legal_ucis_with_turn():
        legacy = me.encode(to_canonical_move(chess.Move.from_uci(uci), turn))
        fast = me.encode_uci_canonical(uci, turn)
        assert fast == legacy, f"mismatch uci={uci} turn={turn}: {fast} != {legacy}"
        checked += 1
    assert checked > 150  # sanity: we actually exercised a lot of moves


def test_fastpath_handles_queen_promotion_like_legacy():
    me = get_move_encoder()
    # Queen promotion is encoded as the plain (from,to) move in the legacy path.
    legacy = me.encode(to_canonical_move(chess.Move.from_uci("a7a8q"), chess.WHITE))
    assert me.encode_uci_canonical("a7a8q", chess.WHITE) == legacy
    # And it equals the non-promo move index.
    assert me.encode_uci_canonical("a7a8q", chess.WHITE) == me.encode_uci_canonical("a7a8", chess.WHITE) \
        if chess.Move.from_uci("a7a8") in me.move_to_idx else True
