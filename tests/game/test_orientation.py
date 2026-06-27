import chess
from src.game.orientation import canonical_board, to_canonical_move, from_canonical_move


def test_white_to_move_is_identity():
    b = chess.Board()
    assert canonical_board(b).fen() == b.fen()
    mv = chess.Move.from_uci("e2e4")
    assert to_canonical_move(mv, chess.WHITE) == mv


def test_black_to_move_mirrors_and_side_is_white():
    b = chess.Board()
    b.push_uci("e2e4")  # now black to move
    cb = canonical_board(b)
    assert cb.turn == chess.WHITE  # side-to-move always White on canonical board


def test_move_roundtrip_is_involution():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move
    for mv in b.legal_moves:
        cmv = to_canonical_move(mv, b.turn)
        assert from_canonical_move(cmv, b.turn) == mv
