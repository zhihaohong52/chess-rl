import chess
import numpy as np
from src.game.token_encoder import encode_square_tokens
from src.game.orientation import canonical_board


def test_startpos_square_tokens():
    toks = encode_square_tokens(canonical_board(chess.Board()))
    assert toks.shape == (64,) and toks.dtype == np.int32
    assert toks[chess.A1] == 4   # White rook (side-to-move) -> id 4
    assert toks[chess.E1] == 6   # White king -> id 6
    assert toks[chess.D8] == 11  # opponent queen -> id 11
    assert toks[chess.E4] == 0   # empty


def test_black_to_move_pieces_become_side_to_move_ids():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move -> canonical mirror
    toks = encode_square_tokens(canonical_board(b))
    # Black's own pieces (now side-to-move) must use ids 1..6 somewhere
    assert toks.max() >= 6 and (toks[(toks >= 1) & (toks <= 6)].size > 0)
