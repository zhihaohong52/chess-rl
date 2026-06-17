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


from src.game.token_encoder import encode_state_features, encode_position, encode_batch, STATE_DIM


def test_state_features_fifty_move_and_repetition():
    # halfmove clock 40 set via FEN field 5; canonical frame preserves it
    b = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 40 80")
    f = encode_state_features(canonical_board(b), repetition_count=2)
    assert f.shape == (STATE_DIM,) and f.dtype == np.float32
    assert abs(f[13] - 0.40) < 1e-6   # halfmove_clock / 100
    assert abs(f[14] - (2 / 3.0)) < 1e-6  # repetition count / 3
    assert f[16] == 1.0  # constant bias


def test_encode_position_and_batch_shapes():
    st, sf = encode_position(chess.Board())
    assert st.shape == (64,) and sf.shape == (STATE_DIM,)
    sts, sfs = encode_batch([chess.Board(), chess.Board()], [0, 0])
    assert sts.shape == (2, 64) and sfs.shape == (2, STATE_DIM)
