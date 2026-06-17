import numpy as np
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import encode_example
from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move


def test_encode_example_canonical_white_to_move():
    lp = LabeledPosition(
        fen=chess.STARTING_FEN,
        policy=[("e2e4", 0.7), ("d2d4", 0.3)],
        wdl=(0.5, 0.4, 0.1),
        moves_left=80.0,
    )
    ex = encode_example(lp)
    assert ex["square_tokens"].shape == (64,) and ex["square_tokens"].dtype == np.int8
    assert ex["state_features"].shape == (18,)
    assert ex["wdl"].shape == (3,) and abs(ex["wdl"].sum() - 1.0) < 1e-6
    # white to move -> canonical == real; e2e4 index present with prob 0.7
    me = get_move_encoder()
    e2e4_idx = me.encode(chess.Move.from_uci("e2e4"))
    pairs = dict(zip(ex["legal_indices"].tolist(), ex["legal_probs"].tolist()))
    assert abs(pairs[e2e4_idx] - 0.7) < 1e-6


def test_encode_example_black_to_move_mirrors_moves():
    b = chess.Board()
    b.push_uci("e2e4")  # black to move
    lp = LabeledPosition(fen=b.fen(), policy=[("e7e5", 1.0)], wdl=(0.4, 0.4, 0.2), moves_left=78.0)
    ex = encode_example(lp)
    me = get_move_encoder()
    expected = me.encode(to_canonical_move(chess.Move.from_uci("e7e5"), chess.BLACK))
    assert ex["legal_indices"].tolist() == [expected]
