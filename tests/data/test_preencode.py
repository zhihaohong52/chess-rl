import numpy as np
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import encode_example, write_shard
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


def test_write_shard_produces_npz_with_correct_schema(tmp_path):
    lps = [
        LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0),
        LabeledPosition(chess.STARTING_FEN, [("d2d4", 0.5), ("c2c4", 0.5)], (0.4, 0.5, 0.1), 79.0),
    ]
    path = str(tmp_path / "shard0.npz")
    n = write_shard(lps, path)
    assert n == 2

    d = np.load(path)
    assert d["square_tokens"].shape == (2, 64) and d["square_tokens"].dtype == np.int8
    assert d["state_features"].shape == (2, 18) and d["state_features"].dtype == np.float32
    assert d["wdl"].shape == (2, 3) and d["wdl"].dtype == np.float32
    assert d["moves_left"].shape == (2,) and d["moves_left"].dtype == np.float32
    # legal_indices and legal_probs are concatenated: lp0 has 1 move, lp1 has 2
    assert d["legal_indices"].dtype == np.int32
    assert d["legal_probs"].dtype == np.float32
    assert d["counts"].shape == (2,) and d["counts"].dtype == np.int32
    assert d["counts"][0] == 1 and d["counts"][1] == 2
    assert len(d["legal_indices"]) == 3  # 1 + 2
