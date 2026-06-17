import numpy as np
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard, encode_example
from src.data.dataset import ShardDataset
from src.game.move_encoder import get_move_encoder


def test_write_shard_roundtrip(tmp_path):
    lps = [
        LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0),
        LabeledPosition(chess.STARTING_FEN, [("d2d4", 0.5), ("c2c4", 0.5)], (0.4, 0.5, 0.1), 79.0),
    ]
    path = str(tmp_path / "shard0.npz")
    n = write_shard(lps, path)
    assert n == 2

    # Raw npz reload
    d = np.load(path)
    assert d["square_tokens"].shape[0] == 2

    # Round-trip via ShardDataset
    P = get_move_encoder().policy_size
    ds = ShardDataset([path], policy_size=P)
    assert len(ds) == 2

    # Check that a known example survives
    ref = encode_example(lps[0])
    (sq, sf), (policy, wdl, ml) = ds[0]

    assert sq.shape == (64,) and sq.dtype == __import__("torch").int64
    assert sf.shape == (18,)
    assert policy.shape == (P,)
    assert wdl.shape == (3,)
    assert ml.shape == (1,)

    # e2e4 should be the only hot index with prob 1.0
    me = get_move_encoder()
    e2e4_idx = me.encode(chess.Move.from_uci("e2e4"))
    assert abs(float(policy[e2e4_idx]) - 1.0) < 1e-5
    assert abs(float(policy.sum()) - 1.0) < 1e-5

    # wdl and moves_left match
    np.testing.assert_allclose(wdl.numpy(), ref["wdl"], atol=1e-6)
    assert abs(float(ml[0]) - ref["moves_left"]) < 1e-6
