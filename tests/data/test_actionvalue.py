import numpy as np
import torch
import chess

from src.data.bagz import write_records
from src.data.chessbench import (
    encode_action_value,
    iter_chessbench_actionvalue,
)
from src.data.preencode import encode_av_example, write_av_shard
from src.data.dataset import make_av_dataloader
from src.game.move_encoder import get_move_encoder
from src.game.orientation import to_canonical_move


def test_iter_actionvalue_roundtrip(tmp_path):
    samples = [
        (chess.STARTING_FEN, "e2e4", 0.55),
        (chess.STARTING_FEN, "d2d4", 0.52),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "e7e5", 0.48),
    ]
    bag = str(tmp_path / "av.bag")
    write_records([encode_action_value(f, m, w) for f, m, w in samples], bag)

    got = list(iter_chessbench_actionvalue(bag))
    assert len(got) == 3
    for (gf, gm, gw), (f, m, w) in zip(got, samples):
        assert gf == f and gm == m
        assert abs(gw - w) < 1e-6


def test_encode_av_example_action_idx_decodes_to_canonical_move():
    me = get_move_encoder()
    fen = chess.STARTING_FEN
    sq, sf, idx, win = encode_av_example(fen, "e2e4", 0.55)
    assert sq.shape == (64,) and sq.dtype == np.int8
    assert sf.shape == (18,) and sf.dtype == np.float32
    assert abs(float(win) - 0.55) < 1e-6
    # action_idx must decode back to the canonical encoding of e2e4 (white to move).
    expected = me.encode(to_canonical_move(chess.Move.from_uci("e2e4"), chess.WHITE))
    assert idx == expected


def test_av_shard_and_loader(tmp_path):
    samples = [
        (chess.STARTING_FEN, "e2e4", 0.55),
        (chess.STARTING_FEN, "g1f3", 0.51),
        (chess.STARTING_FEN, "d2d4", 0.53),
        (chess.STARTING_FEN, "c2c4", 0.50),
    ]
    path = str(tmp_path / "train_00000.npz")
    n = write_av_shard(samples, path)
    assert n == 4

    dl = make_av_dataloader([path], batch_size=2, shuffle=False)
    (sq, sf), (ai, win) = next(iter(dl))
    assert sq.shape == (2, 64) and sq.dtype == torch.int64
    assert sf.shape == (2, 18) and sf.dtype == torch.float32
    assert ai.shape == (2,) and ai.dtype == torch.int64
    assert win.shape == (2,) and win.dtype == torch.float32
    assert abs(float(win[0]) - 0.55) < 1e-6
