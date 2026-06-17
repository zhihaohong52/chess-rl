import numpy as np
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard
from src.data.dataset import make_dataset
from src.game.move_encoder import get_move_encoder


def test_loader_batches_and_scatters(tmp_path):
    P = get_move_encoder().policy_size
    lps = [LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0)] * 4
    path = str(tmp_path / "s.tfrecord")
    write_shard(lps, path)

    ds = make_dataset([path], batch_size=2, policy_size=P, shuffle=False)
    (sq, sf), (pol, wdl, ml) = next(iter(ds))
    assert sq.shape == (2, 64) and sq.dtype.name == "int32"
    assert sf.shape == (2, 18)
    assert pol.shape == (2, P)
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    assert abs(float(pol[0, e2e4]) - 1.0) < 1e-5
    assert abs(float(pol[0].numpy().sum()) - 1.0) < 1e-5
    assert wdl.shape == (2, 3) and ml.shape == (2, 1)
