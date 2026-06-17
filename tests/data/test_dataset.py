import torch
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard
from src.data.dataset import make_dataloader
from src.game.move_encoder import get_move_encoder


def test_loader_batches_and_scatters(tmp_path):
    P = get_move_encoder().policy_size
    lps = [LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0)] * 4
    path = str(tmp_path / "s.npz")
    write_shard(lps, path)

    dl = make_dataloader([path], batch_size=2, policy_size=P, shuffle=False)
    (sq, sf), (pol, wdl, ml) = next(iter(dl))

    # Shape assertions
    assert sq.shape == (2, 64), f"sq shape {sq.shape}"
    assert sq.dtype == torch.int64, f"sq dtype {sq.dtype}"
    assert sf.shape == (2, 18), f"sf shape {sf.shape}"
    assert pol.shape == (2, P), f"pol shape {pol.shape}"
    assert wdl.shape == (2, 3), f"wdl shape {wdl.shape}"
    assert ml.shape == (2, 1), f"ml shape {ml.shape}"

    # Policy correctness: e2e4 should be the only nonzero index, prob 1.0
    e2e4 = get_move_encoder().encode(chess.Move.from_uci("e2e4"))
    assert abs(float(pol[0, e2e4]) - 1.0) < 1e-5
    assert abs(float(pol[0].sum()) - 1.0) < 1e-5

    # dtype checks
    assert sf.dtype == torch.float32
    assert pol.dtype == torch.float32
    assert wdl.dtype == torch.float32
    assert ml.dtype == torch.float32
