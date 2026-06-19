import sys
import chess

from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard
from src.data.targets import winprob_to_wdl
from src.training.checkpoint_meta import read_sidecar


def _tiny_dense_shard(path):
    # one position: e2e4 gets all the policy mass, white-winning value
    lp = LabeledPosition(
        fen=chess.STARTING_FEN,
        policy=[("e2e4", 1.0)],
        wdl=winprob_to_wdl(0.6),
        moves_left=40.0,
    )
    write_shard([lp, lp], path)


def test_distill_cli_writes_routable_sidecar(tmp_path, monkeypatch):
    import scripts.distill as distill

    train = str(tmp_path / "train_0.npz")
    val = str(tmp_path / "val_0.npz")
    _tiny_dense_shard(train)
    _tiny_dense_shard(val)
    ckpt = str(tmp_path / "ck")

    argv = ["distill.py", "--train", train, "--val", val,
            "--preset", "baseline-v1", "--train-data", "unit-test",
            "--batch", "2", "--steps", "1", "--warmup", "1", "--lr", "3e-4",
            "--val-every", "1", "--ckpt", ckpt]
    monkeypatch.setattr(sys, "argv", argv)
    distill.main()

    meta = read_sidecar(str(tmp_path / "ck" / "last.pt"))
    assert meta is not None
    assert meta["objective"] == "policy"
    assert meta["preset"] == "baseline-v1"
    assert meta["train_data"] == "unit-test"
