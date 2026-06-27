from src.training.checkpoint_meta import sidecar_path, write_sidecar, read_sidecar


def test_sidecar_path_swaps_extension(tmp_path):
    p = str(tmp_path / "best.pt")
    assert sidecar_path(p) == str(tmp_path / "best.json")


def test_write_then_read_roundtrip(tmp_path):
    ckpt = str(tmp_path / "best.pt")
    write_sidecar(ckpt, {"preset": "baseline-v1", "objective": "policy",
                         "train_data": "test-bag-62k"})
    meta = read_sidecar(ckpt)
    assert meta["preset"] == "baseline-v1"
    assert meta["objective"] == "policy"
    assert meta["train_data"] == "test-bag-62k"
    assert "git_sha" in meta  # auto-added


def test_read_missing_returns_none(tmp_path):
    assert read_sidecar(str(tmp_path / "nope.pt")) is None
