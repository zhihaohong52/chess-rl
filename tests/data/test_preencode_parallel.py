"""Parallel per-file preencode produces correct, loadable shards."""
import sys

import chess
import msgpack
import numpy as np
import zstandard

import scripts.preencode as pe


def _write_raw(path, n):
    recs = [
        {"fen": chess.STARTING_FEN, "moves": {
            "e2e4": {"win_prob": 0.55, "mate": None},
            "d2d4": {"win_prob": 0.50, "mate": None}}}
        for _ in range(n)
    ]
    blob = b"".join(msgpack.packb(r, use_bin_type=True) for r in recs)
    with open(path, "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(blob))


def _load_total(paths):
    total = 0
    for p in paths:
        d = np.load(p)
        assert set(d.files) == {
            "square_tokens", "state_features", "wdl", "moves_left",
            "legal_indices", "legal_probs", "counts",
        }
        assert d["square_tokens"].dtype == np.int8
        assert d["square_tokens"].shape[1] == 64
        total += d["square_tokens"].shape[0]
    return total


def test_parallel_encode_covers_all_positions(tmp_path, monkeypatch):
    a = tmp_path / "train-00.msgpack.zst"
    b = tmp_path / "train-01.msgpack.zst"
    _write_raw(str(a), 5)
    _write_raw(str(b), 7)
    out = tmp_path / "shards"

    monkeypatch.setattr(sys, "argv", [
        "preencode", "--source", "hf_dense",
        "--input", str(tmp_path / "train-*.msgpack.zst"),
        "--out-dir", str(out), "--shard-size", "10", "--workers", "2",
    ])
    pe.main()

    shards = sorted(out.glob("train_*.npz"))
    assert len(shards) == 2  # one per input file (namespaced by file index)
    assert _load_total([str(s) for s in shards]) == 12


def test_single_process_keeps_legacy_shard_naming(tmp_path, monkeypatch):
    a = tmp_path / "train-00.msgpack.zst"
    _write_raw(str(a), 6)
    out = tmp_path / "shards"

    monkeypatch.setattr(sys, "argv", [
        "preencode", "--source", "hf_dense", "--input", str(a),
        "--out-dir", str(out), "--shard-size", "100",
    ])
    pe.main()

    assert (out / "train_00000.npz").exists()  # historical naming preserved
    assert _load_total([str(out / "train_00000.npz")]) == 6
