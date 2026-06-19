import zstandard
import msgpack
import chess

from src.data.chessbench import iter_hf_dense


def _write(path, records):
    blob = b"".join(msgpack.packb(r, use_bin_type=True) for r in records)
    with open(path, "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(blob))


def test_iter_hf_dense_builds_positions(tmp_path):
    p = str(tmp_path / "s.msgpack.zst")
    recs = [
        {"fen": chess.STARTING_FEN, "moves": {
            "e2e4": {"win_prob": 0.55, "mate": None},
            "d2d4": {"win_prob": 0.50, "mate": None},
            "a2a3": {"win_prob": 0.40, "mate": None}}},
    ]
    _write(p, recs)
    out = list(iter_hf_dense(p, temperature=0.1))
    assert len(out) == 1
    lp = out[0]
    assert lp.fen == chess.STARTING_FEN
    probs = dict(lp.policy)
    assert set(probs) == {"e2e4", "d2d4", "a2a3"}
    # sharper temperature -> best move dominates
    assert probs["e2e4"] > probs["d2d4"] > probs["a2a3"]
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    # value from best win% (0.55 -> W > L)
    assert lp.wdl[0] > lp.wdl[2]


def test_iter_hf_dense_max_positions(tmp_path):
    p = str(tmp_path / "s.msgpack.zst")
    recs = [{"fen": chess.STARTING_FEN,
             "moves": {"e2e4": {"win_prob": 0.5, "mate": None}}} for _ in range(5)]
    _write(p, recs)
    assert len(list(iter_hf_dense(p, max_positions=3))) == 3
