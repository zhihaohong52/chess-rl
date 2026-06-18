import chess
import pytest

from src.data.bagz import write_records
from src.data.chessbench import (
    decode_action_value,
    encode_action_value,
    iter_chessbench,
)
from src.data.labeled_position import LabeledPosition


def test_action_value_codec_roundtrip():
    rec = encode_action_value(chess.STARTING_FEN, "e2e4", 0.55)
    fen, move, win = decode_action_value(rec)
    assert fen == chess.STARTING_FEN
    assert move == "e2e4"
    assert abs(win - 0.55) < 1e-9


def test_decode_matches_apache_beam():
    """If apache_beam is installed, confirm our decoder matches the real coder."""
    pytest.importorskip("apache_beam")
    from apache_beam import coders

    coder = coders.TupleCoder(
        (coders.StrUtf8Coder(), coders.StrUtf8Coder(), coders.FloatCoder())
    )
    encoded = coder.encode(("8/8/8/8/8/8/8/8 w - - 0 1", "a1a2", 0.73))
    fen, move, win = decode_action_value(encoded)
    assert fen == "8/8/8/8/8/8/8/8 w - - 0 1"
    assert move == "a1a2"
    assert abs(win - 0.73) < 1e-9


def test_iter_chessbench_groups_by_fen(tmp_path):
    fen1 = chess.STARTING_FEN
    b = chess.Board()
    b.push_uci("e2e4")
    fen2 = b.fen()
    records = [
        encode_action_value(fen1, "e2e4", 0.60),
        encode_action_value(fen1, "d2d4", 0.55),
        encode_action_value(fen1, "a2a3", 0.40),
        encode_action_value(fen2, "e7e5", 0.52),
        encode_action_value(fen2, "c7c5", 0.50),
    ]
    path = str(tmp_path / "av.bag")
    write_records(records, path)

    positions = list(iter_chessbench(path, temperature=0.1))
    assert len(positions) == 2
    p1, p2 = positions
    assert isinstance(p1, LabeledPosition)
    assert p1.fen == fen1 and len(p1.policy) == 3
    assert p2.fen == fen2 and len(p2.policy) == 2

    assert abs(sum(pr for _, pr in p1.policy) - 1.0) < 1e-6
    best = max(p1.policy, key=lambda mp: mp[1])
    assert best[0] == "e2e4"  # highest win% move
    assert abs(sum(p1.wdl) - 1.0) < 1e-6
    assert p1.wdl[0] > p1.wdl[2]  # win-leaning value from best win% 0.60


def test_iter_chessbench_groups_shuffled(tmp_path):
    """Real ChessBench files interleave a position's moves; dict-grouping handles it."""
    fen1 = chess.STARTING_FEN
    b = chess.Board()
    b.push_uci("e2e4")
    fen2 = b.fen()
    records = [
        encode_action_value(fen1, "e2e4", 0.60),
        encode_action_value(fen2, "e7e5", 0.52),
        encode_action_value(fen1, "d2d4", 0.55),
        encode_action_value(fen2, "c7c5", 0.50),
        encode_action_value(fen1, "a2a3", 0.40),
    ]
    path = str(tmp_path / "shuf.bag")
    write_records(records, path)
    positions = {p.fen: p for p in iter_chessbench(path, temperature=0.1)}
    assert len(positions) == 2
    assert len(positions[fen1].policy) == 3
    assert len(positions[fen2].policy) == 2


def test_iter_chessbench_max_positions(tmp_path):
    fen1 = chess.STARTING_FEN
    b = chess.Board()
    b.push_uci("d2d4")
    fen2 = b.fen()
    records = [
        encode_action_value(fen1, "e2e4", 0.6),
        encode_action_value(fen2, "d7d5", 0.5),
    ]
    path = str(tmp_path / "av2.bag")
    write_records(records, path)
    assert len(list(iter_chessbench(path, max_positions=1))) == 1
