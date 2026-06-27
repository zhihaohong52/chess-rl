"""Offline tests for `OnlineSyzygyTablebase`.

No network: every test injects a fake `_opener(url) -> bytes` so the Lichess HTTP
path is exercised deterministically. We assert the contract the engine relies on:
local 5-man stays an instant, network-free fast-path; the online oracle only fires
for >local / <=max_pieces positions; and *every* failure mode (oversized, empty,
network error, garbage, illegal move) degrades to None so the caller falls back to
the local tables / MCTS.
"""
import json

import chess

from src.engine.online_tb import OnlineSyzygyTablebase

# 7-man, White to move and winning: the lone-rook capture Rxa1 (h1a1) is best.
KRPPP_VS_KR = "4k3/8/8/8/8/4P3/3PKP2/r6R w - - 0 1"


class FakeLocal:
    """Duck-typed `SyzygyTablebase` whose every answer is canned."""

    def __init__(self, move=None, value=None, scope=False):
        self._move = move
        self._value = value
        self._scope = scope
        self.closed = False

    def best_dtz_move(self, board):
        return self._move

    def probe_value(self, board):
        return self._value

    def in_scope(self, board):
        return self._scope

    def close(self):
        self.closed = True


class RecordingOpener:
    """Records every URL fetched and returns a canned body (or raises)."""

    def __init__(self, payload):
        self.payload = payload  # dict (-> json), bytes, or callable(url)->bytes
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if callable(self.payload):
            return self.payload(url)
        if isinstance(self.payload, (bytes, bytearray)):
            return bytes(self.payload)
        return json.dumps(self.payload).encode()


def _tb(local, payload, **kw):
    return OnlineSyzygyTablebase(local=local, _opener=RecordingOpener(payload), **kw)


def test_local_hit_short_circuits_without_network():
    mv = chess.Move.from_uci("e2e4")
    tb = _tb(FakeLocal(move=mv), {"moves": [{"uci": "h1a1"}]})
    # start position (32 men) — only the local fast-path can answer here.
    assert tb.best_dtz_move(chess.Board()) == mv
    assert tb._opener.urls == []   # never reached the network
    assert tb.calls == 0


def test_online_used_when_local_misses():
    board = chess.Board(KRPPP_VS_KR)  # 7 men, beyond local's 5
    tb = _tb(FakeLocal(move=None), {"moves": [{"uci": "h1a1"}, {"uci": "e2e3"}]},
             max_pieces=7)
    assert tb.best_dtz_move(board) == chess.Move.from_uci("h1a1")  # moves[0]
    assert len(tb._opener.urls) == 1
    assert "fen=" in tb._opener.urls[0]
    assert (tb.calls, tb.hits) == (1, 1)


def test_oversized_position_is_gated_without_network():
    tb = _tb(FakeLocal(move=None), {"moves": [{"uci": "e2e4"}]}, max_pieces=7)
    assert tb.best_dtz_move(chess.Board()) is None  # 32 men > 7
    assert tb._opener.urls == []
    assert tb.calls == 0


def test_empty_moves_returns_none():
    tb = _tb(FakeLocal(move=None), {"moves": []}, max_pieces=7)
    assert tb.best_dtz_move(chess.Board(KRPPP_VS_KR)) is None
    assert (tb.calls, tb.hits) == (1, 0)


def test_network_error_returns_none():
    def boom(url):
        raise RuntimeError("network down")

    tb = OnlineSyzygyTablebase(local=FakeLocal(move=None), _opener=boom, max_pieces=7)
    assert tb.best_dtz_move(chess.Board(KRPPP_VS_KR)) is None
    assert (tb.calls, tb.hits) == (1, 0)


def test_garbage_body_returns_none():
    tb = _tb(FakeLocal(move=None), b"not json", max_pieces=7)
    assert tb.best_dtz_move(chess.Board(KRPPP_VS_KR)) is None
    assert (tb.calls, tb.hits) == (1, 0)


def test_illegal_move_in_response_returns_none():
    # a8a1: no piece on a8 in this FEN, so the parsed move isn't legal here.
    tb = _tb(FakeLocal(move=None), {"moves": [{"uci": "a8a1"}]}, max_pieces=7)
    assert tb.best_dtz_move(chess.Board(KRPPP_VS_KR)) is None
    assert (tb.calls, tb.hits) == (1, 0)  # counted as a probe, not a hit


def test_probe_value_and_in_scope_delegate_to_local():
    tb = _tb(FakeLocal(value=1.0, scope=True), {})
    b = chess.Board()
    assert tb.probe_value(b) == 1.0
    assert tb.in_scope(b) is True


def test_no_local_degrades_gracefully():
    board = chess.Board(KRPPP_VS_KR)
    tb = OnlineSyzygyTablebase(local=None,
                               _opener=RecordingOpener({"moves": [{"uci": "h1a1"}]}),
                               max_pieces=7)
    # leaf/scope probing has nothing to delegate to -> None / False, never networked
    assert tb.probe_value(board) is None
    assert tb.in_scope(board) is False
    # but the root oracle still works straight off the network
    assert tb.best_dtz_move(board) == chess.Move.from_uci("h1a1")


def test_close_delegates_and_tolerates_no_local():
    local = FakeLocal()
    OnlineSyzygyTablebase(local=local).close()
    assert local.closed
    OnlineSyzygyTablebase(local=None).close()  # must not raise
