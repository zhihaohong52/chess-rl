import chess

from src.engine.syzygy_tb import SyzygyTablebase

# 3-man winning position (white K+R vs black k), white to move.
KRK = "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"


class FakeTB:
    """Stand-in for chess.syzygy.Tablebase that returns a fixed WDL and counts probes."""

    def __init__(self, wdl):
        self._wdl = wdl
        self.calls = 0

    def get_wdl(self, board, default=None):
        self.calls += 1
        return self._wdl

    def get_dtz(self, board, default=None):
        return None

    def close(self):
        pass


def _tb(wdl, max_pieces=5, cursed=0.0):
    return SyzygyTablebase(max_pieces=max_pieces, cursed_win_value=cursed, _tb=FakeTB(wdl))


def test_in_scope_gate():
    tb = _tb(2)
    assert tb.in_scope(chess.Board(KRK)) is True   # 3 men
    assert tb.in_scope(chess.Board()) is False      # 32 men


def test_probe_value_win_loss_draw_pov():
    # side-to-move POV, mirrors _terminal_value (mate -1.0)
    assert _tb(2).probe_value(chess.Board(KRK)) == 1.0    # win
    assert _tb(-2).probe_value(chess.Board(KRK)) == -1.0  # loss
    assert _tb(0).probe_value(chess.Board(KRK)) == 0.0    # draw


def test_cursed_and_blessed_are_draws_by_default():
    # cursed win (+1) / blessed loss (-1) are draws under the 50-move rule
    assert _tb(1).probe_value(chess.Board(KRK)) == 0.0
    assert _tb(-1).probe_value(chess.Board(KRK)) == 0.0


def test_cursed_win_value_param_keeps_true_draw_zero():
    assert _tb(1, cursed=0.5).probe_value(chess.Board(KRK)) == 0.5
    assert _tb(-1, cursed=0.5).probe_value(chess.Board(KRK)) == -0.5
    assert _tb(0, cursed=0.5).probe_value(chess.Board(KRK)) == 0.0  # true draw unaffected


def test_out_of_scope_returns_none_without_probing():
    fake = FakeTB(2)
    tb = SyzygyTablebase(max_pieces=5, _tb=fake)
    assert tb.probe_value(chess.Board()) is None  # 32 men
    assert fake.calls == 0  # gated before any probe


def test_missing_table_returns_none():
    assert _tb(None).probe_value(chess.Board(KRK)) is None
