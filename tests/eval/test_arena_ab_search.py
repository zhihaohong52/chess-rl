"""Per-side search-config A/B plumbing for arena_eval head-to-head.

These guard that `--c-puct/--fpu/--sims-b/...` actually reach the MCTS object
(not just parse) and that the run label distinguishes asymmetric configs. We
spy on BatchedMCTS so the tests stay fast (no real search / evaluator).
"""

import os
import subprocess
import sys

import scripts.arena_eval as ae
import src.engine.hybrid_mover as hm
from config import Config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _SpyMCTS:
    """Stand-in that records the construction + post-construct overrides."""

    last = None

    def __init__(self, evaluator, config=None, num_simulations=None, tablebase=None):
        cfg = config or Config
        self.num_simulations = num_simulations
        self.c_puct = cfg.c_puct
        self.fpu_reduction = getattr(cfg, "fpu_reduction", None)
        self.tablebase = tablebase
        _SpyMCTS.last = self

    def reset(self):
        pass

    def choose_move(self, board, temperature=0.0):
        return None


def test_build_mcts_mover_default_inherits_config(monkeypatch):
    monkeypatch.setattr(ae, "BatchedMCTS", _SpyMCTS)
    ae.build_mcts_mover(object(), simulations=10)
    assert _SpyMCTS.last.c_puct == Config.c_puct
    assert _SpyMCTS.last.fpu_reduction == Config.fpu_reduction  # None by default


def test_build_mcts_mover_applies_overrides(monkeypatch):
    monkeypatch.setattr(ae, "BatchedMCTS", _SpyMCTS)
    ae.build_mcts_mover(object(), simulations=10, c_puct=2.5, fpu=0.3)
    assert _SpyMCTS.last.c_puct == 2.5
    assert _SpyMCTS.last.fpu_reduction == 0.3
    assert _SpyMCTS.last.num_simulations == 10


def test_build_hybrid_mover_applies_overrides(monkeypatch):
    monkeypatch.setattr(hm, "BatchedMCTS", _SpyMCTS)
    hm.build_hybrid_mover(object(), 20, c_puct=4.0, fpu=0.5)
    assert _SpyMCTS.last.c_puct == 4.0
    assert _SpyMCTS.last.fpu_reduction == 0.5


def test_search_cfg_str_omits_unset_knobs():
    assert ae._search_cfg_str(200, None, None) == "200 sims"
    assert ae._search_cfg_str(800, 2.5, None) == "800 sims, c_puct=2.5"
    assert ae._search_cfg_str(800, 1.5, 0.3) == "800 sims, c_puct=1.5, fpu=0.3"


def test_cli_help_shows_ab_search_flags():
    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/arena_eval.py"), "--help"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert res.returncode == 0
    for flag in ("--c-puct", "--fpu", "--sims-b", "--c-puct-b", "--fpu-b"):
        assert flag in res.stdout
