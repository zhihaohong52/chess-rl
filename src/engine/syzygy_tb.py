"""Syzygy endgame tablebase wrapper for the engine.

Provides exact endgame knowledge in two forms:
  - `probe_value(board)`: a side-to-move-POV scalar for an MCTS leaf (so search
    backs up perfect endgame values), consistent with `_terminal_value` /
    negamax in `src/mcts/batched_mcts.py`.
  - `best_dtz_move(board)`: the DTZ-optimal root move (50-move-rule aware), so the
    engine actually *converts* won endgames instead of shuffling.

All probes self-gate on piece count, and degrade gracefully (return None) when a
required table is missing — callers fall back to the neural net / MCTS.
"""

from __future__ import annotations

import os
from typing import Optional

import chess
import chess.syzygy


class SyzygyTablebase:
    def __init__(
        self,
        directory: Optional[str] = None,
        max_pieces: int = 5,
        cursed_win_value: float = 0.0,
        _tb=None,
    ) -> None:
        """Open the tablebase under `directory` (≤`max_pieces`-man positions).

        `cursed_win_value` maps cursed-wins/blessed-losses (wins/losses that are
        drawn under the 50-move rule) to ±this value; default 0.0 treats them as
        the draws they are. `_tb` injects a pre-opened tablebase for testing.
        """
        self.max_pieces = max_pieces
        self.cursed_win_value = cursed_win_value
        if _tb is not None:
            self._tb = _tb
        else:
            if not directory or not os.path.isdir(directory):
                raise FileNotFoundError(f"Syzygy directory not found: {directory!r}")
            self._tb = chess.syzygy.open_tablebase(directory)

    def in_scope(self, board: chess.Board) -> bool:
        """True if the position is small enough to probe (cheap pre-probe gate)."""
        return chess.popcount(board.occupied) <= self.max_pieces

    def probe_value(self, board: chess.Board) -> Optional[float]:
        """Side-to-move-POV value in [-1, 1], or None if out of scope / no table."""
        if not self.in_scope(board):
            return None
        wdl = self._tb.get_wdl(board)
        if wdl is None:
            return None
        if wdl >= 2:
            return 1.0
        if wdl <= -2:
            return -1.0
        if wdl == 1:       # cursed win — drawn under the 50-move rule
            return self.cursed_win_value
        if wdl == -1:      # blessed loss
            return -self.cursed_win_value
        return 0.0         # true draw

    @staticmethod
    def _rank_key(our_wdl: int, child_dtz: Optional[int]):
        """Move-ranking key (higher = better): WDL first, then DTZ.

        Primary `our_wdl` never discards a win or draw. Secondary `child_dtz`
        (opponent POV) maximized: for winning moves the child DTZ is negative, so
        larger = closer to 0 = faster mate; for losing moves it is positive, so
        larger = longer resistance. (WDL-optimal; the DTZ tie-break is a practical
        fast-convert/long-resist heuristic, not a fully 50-move-exact DTZ solver.)
        """
        return (our_wdl, child_dtz if child_dtz is not None else 0)

    def best_dtz_move(self, board: chess.Board) -> Optional[chess.Move]:
        """The WDL-optimal root move (DTZ-tie-broken), or None out of scope /
        when any child table is missing (caller falls back to MCTS)."""
        if not self.in_scope(board):
            return None
        best_move = None
        best_key = None
        for move in board.legal_moves:
            board.push(move)
            try:
                child_wdl = self._tb.get_wdl(board)
                child_dtz = self._tb.get_dtz(board)
            finally:
                board.pop()
            if child_wdl is None:
                return None  # incomplete tables — fall back rather than guess
            key = self._rank_key(-child_wdl, child_dtz)
            if best_key is None or key > best_key:
                best_key = key
                best_move = move
        return best_move

    def close(self) -> None:
        self._tb.close()

    def __enter__(self) -> "SyzygyTablebase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
