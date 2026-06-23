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


def syzygy_available(directory: Optional[str]) -> bool:
    """True if `directory` exists and holds at least one WDL table (`*.rtbw`).

    Mirrors `stockfish_available` so scripts/tests can gate on real tables being
    present without opening them.
    """
    if not directory or not os.path.isdir(directory):
        return False
    return any(f.endswith(".rtbw") for f in os.listdir(directory))


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
    def _rank_key(our_wdl: int, child_dtz: Optional[int], win_zeroing: int = 0):
        """Move-ranking key (higher = better): WDL, then zeroing, then DTZ.

        Primary `our_wdl` never discards a win or draw. `win_zeroing` (1 only for a
        winning capture/pawn move) is the crucial middle term: raw DTZ magnitudes
        are *not comparable across a zeroing boundary*, because DTZ counts plies to
        the next zeroing move, not to mate. Without it, maximizing `child_dtz`
        toward 0 chases "soonest opponent zeroing" — a harmless forced pawn push —
        over the winning capture that leads to a pawnless mate (whose larger DTZ
        looks worse), so the engine herds the king forever and draws won pawn
        endings (KQ-vs-KP, KR-vs-KP) by repetition/stalemate. Preferring a winning
        zeroing move resets the 50-move counter and guarantees material progress.
        Pawnless wins have no zeroing moves, so this term is inert there and the
        DTZ tie-break alone still picks the fastest mate (KQ-vs-K unchanged).

        Tertiary `child_dtz` (opponent POV) maximized: for winning moves the child
        DTZ is negative, so larger = closer to 0 = faster mate; for losing moves it
        is positive, so larger = longer resistance.
        """
        return (our_wdl, win_zeroing, child_dtz if child_dtz is not None else 0)

    def best_dtz_move(self, board: chess.Board) -> Optional[chess.Move]:
        """The WDL-optimal, 50-move-aware root move, or None out of scope / when
        any child table is missing (caller falls back to MCTS).

        Winning captures/pawn moves are preferred over equal-WDL quiet moves so the
        engine actually converts won endgames with pawns instead of shuffling — see
        `_rank_key` for why raw DTZ alone fails there.
        """
        if not self.in_scope(board):
            return None
        best_move = None
        best_key = None
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)  # capture or pawn move (resets 50mr)
            board.push(move)
            try:
                child_wdl = self._tb.get_wdl(board)
                child_dtz = self._tb.get_dtz(board)
            finally:
                board.pop()
            if child_wdl is None:
                return None  # incomplete tables — fall back rather than guess
            our_wdl = -child_wdl
            win_zeroing = 1 if (our_wdl >= 2 and zeroing) else 0
            key = self._rank_key(our_wdl, child_dtz, win_zeroing)
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
