"""Hybrid move producer: opening book -> Syzygy DTZ -> MCTS.

Composes the strongest available source for each position:
  1. opening book (standard-start lines),
  2. Syzygy DTZ-optimal move in ≤max-piece endgames,
  3. otherwise MCTS (with exact Syzygy WDL at leaves if a tablebase is given).

With `book=None` and `tablebase=None` it is identical to arena_eval's
`build_mcts_mover` — a drop-in, regression-safe replacement.
"""

from __future__ import annotations

from config import Config
from src.mcts.batched_mcts import BatchedMCTS


def build_hybrid_mover(evaluator, simulations, *, book=None, tablebase=None,
                       config=None, c_puct=None, fpu=None):
    """Return a MoveProducer (chess.Board -> Optional[chess.Move]).

    `c_puct` / `fpu` override the search exploration constant and first-play-
    urgency reduction for this mover only (None = inherit from config). Used to
    A/B different search configs of the same checkpoint.
    """
    mcts = BatchedMCTS(evaluator, config or Config, num_simulations=simulations,
                       tablebase=tablebase)
    if c_puct is not None:
        mcts.c_puct = c_puct
    if fpu is not None:
        mcts.fpu_reduction = fpu

    def mover(board):
        if book is not None:
            move = book.lookup(board)
            if move is not None:
                return move
        if tablebase is not None:
            move = tablebase.best_dtz_move(board)
            if move is not None:
                return move
        mcts.reset()
        return mcts.choose_move(board, temperature=0.0)

    return mover
