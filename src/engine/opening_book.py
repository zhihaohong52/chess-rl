"""Polyglot opening-book wrapper for the engine.

Only fires from positions the book knows (i.e. standard-start lines); returns
None otherwise so the caller falls back to MCTS. Every returned move is checked
against `board.legal_moves` to defend against foreign/corrupt books.
"""

from __future__ import annotations

import os
from typing import Optional

import chess
import chess.polyglot


def book_available(path: Optional[str]) -> bool:
    """True if a Polyglot `.bin` exists at `path` (gate before opening it)."""
    return bool(path) and os.path.exists(path)


class OpeningBook:
    def __init__(self, path: Optional[str] = None, _reader=None) -> None:
        """Open the Polyglot book at `path`. `_reader` injects a reader for tests."""
        if _reader is not None:
            self._reader = _reader
        else:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Opening book not found: {path!r}")
            self._reader = chess.polyglot.open_reader(path)

    def lookup(self, board: chess.Board, *, weighted: bool = False,
               rng=None) -> Optional[chess.Move]:
        """Best (or weight-sampled) book move for `board`, or None if out of book.

        A returned move is None unless it is legal in `board` (foreign-book guard).
        """
        try:
            if weighted:
                entry = (self._reader.weighted_choice(board, random=rng)
                         if rng is not None else self._reader.weighted_choice(board))
            else:
                entry = self._reader.find(board)  # highest-weight entry
        except (IndexError, KeyError):
            return None  # no book entry for this position
        move = entry.move
        return move if move in board.legal_moves else None

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> "OpeningBook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
