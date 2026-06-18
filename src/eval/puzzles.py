# src/eval/puzzles.py
"""Lichess-format puzzle loader and top-1 accuracy measurement."""

from __future__ import annotations

import csv
import dataclasses
import io
import os
from typing import Callable, List, Optional

import chess

MoveProducer = Callable[[chess.Board], Optional[chess.Move]]


@dataclasses.dataclass
class Puzzle:
    """A single puzzle: initial FEN (before the opponent's first move) plus
    the ordered solution moves.

    Lichess format (Moves column): `<opponent_move> <solution_move> [<reply> ...]`.
    The engine is expected to play `solution_moves[0]` on the board that results
    after the opponent's forcing move is applied.
    """

    puzzle_id: str
    fen: str  # FEN of the position *before* the opponent's move
    solution_moves: List[chess.Move]  # [0] = our move, [1:] = subsequent line


def load_puzzles(
    path: Optional[str] = None,
    csv_text: Optional[str] = None,
    max_puzzles: Optional[int] = None,
) -> List[Puzzle]:
    """Load puzzles from a Lichess-format CSV file or inline text.

    Lichess puzzle CSV header:
      PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
      Themes, GameUrl, OpeningTags

    The Moves column is space-separated UCI moves. The first is the opponent's
    forcing move; subsequent moves are the expected solution line.

    Args:
        path: Filesystem path to the CSV file.
        csv_text: Inline CSV string (for tests). Mutually exclusive with path.
        max_puzzles: If set, load at most this many puzzles.

    Returns:
        List of Puzzle dataclasses.
    """
    if path is not None and csv_text is not None:
        raise ValueError("Provide exactly one of path or csv_text, not both.")
    if path is None and csv_text is None:
        raise ValueError("Provide either path or csv_text.")

    if path is not None:
        with open(path, newline="", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = csv_text  # type: ignore[assignment]

    reader = csv.DictReader(io.StringIO(text))
    puzzles: List[Puzzle] = []

    for row in reader:
        puzzle_id = row.get("PuzzleId", "").strip()
        fen = row.get("FEN", "").strip()
        moves_str = row.get("Moves", "").strip()
        if not fen or not moves_str:
            continue

        all_uci = moves_str.split()
        if len(all_uci) < 2:
            # Need at least the opponent move + our move
            continue

        # Apply the opponent's first move to get the position the engine sees.
        try:
            board = chess.Board(fen)
            opp_move = chess.Move.from_uci(all_uci[0])
            board.push(opp_move)
            # Solution moves (from the engine's position onward)
            solution_moves = [chess.Move.from_uci(m) for m in all_uci[1:]]
        except (ValueError, chess.IllegalMoveError):
            continue

        puzzles.append(Puzzle(puzzle_id=puzzle_id, fen=board.fen(), solution_moves=solution_moves))

        if max_puzzles is not None and len(puzzles) >= max_puzzles:
            break

    return puzzles


def load_chessbench_puzzles(
    path: Optional[str] = None,
    csv_text: Optional[str] = None,
    max_puzzles: Optional[int] = None,
) -> List[Puzzle]:
    """Load DeepMind ChessBench puzzles.csv.

    Columns: PuzzleId, Rating, PGN, Solution, FEN, Moves. Unlike Lichess, `FEN`
    is already the position to solve (solver to move) and `Moves` is the UCI
    solution line whose FIRST move is the solver's answer (no leading opponent
    move). So the engine is evaluated on `FEN` directly and must play
    `solution_moves[0]`.
    """
    if path is not None and csv_text is not None:
        raise ValueError("Provide exactly one of path or csv_text, not both.")
    if path is None and csv_text is None:
        raise ValueError("Provide either path or csv_text.")
    if path is not None:
        with open(path, newline="", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = csv_text  # type: ignore[assignment]

    reader = csv.DictReader(io.StringIO(text))
    puzzles: List[Puzzle] = []
    for row in reader:
        fen = row.get("FEN", "").strip()
        moves_str = row.get("Moves", "").strip()
        if not fen or not moves_str:
            continue
        try:
            chess.Board(fen)  # validate
            solution_moves = [chess.Move.from_uci(m) for m in moves_str.split()]
        except (ValueError, chess.IllegalMoveError):
            continue
        if not solution_moves:
            continue
        puzzles.append(Puzzle(puzzle_id=row.get("PuzzleId", "").strip(),
                              fen=fen, solution_moves=solution_moves))
        if max_puzzles is not None and len(puzzles) >= max_puzzles:
            break
    return puzzles


def puzzle_accuracy(
    engine: MoveProducer,
    puzzles: List[Puzzle],
) -> float:
    """Measure top-1 accuracy: fraction of puzzles where the engine's first
    move matches the first solution move.

    Args:
        engine: Move-producer callable.
        puzzles: List of Puzzle instances.

    Returns:
        Accuracy in [0.0, 1.0]. Returns 0.0 for an empty list.
    """
    if not puzzles:
        return 0.0

    correct = 0
    for puzzle in puzzles:
        board = chess.Board(puzzle.fen)
        try:
            move = engine(board)
        except Exception:  # noqa: BLE001
            move = None
        if move is not None and move == puzzle.solution_moves[0]:
            correct += 1

    return correct / len(puzzles)
