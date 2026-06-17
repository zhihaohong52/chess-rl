# tests/eval/test_puzzles.py
"""Tests for puzzle accuracy measurement."""
import chess
import pytest
from src.eval.puzzles import load_puzzles, puzzle_accuracy, Puzzle


# ---------------------------------------------------------------------------
# Inline puzzle fixture — two trivial one-move mates.
# Lichess CSV format: PuzzleId,FEN,Moves,Rating,...
# The first move in Moves is the opponent's forcing move that starts the
# puzzle; the engine must play the SECOND move (the solution).
# ---------------------------------------------------------------------------
PUZZLE_CSV = """\
PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
puzzle001,r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 4,d8f6 h5f7,1500,100,95,1000,mate mateIn1,https://lichess.org/abc,
puzzle002,r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 4,d8f6 h5f7,1200,80,90,800,mate mateIn1,https://lichess.org/def,
"""

# ---------------------------------------------------------------------------
# Deterministic move-producers for testing
# ---------------------------------------------------------------------------

def _solve_correctly(board: chess.Board) -> chess.Move:
    """Returns the correct solution: Qxf7# (h5f7 in canonical)."""
    target = chess.Move.from_uci("h5f7")
    if target in board.legal_moves:
        return target
    return sorted(board.legal_moves, key=lambda m: m.uci())[0]


def _solve_wrongly(board: chess.Board) -> chess.Move:
    """Always returns a legal move guaranteed NOT to be the solution (h5f7)."""
    solution = chess.Move.from_uci("h5f7")
    for mv in sorted(board.legal_moves, key=lambda m: m.uci()):
        if mv != solution:
            return mv
    return solution  # unreachable here (position has many legal moves)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_puzzles_from_csv_string():
    puzzles = load_puzzles(csv_text=PUZZLE_CSV)
    assert len(puzzles) == 2


def test_puzzle_dataclass_fields():
    puzzles = load_puzzles(csv_text=PUZZLE_CSV)
    p = puzzles[0]
    assert isinstance(p, Puzzle)
    assert p.puzzle_id == "puzzle001"
    assert isinstance(p.fen, str)
    assert len(p.solution_moves) >= 1


def test_puzzle_solution_is_second_move():
    """The solution move is the second move in the Moves column (index 1)."""
    puzzles = load_puzzles(csv_text=PUZZLE_CSV)
    p = puzzles[0]
    # The first move is the opponent's move; the second is our solution.
    assert p.solution_moves[0].uci() == "h5f7"


def test_puzzle_accuracy_perfect():
    puzzles = load_puzzles(csv_text=PUZZLE_CSV)
    acc = puzzle_accuracy(engine=_solve_correctly, puzzles=puzzles)
    assert acc == pytest.approx(1.0, abs=1e-9)


def test_puzzle_accuracy_zero():
    puzzles = load_puzzles(csv_text=PUZZLE_CSV)
    # _solve_wrongly deterministically returns a legal move != h5f7, so acc == 0.
    acc = puzzle_accuracy(engine=_solve_wrongly, puzzles=puzzles)
    assert acc == pytest.approx(0.0, abs=1e-9)


def test_puzzle_accuracy_empty():
    acc = puzzle_accuracy(engine=_solve_correctly, puzzles=[])
    assert acc == pytest.approx(0.0, abs=1e-9)
