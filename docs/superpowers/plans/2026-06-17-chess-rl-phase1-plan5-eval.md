# Phase 1 Plan 5 — Evaluation Harness Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Build a complete strength-evaluation harness: an arena for engine-vs-engine matches, a Lichess-format puzzle solver, Elo math utilities, a Stockfish ladder opponent, and a `scripts/eval.py` CLI that ties them together and prints an Elo estimate from a checkpoint.

**Architecture:** Five thin, independently testable modules under `src/eval/`. The arena accepts any "move-producer" callable `(chess.Board) -> chess.Move`, making it trivially stubbable in tests. Puzzle accuracy is measured by feeding puzzle FENs through the move-producer and checking the top-1 move. Elo math is pure Python. The Stockfish opponent wraps the existing `StockfishEvaluator` and is gated on binary availability; all unit tests use deterministic stubs. The CLI script wires everything together with argparse.

**Tech Stack:** Python 3.12, python-chess, numpy, pytest (already installed by Plan 1). No TensorFlow import required in `src/eval/` itself — that dependency lives in the evaluator passed in at runtime.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/eval/__init__.py` | Re-export public symbols from the submodules |
| `src/eval/arena.py` | `play_match()` — engine-vs-engine, returns `MatchResult` |
| `src/eval/puzzles.py` | `load_puzzles()`, `puzzle_accuracy()` — Lichess CSV format |
| `src/eval/elo.py` | `elo_diff()`, `expected_score()`, `gauntlet_elo()` |
| `src/eval/stockfish_opponent.py` | `StockfishOpponent` wrapping `StockfishEvaluator`; availability guard |
| `scripts/eval.py` | CLI: arena vs checkpoint, vs Stockfish ladder, puzzles, Elo estimate |
| `tests/eval/test_elo.py` | Elo math tests (known score → known Elo within tolerance) |
| `tests/eval/test_arena.py` | Arena test with deterministic stub move-producers |
| `tests/eval/test_puzzles.py` | Puzzle accuracy test with inline fixture |
| `tests/eval/test_stockfish_opponent.py` | StockfishOpponent tests (skipped if binary absent) |
| `tests/eval/test_eval_cli.py` | CLI smoke tests (`--help`, model-free puzzle path) |
| `tests/eval/__init__.py` | Empty, marks test package |

---

### Task 1: Create `src/eval/__init__.py` and `tests/eval/__init__.py`

**Files:**
- Create: `src/eval/__init__.py`
- Create: `tests/eval/__init__.py`

- [ ] **Step 1: Create the `src/eval` package init.**

```python
# src/eval/__init__.py
"""Evaluation harness: arena, puzzles, Elo utilities, Stockfish opponent.

IMPORTANT: keep this file import-free until Task 6. Submodules are created across
Tasks 2-5; if this __init__ imported them now, an early `from src.eval.elo import
...` (Task 2's test) would trigger the package __init__ and fail on the not-yet-
existing `.stockfish_opponent`. Public symbols are wired up in Task 6.
"""
```

- [ ] **Step 2: Create the test package marker.**

```python
# tests/eval/__init__.py
```

- [ ] **Step 3: Commit scaffolding.**

```bash
rtk git add src/eval/__init__.py tests/eval/__init__.py && rtk git commit -m "eval: create src/eval and tests/eval packages"
```

---

### Task 2: Implement `src/eval/elo.py` (pure Elo math)

**Files:**
- Create: `src/eval/elo.py`
- Create: `tests/eval/test_elo.py`

- [ ] **Step 1: Write the failing test first.**

```python
# tests/eval/test_elo.py
"""Tests for Elo math utilities."""
import math
import pytest
from src.eval.elo import elo_diff, expected_score, gauntlet_elo


def test_expected_score_equal_elo():
    # Equal opponents -> expected score = 0.5
    assert expected_score(0) == pytest.approx(0.5, abs=1e-9)


def test_expected_score_positive_diff():
    # +200 Elo advantage -> expected score ≈ 0.7597
    result = expected_score(200)
    assert result == pytest.approx(0.7597, abs=0.001)


def test_expected_score_negative_diff():
    # Symmetric: score(-d) = 1 - score(d)
    assert expected_score(-200) == pytest.approx(1 - expected_score(200), abs=1e-9)


def test_elo_diff_known_score():
    # Score of 0.76 corresponds to roughly +200 Elo
    diff = elo_diff(0.76, games=1)
    assert diff == pytest.approx(198.0, abs=5.0)


def test_elo_diff_draw():
    # Score of 0.5 -> 0 Elo difference
    assert elo_diff(0.5, games=1) == pytest.approx(0.0, abs=1e-6)


def test_elo_diff_perfect_score_clamps():
    # Perfect score (1.0) should return large finite value, not inf/nan
    diff = elo_diff(1.0, games=10)
    assert math.isfinite(diff)
    assert diff > 600.0


def test_elo_diff_zero_score_clamps():
    # Zero score should return large negative finite value
    diff = elo_diff(0.0, games=10)
    assert math.isfinite(diff)
    assert diff < -600.0


def test_gauntlet_elo_single_opponent():
    # One opponent, known score -> consistent with elo_diff
    result = gauntlet_elo([(0.76, 100, 0)])
    expected = elo_diff(0.76, games=100)
    assert result == pytest.approx(expected, abs=1.0)


def test_gauntlet_elo_multiple_opponents():
    # Two opponents at different base Elos; result is weighted average
    result = gauntlet_elo([(0.76, 50, 0), (0.60, 50, 200)])
    assert isinstance(result, float)
    assert math.isfinite(result)
    assert 100.0 < result < 400.0
```

- [ ] **Step 2: Run the test and confirm it fails (module not found).**

```bash
.venv/bin/python -m pytest tests/eval/test_elo.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'src.eval.elo'` (or `src.eval`).

- [ ] **Step 3: Implement `src/eval/elo.py`.**

```python
# src/eval/elo.py
"""Elo rating utilities: difference from match score and gauntlet aggregation."""

import math
from typing import List, Tuple

# Logistic Elo model: E(d) = 1 / (1 + 10^(-d/400))
_LOG10 = math.log(10)
_EPSILON = 1e-6  # clamp to avoid log(0)
_CLAMP_DIFF = 800.0  # max returnable Elo difference


def expected_score(elo_diff: float) -> float:
    """Expected score for the engine when it has `elo_diff` Elo advantage.

    Args:
        elo_diff: Elo rating of engine minus Elo rating of opponent.

    Returns:
        Expected score in [0, 1] (0.5 = equal).
    """
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def elo_diff(score: float, games: int = 1) -> float:  # noqa: ARG001 (games kept for API clarity)
    """Compute Elo difference from observed match score using logistic model.

    Args:
        score: Observed score fraction (wins + 0.5*draws) / total_games, in [0, 1].
        games: Number of games played (unused in logistic formula but kept for
               future BayesElo / LOS extensions).

    Returns:
        Elo difference (engine Elo minus opponent Elo). Clamped to ±_CLAMP_DIFF
        at extreme scores to avoid infinity.
    """
    # Clamp score away from 0 and 1 to avoid log(0)
    s = max(_EPSILON, min(1.0 - _EPSILON, score))
    # Logistic inverse: d = 400 * log10(s / (1 - s))
    diff = 400.0 * math.log10(s / (1.0 - s))
    return max(-_CLAMP_DIFF, min(_CLAMP_DIFF, diff))


def gauntlet_elo(
    results: List[Tuple[float, int, float]],
) -> float:
    """Estimate engine Elo from gauntlet results against opponents at known Elos.

    Uses weighted average of per-opponent Elo estimates (weighted by game count).

    Args:
        results: List of (score, games, opponent_elo) tuples.
                 score is the fraction in [0,1]; games is total games played;
                 opponent_elo is the known Elo of that opponent (default 0 if
                 unknown, which returns relative Elo difference).

    Returns:
        Estimated engine Elo (or Elo difference from a base of 0 if opponent
        Elos are all 0).
    """
    if not results:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for score, games, opponent_elo in results:
        weight = float(games)
        diff = elo_diff(score, games=games)
        engine_elo = opponent_elo + diff
        weighted_sum += weight * engine_elo
        total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight
```

- [ ] **Step 4: Run the tests and confirm they all pass.**

```bash
.venv/bin/python -m pytest tests/eval/test_elo.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit.**

```bash
rtk git add src/eval/elo.py tests/eval/test_elo.py && rtk git commit -m "eval: add Elo math utilities with tests"
```

---

### Task 3: Implement `src/eval/arena.py` (engine-vs-engine match runner)

**Files:**
- Create: `src/eval/arena.py`
- Create: `tests/eval/test_arena.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/eval/test_arena.py
"""Tests for the arena match runner."""
import chess
import pytest
from src.eval.arena import play_match, MatchResult


# ---------------------------------------------------------------------------
# Deterministic stub move-producers
# ---------------------------------------------------------------------------

def _always_first_legal(board: chess.Board) -> chess.Move:
    """Always picks the first legal move (sorted by UCI for determinism)."""
    return sorted(board.legal_moves, key=lambda m: m.uci())[0]


def _always_last_legal(board: chess.Board) -> chess.Move:
    """Always picks the last legal move (sorted by UCI for determinism)."""
    return sorted(board.legal_moves, key=lambda m: m.uci())[-1]


def _resign_immediately(board: chess.Board) -> chess.Move:
    """Returns None to signal resignation."""
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_match_result_fields():
    r = MatchResult(wins=3, draws=1, losses=1, total=5)
    assert r.wins == 3
    assert r.draws == 1
    assert r.losses == 1
    assert r.total == 5


def test_match_result_score():
    r = MatchResult(wins=3, draws=2, losses=1, total=6)
    expected = (3 + 0.5 * 2) / 6
    assert r.score == pytest.approx(expected, abs=1e-9)


def test_match_result_score_zero_games():
    r = MatchResult(wins=0, draws=0, losses=0, total=0)
    assert r.score == pytest.approx(0.5, abs=1e-9)


def test_arena_runs_and_returns_result():
    """A short match between two deterministic bots returns a valid MatchResult."""
    result = play_match(
        engine=_always_first_legal,
        opponent=_always_last_legal,
        num_games=2,
        max_moves=50,
    )
    assert isinstance(result, MatchResult)
    assert result.total == 2
    assert result.wins + result.draws + result.losses == 2
    assert 0.0 <= result.score <= 1.0


def test_arena_alternates_colours():
    """With 2 games the engine plays White in game 0 and Black in game 1."""
    colours_seen: list = []

    def recording_engine(board: chess.Board) -> chess.Move:
        colours_seen.append(board.turn)  # True=White
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]

    play_match(
        engine=recording_engine,
        opponent=_always_last_legal,
        num_games=2,
        max_moves=2,
    )
    # Engine should have seen at least one White turn and one Black turn
    assert chess.WHITE in colours_seen
    assert chess.BLACK in colours_seen


def test_arena_max_moves_terminates():
    """max_moves prevents infinite games; result is counted as draw."""
    result = play_match(
        engine=_always_first_legal,
        opponent=_always_first_legal,
        num_games=4,
        max_moves=10,
    )
    assert result.total == 4


def test_arena_score_in_valid_range():
    result = play_match(
        engine=_always_first_legal,
        opponent=_always_last_legal,
        num_games=4,
        max_moves=30,
    )
    assert 0.0 <= result.score <= 1.0
```

- [ ] **Step 2: Run and confirm failure.**

```bash
.venv/bin/python -m pytest tests/eval/test_arena.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'src.eval.arena'`.

- [ ] **Step 3: Implement `src/eval/arena.py`.**

```python
# src/eval/arena.py
"""Arena: play a match between two move-producer callables and return a score."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import chess


# A move-producer is any callable that receives a chess.Board and returns a
# chess.Move (or None to resign).
MoveProducer = Callable[[chess.Board], Optional[chess.Move]]


@dataclasses.dataclass
class MatchResult:
    """Aggregated result of a multi-game match from the engine's perspective."""

    wins: int
    draws: int
    losses: int
    total: int

    @property
    def score(self) -> float:
        """Fraction score: (wins + 0.5*draws) / total. Returns 0.5 if total=0."""
        if self.total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.total


def _play_single_game(
    white: MoveProducer,
    black: MoveProducer,
    max_moves: int,
) -> float:
    """Play one game; return result from White's perspective: 1.0/0.5/0.0."""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        mover = white if board.turn == chess.WHITE else black
        move = mover(board)
        if move is None:
            # Resignation
            return 0.0 if board.turn == chess.WHITE else 1.0
        if move not in board.legal_moves:
            # Illegal move counts as resignation
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(move)
        move_count += 1

    if not board.is_game_over():
        # max_moves reached — count as draw
        return 0.5

    outcome = board.outcome()
    if outcome is None:
        return 0.5
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def play_match(
    engine: MoveProducer,
    opponent: MoveProducer,
    num_games: int = 10,
    max_moves: int = 300,
) -> MatchResult:
    """Play `num_games` games, alternating colours each game.

    The engine plays White in even-indexed games (0, 2, …) and Black in
    odd-indexed games (1, 3, …).

    Args:
        engine: The move-producer under evaluation.
        opponent: The move-producer it plays against.
        num_games: Total number of games to play.
        max_moves: Hard cap on plies per game; game is counted as draw if hit.

    Returns:
        MatchResult with wins/draws/losses from the engine's perspective.
    """
    wins = draws = losses = 0

    for game_idx in range(num_games):
        engine_is_white = (game_idx % 2 == 0)
        white = engine if engine_is_white else opponent
        black = opponent if engine_is_white else engine

        white_score = _play_single_game(white, black, max_moves)

        engine_score = white_score if engine_is_white else (1.0 - white_score)

        if engine_score == 1.0:
            wins += 1
        elif engine_score == 0.5:
            draws += 1
        else:
            losses += 1

    return MatchResult(wins=wins, draws=draws, losses=losses, total=num_games)
```

- [ ] **Step 4: Run the tests and confirm they all pass.**

```bash
.venv/bin/python -m pytest tests/eval/test_arena.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit.**

```bash
rtk git add src/eval/arena.py tests/eval/test_arena.py && rtk git commit -m "eval: add arena match runner with deterministic stub tests"
```

---

### Task 4: Implement `src/eval/puzzles.py` (Lichess puzzle accuracy)

**Files:**
- Create: `src/eval/puzzles.py`
- Create: `tests/eval/test_puzzles.py`

- [ ] **Step 1: Write the failing test with an inline fixture.**

```python
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
```

- [ ] **Step 2: Run and confirm failure.**

```bash
.venv/bin/python -m pytest tests/eval/test_puzzles.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'src.eval.puzzles'`.

- [ ] **Step 3: Implement `src/eval/puzzles.py`.**

```python
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
```

- [ ] **Step 4: Run the tests and confirm they all pass.**

```bash
.venv/bin/python -m pytest tests/eval/test_puzzles.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit.**

```bash
rtk git add src/eval/puzzles.py tests/eval/test_puzzles.py && rtk git commit -m "eval: add puzzle loader and accuracy measurement with inline fixture"
```

---

### Task 5: Implement `src/eval/stockfish_opponent.py` (gated Stockfish arena opponent)

**Files:**
- Create: `src/eval/stockfish_opponent.py`

- [ ] **Step 1: Write a test that gates on Stockfish availability.**

```python
# tests/eval/test_stockfish_opponent.py
"""Tests for StockfishOpponent — gated on binary availability."""
import pytest
import chess
from src.eval.stockfish_opponent import StockfishOpponent, stockfish_available


def test_stockfish_available_returns_bool():
    result = stockfish_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish binary not found")
def test_stockfish_opponent_returns_legal_move():
    opp = StockfishOpponent(skill_level=1, depth=1)
    try:
        board = chess.Board()
        move = opp(board)
        assert move in board.legal_moves
    finally:
        opp.close()


@pytest.mark.skipif(not stockfish_available(), reason="Stockfish binary not found")
def test_stockfish_opponent_context_manager():
    with StockfishOpponent(skill_level=5, depth=2) as opp:
        board = chess.Board()
        move = opp(board)
        assert move in board.legal_moves


def test_stockfish_opponent_unavailable_raises(monkeypatch):
    """When binary not found, constructing StockfishOpponent raises RuntimeError."""
    # Patch stockfish_available to False
    import src.eval.stockfish_opponent as module
    monkeypatch.setattr(module, "stockfish_available", lambda: False)
    with pytest.raises(RuntimeError, match="Stockfish"):
        StockfishOpponent(skill_level=1, depth=1)
```

- [ ] **Step 2: Run and confirm failure.**

```bash
.venv/bin/python -m pytest tests/eval/test_stockfish_opponent.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'src.eval.stockfish_opponent'`.

- [ ] **Step 3: Implement `src/eval/stockfish_opponent.py`.**

```python
# src/eval/stockfish_opponent.py
"""Stockfish opponent wrapper for arena matches.

Gates on Stockfish binary availability. Use `stockfish_available()` before
constructing `StockfishOpponent` in scripts; unit tests that need the binary
are skipped when it is absent.
"""

from __future__ import annotations

import shutil
import sys
import os
from typing import Optional

import chess

# Paths to check for the Stockfish binary (same list as StockfishEvaluator)
_CANDIDATE_PATHS = [
    "stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/opt/local/bin/stockfish",
]


def stockfish_available() -> bool:
    """Return True if a Stockfish binary is findable on this system."""
    for candidate in _CANDIDATE_PATHS:
        if os.path.exists(candidate):
            return True
        found = shutil.which(candidate)
        if found:
            return True
    return False


class StockfishOpponent:
    """Move-producer backed by Stockfish at a fixed skill level and depth.

    This is a callable `(chess.Board) -> chess.Move` suitable for use as the
    `opponent` argument of `play_match()`.

    Skill level maps to UCI_Elo via Stockfish's UCI_LimitStrength / UCI_Elo
    options (1=800 Elo ladder rung, 20=full strength).

    Args:
        skill_level: Stockfish skill level 0–20.
        depth: Search depth cap (passed as `time_limit`-equivalent via depth).
        stockfish_path: Override binary path; auto-detected if None.

    Raises:
        RuntimeError: If Stockfish binary cannot be found.
    """

    # Approximate Elo for each skill level 0-20 (rough ladder for Elo reporting)
    SKILL_TO_ELO = {
        0: 800, 1: 900, 2: 1000, 3: 1100, 4: 1200,
        5: 1300, 6: 1400, 7: 1500, 8: 1600, 9: 1700,
        10: 1800, 11: 1900, 12: 2000, 13: 2100, 14: 2200,
        15: 2300, 16: 2400, 17: 2500, 18: 2600, 19: 2700,
        20: 3200,
    }

    def __init__(
        self,
        skill_level: int = 5,
        depth: int = 5,
        stockfish_path: Optional[str] = None,
    ) -> None:
        if not stockfish_available():
            raise RuntimeError(
                "Stockfish binary not found. Install Stockfish and ensure it is "
                "on PATH, or pass stockfish_path explicitly."
            )
        # Import here to avoid hard dependency at module load time
        # (StockfishEvaluator raises RuntimeError in __init__ if binary absent,
        # which we have already checked above).
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        from src.training.stockfish_evaluator import StockfishEvaluator  # noqa: PLC0415

        self.skill_level = skill_level
        self.depth = depth
        self._evaluator = StockfishEvaluator(
            stockfish_path=stockfish_path,
            depth=depth,
            limit_strength=True,
            elo=self.SKILL_TO_ELO.get(skill_level, 1500),
        )
        # Verify the strength limit was actually applied; otherwise the ladder Elo
        # labels are meaningless (Stockfish would run at full strength silently).
        opts = self._evaluator.engine.options
        self.strength_limited = ("UCI_LimitStrength" in opts and "UCI_Elo" in opts)
        if not self.strength_limited:
            print(
                "warning: this Stockfish build lacks UCI_LimitStrength/UCI_Elo; "
                "arena Elo labels will be inaccurate (running near full strength).",
                file=sys.stderr,
            )

    @property
    def approximate_elo(self) -> float:
        """Approximate Elo for this skill level (for gauntlet_elo reporting)."""
        return float(self.SKILL_TO_ELO.get(self.skill_level, 1500))

    def __call__(self, board: chess.Board) -> Optional[chess.Move]:
        """Return Stockfish's best move for the given position."""
        return self._evaluator.get_best_move(board)

    def close(self) -> None:
        """Shut down the Stockfish engine process."""
        self._evaluator.close()

    def __enter__(self) -> "StockfishOpponent":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```

- [ ] **Step 4: Run the tests.**

```bash
.venv/bin/python -m pytest tests/eval/test_stockfish_opponent.py -v
```

Expected: the two `skipif` tests skip if Stockfish is absent; `test_stockfish_available_returns_bool` and `test_stockfish_opponent_unavailable_raises` always pass.

- [ ] **Step 5: Commit.**

```bash
rtk git add src/eval/stockfish_opponent.py tests/eval/test_stockfish_opponent.py && rtk git commit -m "eval: add StockfishOpponent with binary-availability guard and tests"
```

---

### Task 6: Update `src/eval/__init__.py` to expose `StockfishOpponent`

**Files:**
- Modify: `src/eval/__init__.py`

- [ ] **Step 1: Add `StockfishOpponent` and `stockfish_available` to the public API.**

```python
# src/eval/__init__.py
"""Evaluation harness: arena, puzzles, Elo utilities, Stockfish opponent."""

from .arena import play_match, MatchResult
from .elo import elo_diff, expected_score, gauntlet_elo
from .puzzles import load_puzzles, puzzle_accuracy, Puzzle
from .stockfish_opponent import StockfishOpponent, stockfish_available

__all__ = [
    "play_match",
    "MatchResult",
    "elo_diff",
    "expected_score",
    "gauntlet_elo",
    "load_puzzles",
    "puzzle_accuracy",
    "Puzzle",
    "StockfishOpponent",
    "stockfish_available",
]
```

- [ ] **Step 2: Verify existing tests still pass.**

```bash
.venv/bin/python -m pytest tests/eval/ -v
```

Expected: all tests pass (or skip for Stockfish-gated ones when binary absent).

- [ ] **Step 3: Commit.**

```bash
rtk git add src/eval/__init__.py && rtk git commit -m "eval: expose StockfishOpponent in package __init__"
```

---

### Task 7: Implement `scripts/eval.py` CLI

**Files:**
- Create: `scripts/eval.py`

- [ ] **Step 1: Write a smoke test for the CLI help flag (no model needed).**

```python
# tests/eval/test_eval_cli.py
"""Smoke test for the eval.py CLI — --help and a model-free puzzle path."""
import os
import subprocess
import sys

# Repo root derived from this test's location (portable; no hardcoded path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_eval_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/eval.py", "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "eval" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_eval_cli_puzzle_only_inline(tmp_path):
    """--puzzle-csv with inline file runs without a model by using random moves."""
    puzzle_csv = tmp_path / "puzzles.csv"
    puzzle_csv.write_text(
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
        "p1,r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 4,"
        "d8f6 h5f7,1500,100,95,1000,mate,https://lichess.org/abc,\n"
    )
    result = subprocess.run(
        [
            sys.executable, "scripts/eval.py",
            "--puzzle-csv", str(puzzle_csv),
            "--random-engine",
            "--no-arena",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    # Should succeed or fail with a clear message, not an unhandled exception
    assert result.returncode == 0 or "Error" in result.stdout or "error" in result.stderr
```

- [ ] **Step 2: Run and confirm failure.**

```bash
.venv/bin/python -m pytest tests/eval/test_eval_cli.py::test_eval_cli_help -v
```

Expected failure: `FileNotFoundError` or returncode != 0 since `scripts/eval.py` does not exist.

- [ ] **Step 3: Implement `scripts/eval.py`.**

```python
#!/usr/bin/env python
# scripts/eval.py
"""CLI: evaluate engine strength via arena matches, puzzles, and Elo estimation.

Usage examples
--------------
# Evaluate checkpoint/model_iter_5 vs Stockfish skill ladder (requires binary)
  python scripts/eval.py --model checkpoint/model_iter_5.weights.h5

# Puzzles only, using the bundled transformer evaluator
  python scripts/eval.py --model checkpoint/model_iter_5.weights.h5 \\
      --puzzle-csv data/puzzles.csv --no-arena

# Quick smoke test with a random engine (no model needed)
  python scripts/eval.py --random-engine --no-arena \\
      --puzzle-csv data/puzzles.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Optional

import chess

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.eval.arena import play_match, MatchResult
from src.eval.elo import elo_diff, gauntlet_elo
from src.eval.puzzles import load_puzzles, puzzle_accuracy
from src.eval.stockfish_opponent import StockfishOpponent, stockfish_available


# ---------------------------------------------------------------------------
# Move-producers
# ---------------------------------------------------------------------------

def _random_mover(board: chess.Board) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    return random.choice(legal) if legal else None


def _build_transformer_mover(model_path: str, num_simulations: int):
    """Build a move-producer backed by TransformerEvaluator + BatchedMCTS (Plan 4)."""
    # Lazy imports so the CLI can run --help without TensorFlow.
    from src.model.evaluator import TransformerEvaluator  # noqa: PLC0415
    from src.model.transformer import ChessTransformer  # noqa: PLC0415
    from src.mcts.batched_mcts import BatchedMCTS  # noqa: PLC0415
    from src.game.token_encoder import encode_batch  # noqa: PLC0415
    from config import Config  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    config = Config()
    net = ChessTransformer(config)
    # Build the network once (forward pass) before loading weights.
    sq, sf = encode_batch([chess.Board()], [0])
    net(tf.constant(sq), tf.constant(sf))
    net.load_weights(model_path)
    evaluator = TransformerEvaluator(net, use_fp16=True)
    mcts = BatchedMCTS(evaluator, config, num_simulations=num_simulations)

    def _mover(board: chess.Board) -> Optional[chess.Move]:
        mcts.reset()  # evaluate each position independently for arena/puzzles
        return mcts.choose_move(board, temperature=0.0)

    return _mover


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="Evaluate chess engine strength via arena, puzzles, and Elo.",
    )
    parser.add_argument(
        "--model",
        default="checkpoint/model_iter_5.weights.h5",
        help="Path to model weights (.weights.h5). Ignored if --random-engine.",
    )
    parser.add_argument(
        "--random-engine",
        action="store_true",
        help="Use a random mover instead of the neural network (for smoke tests).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations per move (default 50).",
    )
    parser.add_argument(
        "--arena-games",
        type=int,
        default=10,
        help="Games per Stockfish skill-level rung (default 10).",
    )
    parser.add_argument(
        "--skill-levels",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15],
        help="Stockfish skill levels to use as ladder opponents.",
    )
    parser.add_argument(
        "--arena-depth",
        type=int,
        default=5,
        help="Stockfish search depth for arena opponents (default 5).",
    )
    parser.add_argument(
        "--puzzle-csv",
        default=None,
        help="Path to Lichess-format puzzle CSV. Skip puzzles if not set.",
    )
    parser.add_argument(
        "--max-puzzles",
        type=int,
        default=200,
        help="Max puzzles to evaluate (default 200).",
    )
    parser.add_argument(
        "--no-arena",
        action="store_true",
        help="Skip arena matches (useful when Stockfish is absent).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Max moves per arena game before counting as draw (default 200).",
    )
    return parser.parse_args()


def _print_match_result(tag: str, result: MatchResult, opponent_elo: float) -> None:
    diff = elo_diff(result.score, games=result.total)
    est_elo = opponent_elo + diff
    print(
        f"  {tag}: W{result.wins}/D{result.draws}/L{result.losses} "
        f"score={result.score:.3f}  Elo diff vs opp({opponent_elo:.0f}) = "
        f"{diff:+.0f}  => est. engine Elo ~ {est_elo:.0f}"
    )


def main() -> int:
    args = _parse_args()

    # --- Build engine ---
    if args.random_engine:
        print("Using RANDOM ENGINE (smoke test mode).")
        engine = _random_mover
    else:
        print(f"Loading model from: {args.model}")
        try:
            engine = _build_transformer_mover(args.model, args.simulations)
        except Exception as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            return 1

    gauntlet_entries = []

    # --- Arena vs Stockfish ladder ---
    if args.no_arena:
        print("Skipping arena (--no-arena).")
    elif not stockfish_available():
        print("Stockfish binary not found — skipping arena. Install Stockfish to enable.")
    else:
        print(f"\n=== Arena vs Stockfish skill ladder ({args.arena_games} games/level) ===")
        for skill in args.skill_levels:
            with StockfishOpponent(skill_level=skill, depth=args.arena_depth) as opp:
                result = play_match(
                    engine=engine,
                    opponent=opp,
                    num_games=args.arena_games,
                    max_moves=args.max_moves,
                )
                _print_match_result(f"Skill {skill:2d}", result, opp.approximate_elo)
                gauntlet_entries.append((result.score, result.total, opp.approximate_elo))

        if gauntlet_entries:
            est = gauntlet_elo(gauntlet_entries)
            print(f"\n  Gauntlet Elo estimate: {est:.0f}")

    # --- Puzzles ---
    if args.puzzle_csv is None:
        print("\nNo --puzzle-csv provided; skipping puzzle evaluation.")
    else:
        print(f"\n=== Puzzle accuracy (max {args.max_puzzles}) ===")
        try:
            puzzles = load_puzzles(path=args.puzzle_csv, max_puzzles=args.max_puzzles)
            acc = puzzle_accuracy(engine=engine, puzzles=puzzles)
            print(f"  Puzzles loaded: {len(puzzles)}")
            print(f"  Top-1 accuracy: {acc:.3f} ({int(acc * len(puzzles))}/{len(puzzles)})")
        except FileNotFoundError:
            print(f"  Puzzle file not found: {args.puzzle_csv}", file=sys.stderr)
            return 1

    # --- Summary ---
    print("\n=== Done ===")
    if not gauntlet_entries and args.puzzle_csv is None:
        print("  Nothing evaluated. Pass --puzzle-csv and/or ensure Stockfish is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the CLI smoke tests.**

```bash
.venv/bin/python -m pytest tests/eval/test_eval_cli.py -v
```

Expected: both tests pass (help exits 0; puzzle-only smoke test exits cleanly).

- [ ] **Step 5: Commit.**

```bash
rtk git add scripts/eval.py tests/eval/test_eval_cli.py && rtk git commit -m "eval: add scripts/eval.py CLI with arena, puzzle, and Elo subcommands"
```

---

### Task 8: Full integration test pass and final commit

**Files:**
- Test only: `tests/eval/`

- [ ] **Step 1: Run the entire eval test suite.**

```bash
.venv/bin/python -m pytest tests/eval/ -v
```

Expected: all non-Stockfish tests pass; Stockfish-gated tests show `SKIPPED` or `PASSED` depending on binary availability.

- [ ] **Step 2: Run a quick end-to-end CLI smoke test (random engine, no arena, no model).**

```bash
.venv/bin/python scripts/eval.py --random-engine --no-arena
```

Expected output includes `Skipping arena` and `No --puzzle-csv provided`.

- [ ] **Step 3: Verify the Elo math is self-consistent via a quick sanity check.**

```bash
.venv/bin/python -c "
from src.eval.elo import expected_score, elo_diff, gauntlet_elo
# Round-trip: expected_score -> elo_diff should recover original diff
for diff in [-400, -200, 0, 200, 400]:
    s = expected_score(diff)
    recovered = elo_diff(s)
    assert abs(recovered - diff) < 0.01, f'round-trip failed for diff={diff}: got {recovered}'
print('Elo round-trip sanity check: PASSED')
# Gauntlet test
est = gauntlet_elo([(0.76, 100, 1500), (0.60, 100, 1700)])
print(f'Gauntlet estimate (0.76 vs 1500, 0.60 vs 1700) => {est:.0f} Elo')
"
```

Expected output: `Elo round-trip sanity check: PASSED` followed by a plausible Elo estimate (roughly 1680-1720).

- [ ] **Step 4: Run the full test suite to confirm nothing is broken.**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: all tests from Plans 1–4 continue to pass; eval tests pass.

- [ ] **Step 5: Final commit.**

```bash
rtk git add tests/eval/ && rtk git commit -m "eval: final integration pass — all eval tests green"
```

---

## Summary of Module Contracts Consumed by This Plan

| Symbol | Defined In | How Used |
|--------|-----------|----------|
| `chess.Board`, `chess.Move` | python-chess | Throughout arena, puzzles, opponent |
| `StockfishEvaluator(stockfish_path, depth, time_limit, multipv, limit_strength, elo, skill_level)` | `src/training/stockfish_evaluator.py` | Wrapped by `StockfishOpponent` (uses `stockfish_path`, `depth`, `limit_strength=True`, `elo`) |
| `StockfishEvaluator.get_best_move(board)` | `src/training/stockfish_evaluator.py` | Called by `StockfishOpponent.__call__` |
| `StockfishEvaluator.close()` / `.engine.options` | `src/training/stockfish_evaluator.py` | Shutdown; strength-limit verification |
| `TransformerEvaluator(net, use_fp16)` | `src/model/evaluator.py` (Plan 4) | Built in `scripts/eval.py` |
| `ChessTransformer(config)` + `encode_batch` | Plan 1 | Built/loaded in `_build_transformer_mover` |
| `BatchedMCTS(evaluator, config, num_simulations).choose_move(board, temperature)` | `src/mcts/batched_mcts.py` (Plan 4) | The move-producer in `scripts/eval.py` |

## Resolved Interfaces (verified against Plans 1 & 4)

The drafted version of this plan predated Plan 4; the following are now resolved:

1. **`StockfishEvaluator` signature is confirmed** as `StockfishEvaluator(stockfish_path, depth, time_limit, multipv, limit_strength, elo, skill_level)`. `StockfishOpponent` passes `stockfish_path`, `depth`, `limit_strength=True`, `elo`, and verifies `UCI_LimitStrength`/`UCI_Elo` are present (warning otherwise) so ladder Elo labels are trustworthy.
2. **The move-producer is `BatchedMCTS.choose_move(board, temperature=0.0) -> chess.Move`** (Plan 4), not the legacy `MCTS.search(game)`. `_build_transformer_mover` builds the network, loads weights, wraps it in `TransformerEvaluator`, and returns a closure calling `mcts.reset()` + `choose_move`. No `move_encoder.decode` is needed — `choose_move` returns a real `chess.Move`.
3. **`src/eval/__init__.py` is import-free until Task 6**, so importing `src.eval.elo`/`arena`/`puzzles` in Tasks 2–4 never triggers a not-yet-existing submodule import.
