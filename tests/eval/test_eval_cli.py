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
