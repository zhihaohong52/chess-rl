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
