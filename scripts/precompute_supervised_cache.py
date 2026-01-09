#!/usr/bin/env python3
"""Precompute supervised training cache without starting training."""

import argparse
import os
import sys
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.model.network import ChessNetwork
from src.training.supervised_trainer import SupervisedTrainer


def find_stockfish() -> str:
    """Auto-detect Stockfish executable."""
    paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/usr/local/Cellar/stockfish/*/bin/stockfish",
        "C:/Program Files/Stockfish/stockfish.exe",
        "C:/Program Files (x86)/Stockfish/stockfish.exe",
        "./stockfish",
        "./stockfish.exe",
    ]

    stockfish_in_path = shutil.which("stockfish")
    if stockfish_in_path:
        return stockfish_in_path

    for path in paths:
        if "*" in path:
            import glob
            matches = glob.glob(path)
            if matches:
                return matches[0]
        elif os.path.isfile(path):
            return path

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute supervised cache for Chess RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/precompute_supervised_cache.py
  python scripts/precompute_supervised_cache.py --cache-size 50000 --cache-dtype float16
  python scripts/precompute_supervised_cache.py --stockfish /path/to/stockfish --depth 8
        """,
    )

    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish executable (auto-detected if not specified)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Stockfish depth for supervised data (default: config)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: config)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=None,
        help="Number of cached supervised examples (default: config)",
    )
    parser.add_argument(
        "--cache-dtype",
        type=str,
        choices=["float16", "float32"],
        default=None,
        help="Cache dtype for states/policies (default: config)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild cache even if present",
    )
    parser.add_argument(
        "--multipv",
        type=int,
        default=None,
        help="Stockfish MultiPV for policy targets (default: config)",
    )
    parser.add_argument(
        "--position-source",
        type=str,
        choices=["stockfish", "random"],
        default=None,
        help="Position sampling source (default: config)",
    )
    parser.add_argument(
        "--position-max-moves",
        type=int,
        default=None,
        help="Maximum plies for position sampling (default: config)",
    )
    parser.add_argument(
        "--position-temperature",
        type=float,
        default=None,
        help="Sampling temperature for position generation (default: config)",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=None,
        help="Softmax temperature for Stockfish policy labels (default: config)",
    )

    args = parser.parse_args()

    config = Config()
    config.supervised_cache_enabled = True
    if args.cache_dir is not None:
        config.supervised_cache_dir = args.cache_dir
    if args.cache_size is not None:
        config.supervised_cache_size = args.cache_size
    if args.cache_dtype is not None:
        config.supervised_cache_dtype = args.cache_dtype
    if args.rebuild:
        config.supervised_cache_rebuild = True
    if args.multipv is not None:
        config.supervised_multipv = args.multipv
    if args.position_source is not None:
        config.supervised_position_source = args.position_source
    if args.position_max_moves is not None:
        config.supervised_position_max_moves = args.position_max_moves
    if args.position_temperature is not None:
        config.supervised_position_temperature = args.position_temperature
    if args.policy_temperature is not None:
        config.supervised_policy_temperature = args.policy_temperature

    stockfish_path = args.stockfish or find_stockfish()
    if stockfish_path is None:
        print("ERROR: Stockfish not found!")
        print("Please install Stockfish and either:")
        print("  1. Add it to your PATH")
        print("  2. Use --stockfish /path/to/stockfish")
        return 1

    depth = args.depth if args.depth is not None else 10

    print("=" * 60)
    print("Chess RL - Supervised Cache Precompute")
    print("=" * 60)
    print(f"Cache dir: {config.supervised_cache_dir}")
    print(f"Cache size: {config.supervised_cache_size}")
    print(f"Cache dtype: {config.supervised_cache_dtype}")
    print(f"Stockfish path: {stockfish_path}")
    print(f"Stockfish depth: {depth}")
    print()

    network = ChessNetwork(config)
    trainer = SupervisedTrainer(
        network=network,
        stockfish_path=stockfish_path,
        stockfish_depth=depth,
        config=config,
    )
    trainer.close()

    print(f"Cache ready: {config.supervised_cache_dir} ({trainer.cache_count} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
