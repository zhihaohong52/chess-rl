#!/usr/bin/env python3
"""Local training script for Chess RL."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Chess RL model")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--games", type=int, default=None, help="Games per iteration (override config)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="MCTS simulations per move (override config)",
    )
    args = parser.parse_args()

    # Configure
    config = Config()
    if args.games:
        config.games_per_iteration = args.games
    if args.simulations:
        config.num_simulations = args.simulations

    # Initialize trainer
    trainer = Trainer(config, args.checkpoint_dir)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint()

    # Train
    trainer.train(args.iterations)


if __name__ == "__main__":
    main()
