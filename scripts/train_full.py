#!/usr/bin/env python3
"""Full training pipeline script for local machine.

Usage:
    # Run full pipeline with defaults
    python scripts/train_full.py

    # Custom checkpoint directory
    python scripts/train_full.py --checkpoint-dir ./my_checkpoints

    # Specify Stockfish path
    python scripts/train_full.py --stockfish /usr/local/bin/stockfish

    # Run only specific phases
    python scripts/train_full.py --phase 1        # Supervised only
    python scripts/train_full.py --phase 2        # Curriculum only
    python scripts/train_full.py --phase 3        # Self-play only
    python scripts/train_full.py --phase 1,2      # Supervised + Curriculum

    # Quick test run
    python scripts/train_full.py --quick

    # Resume from checkpoint
    python scripts/train_full.py --resume phase2_curriculum_depth3
"""

import argparse
import os
import sys
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.training.full_pipeline import FullTrainingPipeline


def find_stockfish() -> str:
    """Auto-detect Stockfish executable."""
    # Common paths to check
    paths = [
        # Linux
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        # macOS (Homebrew)
        "/opt/homebrew/bin/stockfish",
        "/usr/local/Cellar/stockfish/*/bin/stockfish",
        # Windows
        "C:/Program Files/Stockfish/stockfish.exe",
        "C:/Program Files (x86)/Stockfish/stockfish.exe",
        # Current directory
        "./stockfish",
        "./stockfish.exe",
    ]

    # Check PATH first
    stockfish_in_path = shutil.which("stockfish")
    if stockfish_in_path:
        return stockfish_in_path

    # Check common paths
    for path in paths:
        if "*" in path:
            # Handle glob pattern
            import glob
            matches = glob.glob(path)
            if matches:
                return matches[0]
        elif os.path.isfile(path):
            return path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Train Chess RL model with 3-phase pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_full.py                    # Full training
  python scripts/train_full.py --quick            # Quick test run
  python scripts/train_full.py --phase 1,2       # Supervised + Curriculum only
  python scripts/train_full.py --resume model    # Resume from checkpoint
        """
    )

    # Model size presets
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default=None,
        help="Model size preset: small (~600K), medium (~2.5M), large (~10M params)",
    )

    # General options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish executable (auto-detected if not specified)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1,2,3",
        help="Phases to run: 1=supervised, 2=curriculum, 3=self-play (default: 1,2,3)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint name (e.g., 'phase2_curriculum_depth3')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with minimal iterations",
    )

    # Phase 1: Supervised Learning
    parser.add_argument(
        "--p1-iterations",
        type=int,
        default=50,
        help="Phase 1: Number of supervised iterations (default: 50)",
    )
    parser.add_argument(
        "--p1-batches",
        type=int,
        default=100,
        help="Phase 1: Batches per iteration (default: 100)",
    )
    parser.add_argument(
        "--p1-batch-size",
        type=int,
        default=256,
        help="Phase 1: Batch size (default: 256)",
    )
    parser.add_argument(
        "--p1-depth",
        type=int,
        default=8,
        help="Phase 1: Stockfish depth for training data (default: 8)",
    )
    parser.add_argument(
        "--p1-cache-dir",
        type=str,
        default=None,
        help="Phase 1: Supervised cache directory (default: config)",
    )
    parser.add_argument(
        "--p1-cache-size",
        type=int,
        default=None,
        help="Phase 1: Number of cached supervised examples (default: config)",
    )
    parser.add_argument(
        "--p1-cache-dtype",
        type=str,
        choices=["float16", "float32"],
        default=None,
        help="Phase 1: Cache dtype for states/policies (default: config)",
    )
    parser.add_argument(
        "--p1-no-cache",
        action="store_true",
        help="Phase 1: Disable supervised cache",
    )
    parser.add_argument(
        "--p1-rebuild-cache",
        action="store_true",
        help="Phase 1: Rebuild supervised cache even if present",
    )

    # Phase 2: Curriculum Learning
    parser.add_argument(
        "--p2-iterations",
        type=int,
        default=1000,
        help="Phase 2: Max iterations, will early stop when mastered (default: 1000)",
    )
    parser.add_argument(
        "--p2-no-early-stop",
        action="store_true",
        help="Phase 2: Disable early stopping, run all iterations",
    )
    parser.add_argument(
        "--p2-games",
        type=int,
        default=40,
        help="Phase 2: Games per iteration (default: 40)",
    )
    parser.add_argument(
        "--p2-training-steps",
        type=int,
        default=200,
        help="Phase 2: Training steps per iteration (default: 200)",
    )
    parser.add_argument(
        "--p2-max-depth",
        type=int,
        default=12,
        help="Phase 2: Maximum Stockfish depth (default: 12, ~2500 Elo)",
    )
    parser.add_argument(
        "--p2-simulations",
        type=int,
        default=150,
        help="Phase 2: MCTS simulations per move (default: 150)",
    )
    parser.add_argument(
        "--p2-elo",
        type=int,
        default=None,
        help="Phase 2: Limit Stockfish strength with UCI_Elo (overrides config)",
    )
    parser.add_argument(
        "--p2-skill-level",
        type=int,
        default=None,
        help="Phase 2: Stockfish Skill Level (overrides config)",
    )
    parser.add_argument(
        "--p2-policy-weight",
        type=float,
        default=None,
        help="Phase 2: Blend weight for Stockfish policy targets (0-1)",
    )
    parser.add_argument(
        "--p2-value-weight",
        type=float,
        default=None,
        help="Phase 2: Blend weight for Stockfish value targets (0-1)",
    )
    parser.add_argument(
        "--p2-parallel",
        type=int,
        default=None,
        help="Phase 2: Number of parallel games for batched inference (default: 4)",
    )

    # Phase 3: Self-Play
    parser.add_argument(
        "--p3-iterations",
        type=int,
        default=50,
        help="Phase 3: Number of self-play iterations (default: 50)",
    )
    parser.add_argument(
        "--p3-games",
        type=int,
        default=64,
        help="Phase 3: Games per iteration (default: 64)",
    )
    parser.add_argument(
        "--p3-simulations",
        type=int,
        default=200,
        help="Phase 3: MCTS simulations per move (default: 200)",
    )
    parser.add_argument(
        "--p3-parallel",
        type=int,
        default=16,
        help="Phase 3: Number of parallel games (default: 16)",
    )

    args = parser.parse_args()

    # Parse phases
    phases = set(int(p.strip()) for p in args.phase.split(",") if p.strip())

    # Quick mode overrides
    if args.quick:
        print("Quick test mode - using minimal iterations")
        args.p1_iterations = 2
        args.p1_batches = 10
        args.p2_iterations = 3
        args.p2_games = 5
        args.p2_training_steps = 20
        args.p3_iterations = 3
        args.p3_games = 8
        if args.p1_cache_size is None:
            args.p1_cache_size = 512

    # Find Stockfish
    stockfish_path = args.stockfish or find_stockfish()

    if stockfish_path is None and (1 in phases or 2 in phases):
        print("ERROR: Stockfish not found!")
        print("Please install Stockfish and either:")
        print("  1. Add it to your PATH")
        print("  2. Use --stockfish /path/to/stockfish")
        print("\nInstallation:")
        print("  macOS:   brew install stockfish")
        print("  Ubuntu:  sudo apt install stockfish")
        print("  Windows: Download from https://stockfishchess.org/download/")
        sys.exit(1)

    print("=" * 60)
    print("Chess RL - Full Training Pipeline")
    print("=" * 60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if stockfish_path:
        print(f"Stockfish path: {stockfish_path}")
    print(f"Phases to run: {sorted(phases)}")
    print()

    # Initialize config
    config = Config()

    # Apply model size preset
    if args.model_size == "small":
        config.residual_blocks = 4
        config.residual_filters = 64
        config.value_hidden = 64
        config.policy_channels = 2   # Minimal policy head
        config.value_channels = 1    # Minimal value head
        config.use_se_blocks = False
        print("Using SMALL model (~600K params)")
    elif args.model_size == "medium":
        config.residual_blocks = 8
        config.residual_filters = 128
        config.value_hidden = 128
        config.policy_channels = 8
        config.value_channels = 8
        config.use_se_blocks = True
        print("Using MEDIUM model (~2.5M params)")
    elif args.model_size == "large":
        config.residual_blocks = 12
        config.residual_filters = 192
        config.value_hidden = 256
        config.policy_channels = 32
        config.value_channels = 32
        config.use_se_blocks = True
        print("Using LARGE model (~10M params)")

    # Apply config overrides
    if args.p1_no_cache:
        config.supervised_cache_enabled = False
    if args.p1_cache_dir is not None:
        config.supervised_cache_dir = args.p1_cache_dir
    if args.p1_cache_size is not None:
        config.supervised_cache_size = args.p1_cache_size
    if args.p1_cache_dtype is not None:
        config.supervised_cache_dtype = args.p1_cache_dtype
    if args.p1_rebuild_cache:
        config.supervised_cache_rebuild = True

    # Initialize pipeline
    pipeline = FullTrainingPipeline(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        stockfish_path=stockfish_path,
    )

    # Phase 2 overrides
    if args.p2_elo is not None or args.p2_skill_level is not None:
        config.curriculum_limit_strength = True
    if args.p2_elo is not None:
        config.curriculum_elo = args.p2_elo
    if args.p2_skill_level is not None:
        config.curriculum_skill_level = args.p2_skill_level
    if args.p2_policy_weight is not None:
        config.curriculum_policy_weight = args.p2_policy_weight
    if args.p2_value_weight is not None:
        config.curriculum_value_weight = args.p2_value_weight
    if args.p2_parallel is not None:
        config.curriculum_num_parallel = args.p2_parallel

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        pipeline.network.load(checkpoint_path)

    results = {}

    # Phase 1: Supervised Learning
    if 1 in phases:
        print("\n" + "=" * 60)
        print("Starting Phase 1: Supervised Learning")
        print("=" * 60)
        results["phase1"] = pipeline.phase1_supervised(
            num_iterations=args.p1_iterations,
            batch_size=args.p1_batch_size,
            batches_per_iteration=args.p1_batches,
            stockfish_depth=args.p1_depth,
        )

    # Phase 2: Curriculum Learning
    if 2 in phases:
        print("\n" + "=" * 60)
        print("Starting Phase 2: Curriculum Learning")
        print("=" * 60)
        results["phase2"] = pipeline.phase2_curriculum(
            num_iterations=args.p2_iterations,
            games_per_iteration=args.p2_games,
            training_steps=args.p2_training_steps,
            max_depth=args.p2_max_depth,
            num_simulations=args.p2_simulations,
            early_stop=not args.p2_no_early_stop,
        )

    # Phase 3: Self-Play
    if 3 in phases:
        print("\n" + "=" * 60)
        print("Starting Phase 3: Self-Play Refinement")
        print("=" * 60)
        results["phase3"] = pipeline.phase3_self_play(
            num_iterations=args.p3_iterations,
            games_per_iteration=args.p3_games,
            num_simulations=args.p3_simulations,
            num_parallel=args.p3_parallel,
        )

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "model_final")
    pipeline.network.save(final_path)

    keras_path = os.path.join(args.checkpoint_dir, "model_final.keras")
    pipeline.network.save_full_model(keras_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final model saved to: {final_path}.weights.h5")
    print(f"Keras model saved to: {keras_path}")
    print()
    print("To play against the model:")
    print(f"  python scripts/play.py --model {final_path}")
    print()
    print("To use with UCI (Arena, etc.):")
    print(f"  python uci.py --model {final_path}")


if __name__ == "__main__":
    main()
