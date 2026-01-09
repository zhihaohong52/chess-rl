"""Full training pipeline: Supervised → Curriculum → Self-Play."""

import numpy as np
from typing import Optional, List
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.model.network import ChessNetwork
from src.training.supervised_trainer import SupervisedTrainer
from src.training.curriculum_trainer import CurriculumTrainer
from src.training.trainer import Trainer


class FullTrainingPipeline:
    """Complete training pipeline combining all three phases.

    Phase 1: Supervised Learning
        - Learn from Stockfish moves and evaluations
        - Quick bootstrap of basic chess knowledge

    Phase 2: Curriculum Learning
        - Play against Stockfish at increasing depths
        - Learn to beat progressively stronger opponents

    Phase 3: Self-Play
        - Refine strategies through self-play
        - Develop unique playing style
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_dir: str = "checkpoints",
        stockfish_path: Optional[str] = None,
    ):
        """Initialize the training pipeline.

        Args:
            config: Configuration object.
            checkpoint_dir: Directory for saving checkpoints.
            stockfish_path: Path to Stockfish executable.
        """
        self.config = config or Config()
        self.checkpoint_dir = checkpoint_dir
        self.stockfish_path = stockfish_path

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize network
        self.network = ChessNetwork(self.config)
        self.network.compile()

        print(f"Initialized network with {self.network.trainable_params:,} parameters")

    def phase1_supervised(
        self,
        num_iterations: int = 20,
        batch_size: int = 256,
        batches_per_iteration: int = 100,
        stockfish_depth: int = 10,
    ) -> List[dict]:
        """Phase 1: Supervised learning from Stockfish.

        Args:
            num_iterations: Number of supervised iterations.
            batch_size: Batch size for training.
            batches_per_iteration: Batches per iteration.
            stockfish_depth: Stockfish depth for generating training data.

        Returns:
            Training history.
        """
        print("\n" + "="*60)
        print("PHASE 1: SUPERVISED LEARNING")
        print("Learning from Stockfish moves and evaluations")
        print("="*60)

        with SupervisedTrainer(
            network=self.network,
            stockfish_path=self.stockfish_path,
            stockfish_depth=stockfish_depth,
            config=self.config,
        ) as trainer:
            history = trainer.train(
                num_iterations=num_iterations,
                batch_size=batch_size,
                batches_per_iteration=batches_per_iteration,
                show_progress=True,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_interval=5,
            )

        # Save phase 1 checkpoint
        self.network.save(os.path.join(self.checkpoint_dir, "phase1_supervised"))
        print("\nPhase 1 complete! Checkpoint saved.")

        return history

    def phase2_curriculum(
        self,
        num_iterations: int = 50,
        games_per_iteration: Optional[int] = None,
        training_steps: Optional[int] = None,
        initial_depth: int = 1,
        max_depth: int = 6,
        promotion_threshold: float = 0.55,
        num_simulations: int = 100,
    ) -> List[dict]:
        """Phase 2: Curriculum learning against Stockfish.

        Args:
            num_iterations: Number of curriculum iterations.
            games_per_iteration: Games per iteration.
            training_steps: Training steps per iteration.
            initial_depth: Starting Stockfish depth.
            max_depth: Maximum Stockfish depth.
            promotion_threshold: Win rate to advance depth.
            num_simulations: MCTS simulations per move.

        Returns:
            Training history.
        """
        print("\n" + "="*60)
        print("PHASE 2: CURRICULUM LEARNING")
        print(f"Playing against Stockfish (depth {initial_depth} → {max_depth})")
        print("="*60)

        with CurriculumTrainer(
            network=self.network,
            config=self.config,
            stockfish_path=self.stockfish_path,
            initial_depth=initial_depth,
            max_depth=max_depth,
            promotion_threshold=promotion_threshold,
            num_simulations=num_simulations,
        ) as trainer:
            history = trainer.train(
                num_iterations=num_iterations,
                games_per_iteration=games_per_iteration or self.config.curriculum_games_per_iteration,
                training_steps=training_steps,
                show_progress=True,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_interval=10,
            )

        # Save phase 2 checkpoint
        final_depth = history[-1]["depth"] if history else initial_depth
        self.network.save(os.path.join(self.checkpoint_dir, f"phase2_curriculum_depth{final_depth}"))
        print(f"\nPhase 2 complete! Reached Stockfish depth {final_depth}")

        return history

    def phase3_self_play(
        self,
        num_iterations: int = 50,
        games_per_iteration: int = 64,
        training_steps: int = 200,
        num_simulations: int = 200,
        num_parallel: int = 16,
    ) -> List[dict]:
        """Phase 3: Self-play refinement.

        Args:
            num_iterations: Number of self-play iterations.
            games_per_iteration: Games per iteration.
            training_steps: Training steps per iteration.
            num_simulations: MCTS simulations per move.
            num_parallel: Number of parallel games.

        Returns:
            Training history.
        """
        print("\n" + "="*60)
        print("PHASE 3: SELF-PLAY REFINEMENT")
        print("Developing unique strategies through self-play")
        print("="*60)

        # Update config for self-play
        self.config.num_simulations = num_simulations
        self.config.games_per_iteration = games_per_iteration
        self.config.training_steps = training_steps
        self.config.warmup_games = games_per_iteration
        self.config.main_games = games_per_iteration

        # Create trainer with existing network
        trainer = Trainer(
            config=self.config,
            checkpoint_dir=self.checkpoint_dir,
            num_parallel=num_parallel,
            use_parallel=True,
        )

        # Replace trainer's network with our trained one
        trainer.network = self.network

        # Run self-play
        trainer.train(num_iterations=num_iterations, show_progress=True)

        # Save final checkpoint
        self.network.save(os.path.join(self.checkpoint_dir, "phase3_final"))
        print("\nPhase 3 complete! Final model saved.")

        return trainer.training_history

    def train_full_pipeline(
        self,
        # Phase 1 params
        supervised_iterations: int = 20,
        supervised_batches: int = 100,
        # Phase 2 params
        curriculum_iterations: int = 50,
        curriculum_games: Optional[int] = None,
        curriculum_max_depth: int = 6,
        # Phase 3 params
        selfplay_iterations: int = 50,
        selfplay_games: int = 64,
        selfplay_simulations: int = 200,
    ) -> dict:
        """Run the complete training pipeline.

        Returns:
            Dictionary with history from all phases.
        """
        print("\n" + "#"*60)
        print("# FULL TRAINING PIPELINE")
        print("# Phase 1: Supervised → Phase 2: Curriculum → Phase 3: Self-Play")
        print("#"*60)

        results = {}

        # Phase 1
        results["phase1"] = self.phase1_supervised(
            num_iterations=supervised_iterations,
            batches_per_iteration=supervised_batches,
        )

        # Phase 2
        results["phase2"] = self.phase2_curriculum(
            num_iterations=curriculum_iterations,
            games_per_iteration=curriculum_games,
            max_depth=curriculum_max_depth,
        )

        # Phase 3
        results["phase3"] = self.phase3_self_play(
            num_iterations=selfplay_iterations,
            games_per_iteration=selfplay_games,
            num_simulations=selfplay_simulations,
        )

        # Save final model
        self.network.save(os.path.join(self.checkpoint_dir, "model_final"))
        self.network.save_full_model(os.path.join(self.checkpoint_dir, "model_final.keras"))

        print("\n" + "#"*60)
        print("# TRAINING COMPLETE!")
        print(f"# Final model saved to: {self.checkpoint_dir}/model_final")
        print("#"*60)

        return results


def run_full_training(
    checkpoint_dir: str = "/content/drive/MyDrive/chess-rl/checkpoints",
    stockfish_path: Optional[str] = None,
):
    """Convenience function to run full training pipeline.

    Args:
        checkpoint_dir: Directory for checkpoints.
        stockfish_path: Path to Stockfish (None = auto-detect).

    Returns:
        Trained pipeline instance.
    """
    pipeline = FullTrainingPipeline(
        checkpoint_dir=checkpoint_dir,
        stockfish_path=stockfish_path,
    )

    pipeline.train_full_pipeline(
        # Phase 1: ~1-2 hours
        supervised_iterations=20,
        supervised_batches=100,
        # Phase 2: ~2-4 hours
        curriculum_iterations=50,
        curriculum_games=20,
        curriculum_max_depth=6,
        # Phase 3: ~2-4 hours
        selfplay_iterations=50,
        selfplay_games=64,
        selfplay_simulations=200,
    )

    return pipeline
