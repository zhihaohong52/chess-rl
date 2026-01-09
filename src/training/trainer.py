"""Training orchestration for AlphaZero-style learning."""

import numpy as np
from typing import Optional
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.model.network import ChessNetwork
from src.training.self_play import SelfPlay
from src.training.parallel_self_play import ParallelSelfPlay
from src.training.replay_buffer import ReplayBuffer
from src.game.move_encoder import get_move_encoder


class Trainer:
    """Orchestrates AlphaZero-style training."""

    def __init__(self, config: Optional[Config] = None, checkpoint_dir: Optional[str] = None,
                 num_parallel: int = 8, use_parallel: bool = True):
        """Initialize the trainer.

        Args:
            config: Configuration object.
            checkpoint_dir: Directory for saving checkpoints.
            num_parallel: Number of games to play in parallel.
            use_parallel: Whether to use parallel self-play (recommended for GPU).
        """
        self.config = config or Config()
        self.checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        self.num_parallel = num_parallel
        self.use_parallel = use_parallel

        # Initialize components
        self.network = ChessNetwork(self.config)
        self.network.compile()
        self.move_encoder = get_move_encoder()
        self.replay_buffer = ReplayBuffer(
            max_size=self.config.buffer_size,
            state_shape=self.config.input_shape,
            policy_size=self.network.policy_size
        )

        # Training stats
        self.iteration = 0
        self.total_games = 0
        self.training_history = []

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_training_params(self, iteration: int) -> tuple:
        """Get training parameters based on current iteration.

        Progressive training schedule:
        - Warmup: fewer simulations and games
        - Main: standard training
        - Refinement: more simulations

        Args:
            iteration: Current iteration number.

        Returns:
            Tuple of (num_simulations, num_games).
        """
        if iteration < self.config.warmup_iterations:
            return self.config.warmup_simulations, self.config.warmup_games
        elif iteration < 50:
            return self.config.main_simulations, self.config.main_games
        else:
            return self.config.refinement_simulations, self.config.main_games

    def run_iteration(self, show_progress: bool = True) -> dict:
        """Run a single training iteration.

        Each iteration consists of:
        1. Generate self-play games
        2. Add examples to replay buffer
        3. Train network on replay buffer

        Args:
            show_progress: Whether to show progress bars.

        Returns:
            Dictionary with iteration statistics.
        """
        self.iteration += 1
        num_sims, num_games = self.get_training_params(self.iteration)

        stats = {
            "iteration": self.iteration,
            "num_simulations": num_sims,
            "num_games": num_games,
        }

        # Generate self-play games
        if show_progress:
            mode = "parallel" if self.use_parallel else "sequential"
            print(f"\nIteration {self.iteration}: Generating {num_games} self-play games ({mode})...")

        if self.use_parallel:
            self_play = ParallelSelfPlay(
                self.network, self.config,
                num_parallel=self.num_parallel,
                num_simulations=num_sims
            )
        else:
            self_play = SelfPlay(self.network, self.config, num_simulations=num_sims)

        examples = self_play.generate_games(num_games, show_progress=show_progress)
        self.replay_buffer.add(examples)
        self.total_games += num_games

        stats["num_examples"] = len(examples)
        stats["buffer_size"] = len(self.replay_buffer)

        # Train on replay buffer
        if len(self.replay_buffer) >= self.config.batch_size:
            if show_progress:
                print(f"Training for {self.config.training_steps} steps...")

            losses = self._train_network(show_progress)
            stats["avg_total_loss"] = np.mean([l["total_loss"] for l in losses])
            stats["avg_policy_loss"] = np.mean([l["policy_loss"] for l in losses])
            stats["avg_value_loss"] = np.mean([l["value_loss"] for l in losses])

        # Save checkpoint
        if self.iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint()
            stats["checkpoint_saved"] = True

        self.training_history.append(stats)
        return stats

    def _train_network(self, show_progress: bool = True) -> list:
        """Train the network on replay buffer samples.

        Args:
            show_progress: Whether to show progress bar.

        Returns:
            List of loss dictionaries for each training step.
        """
        losses = []

        iterator = range(self.config.training_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Training")

        for _ in iterator:
            states, policies, values = self.replay_buffer.sample(
                self.config.batch_size,
                augment=True,
                move_encoder=self.move_encoder,
            )
            loss = self.network.train_on_batch(states, policies, values)
            losses.append(loss)

        return losses

    def _save_checkpoint(self):
        """Save a training checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"model_iter_{self.iteration}"
        )
        self.network.save(checkpoint_path)

        # Save training state
        state_path = os.path.join(self.checkpoint_dir, "training_state.npz")
        np.savez(
            state_path,
            iteration=self.iteration,
            total_games=self.total_games,
        )

    def load_checkpoint(self, iteration: Optional[int] = None):
        """Load a checkpoint.

        Args:
            iteration: Specific iteration to load. If None, loads latest.
        """
        if iteration is None:
            # Find latest checkpoint
            state_path = os.path.join(self.checkpoint_dir, "training_state.npz")
            if os.path.exists(state_path):
                state = np.load(state_path)
                iteration = int(state["iteration"])
                self.total_games = int(state["total_games"])

        if iteration:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_iter_{iteration}"
            )
            if os.path.exists(checkpoint_path + ".index") or os.path.exists(checkpoint_path + ".weights.h5"):
                self.network.load(checkpoint_path)
                self.iteration = iteration
                print(f"Loaded checkpoint from iteration {iteration}")

    def train(self, num_iterations: Optional[int] = None, show_progress: bool = True):
        """Run the full training loop.

        Args:
            num_iterations: Number of iterations to run. If None, uses config.
            show_progress: Whether to show progress.
        """
        num_iterations = num_iterations or self.config.num_iterations

        print(f"Starting training for {num_iterations} iterations")
        print(f"Network has {self.network.trainable_params:,} trainable parameters")

        for _ in range(num_iterations):
            stats = self.run_iteration(show_progress)

            if show_progress:
                print(f"Iteration {stats['iteration']}: "
                      f"games={stats['num_games']}, "
                      f"examples={stats['num_examples']}, "
                      f"buffer={stats['buffer_size']}")
                if "avg_total_loss" in stats:
                    print(f"  Loss: total={stats['avg_total_loss']:.4f}, "
                          f"policy={stats['avg_policy_loss']:.4f}, "
                          f"value={stats['avg_value_loss']:.4f}")

        print("\nTraining complete!")
        self._save_checkpoint()


def train_colab(num_iterations: int = 10, checkpoint_dir: str = None):
    """Convenience function for training in Google Colab.

    Args:
        num_iterations: Number of training iterations.
        checkpoint_dir: Directory for checkpoints (defaults to Colab drive).

    Returns:
        Trained Trainer instance.
    """
    config = Config()

    # Use Colab checkpoint dir if not specified
    if checkpoint_dir is None:
        checkpoint_dir = config.colab_checkpoint_dir

    # Enable mixed precision if configured
    if config.use_mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")

    trainer = Trainer(config, checkpoint_dir)
    trainer.train(num_iterations)
    return trainer
