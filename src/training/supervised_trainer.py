"""Supervised learning from Stockfish moves and evaluations."""

import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm
import chess
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.chess_game import ChessGame
from src.game.board_encoder import BoardEncoder
from src.game.move_encoder import get_move_encoder
from src.model.network import ChessNetwork
from src.training.stockfish_evaluator import StockfishEvaluator
from src.training.replay_buffer import ReplayBuffer


class SupervisedTrainer:
    """Train the network by imitating Stockfish moves and evaluations.

    This is much faster than self-play for initial training because:
    1. We get expert moves immediately (no MCTS needed)
    2. We get accurate evaluations (not just win/loss at end)
    3. More diverse positions (random sampling)
    """

    def __init__(
        self,
        network: ChessNetwork,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 10,
        config: Optional[Config] = None,
    ):
        """Initialize supervised trainer.

        Args:
            network: Neural network to train.
            stockfish_path: Path to Stockfish executable.
            stockfish_depth: Stockfish search depth.
            config: Configuration object.
        """
        self.network = network
        self.config = config or Config()
        self.move_encoder = get_move_encoder()

        # Initialize Stockfish
        self.stockfish = StockfishEvaluator(
            stockfish_path=stockfish_path,
            depth=stockfish_depth,
            time_limit=0.1
        )

    def generate_random_position(self, max_moves: int = 40) -> chess.Board:
        """Generate a random legal chess position by playing random moves.

        Args:
            max_moves: Maximum number of random moves to play.

        Returns:
            A chess board in a random position.
        """
        board = chess.Board()
        num_moves = np.random.randint(1, max_moves + 1)

        for _ in range(num_moves):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            move = np.random.choice(moves)
            board.push(move)

        return board

    def generate_training_batch(
        self,
        batch_size: int,
        max_position_moves: int = 40,
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of training examples from random positions.

        Args:
            batch_size: Number of examples to generate.
            max_position_moves: Maximum moves for random position generation.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (states, policies, values).
        """
        states = []
        policies = []
        values = []

        iterator = range(batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating supervised data")

        for _ in iterator:
            # Generate random position
            board = self.generate_random_position(max_position_moves)

            # Skip terminal positions
            if board.is_game_over():
                continue

            # Get Stockfish move and evaluation
            best_move, evaluation = self.stockfish.get_move_and_eval(board)

            if best_move is None:
                continue

            # Encode state
            state = BoardEncoder.encode(board)

            # Create policy (one-hot for Stockfish's move)
            policy = np.zeros(self.move_encoder.policy_size, dtype=np.float32)
            try:
                move_idx = self.move_encoder.encode(best_move)
                policy[move_idx] = 1.0
            except KeyError:
                # Move not in encoding, skip
                continue

            states.append(state)
            policies.append(policy)
            values.append(evaluation)

        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    def train_step(self, batch_size: int = 256) -> dict:
        """Run one training step.

        Args:
            batch_size: Number of examples per step.

        Returns:
            Dictionary with loss values.
        """
        states, policies, values = self.generate_training_batch(batch_size)

        if len(states) == 0:
            return {"total_loss": 0, "policy_loss": 0, "value_loss": 0}

        return self.network.train_on_batch(states, policies, values)

    def train(
        self,
        num_iterations: int = 100,
        batch_size: int = 256,
        batches_per_iteration: int = 50,
        show_progress: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> List[dict]:
        """Run supervised training.

        Args:
            num_iterations: Number of training iterations.
            batch_size: Batch size for training.
            batches_per_iteration: Number of batches per iteration.
            show_progress: Whether to show progress.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_interval: Save checkpoint every N iterations.

        Returns:
            List of training statistics per iteration.
        """
        history = []

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for iteration in range(1, num_iterations + 1):
            if show_progress:
                print(f"\nSupervised Iteration {iteration}/{num_iterations}")

            iteration_losses = []

            iterator = range(batches_per_iteration)
            if show_progress:
                iterator = tqdm(iterator, desc="Training batches")

            for _ in iterator:
                loss = self.train_step(batch_size)
                iteration_losses.append(loss)

            # Average losses
            avg_loss = {
                "total_loss": np.mean([l["total_loss"] for l in iteration_losses]),
                "policy_loss": np.mean([l["policy_loss"] for l in iteration_losses]),
                "value_loss": np.mean([l["value_loss"] for l in iteration_losses]),
            }

            history.append(avg_loss)

            if show_progress:
                print(f"  Loss: total={avg_loss['total_loss']:.4f}, "
                      f"policy={avg_loss['policy_loss']:.4f}, "
                      f"value={avg_loss['value_loss']:.4f}")

            # Save checkpoint
            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"supervised_iter_{iteration}")
                self.network.save(path)
                if show_progress:
                    print(f"  Checkpoint saved: {path}")

        return history

    def close(self):
        """Clean up resources."""
        self.stockfish.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
