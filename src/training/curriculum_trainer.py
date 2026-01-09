"""Curriculum learning by playing against Stockfish at increasing depths."""

import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm
import chess
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.chess_game import ChessGame
from src.game.move_encoder import get_move_encoder
from src.model.network import ChessNetwork
from src.mcts.mcts import MCTS
from src.training.stockfish_evaluator import StockfishEvaluator
from src.training.replay_buffer import ReplayBuffer


class CurriculumTrainer:
    """Train by playing against Stockfish at progressively increasing difficulty.

    Starts at depth 1 (very weak) and increases when the bot wins consistently.
    This provides a natural curriculum that adapts to the bot's skill level.
    """

    def __init__(
        self,
        network: ChessNetwork,
        config: Optional[Config] = None,
        stockfish_path: Optional[str] = None,
        initial_depth: int = 1,
        max_depth: int = 8,
        promotion_threshold: float = 0.55,  # Win rate to advance
        num_simulations: int = 100,
    ):
        """Initialize curriculum trainer.

        Args:
            network: Neural network to train.
            config: Configuration object.
            stockfish_path: Path to Stockfish executable.
            initial_depth: Starting Stockfish depth.
            max_depth: Maximum Stockfish depth.
            promotion_threshold: Win rate required to increase depth.
            num_simulations: MCTS simulations per move.
        """
        self.network = network
        self.config = config or Config()
        self.move_encoder = get_move_encoder()

        self.current_depth = initial_depth
        self.max_depth = max_depth
        self.promotion_threshold = promotion_threshold
        self.num_simulations = num_simulations
        self.teacher_policy_weight = self.config.curriculum_policy_weight
        self.teacher_value_weight = self.config.curriculum_value_weight
        self.teacher_multipv = self.config.curriculum_multipv
        self.teacher_policy_temperature = self.config.curriculum_policy_temperature

        # Initialize Stockfish
        self.stockfish = StockfishEvaluator(
            stockfish_path=stockfish_path,
            depth=initial_depth,
            time_limit=0.05,  # Fast for low depths
            multipv=self.teacher_multipv,
            limit_strength=self.config.curriculum_limit_strength,
            elo=self.config.curriculum_elo,
            skill_level=self.config.curriculum_skill_level,
        )

        # Stats tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Replay buffer for training
        self.replay_buffer = ReplayBuffer(
            max_size=self.config.buffer_size,
            state_shape=self.config.input_shape,
            policy_size=self.move_encoder.policy_size
        )

    def play_game_against_stockfish(
        self,
        bot_color: bool = True,  # True = bot plays white
    ) -> Tuple[List[Tuple], float]:
        """Play a game against Stockfish.

        Args:
            bot_color: True if bot plays white, False for black.

        Returns:
            Tuple of (training_examples, outcome).
            outcome: 1 if bot wins, -1 if bot loses, 0 for draw.
        """
        game = ChessGame()
        mcts = MCTS(self.network, self.config, self.num_simulations)

        examples = []  # (state, policy, teacher_value)

        while not game.is_terminal() and game.move_count < self.config.max_moves:
            is_bot_turn = (game.turn == chess.WHITE) == bot_color

            if is_bot_turn:
                # Bot's turn - use MCTS
                state = game.get_state()
                add_noise = game.move_count < self.config.dirichlet_moves
                action, policy, _ = mcts.get_action(game, temperature=0.5, add_noise=add_noise)

                # Teacher policy + value from Stockfish
                move_probs, teacher_value = self.stockfish.get_policy_and_value(
                    game.board,
                    multipv=self.teacher_multipv,
                    policy_temperature=self.teacher_policy_temperature,
                )

                teacher_policy = np.zeros(self.move_encoder.policy_size, dtype=np.float32)
                for move, prob in move_probs:
                    try:
                        move_idx = self.move_encoder.encode(move)
                        teacher_policy[move_idx] = prob
                    except KeyError:
                        continue

                if teacher_policy.sum() > 0:
                    teacher_policy /= teacher_policy.sum()
                    policy = (
                        (1.0 - self.teacher_policy_weight) * policy
                        + self.teacher_policy_weight * teacher_policy
                    )

                if policy.sum() > 0:
                    policy /= policy.sum()
                else:
                    legal_moves = game.get_legal_move_indices()
                    if legal_moves:
                        policy = np.zeros(self.move_encoder.policy_size, dtype=np.float32)
                        policy[legal_moves] = 1.0 / len(legal_moves)

                if action < 0:
                    if move_probs:
                        game.apply_move(move_probs[0][0])
                        examples.append((state, teacher_policy, teacher_value))
                    else:
                        break
                else:
                    examples.append((state, policy, teacher_value))
                    game.apply_move_index(action)

            else:
                # Stockfish's turn
                sf_move = self.stockfish.get_best_move(game.board)

                if sf_move is None:
                    break

                game.apply_move(sf_move)

        # Determine outcome from bot's perspective
        game_result = game.get_outcome()  # 1=white wins, -1=black wins, 0=draw

        if bot_color:  # Bot is white
            bot_outcome = game_result
        else:  # Bot is black
            bot_outcome = -game_result

        # Convert examples with final outcome
        training_examples = []
        for state, policy, teacher_value in examples:
            value = (
                self.teacher_value_weight * teacher_value
                + (1.0 - self.teacher_value_weight) * bot_outcome
            )
            training_examples.append((state, policy, value))

        return training_examples, bot_outcome

    def run_games(
        self,
        num_games: int,
        show_progress: bool = True,
    ) -> Tuple[List[Tuple], dict]:
        """Run multiple games against Stockfish.

        Args:
            num_games: Number of games to play.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (all_examples, stats).
        """
        all_examples = []
        wins, losses, draws = 0, 0, 0

        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Playing vs Stockfish (depth {self.current_depth})")

        for i in iterator:
            # Alternate colors
            bot_color = (i % 2 == 0)
            examples, outcome = self.play_game_against_stockfish(bot_color)

            all_examples.extend(examples)

            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1

        stats = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0,
            "depth": self.current_depth,
        }

        return all_examples, stats

    def check_promotion(self) -> bool:
        """Check if bot should be promoted to next depth.

        Returns:
            True if promoted.
        """
        total = self.wins + self.losses + self.draws
        if total < 10:
            return False

        win_rate = self.wins / total

        if win_rate >= self.promotion_threshold and self.current_depth < self.max_depth:
            self.current_depth += 1
            self.stockfish.set_depth(self.current_depth)
            # Increase time limit slightly for higher depths
            self.stockfish.set_time_limit(0.05 + self.current_depth * 0.02)

            # Reset stats for new level
            self.wins = 0
            self.losses = 0
            self.draws = 0

            return True

        return False

    def train_iteration(
        self,
        num_games: int = 20,
        training_steps: Optional[int] = None,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> dict:
        """Run one curriculum training iteration.

        Args:
            num_games: Games to play against Stockfish.
            training_steps: Training steps on collected data.
            batch_size: Batch size for training.
            show_progress: Whether to show progress.

        Returns:
            Dictionary with iteration statistics.
        """
        training_steps = training_steps or self.config.curriculum_training_steps

        # Play games
        examples, game_stats = self.run_games(num_games, show_progress)

        # Update global stats
        self.wins += game_stats["wins"]
        self.losses += game_stats["losses"]
        self.draws += game_stats["draws"]

        # Add to replay buffer
        self.replay_buffer.add(examples)

        # Train
        losses = []
        if len(self.replay_buffer) >= batch_size:
            iterator = range(training_steps)
            if show_progress:
                iterator = tqdm(iterator, desc="Training")

            for _ in iterator:
                states, policies, values = self.replay_buffer.sample(
                    batch_size,
                    augment=True,
                    move_encoder=self.move_encoder,
                )
                loss = self.network.train_on_batch(states, policies, values)
                losses.append(loss)

        # Check for promotion
        promoted = self.check_promotion()

        # Compile stats
        stats = {
            **game_stats,
            "total_wins": self.wins,
            "total_losses": self.losses,
            "total_draws": self.draws,
            "promoted": promoted,
            "buffer_size": len(self.replay_buffer),
        }

        if losses:
            stats["avg_total_loss"] = np.mean([l["total_loss"] for l in losses])
            stats["avg_policy_loss"] = np.mean([l["policy_loss"] for l in losses])
            stats["avg_value_loss"] = np.mean([l["value_loss"] for l in losses])

        return stats

    def train(
        self,
        num_iterations: int = 50,
        games_per_iteration: Optional[int] = None,
        training_steps: Optional[int] = None,
        batch_size: int = 256,
        show_progress: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> List[dict]:
        """Run full curriculum training.

        Args:
            num_iterations: Number of training iterations.
            games_per_iteration: Games per iteration.
            training_steps: Training steps per iteration.
            batch_size: Batch size.
            show_progress: Whether to show progress.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_interval: Save every N iterations.

        Returns:
            Training history.
        """
        history = []

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        games_per_iteration = games_per_iteration or self.config.curriculum_games_per_iteration

        for iteration in range(1, num_iterations + 1):
            if show_progress:
                print(f"\n{'='*50}")
                print(f"Curriculum Iteration {iteration}/{num_iterations} (Stockfish depth {self.current_depth})")
                print(f"{'='*50}")

            stats = self.train_iteration(
                num_games=games_per_iteration,
                training_steps=training_steps,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            stats["iteration"] = iteration

            history.append(stats)

            if show_progress:
                print(f"\nResults: W={stats['wins']} L={stats['losses']} D={stats['draws']} "
                      f"(win rate: {stats['win_rate']:.1%})")
                if "avg_total_loss" in stats:
                    print(f"Loss: total={stats['avg_total_loss']:.4f}, "
                          f"policy={stats['avg_policy_loss']:.4f}, "
                          f"value={stats['avg_value_loss']:.4f}")
                if stats["promoted"]:
                    print(f"*** PROMOTED to depth {self.current_depth}! ***")

            # Save checkpoint
            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"curriculum_iter_{iteration}_depth_{self.current_depth}")
                self.network.save(path)
                if show_progress:
                    print(f"Checkpoint saved: {path}")

        return history

    def close(self):
        """Clean up resources."""
        self.stockfish.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
