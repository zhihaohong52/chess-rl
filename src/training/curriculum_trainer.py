"""Curriculum learning by playing against Stockfish at increasing depths."""

import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        num_simulations: Optional[int] = None,
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
        self.stockfish_path = stockfish_path

        self.current_depth = initial_depth
        self.max_depth = max_depth
        self.promotion_threshold = promotion_threshold
        self.num_simulations = num_simulations or getattr(self.config, "curriculum_num_simulations", 100)
        self.teacher_policy_weight = self.config.curriculum_policy_weight
        self.teacher_value_weight = self.config.curriculum_value_weight
        self.teacher_multipv = self.config.curriculum_multipv
        self.teacher_policy_temperature = self.config.curriculum_policy_temperature
        self.num_parallel = getattr(self.config, "curriculum_num_parallel", 4)

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
        if self.stockfish_path is None:
            self.stockfish_path = self.stockfish.stockfish_path

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

    def run_games_parallel(
        self,
        num_games: int,
        show_progress: bool = True,
    ) -> Tuple[List[Tuple], dict]:
        """Run multiple games against Stockfish with parallel batched inference.

        Uses batched neural network calls across multiple simultaneous games
        for significantly faster throughput.

        Args:
            num_games: Number of games to play.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (all_examples, stats).
        """
        from src.mcts.node import Node

        all_examples = []
        wins, losses, draws = 0, 0, 0
        games_completed = 0

        # Create multiple Stockfish instances for parallel games
        stockfish_instances = []
        for _ in range(self.num_parallel):
            sf = StockfishEvaluator(
                stockfish_path=self.stockfish_path,
                depth=self.current_depth,
                time_limit=0.05 + self.current_depth * 0.02,
                multipv=self.teacher_multipv,
            )
            stockfish_instances.append(sf)

        pbar = tqdm(total=num_games, desc=f"Playing vs Stockfish (depth {self.current_depth})",
                    disable=not show_progress)

        try:
            # Active game state
            active_games = []      # ChessGame instances
            active_histories = []  # [(state, policy, teacher_value), ...]
            active_colors = []     # bot_color for each game
            active_stockfish = []  # Stockfish instance for each game

            while games_completed < num_games:
                # Start new games if we have capacity
                while len(active_games) < self.num_parallel and games_completed + len(active_games) < num_games:
                    game_idx = games_completed + len(active_games)
                    bot_color = (game_idx % 2 == 0)
                    active_games.append(ChessGame())
                    active_histories.append([])
                    active_colors.append(bot_color)
                    active_stockfish.append(stockfish_instances[len(active_games) - 1])

                if not active_games:
                    break

                # Separate games by whose turn it is
                bot_turn_indices = []
                sf_turn_indices = []

                for i, (game, bot_color) in enumerate(zip(active_games, active_colors)):
                    if game.is_terminal() or game.move_count >= self.config.max_moves:
                        continue
                    is_bot_turn = (game.turn == chess.WHITE) == bot_color
                    if is_bot_turn:
                        bot_turn_indices.append(i)
                    else:
                        sf_turn_indices.append(i)

                # Handle Stockfish turns (can't batch these, but they're fast)
                for i in sf_turn_indices:
                    sf_move = active_stockfish[i].get_best_move(active_games[i].board)
                    if sf_move:
                        active_games[i].apply_move(sf_move)

                # Handle bot turns with batched MCTS
                if bot_turn_indices:
                    self._run_batched_mcts_step(
                        [active_games[i] for i in bot_turn_indices],
                        [active_histories[i] for i in bot_turn_indices],
                        [active_stockfish[i] for i in bot_turn_indices],
                    )

                # Check for completed games
                games_to_remove = []
                for i, game in enumerate(active_games):
                    if game.is_terminal() or game.move_count >= self.config.max_moves:
                        games_to_remove.append(i)

                # Finalize completed games
                for i in reversed(games_to_remove):
                    game = active_games[i]
                    history = active_histories[i]
                    bot_color = active_colors[i]

                    game_result = game.get_outcome()
                    bot_outcome = game_result if bot_color else -game_result

                    # Convert examples with final outcome
                    for state, policy, teacher_value in history:
                        value = (
                            self.teacher_value_weight * teacher_value
                            + (1.0 - self.teacher_value_weight) * bot_outcome
                        )
                        all_examples.append((state, policy, value))

                    if bot_outcome > 0:
                        wins += 1
                    elif bot_outcome < 0:
                        losses += 1
                    else:
                        draws += 1

                    active_games.pop(i)
                    active_histories.pop(i)
                    active_colors.pop(i)
                    active_stockfish.pop(i)
                    games_completed += 1
                    pbar.update(1)

        finally:
            pbar.close()
            for sf in stockfish_instances:
                sf.close()

        stats = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0,
            "depth": self.current_depth,
        }

        return all_examples, stats

    def _run_batched_mcts_step(
        self,
        games: List[ChessGame],
        histories: List[List],
        stockfish_instances: List[StockfishEvaluator],
    ):
        """Run one MCTS step for multiple games with batched neural network calls."""
        from src.mcts.node import Node

        num_games = len(games)
        if num_games == 0:
            return

        # Initialize MCTS roots for each game
        roots = [Node(prior=0) for _ in range(num_games)]

        # Batch initial evaluation
        states = np.array([g.get_state() for g in games], dtype=np.float32)
        policies, _ = self.network.predict_batch(states)

        for i, (game, root, policy) in enumerate(zip(games, roots, policies)):
            legal_moves = game.get_legal_move_indices()
            if legal_moves:
                root.expand(policy, legal_moves)
                if game.move_count < self.config.dirichlet_moves:
                    root.add_dirichlet_noise(
                        self.config.dirichlet_alpha,
                        self.config.dirichlet_epsilon
                    )

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            leaves = []

            for i in range(num_games):
                if not roots[i].is_expanded():
                    continue

                node = roots[i]
                scratch_game = games[i].clone()
                search_path = [node]

                while node.is_expanded():
                    move_idx, node = node.select_child(self.config.c_puct)
                    scratch_game.apply_move_index(move_idx)
                    search_path.append(node)

                leaves.append((i, node, scratch_game, search_path))

            if not leaves:
                break

            # Batch evaluate leaves
            leaf_states = []
            leaf_terminal = []
            leaf_values = []

            for _, node, scratch_game, _ in leaves:
                if scratch_game.is_terminal():
                    leaf_terminal.append(True)
                    leaf_values.append(scratch_game.get_outcome_for_current_player())
                    leaf_states.append(np.zeros(self.config.input_shape, dtype=np.float32))
                else:
                    leaf_terminal.append(False)
                    leaf_values.append(0)
                    leaf_states.append(scratch_game.get_state())

            policies_batch = None
            values_batch = None
            if any(not t for t in leaf_terminal):
                leaf_states = np.array(leaf_states, dtype=np.float32)
                policies_batch, values_batch = self.network.predict_batch(leaf_states)

            # Expand and backpropagate
            for leaf_idx, (game_idx, node, scratch_game, search_path) in enumerate(leaves):
                if leaf_terminal[leaf_idx]:
                    value = leaf_values[leaf_idx]
                else:
                    policy = policies_batch[leaf_idx]
                    value = float(values_batch[leaf_idx])
                    legal_moves = scratch_game.get_legal_move_indices()
                    if legal_moves:
                        node.expand(policy, legal_moves)

                for path_node in reversed(search_path):
                    path_node.visit_count += 1
                    path_node.value_sum += value
                    value = -value

        # Get actions and create training examples
        for i, (game, root, sf) in enumerate(zip(games, roots, stockfish_instances)):
            if not root.is_expanded():
                continue

            state = game.get_state()
            move_indices, probs = root.get_policy()

            # Create policy array
            policy = np.zeros(self.move_encoder.policy_size, dtype=np.float32)
            for move_idx, prob in zip(move_indices, probs):
                policy[move_idx] = prob

            # Get teacher policy/value from Stockfish
            move_probs, teacher_value = sf.get_policy_and_value(
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

            # Select action
            if len(move_indices) > 0:
                adjusted_probs = probs ** 2.0  # temperature=0.5
                adjusted_probs /= adjusted_probs.sum()
                action = np.random.choice(move_indices, p=adjusted_probs)

                histories[i].append((state, policy, teacher_value))
                game.apply_move_index(action)

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

        # Play games (use parallel version if num_parallel > 1)
        if self.num_parallel > 1:
            examples, game_stats = self.run_games_parallel(num_games, show_progress)
        else:
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
        num_iterations: int = 1000,
        games_per_iteration: Optional[int] = None,
        training_steps: Optional[int] = None,
        batch_size: int = 256,
        show_progress: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
        early_stop: bool = True,
        early_stop_wins: int = 3,
    ) -> List[dict]:
        """Run full curriculum training.

        Args:
            num_iterations: Maximum training iterations (default high for early stopping).
            games_per_iteration: Games per iteration.
            training_steps: Training steps per iteration.
            batch_size: Batch size.
            show_progress: Whether to show progress.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_interval: Save every N iterations.
            early_stop: Stop when max depth is mastered.
            early_stop_wins: Consecutive wins at max depth to trigger early stop.

        Returns:
            Training history.
        """
        history = []
        max_depth_wins = 0  # Track consecutive winning iterations at max depth

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        games_per_iteration = games_per_iteration or self.config.curriculum_games_per_iteration

        for iteration in range(1, num_iterations + 1):
            if show_progress:
                print(f"\n{'='*50}")
                print(f"Curriculum Iteration {iteration} (Stockfish depth {self.current_depth}/{self.max_depth})")
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

            # Early stopping: check if we've mastered max depth
            if early_stop and self.current_depth >= self.max_depth:
                if stats["win_rate"] >= self.promotion_threshold:
                    max_depth_wins += 1
                    if show_progress:
                        print(f"Max depth win streak: {max_depth_wins}/{early_stop_wins}")
                    if max_depth_wins >= early_stop_wins:
                        if show_progress:
                            print(f"\n*** EARLY STOP: Mastered Stockfish depth {self.max_depth}! ***")
                        break
                else:
                    max_depth_wins = 0  # Reset streak on loss

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
