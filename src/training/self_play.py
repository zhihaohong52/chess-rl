"""Self-play game generation for training."""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS


class SelfPlay:
    """Generate training games through self-play."""

    def __init__(
        self,
        network,
        config: Optional[Config] = None,
        num_simulations: Optional[int] = None,
    ):
        """Initialize self-play.

        Args:
            network: Neural network for move selection.
            config: Configuration object.
            num_simulations: MCTS simulations per move (overrides config).
        """
        self.network = network
        self.config = config or Config()
        self.num_simulations = num_simulations or self.config.num_simulations

    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Play a single self-play game.

        Returns:
            List of (state, policy, value) training examples.
        """
        game = ChessGame()
        mcts = MCTS(self.network, self.config, self.num_simulations)

        # Store (state, policy, current_player, root_value) during game
        game_history = []

        move_count = 0
        while not game.is_terminal() and move_count < self.config.max_moves:
            state = game.get_state()

            # Determine temperature based on move count
            if move_count < self.config.temp_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1  # Near-greedy after threshold

            # Run MCTS and get action
            add_noise = game.move_count < self.config.dirichlet_moves
            action, policy, root_value = mcts.get_action(
                game, temperature=temperature, add_noise=add_noise
            )

            if action < 0:
                # No legal moves (shouldn't happen, but safety check)
                break

            # Store example with current player
            current_player = game.turn  # True = White
            game_history.append((state, policy, current_player, root_value))

            # Apply move
            game.apply_move_index(action)
            move_count += 1

        # Get game outcome
        outcome = game.get_outcome()  # 1 for white win, -1 for black win, 0 for draw

        # Create final training examples with proper values
        examples = []
        for state, policy, player_was_white, root_value in game_history:
            # Value from the perspective of the player who made the move
            if player_was_white:
                outcome_value = outcome
            else:
                outcome_value = -outcome
            value = (
                (1.0 - self.config.value_target_mix) * outcome_value
                + self.config.value_target_mix * root_value
            )
            examples.append((state, policy, value))

        return examples

    def generate_games(
        self, num_games: int, show_progress: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate multiple self-play games.

        Args:
            num_games: Number of games to play.
            show_progress: Whether to show progress bar.

        Returns:
            List of all training examples from all games.
        """
        all_examples = []

        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc="Self-play games")

        for _ in iterator:
            examples = self.play_game()
            all_examples.extend(examples)

        return all_examples

    def play_game_with_stats(self) -> Tuple[List, dict]:
        """Play a game and return statistics.

        Returns:
            Tuple of (examples, stats_dict).
        """
        game = ChessGame()
        mcts = MCTS(self.network, self.config, self.num_simulations)

        game_history = []
        move_count = 0

        while not game.is_terminal() and move_count < self.config.max_moves:
            state = game.get_state()

            if move_count < self.config.temp_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1

            add_noise = game.move_count < self.config.dirichlet_moves
            action, policy, root_value = mcts.get_action(game, temperature=temperature, add_noise=add_noise)

            if action < 0:
                break

            current_player = game.turn
            game_history.append((state, policy, current_player, root_value))
            game.apply_move_index(action)
            move_count += 1

        outcome = game.get_outcome()

        examples = []
        for state, policy, player_was_white, root_value in game_history:
            outcome_value = outcome if player_was_white else -outcome
            value = (
                (1.0 - self.config.value_target_mix) * outcome_value
                + self.config.value_target_mix * root_value
            )
            examples.append((state, policy, value))

        stats = {
            "num_moves": move_count,
            "outcome": outcome,
            "result": "white" if outcome > 0 else ("black" if outcome < 0 else "draw"),
        }

        return examples, stats
