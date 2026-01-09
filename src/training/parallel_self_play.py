"""Parallel self-play with batched neural network inference."""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.chess_game import ChessGame
from src.mcts.node import Node


class ParallelMCTS:
    """MCTS that batches evaluations across multiple games."""

    def __init__(self, network, config: Config, num_simulations: int):
        self.network = network
        self.config = config
        self.num_simulations = num_simulations
        self.c_puct = config.c_puct

    def search_batch(
        self,
        games: List[ChessGame],
        add_noise: bool = True,
        noise_mask: Optional[List[bool]] = None,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """Run MCTS for multiple games with batched neural network calls.

        Args:
            games: List of games to search.
            add_noise: Whether to add Dirichlet noise (fallback if no mask).
            noise_mask: Optional per-game noise mask.

        Returns:
            List of (action, policy, root_value) tuples for each game.
        """
        num_games = len(games)
        if noise_mask is None:
            noise_mask = [add_noise] * num_games

        # Initialize roots for each game
        roots = [Node(prior=0) for _ in range(num_games)]

        # Initial expansion - batch evaluate all root positions
        states = np.array([g.get_state() for g in games], dtype=np.float32)
        policies, values = self.network.predict_batch(states)

        for i, (game, root, policy) in enumerate(zip(games, roots, policies)):
            legal_moves = game.get_legal_move_indices()
            if legal_moves:
                root.expand(policy, legal_moves)
                if noise_mask[i]:
                    root.add_dirichlet_noise(
                        self.config.dirichlet_alpha,
                        self.config.dirichlet_epsilon
                    )

        # Run simulations
        for _ in range(self.num_simulations):
            # For each game, traverse tree and collect leaf positions
            leaves = []  # (game_idx, node, scratch_game, search_path)

            for i in range(num_games):
                if not roots[i].is_expanded():
                    continue

                node = roots[i]
                scratch_game = games[i].clone()
                search_path = [node]

                # SELECT: traverse to leaf
                while node.is_expanded():
                    move_idx, node = node.select_child(self.c_puct)
                    scratch_game.apply_move_index(move_idx)
                    search_path.append(node)

                leaves.append((i, node, scratch_game, search_path))

            if not leaves:
                break

            # Batch evaluate all leaf positions
            leaf_states = []
            leaf_terminal = []
            leaf_values = []

            for game_idx, node, scratch_game, search_path in leaves:
                if scratch_game.is_terminal():
                    leaf_terminal.append(True)
                    leaf_values.append(scratch_game.get_outcome_for_current_player())
                    leaf_states.append(np.zeros(self.config.input_shape, dtype=np.float32))  # Placeholder
                else:
                    leaf_terminal.append(False)
                    leaf_values.append(0)
                    leaf_states.append(scratch_game.get_state())

            # Batch predict for non-terminal leaves
            leaf_states = np.array(leaf_states, dtype=np.float32)
            non_terminal_mask = [not t for t in leaf_terminal]

            if any(non_terminal_mask):
                policies_batch, values_batch = self.network.predict_batch(leaf_states)

            # Expand and backpropagate
            batch_idx = 0
            for leaf_idx, (game_idx, node, scratch_game, search_path) in enumerate(leaves):
                if leaf_terminal[leaf_idx]:
                    value = leaf_values[leaf_idx]
                else:
                    policy = policies_batch[leaf_idx]
                    value = float(values_batch[leaf_idx])
                    legal_moves = scratch_game.get_legal_move_indices()
                    if legal_moves:
                        node.expand(policy, legal_moves)

                # Backpropagate
                for path_node in reversed(search_path):
                    path_node.visit_count += 1
                    path_node.value_sum += value
                    value = -value

        # Get actions from roots
        results = []
        for i, (game, root) in enumerate(zip(games, roots)):
            if not root.is_expanded():
                results.append((-1, np.zeros(self.network.policy_size, dtype=np.float32), 0.0))
                continue

            move_indices, probs = root.get_policy()

            # Create full policy array
            full_policy = np.zeros(self.network.policy_size, dtype=np.float32)
            for move_idx, prob in zip(move_indices, probs):
                full_policy[move_idx] = prob

            # Get root value (average value from MCTS)
            root_value = root.value

            # Select action based on temperature
            if game.move_count < self.config.temp_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1

            if temperature == 0 or len(move_indices) == 1:
                action = move_indices[np.argmax(probs)]
            else:
                adjusted_probs = probs ** (1.0 / temperature)
                adjusted_probs /= adjusted_probs.sum()
                action = np.random.choice(move_indices, p=adjusted_probs)

            results.append((action, full_policy, root_value))

        return results


class ParallelSelfPlay:
    """Generate training games through parallel self-play."""

    def __init__(
        self,
        network,
        config: Optional[Config] = None,
        num_parallel: int = 8,
        num_simulations: Optional[int] = None,
    ):
        """Initialize parallel self-play.

        Args:
            network: Neural network for move selection.
            config: Configuration object.
            num_parallel: Number of games to play in parallel.
            num_simulations: MCTS simulations per move.
        """
        self.network = network
        self.config = config or Config()
        self.num_parallel = num_parallel
        self.num_simulations = num_simulations or self.config.num_simulations
        self.mcts = ParallelMCTS(network, self.config, self.num_simulations)

    def generate_games(
        self, num_games: int, show_progress: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate multiple self-play games with parallel execution.

        Args:
            num_games: Total number of games to play.
            show_progress: Whether to show progress bar.

        Returns:
            List of all training examples from all games.
        """
        all_examples = []
        games_completed = 0

        # Active games and their histories
        active_games = []
        game_histories = []

        pbar = tqdm(total=num_games, desc="Parallel self-play", disable=not show_progress)

        while games_completed < num_games:
            # Start new games if we have capacity
            while len(active_games) < self.num_parallel and games_completed + len(active_games) < num_games:
                active_games.append(ChessGame())
                game_histories.append([])

            if not active_games:
                break

            # Run batched MCTS for all active games
            noise_mask = [g.move_count < self.config.dirichlet_moves for g in active_games]
            results = self.mcts.search_batch(active_games, noise_mask=noise_mask)

            # Process results and advance games
            games_to_remove = []
            resigned_outcomes = {}  # Track resigned games and their outcomes

            for i, (game, history, (action, policy, root_value)) in enumerate(zip(active_games, game_histories, results)):
                if action < 0 or game.is_terminal():
                    games_to_remove.append(i)
                    continue

                # Store training example
                state = game.get_state()
                current_player = game.turn
                history.append((state, policy, current_player, root_value))

                # Apply move
                game.apply_move_index(action)

                # Check if game ended
                if game.is_terminal() or game.move_count >= self.config.max_moves:
                    games_to_remove.append(i)
                # Check resign threshold (only after some moves to avoid early resignations)
                elif (game.move_count >= self.config.resign_check_moves and
                      root_value < self.config.resign_threshold):
                    games_to_remove.append(i)
                    # The current player (after move) is losing, so the side that just moved wins
                    # value is from perspective of player who just moved, and it's very negative
                    # meaning they think they're losing, so opponent wins
                    resigned_outcomes[i] = -1.0 if game.turn else 1.0  # Opponent of current player wins

            # Finalize completed games
            for i in reversed(games_to_remove):
                game = active_games[i]
                history = game_histories[i]

                # Get outcome (use resigned outcome if available)
                if i in resigned_outcomes:
                    outcome = resigned_outcomes[i]
                else:
                    outcome = game.get_outcome()

                # Create training examples
                for state, policy, player_was_white, root_value in history:
                    outcome_value = outcome if player_was_white else -outcome
                    value = (
                        (1.0 - self.config.value_target_mix) * outcome_value
                        + self.config.value_target_mix * root_value
                    )
                    all_examples.append((state, policy, value))

                # Remove completed game
                active_games.pop(i)
                game_histories.pop(i)
                games_completed += 1
                pbar.update(1)

        pbar.close()
        return all_examples
