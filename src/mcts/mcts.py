"""Monte Carlo Tree Search with neural network guidance."""

import numpy as np
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.chess_game import ChessGame
from src.mcts.node import Node


class MCTS:
    """Monte Carlo Tree Search with PUCT selection and neural network guidance."""

    def __init__(
        self,
        network,
        config: Optional[Config] = None,
        num_simulations: Optional[int] = None,
    ):
        """Initialize MCTS.

        Args:
            network: Neural network for position evaluation.
            config: Configuration object.
            num_simulations: Number of simulations per search (overrides config).
        """
        self.network = network
        self.config = config or Config()
        self.num_simulations = num_simulations or self.config.num_simulations
        self.c_puct = self.config.c_puct

    def search(self, game: ChessGame, add_noise: bool = False) -> Tuple[list, np.ndarray, float]:
        """Run MCTS from the current position.

        Args:
            game: Current game state.
            add_noise: Whether to add Dirichlet noise to root priors.

        Returns:
            Tuple of (move_indices, probabilities, root_value) representing the search policy.
        """
        # Initialize root node
        root = Node(prior=0)

        # Get initial evaluation
        state = game.get_state()
        policy, _ = self.network.predict(state)
        legal_moves = game.get_legal_move_indices()

        if not legal_moves:
            return [], np.array([]), 0.0

        # Expand root
        root.expand(policy, legal_moves)

        # Add Dirichlet noise for exploration during training
        if add_noise:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha, self.config.dirichlet_epsilon
            )

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            scratch_game = game.clone()
            search_path = [node]

            # SELECT: traverse tree using PUCT
            while node.is_expanded():
                move_idx, node = node.select_child(self.c_puct)
                scratch_game.apply_move_index(move_idx)
                search_path.append(node)

            # Get value for leaf node
            if scratch_game.is_terminal():
                # Terminal node: use actual game outcome
                value = scratch_game.get_outcome_for_current_player()
            else:
                # Non-terminal: expand and evaluate
                state = scratch_game.get_state()
                policy, value = self.network.predict(state)
                legal_moves = scratch_game.get_legal_move_indices()

                if legal_moves:
                    node.expand(policy, legal_moves)

            # BACKPROPAGATE
            self._backpropagate(search_path, value)

        move_indices, probs = root.get_policy()
        return move_indices, probs, root.value

    def _backpropagate(self, search_path: list, value: float):
        """Backpropagate the value through the search path.

        Args:
            search_path: List of nodes from root to leaf.
            value: Value to backpropagate (from leaf's perspective).
        """
        # Value alternates sign as we go up the tree (negamax)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent

    def get_action(
        self, game: ChessGame, temperature: float = 1.0, add_noise: bool = False
    ) -> Tuple[int, np.ndarray, float]:
        """Get the best action after running MCTS.

        Args:
            game: Current game state.
            temperature: Temperature for action selection.
            add_noise: Whether to add exploration noise.

        Returns:
            Tuple of (selected_move_index, full_policy_array).
        """
        move_indices, probs, root_value = self.search(game, add_noise=add_noise)

        if not move_indices:
            return -1, np.zeros(self.network.policy_size), 0.0

        # Create full policy array
        full_policy = np.zeros(self.network.policy_size, dtype=np.float32)
        for move_idx, prob in zip(move_indices, probs):
            full_policy[move_idx] = prob

        # Select action based on temperature
        if temperature == 0:
            action = move_indices[np.argmax(probs)]
        else:
            # Sample based on visit counts
            adjusted_probs = probs ** (1.0 / temperature)
            adjusted_probs /= adjusted_probs.sum()
            action = np.random.choice(move_indices, p=adjusted_probs)

        return action, full_policy, root_value
