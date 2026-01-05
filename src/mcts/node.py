"""MCTS Node implementation."""

from typing import Dict, Optional
import numpy as np


class Node:
    """A node in the MCTS tree.

    Each node stores statistics for the PUCT algorithm:
    - visit_count: Number of times this node has been visited
    - prior: Prior probability from the neural network P(s,a)
    - value_sum: Sum of values from backpropagation
    - children: Dictionary mapping move indices to child nodes
    """

    def __init__(self, prior: float):
        """Initialize a node.

        Args:
            prior: Prior probability from the neural network.
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}

    @property
    def value(self) -> float:
        """Average value of this node (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return len(self.children) > 0

    def expand(self, policy: np.ndarray, legal_moves: list):
        """Expand this node with children for legal moves.

        Args:
            policy: Policy distribution from neural network (over all moves).
            legal_moves: List of legal move indices.
        """
        # Mask illegal moves and renormalize
        masked_policy = np.zeros_like(policy)
        for move_idx in legal_moves:
            masked_policy[move_idx] = policy[move_idx]

        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # Uniform distribution if all priors are zero
            for move_idx in legal_moves:
                masked_policy[move_idx] = 1.0 / len(legal_moves)

        # Create child nodes
        for move_idx in legal_moves:
            self.children[move_idx] = Node(prior=masked_policy[move_idx])

    def select_child(self, c_puct: float) -> tuple:
        """Select the best child using PUCT formula.

        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant.

        Returns:
            Tuple of (move_index, child_node) for the best child.
        """
        best_score = float("-inf")
        best_move = None
        best_child = None

        sqrt_total_visits = np.sqrt(self.visit_count)

        for move_idx, child in self.children.items():
            # PUCT formula
            q_value = -child.value  # Negamax: opponent's value is negated
            exploration = c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_move = move_idx
                best_child = child

        return best_move, best_child

    def select_action(self, temperature: float = 1.0) -> int:
        """Select an action based on visit counts.

        Args:
            temperature: Temperature for action selection.
                - temperature = 0: Select most visited child (greedy)
                - temperature = 1: Sample proportional to visit counts
                - temperature > 1: More exploration

        Returns:
            Selected move index.
        """
        moves = list(self.children.keys())
        visit_counts = np.array([self.children[m].visit_count for m in moves])

        if temperature == 0:
            # Greedy selection
            return moves[np.argmax(visit_counts)]

        # Sample based on visit counts
        visit_counts = visit_counts.astype(np.float64)
        if temperature != 1.0:
            visit_counts = visit_counts ** (1.0 / temperature)

        probs = visit_counts / visit_counts.sum()
        return np.random.choice(moves, p=probs)

    def get_policy(self) -> tuple:
        """Get the policy (visit count distribution) from this node.

        Returns:
            Tuple of (move_indices, probabilities).
        """
        moves = list(self.children.keys())
        visit_counts = np.array([self.children[m].visit_count for m in moves], dtype=np.float32)
        total = visit_counts.sum()
        if total > 0:
            probs = visit_counts / total
        else:
            probs = np.ones(len(moves), dtype=np.float32) / len(moves)
        return moves, probs

    def add_dirichlet_noise(self, alpha: float, epsilon: float):
        """Add Dirichlet noise to prior probabilities for exploration.

        Args:
            alpha: Dirichlet alpha parameter.
            epsilon: Weight of noise (0 = no noise, 1 = all noise).
        """
        moves = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for i, move_idx in enumerate(moves):
            self.children[move_idx].prior = (
                (1 - epsilon) * self.children[move_idx].prior + epsilon * noise[i]
            )
