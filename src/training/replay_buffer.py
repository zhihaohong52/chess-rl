"""Experience replay buffer for storing training examples."""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional
import random


def mirror_square(sq: int) -> int:
    """Mirror a square horizontally (flip file)."""
    rank = sq // 8
    file = sq % 8
    return rank * 8 + (7 - file)


def mirror_state(state: np.ndarray) -> np.ndarray:
    """Mirror a board state horizontally.

    Flips all bitboards so that files a-h become h-a.
    """
    mirrored = state.copy()

    # Mirror each of the 12 bitboards (768 values)
    for i in range(12):
        offset = i * 64
        for sq in range(64):
            mirrored[offset + mirror_square(sq)] = state[offset + sq]

    # En passant file (positions 772-779) also needs to be mirrored
    ep_original = state[772:780].copy()
    for i in range(8):
        mirrored[772 + (7 - i)] = ep_original[i]

    return mirrored


def mirror_policy(policy: np.ndarray, move_encoder) -> np.ndarray:
    """Mirror a policy horizontally.

    Each move's from and to squares have their files flipped.
    """
    mirrored = np.zeros_like(policy)

    for idx in range(len(policy)):
        if policy[idx] > 0:
            move = move_encoder.decode(idx)
            from_sq = mirror_square(move.from_square)
            to_sq = mirror_square(move.to_square)

            import chess
            mirrored_move = chess.Move(from_sq, to_sq, promotion=move.promotion)
            try:
                new_idx = move_encoder.encode(mirrored_move)
                mirrored[new_idx] = policy[idx]
            except KeyError:
                # Move not in encoding (shouldn't happen)
                mirrored[idx] = policy[idx]

    return mirrored


class ReplayBuffer:
    """Replay buffer using numpy arrays for fast sampling.

    Each example is a tuple of (state, policy, value):
    - state: Board state as 781-dim vector
    - policy: Target policy distribution
    - value: Target value (-1 to 1)
    """

    def __init__(self, max_size: int = 100000, state_size: int = 781, policy_size: int = 1924):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of examples to store.
            state_size: Size of state vector.
            policy_size: Size of policy vector.
        """
        self.max_size = max_size
        self.state_size = state_size
        self.policy_size = policy_size

        # Pre-allocate numpy arrays
        self.states = np.zeros((max_size, state_size), dtype=np.float32)
        self.policies = np.zeros((max_size, policy_size), dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)

        self.size = 0  # Current number of examples
        self.pos = 0   # Current write position (circular buffer)

    def add(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add training examples to the buffer.

        Args:
            examples: List of (state, policy, value) tuples.
        """
        for state, policy, value in examples:
            self.states[self.pos] = state
            self.policies[self.pos] = policy
            self.values[self.pos] = value

            self.pos = (self.pos + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def add_single(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training example.

        Args:
            state: Board state.
            policy: Target policy.
            value: Target value.
        """
        self.states[self.pos] = state
        self.policies[self.pos] = policy
        self.values[self.pos] = value

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, augment: bool = False,
               move_encoder=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of training examples.

        Args:
            batch_size: Number of examples to sample.
            augment: Whether to apply random horizontal mirroring (doubles effective data).
            move_encoder: Required if augment=True, for mirroring policies.

        Returns:
            Tuple of (states, policies, values) as numpy arrays.
        """
        batch_size = min(batch_size, self.size)

        # Fast random sampling with numpy
        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices].copy()
        policies = self.policies[indices].copy()
        values = self.values[indices].copy()

        if augment and move_encoder is not None:
            # Randomly mirror each example with 50% probability
            mirror_mask = np.random.random(batch_size) < 0.5
            for i in np.where(mirror_mask)[0]:
                states[i] = mirror_state(states[i])
                policies[i] = mirror_policy(policies[i], move_encoder)

        return states, policies, values

    def __len__(self) -> int:
        """Return the number of examples in the buffer."""
        return self.size

    def clear(self):
        """Clear all examples from the buffer."""
        self.size = 0
        self.pos = 0

    def save(self, path: str):
        """Save the buffer to disk.

        Args:
            path: Path to save the buffer.
        """
        np.savez_compressed(
            path,
            states=self.states[:self.size],
            policies=self.policies[:self.size],
            values=self.values[:self.size],
        )

    def load(self, path: str):
        """Load the buffer from disk.

        Args:
            path: Path to the saved buffer.
        """
        data = np.load(path)
        loaded_size = len(data["states"])

        self.size = min(loaded_size, self.max_size)
        self.pos = self.size % self.max_size

        self.states[:self.size] = data["states"][:self.size]
        self.policies[:self.size] = data["policies"][:self.size]
        self.values[:self.size] = data["values"][:self.size]
