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
    """Replay buffer for storing and sampling training examples.

    Each example is a tuple of (state, policy, value):
    - state: Board state as 781-dim vector
    - policy: Target policy distribution
    - value: Target value (-1 to 1)
    """

    def __init__(self, max_size: int = 100000):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of examples to store.
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add training examples to the buffer.

        Args:
            examples: List of (state, policy, value) tuples.
        """
        self.buffer.extend(examples)

    def add_single(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training example.

        Args:
            state: Board state.
            policy: Target policy.
            value: Target value.
        """
        self.buffer.append((state, policy, value))

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
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        states = [ex[0] for ex in batch]
        policies = [ex[1] for ex in batch]
        values = np.array([ex[2] for ex in batch], dtype=np.float32)

        if augment and move_encoder is not None:
            # Randomly mirror each example with 50% probability
            for i in range(len(states)):
                if random.random() < 0.5:
                    states[i] = mirror_state(states[i])
                    policies[i] = mirror_policy(policies[i], move_encoder)

        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)

        return states, policies, values

    def __len__(self) -> int:
        """Return the number of examples in the buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all examples from the buffer."""
        self.buffer.clear()

    def save(self, path: str):
        """Save the buffer to disk.

        Args:
            path: Path to save the buffer.
        """
        data = {
            "states": np.array([ex[0] for ex in self.buffer]),
            "policies": np.array([ex[1] for ex in self.buffer]),
            "values": np.array([ex[2] for ex in self.buffer]),
        }
        np.savez_compressed(path, **data)

    def load(self, path: str):
        """Load the buffer from disk.

        Args:
            path: Path to the saved buffer.
        """
        data = np.load(path)
        states = data["states"]
        policies = data["policies"]
        values = data["values"]

        self.buffer.clear()
        for i in range(len(states)):
            self.buffer.append((states[i], policies[i], values[i]))
