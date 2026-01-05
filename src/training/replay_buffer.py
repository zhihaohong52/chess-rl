"""Experience replay buffer for storing training examples."""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional
import random


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

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of training examples.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            Tuple of (states, policies, values) as numpy arrays.
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        states = np.array([ex[0] for ex in batch], dtype=np.float32)
        policies = np.array([ex[1] for ex in batch], dtype=np.float32)
        values = np.array([ex[2] for ex in batch], dtype=np.float32)

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
