"""Lightweight neural network for chess RL."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.move_encoder import get_move_encoder


class ChessNetwork:
    """Lightweight neural network with policy and value heads.

    Architecture:
        Input (781)
            |
        Dense(512) + ReLU
            |
        Dense(512) + ReLU
            |
        Dense(256) + ReLU
            |
        Dense(256) + ReLU
            |
        +---------------+
        |               |
    Policy Head    Value Head
        |               |
    Dense(policy_size)  Dense(64) + ReLU
        |               |
    Softmax         Dense(1) + tanh
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the network.

        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.move_encoder = get_move_encoder()
        self.policy_size = self.move_encoder.policy_size
        self.model = self._build_model()
        self._predict_fn = None  # Cached compiled prediction function

    def _build_model(self) -> keras.Model:
        """Build the neural network model."""
        # Input layer
        inputs = layers.Input(shape=(Config.input_size,), name="board_input")

        # Shared hidden layers
        x = inputs
        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(units, activation="relu", name=f"hidden_{i}")(x)

        # Policy head
        policy = layers.Dense(self.policy_size, name="policy_logits")(x)
        policy_output = layers.Softmax(name="policy")(policy)

        # Value head
        value = layers.Dense(self.config.value_hidden, activation="relu", name="value_hidden")(x)
        value_output = layers.Dense(1, activation="tanh", name="value")(value)

        model = keras.Model(inputs=inputs, outputs=[policy_output, value_output])
        return model

    def compile(self, learning_rate: Optional[float] = None):
        """Compile the model with optimizer and loss functions.

        Args:
            learning_rate: Learning rate for optimizer. If None, uses config value.
        """
        lr = learning_rate or self.config.learning_rate
        optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(
            optimizer=optimizer,
            loss={
                "policy": keras.losses.CategoricalCrossentropy(),
                "value": keras.losses.MeanSquaredError(),
            },
            loss_weights={"policy": 1.0, "value": 1.0},
        )

    @tf.function(reduce_retracing=True)
    def _fast_predict(self, states):
        """Compiled prediction function for speed."""
        return self.model(states, training=False)

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a single state.

        Uses direct model call instead of model.predict() for 10-50x speedup.

        Args:
            state: Board state as a 781-dimensional vector.

        Returns:
            Tuple of (policy, value) where policy is a probability distribution
            over moves and value is a scalar.
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state_tensor = tf.constant(state, dtype=tf.float32)
        policy, value = self._fast_predict(state_tensor)
        return policy[0].numpy(), float(value[0, 0])

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict policy and value for a batch of states.

        Args:
            states: Batch of board states, shape (batch_size, 781).

        Returns:
            Tuple of (policies, values) where policies has shape (batch_size, policy_size)
            and values has shape (batch_size,).
        """
        state_tensor = tf.constant(states, dtype=tf.float32)
        policies, values = self._fast_predict(state_tensor)
        return policies.numpy(), values.numpy().flatten()

    def train_on_batch(
        self,
        states: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
    ) -> dict:
        """Train the network on a single batch.

        Args:
            states: Batch of board states, shape (batch_size, 781).
            target_policies: Target policy distributions, shape (batch_size, policy_size).
            target_values: Target values, shape (batch_size,).

        Returns:
            Dictionary with loss values.
        """
        target_values = target_values.reshape(-1, 1)
        loss = self.model.train_on_batch(
            states, {"policy": target_policies, "value": target_values}
        )

        # loss is [total_loss, policy_loss, value_loss]
        if isinstance(loss, list):
            return {
                "total_loss": loss[0],
                "policy_loss": loss[1],
                "value_loss": loss[2],
            }
        return {"total_loss": loss}

    def save(self, path: str):
        """Save the model weights.

        Args:
            path: Path to save the weights (without extension).
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """Load model weights.

        Args:
            path: Path to the weights file.
        """
        self.model.load_weights(path)

    def save_full_model(self, path: str):
        """Save the entire model (architecture + weights).

        Args:
            path: Path to save the model.
        """
        self.model.save(path)

    @classmethod
    def load_full_model(cls, path: str, config: Optional[Config] = None) -> "ChessNetwork":
        """Load a full model from disk.

        Args:
            path: Path to the saved model.
            config: Configuration object.

        Returns:
            ChessNetwork instance with loaded model.
        """
        network = cls(config)
        network.model = keras.models.load_model(path)
        return network

    def summary(self):
        """Print model summary."""
        self.model.summary()

    @property
    def trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            np.prod(v.shape.as_list()) for v in self.model.trainable_variables
        )
