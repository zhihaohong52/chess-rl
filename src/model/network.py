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
    """Residual CNN with policy and value heads, supporting SE attention."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the network.

        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.move_encoder = get_move_encoder()
        self.policy_size = self.move_encoder.policy_size
        self.use_se = getattr(self.config, "use_se_blocks", True)
        self.se_ratio = getattr(self.config, "se_ratio", 4)
        self.policy_channels = getattr(self.config, "policy_channels", 32)
        self.value_channels = getattr(self.config, "value_channels", 32)
        self.model = self._build_model()
        self._predict_fn = None  # Cached compiled prediction function

        print(self.model.summary())

    def _se_block(self, x: tf.Tensor, filters: int, idx: int) -> tf.Tensor:
        """Squeeze-and-Excitation block for channel attention."""
        se = layers.GlobalAveragePooling2D(name=f"res{idx}_se_pool")(x)
        se = layers.Dense(filters // self.se_ratio, activation="relu", name=f"res{idx}_se_fc1")(se)
        se = layers.Dense(filters, activation="sigmoid", name=f"res{idx}_se_fc2")(se)
        se = layers.Reshape((1, 1, filters), name=f"res{idx}_se_reshape")(se)
        return layers.Multiply(name=f"res{idx}_se_mul")([x, se])

    def _residual_block(self, x: tf.Tensor, filters: int, idx: int) -> tf.Tensor:
        """Residual block with two 3x3 convolutions and optional SE attention."""
        shortcut = x
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"res{idx}_conv1")(x)
        x = layers.BatchNormalization(name=f"res{idx}_bn1")(x)
        x = layers.ReLU(name=f"res{idx}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"res{idx}_conv2")(x)
        x = layers.BatchNormalization(name=f"res{idx}_bn2")(x)

        # Squeeze-and-Excitation attention
        if self.use_se:
            x = self._se_block(x, filters, idx)

        x = layers.Add(name=f"res{idx}_add")([shortcut, x])
        x = layers.ReLU(name=f"res{idx}_relu2")(x)
        return x

    def _build_model(self) -> keras.Model:
        """Build the residual CNN model."""
        inputs = layers.Input(shape=self.config.input_shape, name="board_input")

        x = layers.Conv2D(
            self.config.residual_filters,
            3,
            padding="same",
            use_bias=False,
            name="stem_conv",
        )(inputs)
        x = layers.BatchNormalization(name="stem_bn")(x)
        x = layers.ReLU(name="stem_relu")(x)

        for i in range(self.config.residual_blocks):
            x = self._residual_block(x, self.config.residual_filters, i)

        # Policy head - 1x1 conv to reduce channels, then flatten to policy
        policy = layers.Conv2D(self.policy_channels, 1, use_bias=False, name="policy_conv")(x)
        policy = layers.BatchNormalization(name="policy_bn")(policy)
        policy = layers.ReLU(name="policy_relu")(policy)
        policy = layers.Flatten(name="policy_flatten")(policy)
        policy = layers.Dense(self.policy_size, name="policy_logits")(policy)
        policy_output = layers.Softmax(name="policy")(policy)

        # Value head - 1x1 conv then combine global + spatial features
        value_spatial = layers.Conv2D(self.value_channels, 1, use_bias=False, name="value_conv")(x)
        value_spatial = layers.BatchNormalization(name="value_bn")(value_spatial)
        value_spatial = layers.ReLU(name="value_relu")(value_spatial)

        # Global features (position-invariant) - key improvement over original
        value_global = layers.GlobalAveragePooling2D(name="value_global_pool")(value_spatial)

        # Combine with hidden layer
        value = layers.Dense(self.config.value_hidden, activation="relu", name="value_dense")(value_global)
        value_output = layers.Dense(1, activation="tanh", name="value")(value)

        model = keras.Model(inputs=inputs, outputs=[policy_output, value_output])
        return model

    def compile(self, learning_rate: Optional[float] = None, use_schedule: Optional[bool] = None,
                total_steps: Optional[int] = None):
        """Compile the model with optimizer and loss functions.

        Args:
            learning_rate: Learning rate for optimizer. If None, uses config value.
            use_schedule: Whether to use cosine decay learning rate schedule.
            total_steps: Total training steps for LR schedule.
        """
        lr = learning_rate or self.config.learning_rate
        use_schedule = self.config.use_lr_schedule if use_schedule is None else use_schedule
        total_steps = total_steps or self.config.lr_total_steps

        if use_schedule:
            # Cosine decay with warmup
            lr = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps,
                alpha=self.config.lr_final_alpha
            )

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=self.config.weight_decay,
            clipnorm=1.0,
        )

        self.model.compile(
            optimizer=optimizer,
            loss={
                "policy": keras.losses.CategoricalCrossentropy(from_logits=False),
                "value": keras.losses.Huber(),
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
            state: Board state as an 8x8xN tensor.

        Returns:
            Tuple of (policy, value) where policy is a probability distribution
            over moves and value is a scalar.
        """
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)

        state_tensor = tf.constant(state, dtype=tf.float32)
        policy, value = self._fast_predict(state_tensor)
        return policy[0].numpy(), float(value[0, 0])

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict policy and value for a batch of states.

        Args:
            states: Batch of board states, shape (batch_size, 8, 8, planes).

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
            states: Batch of board states, shape (batch_size, 8, 8, planes).
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
        if isinstance(loss, list) and len(loss) >= 3:
            return {
                "total_loss": float(loss[0]),
                "policy_loss": float(loss[1]),
                "value_loss": float(loss[2]),
            }
        # Fallback: return same value for all (shouldn't happen with multi-output model)
        total = float(loss) if not isinstance(loss, list) else float(loss[0])
        return {"total_loss": total, "policy_loss": total, "value_loss": total}

    def save(self, path: str):
        """Save the model weights.

        Args:
            path: Path to save the weights (without extension).
        """
        if not path.endswith(".weights.h5"):
            path = path + ".weights.h5"
        self.model.save_weights(path)

    def load(self, path: str):
        """Load model weights.

        Args:
            path: Path to the weights file.
        """
        if not path.endswith(".weights.h5"):
            path = path + ".weights.h5"
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
