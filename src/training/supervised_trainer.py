"""Supervised learning from Stockfish moves and evaluations."""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple

import chess
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from src.game.board_encoder import BoardEncoder
from src.game.move_encoder import get_move_encoder
from src.model.network import ChessNetwork
from src.training.stockfish_evaluator import StockfishEvaluator
from src.training.replay_buffer import mirror_state, mirror_policy


def _generate_random_position(max_moves: int) -> Tuple[chess.Board, List[chess.Board]]:
    """Generate a random legal chess position by playing random moves."""
    board = chess.Board()
    history = [board.copy()]
    num_moves = np.random.randint(1, max_moves + 1)

    for _ in range(num_moves):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        if not moves:
            break
        move = np.random.choice(moves)
        board.push(move)
        history.append(board.copy())

    return board, history[-Config.history_length:]


def _generate_stockfish_position(
    stockfish: StockfishEvaluator,
    multipv: int,
    position_temperature: float,
    max_moves: int,
) -> Tuple[chess.Board, List[chess.Board]]:
    """Generate a position by playing Stockfish moves with sampling."""
    board = chess.Board()
    history = [board.copy()]
    num_moves = np.random.randint(1, max_moves + 1)

    for _ in range(num_moves):
        if board.is_game_over():
            break
        move_probs, _ = stockfish.get_policy_and_value(
            board,
            multipv=multipv,
            policy_temperature=position_temperature,
        )
        move = SupervisedTrainer._sample_move_static(move_probs)
        if move is None:
            break
        board.push(move)
        history.append(board.copy())

    return board, history[-Config.history_length:]


def _generate_supervised_batch_worker(
    batch_size: int,
    stockfish_path: Optional[str],
    stockfish_depth: int,
    multipv: int,
    policy_temperature: float,
    position_source: str,
    position_max_moves: int,
    position_temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker to generate supervised samples with a private Stockfish evaluator."""
    move_encoder = get_move_encoder()
    stockfish = StockfishEvaluator(
        stockfish_path=stockfish_path,
        depth=stockfish_depth,
        time_limit=0.1,
        multipv=multipv,
    )
    try:
        states = []
        policies = []
        values = []

        for _ in range(batch_size):
            if position_source == "stockfish":
                board, history = _generate_stockfish_position(
                    stockfish,
                    multipv,
                    position_temperature,
                    position_max_moves,
                )
            else:
                board, history = _generate_random_position(position_max_moves)

            if board.is_game_over():
                continue

            move_probs, evaluation = stockfish.get_policy_and_value(
                board,
                multipv=multipv,
                policy_temperature=policy_temperature,
            )

            if not move_probs:
                continue

            state = BoardEncoder.encode(board, history=history)

            policy = np.zeros(move_encoder.policy_size, dtype=np.float32)
            for move, prob in move_probs:
                try:
                    move_idx = move_encoder.encode(move)
                    policy[move_idx] = prob
                except KeyError:
                    continue

            if policy.sum() <= 0:
                continue
            policy /= policy.sum()

            states.append(state)
            policies.append(policy)
            values.append(evaluation)

        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )
    finally:
        stockfish.close()


class SupervisedTrainer:
    """Train the network by imitating Stockfish moves and evaluations.

    This is much faster than self-play for initial training because:
    1. We get expert moves immediately (no MCTS needed)
    2. We get accurate evaluations (not just win/loss at end)
    3. More diverse positions (engine-guided sampling)
    """

    def __init__(
        self,
        network: ChessNetwork,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 10,
        multipv: Optional[int] = None,
        policy_temperature: Optional[float] = None,
        config: Optional[Config] = None,
    ):
        """Initialize supervised trainer.

        Args:
            network: Neural network to train.
            stockfish_path: Path to Stockfish executable.
            stockfish_depth: Stockfish search depth.
            config: Configuration object.
        """
        self.network = network
        self.config = config or Config()
        self.move_encoder = get_move_encoder()
        self.multipv = multipv or self.config.supervised_multipv
        self.policy_temperature = (
            self.config.supervised_policy_temperature if policy_temperature is None else policy_temperature
        )
        self.position_source = self.config.supervised_position_source
        self.position_max_moves = self.config.supervised_position_max_moves
        self.position_temperature = self.config.supervised_position_temperature
        self.augment = self.config.supervised_augment
        self.cache_enabled = self.config.supervised_cache_enabled
        self.cache_dir = self.config.supervised_cache_dir
        self.cache_size = self.config.supervised_cache_size
        self.cache_dtype = self.config.supervised_cache_dtype
        self.cache_rebuild = self.config.supervised_cache_rebuild
        self.cache_workers = getattr(self.config, "supervised_cache_workers", 1)
        self.cache_batch_size = getattr(self.config, "supervised_cache_batch_size", 512)
        self.cache_states = None
        self.cache_policies = None
        self.cache_values = None
        self.cache_count = 0
        self.cache_build_seconds = None
        self.cache_build_throughput = None
        self.cache_build_workers = None
        self.cache_build_batch_size = None
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth

        # Initialize Stockfish
        self.stockfish = StockfishEvaluator(
            stockfish_path=stockfish_path,
            depth=stockfish_depth,
            time_limit=0.1,
            multipv=self.multipv,
        )

        if self.cache_enabled:
            self._ensure_cache()

    def _cache_paths(self) -> Tuple[str, str, str, str]:
        """Return paths for cached arrays and metadata."""
        base = self.cache_dir
        return (
            os.path.join(base, "states.npy"),
            os.path.join(base, "policies.npy"),
            os.path.join(base, "values.npy"),
            os.path.join(base, "meta.json"),
        )

    def _cache_meta(self) -> dict:
        """Build metadata for cache validation."""
        return {
            "version": 1,
            "input_shape": list(self.config.input_shape),
            "policy_size": self.move_encoder.policy_size,
            "cache_size": int(self.cache_size),
            "dtype": str(self.cache_dtype),
            "stockfish_depth": int(self.stockfish.depth),
            "multipv": int(self.multipv),
            "position_source": self.position_source,
            "position_max_moves": int(self.position_max_moves),
            "position_temperature": float(self.position_temperature),
            "policy_temperature": float(self.policy_temperature),
        }

    def _load_cache(self) -> bool:
        """Load cache if it matches current configuration."""
        states_path, policies_path, values_path, meta_path = self._cache_paths()
        if not (os.path.exists(states_path) and os.path.exists(policies_path) and
                os.path.exists(values_path) and os.path.exists(meta_path)):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            return False

        expected = self._cache_meta()
        for key in ("version", "input_shape", "policy_size", "cache_size", "dtype",
                    "stockfish_depth", "multipv", "position_source",
                    "position_max_moves", "position_temperature", "policy_temperature"):
            if meta.get(key) != expected.get(key):
                return False

        self.cache_count = int(meta.get("cache_size", 0))
        if self.cache_count <= 0:
            return False

        self.cache_states = np.load(states_path, mmap_mode="r")
        self.cache_policies = np.load(policies_path, mmap_mode="r")
        self.cache_values = np.load(values_path, mmap_mode="r")
        return True

    def _build_cache(self):
        """Generate a supervised dataset cache on disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        states_path, policies_path, values_path, meta_path = self._cache_paths()

        dtype = np.float16 if str(self.cache_dtype).lower() == "float16" else np.float32
        states_mm = np.lib.format.open_memmap(
            states_path,
            mode="w+",
            dtype=dtype,
            shape=(self.cache_size, *self.config.input_shape),
        )
        policies_mm = np.lib.format.open_memmap(
            policies_path,
            mode="w+",
            dtype=dtype,
            shape=(self.cache_size, self.move_encoder.policy_size),
        )
        values_mm = np.lib.format.open_memmap(
            values_path,
            mode="w+",
            dtype=dtype,
            shape=(self.cache_size,),
        )

        if self.cache_workers and self.cache_workers > 1:
            self._build_cache_parallel(states_mm, policies_mm, values_mm, dtype)
            meta = self._cache_meta()
            meta["cache_size"] = int(self.cache_size)
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)

            self.cache_states = np.load(states_path, mmap_mode="r")
            self.cache_policies = np.load(policies_path, mmap_mode="r")
            self.cache_values = np.load(values_path, mmap_mode="r")
            self.cache_count = self.cache_size
            return

        start_time = time.time()
        filled = 0
        pbar = tqdm(total=self.cache_size, desc="Building supervised cache", leave=True)
        while filled < self.cache_size:
            batch = min(max(1, int(self.cache_batch_size)), self.cache_size - filled)
            states, policies, values = self.generate_training_batch(
                batch,
                max_position_moves=self.position_max_moves,
                augment=False,
                show_progress=False,
            )
            if len(states) == 0:
                continue

            take = min(len(states), self.cache_size - filled)
            states_mm[filled:filled + take] = states[:take].astype(dtype, copy=False)
            policies_mm[filled:filled + take] = policies[:take].astype(dtype, copy=False)
            values_mm[filled:filled + take] = values[:take].astype(dtype, copy=False)
            filled += take
            pbar.update(take)

        pbar.close()
        states_mm.flush()
        policies_mm.flush()
        values_mm.flush()

        meta = self._cache_meta()
        meta["cache_size"] = int(self.cache_size)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        self.cache_states = np.load(states_path, mmap_mode="r")
        self.cache_policies = np.load(policies_path, mmap_mode="r")
        self.cache_values = np.load(values_path, mmap_mode="r")
        self.cache_count = self.cache_size
        elapsed = max(0.001, time.time() - start_time)
        self.cache_build_seconds = elapsed
        self.cache_build_throughput = self.cache_size / elapsed
        self.cache_build_workers = 1
        self.cache_build_batch_size = self.cache_batch_size

    def _build_cache_parallel(self, states_mm, policies_mm, values_mm, dtype):
        """Build cache using multiprocessing workers."""
        start_time = time.time()
        filled = 0
        pbar = tqdm(total=self.cache_size, desc="Building supervised cache", leave=True)
        worker_count = max(1, int(self.cache_workers))
        batch_size = max(1, int(self.cache_batch_size))

        worker_args = (
            batch_size,
            self.stockfish_path,
            int(self.stockfish_depth),
            int(self.multipv),
            float(self.policy_temperature),
            self.position_source,
            int(self.position_max_moves),
            float(self.position_temperature),
        )

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending = {
                executor.submit(_generate_supervised_batch_worker, *worker_args): worker_args
                for _ in range(worker_count)
            }
            while pending and filled < self.cache_size:
                for future in as_completed(list(pending)):
                    pending.pop(future, None)
                    states, policies, values = future.result()
                    if len(states) > 0:
                        take = min(len(states), self.cache_size - filled)
                        states_mm[filled:filled + take] = states[:take].astype(dtype, copy=False)
                        policies_mm[filled:filled + take] = policies[:take].astype(dtype, copy=False)
                        values_mm[filled:filled + take] = values[:take].astype(dtype, copy=False)
                        filled += take
                        pbar.update(take)
                    if filled < self.cache_size:
                        pending[executor.submit(_generate_supervised_batch_worker, *worker_args)] = worker_args
                    if filled >= self.cache_size:
                        for pending_future in pending:
                            pending_future.cancel()
                        pending.clear()
                        break

        pbar.close()
        states_mm.flush()
        policies_mm.flush()
        values_mm.flush()
        elapsed = max(0.001, time.time() - start_time)
        self.cache_build_seconds = elapsed
        self.cache_build_throughput = self.cache_size / elapsed
        self.cache_build_workers = worker_count
        self.cache_build_batch_size = batch_size

    def _ensure_cache(self):
        """Load cache or build it if missing/mismatched."""
        if not self.cache_dir or self.cache_size <= 0:
            self.cache_enabled = False
            return
        if not self.cache_rebuild and self._load_cache():
            return
        self._build_cache()

    def _sample_from_cache(self, batch_size: int, augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a training batch from the cached dataset."""
        if self.cache_states is None or self.cache_count <= 0:
            return np.empty((0, *self.config.input_shape), dtype=np.float32), np.empty(
                (0, self.move_encoder.policy_size), dtype=np.float32), np.empty((0,), dtype=np.float32)

        indices = np.random.randint(0, self.cache_count, size=batch_size)
        states = np.array(self.cache_states[indices], dtype=np.float32, copy=False)
        policies = np.array(self.cache_policies[indices], dtype=np.float32, copy=False)
        values = np.array(self.cache_values[indices], dtype=np.float32, copy=False)

        if augment:
            mirror_mask = np.random.random(len(states)) < 0.5
            for i in np.where(mirror_mask)[0]:
                states[i] = mirror_state(states[i])
                policies[i] = mirror_policy(policies[i], self.move_encoder)

        return states, policies, values

    @staticmethod
    def _sample_move_static(move_probs: List[Tuple[chess.Move, float]]) -> Optional[chess.Move]:
        """Sample a move from a list of (move, prob)."""
        if not move_probs:
            return None
        moves, probs = zip(*move_probs)
        probs = np.array(probs, dtype=np.float32)
        probs = probs / probs.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    def _sample_move(self, move_probs: List[Tuple[chess.Move, float]]) -> Optional[chess.Move]:
        return self._sample_move_static(move_probs)

    def generate_random_position(self, max_moves: int = 40) -> Tuple[chess.Board, List[chess.Board]]:
        """Generate a random legal chess position by playing random moves.

        Args:
            max_moves: Maximum number of random moves to play.

        Returns:
            Tuple of (board, history) with history containing recent boards.
        """
        return _generate_random_position(max_moves)

    def generate_stockfish_position(self, max_moves: int = 40) -> Tuple[chess.Board, List[chess.Board]]:
        """Generate a position by playing Stockfish moves with sampling."""
        return _generate_stockfish_position(
            self.stockfish,
            self.multipv,
            self.position_temperature,
            max_moves,
        )

    def generate_training_batch(
        self,
        batch_size: int,
        max_position_moves: int = 40,
        show_progress: bool = False,
        augment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of training examples from sampled positions.

        Args:
            batch_size: Number of examples to generate.
            max_position_moves: Maximum moves for random position generation.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (states, policies, values).
        """
        if self.cache_enabled and self.cache_states is not None:
            return self._sample_from_cache(batch_size, augment=augment)

        states = []
        policies = []
        values = []

        iterator = range(batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating supervised data")

        for _ in iterator:
            # Generate random position
            if self.position_source == "stockfish":
                board, history = self.generate_stockfish_position(max_position_moves)
            else:
                board, history = self.generate_random_position(max_position_moves)

            # Skip terminal positions
            if board.is_game_over():
                continue

            # Get Stockfish policy and evaluation
            move_probs, evaluation = self.stockfish.get_policy_and_value(
                board,
                multipv=self.multipv,
                policy_temperature=self.policy_temperature,
            )

            if not move_probs:
                continue

            # Encode state
            state = BoardEncoder.encode(board, history=history)

            # Create policy from Stockfish MultiPV
            policy = np.zeros(self.move_encoder.policy_size, dtype=np.float32)
            for move, prob in move_probs:
                try:
                    move_idx = self.move_encoder.encode(move)
                    policy[move_idx] = prob
                except KeyError:
                    continue

            if policy.sum() <= 0:
                continue
            policy /= policy.sum()

            states.append(state)
            policies.append(policy)
            values.append(evaluation)

        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        if augment and len(states) > 0:
            mirror_mask = np.random.random(len(states)) < 0.5
            for i in np.where(mirror_mask)[0]:
                states[i] = mirror_state(states[i])
                policies[i] = mirror_policy(policies[i], self.move_encoder)

        return states, policies, values

    def train_step(self, batch_size: int = 256) -> dict:
        """Run one training step.

        Args:
            batch_size: Number of examples per step.

        Returns:
            Dictionary with loss values.
        """
        states, policies, values = self.generate_training_batch(
            batch_size,
            max_position_moves=self.position_max_moves,
            augment=self.augment,
        )

        if len(states) == 0:
            return {"total_loss": 0, "policy_loss": 0, "value_loss": 0}

        return self.network.train_on_batch(states, policies, values)

    def train(
        self,
        num_iterations: int = 100,
        batch_size: int = 256,
        batches_per_iteration: int = 50,
        show_progress: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> List[dict]:
        """Run supervised training.

        Args:
            num_iterations: Number of training iterations.
            batch_size: Batch size for training.
            batches_per_iteration: Number of batches per iteration.
            show_progress: Whether to show progress.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_interval: Save checkpoint every N iterations.

        Returns:
            List of training statistics per iteration.
        """
        history = []

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for iteration in range(1, num_iterations + 1):
            if show_progress:
                print(f"\nSupervised Iteration {iteration}/{num_iterations}")

            iteration_losses = []

            iterator = range(batches_per_iteration)
            if show_progress:
                iterator = tqdm(iterator, desc="Training batches")

            for _ in iterator:
                loss = self.train_step(batch_size)
                iteration_losses.append(loss)

            # Average losses
            avg_loss = {
                "total_loss": np.mean([l["total_loss"] for l in iteration_losses]),
                "policy_loss": np.mean([l["policy_loss"] for l in iteration_losses]),
                "value_loss": np.mean([l["value_loss"] for l in iteration_losses]),
            }

            history.append(avg_loss)

            if show_progress:
                print(f"  Loss: total={avg_loss['total_loss']:.4f}, "
                      f"policy={avg_loss['policy_loss']:.4f}, "
                      f"value={avg_loss['value_loss']:.4f}")

            # Save checkpoint
            if checkpoint_dir and iteration % checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"supervised_iter_{iteration}")
                self.network.save(path)
                if show_progress:
                    print(f"  Checkpoint saved: {path}")

        return history

    def close(self):
        """Clean up resources."""
        self.stockfish.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
