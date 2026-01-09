"""Stockfish integration for evaluation and move generation."""

import chess
import chess.engine
import numpy as np
from typing import Optional, Tuple, List
import os


class StockfishEvaluator:
    """Interface to Stockfish for position evaluation and move generation."""

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 10,
        time_limit: float = 0.1,
        multipv: int = 1,
        limit_strength: Optional[bool] = None,
        elo: Optional[int] = None,
        skill_level: Optional[int] = None,
        hash_mb: Optional[int] = None,
        threads: Optional[int] = None,
    ):
        """Initialize Stockfish evaluator.

        Args:
            stockfish_path: Path to Stockfish executable. If None, tries common locations.
            depth: Search depth for evaluation.
            time_limit: Time limit per move in seconds.
            multipv: Number of candidate moves to return for policy targets.
            limit_strength: Enable UCI_LimitStrength if supported.
            elo: Target UCI_Elo if supported.
            skill_level: Target Skill Level if supported.
            hash_mb: Hash table size in MB (larger = faster for deep searches).
            threads: Number of threads for Stockfish (1 recommended when using multiprocessing).
        """
        self.depth = depth
        self.time_limit = time_limit
        self.multipv = multipv

        # Find Stockfish
        if stockfish_path is None:
            stockfish_path = self._find_stockfish()

        if stockfish_path is None:
            raise RuntimeError(
                "Stockfish not found. Install it or provide path.\n"
                "Colab: !apt-get install stockfish\n"
                "Mac: brew install stockfish\n"
                "Windows: Download from stockfishchess.org"
            )

        self.stockfish_path = stockfish_path
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._configure_engine(limit_strength, elo, skill_level, hash_mb, threads)

    def _configure_engine(
        self,
        limit_strength: Optional[bool],
        elo: Optional[int],
        skill_level: Optional[int],
        hash_mb: Optional[int] = None,
        threads: Optional[int] = None,
    ) -> None:
        """Configure optional strength limits if supported by the engine."""
        options = self.engine.options
        config = {}
        if limit_strength is not None and "UCI_LimitStrength" in options:
            config["UCI_LimitStrength"] = limit_strength
        if elo is not None and "UCI_Elo" in options:
            config["UCI_Elo"] = int(elo)
        if skill_level is not None and "Skill Level" in options:
            config["Skill Level"] = int(skill_level)
        if hash_mb is not None and "Hash" in options:
            config["Hash"] = int(hash_mb)
        if threads is not None and "Threads" in options:
            config["Threads"] = int(threads)
        if config:
            self.engine.configure(config)

    def _find_stockfish(self) -> Optional[str]:
        """Try to find Stockfish in common locations."""
        common_paths = [
            "/usr/games/stockfish",           # Linux/Colab
            "/usr/local/bin/stockfish",       # Mac Homebrew
            "/opt/homebrew/bin/stockfish",    # Mac M1 Homebrew
            "stockfish",                       # In PATH
            "stockfish.exe",                   # Windows in PATH
            r"C:\stockfish\stockfish.exe",    # Windows common
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path
            # Try running it (might be in PATH)
            try:
                import shutil
                found = shutil.which(path)
                if found:
                    return found
            except (OSError, IOError):
                pass

        return None

    def evaluate(self, board: chess.Board) -> float:
        """Get Stockfish evaluation normalized to [-1, 1].

        Args:
            board: Chess board position.

        Returns:
            Evaluation from current player's perspective, normalized to [-1, 1].
        """
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit)
            )
            score = info["score"].relative

            if score.is_mate():
                mate_in = score.mate()
                if mate_in > 0:
                    return 1.0  # Winning
                else:
                    return -1.0  # Losing
            else:
                # Centipawns to [-1, 1] using tanh
                cp = score.score()
                return float(np.tanh(cp / 400))  # 400cp ≈ 0.76
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get Stockfish's best move.

        Args:
            board: Chess board position.

        Returns:
            Best move, or None if no legal moves.
        """
        if board.is_game_over():
            return None

        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit)
            )
            return result.move
        except Exception as e:
            print(f"Move generation error: {e}")
            return None

    def get_move_and_eval(self, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
        """Get best move and evaluation together (more efficient).

        Args:
            board: Chess board position.

        Returns:
            Tuple of (best_move, evaluation).
        """
        if board.is_game_over():
            return None, 0.0

        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit),
                info=chess.engine.INFO_SCORE
            )

            move = result.move
            eval_score = 0.0

            if result.info and "score" in result.info:
                score = result.info["score"].relative
                if score.is_mate():
                    eval_score = 1.0 if score.mate() > 0 else -1.0
                else:
                    eval_score = float(np.tanh(score.score() / 400))

            return move, eval_score
        except Exception as e:
            print(f"Error: {e}")
            return None, 0.0

    def play_game(self, board: chess.Board, our_color: bool) -> chess.Move:
        """Play a move as Stockfish (for curriculum training).

        Args:
            board: Current board position.
            our_color: The color Stockfish is playing (True=White).

        Returns:
            Stockfish's move.
        """
        return self.get_best_move(board)

    def _score_to_cp(self, score: chess.engine.PovScore) -> int:
        """Convert a python-chess score to centipawns with mate handling."""
        if score.is_mate():
            mate_in = score.mate()
            return 10000 if mate_in and mate_in > 0 else -10000
        cp = score.score()
        return int(cp) if cp is not None else 0

    def get_policy_and_value(
        self,
        board: chess.Board,
        multipv: Optional[int] = None,
        policy_temperature: float = 1.0,
    ) -> Tuple[List[Tuple[chess.Move, float]], float]:
        """Get a move distribution (MultiPV) and value for a position.

        Args:
            board: Chess position.
            multipv: Number of top moves to request.
            policy_temperature: Temperature for softmax over scores.

        Returns:
            Tuple of ([(move, prob), ...], value) where value is in [-1, 1].
        """
        if board.is_game_over():
            return [], 0.0

        multipv = multipv or self.multipv
        try:
            infos = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit),
                multipv=multipv,
            )
        except Exception:
            infos = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit),
            )

        if isinstance(infos, dict):
            infos = [infos]

        moves = []
        scores = []

        for info in infos:
            move = None
            if "pv" in info and info["pv"]:
                move = info["pv"][0]
            elif "move" in info:
                move = info["move"]

            if move is None or "score" not in info:
                continue

            cp = self._score_to_cp(info["score"].relative)
            moves.append(move)
            scores.append(cp)

        if not moves:
            return [], 0.0

        score_array = np.clip(np.array(scores, dtype=np.float32), -10000, 10000)
        denom = max(1e-3, 400.0 * max(policy_temperature, 1e-3))
        logits = score_array / denom
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()

        value = float(np.tanh(score_array[0] / 400.0))
        return list(zip(moves, probs.tolist())), value

    def set_depth(self, depth: int):
        """Change the search depth."""
        self.depth = depth

    def set_time_limit(self, time_limit: float):
        """Change the time limit per move."""
        self.time_limit = time_limit

    def close(self):
        """Close the Stockfish engine."""
        try:
            self.engine.quit()
        except (chess.engine.EngineTerminatedError, OSError, AttributeError):
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
