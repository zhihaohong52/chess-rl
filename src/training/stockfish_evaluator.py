"""Stockfish integration for evaluation and move generation."""

import chess
import chess.engine
import numpy as np
from typing import Optional, Tuple, List
import os


class StockfishEvaluator:
    """Interface to Stockfish for position evaluation and best move generation."""

    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 10, time_limit: float = 0.1):
        """Initialize Stockfish evaluator.

        Args:
            stockfish_path: Path to Stockfish executable. If None, tries common locations.
            depth: Search depth for evaluation.
            time_limit: Time limit per move in seconds.
        """
        self.depth = depth
        self.time_limit = time_limit

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

        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

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
            except:
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
        except:
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
