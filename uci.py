#!/usr/bin/env python3
"""UCI (Universal Chess Interface) implementation for Chess RL engine.

This allows the engine to be used with chess GUIs like Arena, CuteChess, etc.

Usage:
    python uci.py [--model PATH] [--simulations N]

The engine reads UCI commands from stdin and writes responses to stdout.
"""

import sys
import os
import argparse
import chess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.game.chess_game import ChessGame
from src.mcts.batched_mcts import BatchedMCTS


class UCIEngine:
    """UCI-compatible chess engine using MCTS and neural network."""

    ENGINE_NAME = "ChessRL"
    ENGINE_AUTHOR = "Chess RL Project"

    def __init__(self, model_path: str = None, num_simulations: int = 400):
        """Initialize the UCI engine.

        Args:
            model_path: Path to model weights. If None, uses random initialization.
            num_simulations: Number of MCTS simulations per move.
        """
        self.config = Config()
        self.num_simulations = num_simulations
        self.debug = False

        self.net = ChessTransformer(self.config)
        # Build once so weights can load.
        import chess as _chess
        from src.game.token_encoder import encode_batch as _eb
        import tensorflow as _tf
        _sq, _sf = _eb([_chess.Board()], [0])
        self.net(_tf.constant(_sq), _tf.constant(_sf))
        if model_path:
            wpath = model_path if model_path.endswith(".weights.h5") else model_path + ".weights.h5"
            if os.path.exists(wpath):
                try:
                    self.net.load_weights(wpath)
                    self._debug(f"Loaded model from {wpath}")
                except Exception as e:
                    self._debug(f"Failed to load model: {e}")

        self.evaluator = TransformerEvaluator(self.net, use_fp16=True)
        self.game = ChessGame()
        self.mcts = BatchedMCTS(self.evaluator, self.config, self.num_simulations)

    def _debug(self, message: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"info string {message}", flush=True)

    def _send(self, message: str):
        """Send a message to the GUI."""
        print(message, flush=True)

    def uci(self):
        """Handle 'uci' command - identify the engine."""
        self._send(f"id name {self.ENGINE_NAME}")
        self._send(f"id author {self.ENGINE_AUTHOR}")
        # UCI options
        self._send(f"option name Simulations type spin default {self.num_simulations} min 10 max 10000")
        self._send("option name Debug type check default false")
        self._send("uciok")

    def isready(self):
        """Handle 'isready' command."""
        self._send("readyok")

    def ucinewgame(self):
        """Handle 'ucinewgame' command - reset for new game."""
        self.game = ChessGame()
        self._debug("New game started")

    def setoption(self, name: str, value: str):
        """Handle 'setoption' command.

        Args:
            name: Option name.
            value: Option value.
        """
        name_lower = name.lower()
        if name_lower == "simulations":
            try:
                self.num_simulations = int(value)
                self.mcts.num_simulations = self.num_simulations
                self._debug(f"Simulations set to {self.num_simulations}")
            except ValueError:
                pass
        elif name_lower == "debug":
            self.debug = value.lower() == "true"

    def position(self, args: list):
        """Handle 'position' command - set up a position.

        Args:
            args: Position arguments (startpos/fen and optional moves).
        """
        if not args:
            return

        # Parse position
        if args[0] == "startpos":
            self.game = ChessGame()
            args = args[1:]
        elif args[0] == "fen":
            # Find where FEN ends (either at "moves" or end of args)
            fen_parts = []
            i = 1
            while i < len(args) and args[i] != "moves":
                fen_parts.append(args[i])
                i += 1
            fen = " ".join(fen_parts)
            self.game = ChessGame(fen=fen)
            args = args[i:]

        # Apply moves if present
        if args and args[0] == "moves":
            for move_uci in args[1:]:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.game.board.legal_moves:
                        self.game.apply_move(move)
                        self.mcts.advance(move)
                except ValueError:
                    self._debug(f"Invalid move: {move_uci}")

    def get_best_move_uci(self) -> str:
        if self.game.is_terminal():
            return "0000"
        move = self.mcts.choose_move(self.game.board, temperature=0.0)
        if move is None:
            return "0000"
        return move.uci()

    def go(self, args: list):
        """Handle 'go' command - start calculating.

        Args:
            args: Go parameters (wtime, btime, winc, binc, movetime, etc.)
        """
        sims = self.num_simulations
        i = 0
        while i < len(args):
            if args[i] == "movetime" and i + 1 < len(args):
                try:
                    sims = max(10, min(int(args[i + 1]) // 2, self.num_simulations))
                except ValueError:
                    pass
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                try:
                    sims = max(10, int(args[i + 1]) * 50)
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        self.mcts.num_simulations = sims
        self._send(f"bestmove {self.get_best_move_uci()}")

    def quit(self):
        """Handle 'quit' command."""
        sys.exit(0)

    def run(self):
        """Main UCI loop - read commands and process them."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            parts = line.split()
            command = parts[0].lower()
            args = parts[1:]

            if command == "uci":
                self.uci()
            elif command == "isready":
                self.isready()
            elif command == "ucinewgame":
                self.ucinewgame()
            elif command == "setoption":
                # Parse "setoption name X value Y"
                name = ""
                value = ""
                if "name" in args:
                    name_idx = args.index("name") + 1
                    if "value" in args:
                        value_idx = args.index("value")
                        name = " ".join(args[name_idx:value_idx])
                        value = " ".join(args[value_idx + 1:])
                    else:
                        name = " ".join(args[name_idx:])
                self.setoption(name, value)
            elif command == "position":
                self.position(args)
            elif command == "go":
                self.go(args)
            elif command == "stop":
                # We don't support pondering, so stop is a no-op
                pass
            elif command == "quit":
                self.quit()
            elif command == "d":
                # Debug: print board
                print(self.game.board, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Chess RL UCI Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_final",
        help="Path to model weights",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move",
    )
    args = parser.parse_args()

    engine = UCIEngine(args.model, args.simulations)
    engine.run()


if __name__ == "__main__":
    main()
