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
from src.model.network import ChessNetwork
from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS


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

        # Initialize network
        self.network = ChessNetwork(self.config)
        self.network.compile()

        # Load model if provided
        weights_path = model_path + ".weights.h5" if model_path and not model_path.endswith(".weights.h5") else model_path
        if model_path and os.path.exists(weights_path):
            try:
                self.network.load(model_path)
                self._debug(f"Loaded model from {model_path}")
            except Exception as e:
                self._debug(f"Failed to load model: {e}")

        # Game state
        self.game = ChessGame()
        self.mcts = MCTS(self.network, self.config, self.num_simulations)

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
                self.mcts = MCTS(self.network, self.config, self.num_simulations)
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
                except ValueError:
                    self._debug(f"Invalid move: {move_uci}")

    def go(self, args: list):
        """Handle 'go' command - start calculating.

        Args:
            args: Go parameters (wtime, btime, winc, binc, movetime, etc.)
        """
        # Parse time controls (simplified - just use fixed simulations for now)
        movetime = None
        depth = None

        i = 0
        while i < len(args):
            if args[i] == "movetime" and i + 1 < len(args):
                try:
                    movetime = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                try:
                    depth = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        # Adjust simulations based on time (rough heuristic)
        sims = self.num_simulations
        if movetime:
            # Scale simulations with time (assume ~1ms per simulation)
            sims = max(10, min(movetime // 2, self.num_simulations))
        if depth:
            # Use depth as a proxy for simulations
            sims = max(10, depth * 50)

        # Create MCTS with adjusted simulations
        mcts = MCTS(self.network, self.config, sims)

        # Check if game is already over
        if self.game.is_terminal():
            self._send("bestmove 0000")
            return

        # Get best move
        action, policy = mcts.get_action(self.game, temperature=0.1, add_noise=False)

        if action < 0:
            self._send("bestmove 0000")
            return

        # Decode move
        move = self.game.move_encoder.decode(action)

        # Handle queen promotion
        piece = self.game.board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if to_rank == 0 or to_rank == 7:
                if move.promotion is None:
                    move = chess.Move(move.from_square, move.to_square, chess.QUEEN)

        # Send best move
        self._send(f"bestmove {move.uci()}")

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
