#!/usr/bin/env python3
"""Play against the trained Chess RL model."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from config import Config
from src.model.network import ChessNetwork
from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS


def play_game(network, player_color: bool, num_simulations: int = 400):
    """Play a game against the model.

    Args:
        network: Trained neural network.
        player_color: True if player is white, False if black.
        num_simulations: MCTS simulations per move.
    """
    game = ChessGame()
    mcts = MCTS(network, num_simulations=num_simulations)

    print("\nChess RL - Play against AI")
    print("=" * 40)
    print(f"You are playing as {'White' if player_color else 'Black'}")
    print("Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("Type 'quit' to exit, 'board' to show board")
    print("=" * 40)

    while not game.is_terminal():
        print(f"\n{game.board}")
        print(f"\nMove {game.move_count + 1}")

        is_player_turn = game.turn == player_color

        if is_player_turn:
            # Player's turn
            while True:
                move_str = input("Your move: ").strip().lower()

                if move_str == "quit":
                    print("Game ended by player.")
                    return
                elif move_str == "board":
                    print(f"\n{game.board}")
                    continue

                try:
                    move = chess.Move.from_uci(move_str)
                    if move in game.board.legal_moves:
                        game.apply_move(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid move format. Use UCI (e.g., e2e4)")
        else:
            # AI's turn
            print("AI is thinking...")
            action, policy, _ = mcts.get_action(game, temperature=0.1)

            if action < 0:
                print("AI has no legal moves.")
                break

            move = game.move_encoder.decode(action)

            # Handle queen promotion
            if game.board.piece_at(move.from_square):
                piece = game.board.piece_at(move.from_square)
                if piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(move.to_square)
                    if to_rank == 0 or to_rank == 7:
                        move = chess.Move(move.from_square, move.to_square, chess.QUEEN)

            san = game.board.san(move)
            print(f"AI plays: {san} ({move.uci()})")
            game.apply_move(move)

    # Game over
    print(f"\n{game.board}")
    print("\n" + "=" * 40)
    outcome = game.get_outcome()
    if outcome > 0:
        result = "White wins!"
    elif outcome < 0:
        result = "Black wins!"
    else:
        result = "Draw!"
    print(f"Game Over: {result}")


def watch_game(network, num_simulations: int = 400, max_moves: int = 200):
    """Watch the AI play against itself.

    Args:
        network: Trained neural network.
        num_simulations: MCTS simulations per move.
        max_moves: Maximum moves before stopping.
    """
    game = ChessGame()
    mcts = MCTS(network, num_simulations=num_simulations)

    print("\nChess RL - Self-Play Demo")
    print("=" * 40)

    move_history = []

    while not game.is_terminal() and game.move_count < max_moves:
        action, _, _ = mcts.get_action(game, temperature=0.1)

        if action < 0:
            break

        move = game.move_encoder.decode(action)

        # Handle queen promotion
        if game.board.piece_at(move.from_square):
            piece = game.board.piece_at(move.from_square)
            if piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(move.to_square)
                if to_rank == 0 or to_rank == 7:
                    move = chess.Move(move.from_square, move.to_square, chess.QUEEN)

        san = game.board.san(move)
        move_history.append(san)

        turn = "White" if game.turn else "Black"
        print(f"{game.move_count + 1}. {turn}: {san}")

        game.apply_move(move)

    print(f"\n{game.board}")
    print("\n" + "=" * 40)

    outcome = game.get_outcome()
    if outcome > 0:
        result = "White wins!"
    elif outcome < 0:
        result = "Black wins!"
    else:
        result = "Draw!"

    print(f"Game Over: {result}")
    print(f"\nMoves: {' '.join(move_history)}")


def main():
    parser = argparse.ArgumentParser(description="Play against Chess RL model")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_final",
        help="Path to model weights",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Player color",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch AI play against itself instead of playing",
    )
    args = parser.parse_args()

    # Load model
    config = Config()
    network = ChessNetwork(config)
    network.compile()

    weights_path = args.model if args.model.endswith(".weights.h5") else args.model + ".weights.h5"
    if os.path.exists(weights_path):
        network.load(args.model)
        print(f"Loaded model from {weights_path}")
    else:
        print(f"Warning: Model not found at {args.model}")
        print("Playing with randomly initialized network")

    if args.watch:
        watch_game(network, args.simulations)
    else:
        player_color = args.color == "white"
        play_game(network, player_color, args.simulations)


if __name__ == "__main__":
    main()
