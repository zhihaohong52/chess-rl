"""Chess game wrapper around python-chess."""

import chess
import numpy as np
from typing import Optional, List, Tuple
from .board_encoder import BoardEncoder
from .move_encoder import MoveEncoder, get_move_encoder


class ChessGame:
    """Wrapper around python-chess for RL training."""

    def __init__(self, fen: Optional[str] = None):
        """Initialize a new chess game.

        Args:
            fen: Optional FEN string to initialize from. If None, starts from
                 the standard starting position.
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.move_encoder = get_move_encoder()
        self.move_history: List[chess.Move] = []

    def clone(self) -> "ChessGame":
        """Create a deep copy of the game."""
        game = ChessGame()
        game.board = self.board.copy()
        game.move_history = self.move_history.copy()
        return game

    def get_state(self) -> np.ndarray:
        """Get the current board state as a 781-dimensional vector."""
        return BoardEncoder.encode(self.board)

    def get_legal_moves(self) -> List[chess.Move]:
        """Get list of legal moves in the current position."""
        return list(self.board.legal_moves)

    def get_legal_move_mask(self) -> np.ndarray:
        """Get a mask of legal moves (1 for legal, 0 for illegal)."""
        return self.move_encoder.encode_legal_moves(self.board)

    def get_legal_move_indices(self) -> List[int]:
        """Get indices of legal moves."""
        return self.move_encoder.get_legal_move_indices(self.board)

    def apply_move(self, move: chess.Move) -> None:
        """Apply a move to the board.

        Args:
            move: A python-chess Move object.
        """
        self.move_history.append(move)
        self.board.push(move)

    def apply_move_index(self, move_idx: int) -> None:
        """Apply a move by its index.

        Args:
            move_idx: Index of the move to apply.
        """
        move = self.move_encoder.decode(move_idx)
        # Handle queen promotion (decode returns move without promotion)
        if self._is_pawn_promotion(move):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        self.apply_move(move)

    def _is_pawn_promotion(self, move: chess.Move) -> bool:
        """Check if a move is a pawn promotion."""
        piece = self.board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        to_rank = chess.square_rank(move.to_square)
        return to_rank == 0 or to_rank == 7

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()

    def get_outcome(self) -> float:
        """Get the game outcome from white's perspective.

        Returns:
            1.0 if white wins, -1.0 if black wins, 0.0 for draw.
            Returns 0.0 if game is not over.
        """
        if not self.board.is_game_over():
            return 0.0

        result = self.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    def get_outcome_for_current_player(self) -> float:
        """Get the game outcome from the current player's perspective.

        Returns:
            1.0 if current player wins, -1.0 if opponent wins, 0.0 for draw.
        """
        outcome = self.get_outcome()
        if self.board.turn == chess.BLACK:
            outcome = -outcome
        return outcome

    @property
    def turn(self) -> bool:
        """Return whose turn it is (True=White, False=Black)."""
        return self.board.turn

    @property
    def move_count(self) -> int:
        """Return the number of moves played."""
        return len(self.move_history)

    @property
    def fen(self) -> str:
        """Return the FEN string of the current position."""
        return self.board.fen()

    def __str__(self) -> str:
        """Return a string representation of the board."""
        return str(self.board)

    def __repr__(self) -> str:
        return f"ChessGame(fen='{self.fen}')"
