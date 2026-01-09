"""Board encoder: converts chess.Board to an 8x8xN plane representation."""

import numpy as np
import chess
from typing import Optional

from config import Config


class BoardEncoder:
    """Encodes a chess board state into an 8x8xN stack of planes.

    Planes:
        - 12 planes per historical position (piece bitboards)
        - 4 castling planes
        - 1 side-to-move plane
        - 1 repetition plane
        - 1 move-count plane (normalized)
        - 1 en passant plane (single square, optional)
    """

    INPUT_SHAPE = Config.input_shape

    # Piece types in order for encoding
    PIECE_ORDER = [
        (chess.WHITE, chess.PAWN),
        (chess.WHITE, chess.KNIGHT),
        (chess.WHITE, chess.BISHOP),
        (chess.WHITE, chess.ROOK),
        (chess.WHITE, chess.QUEEN),
        (chess.WHITE, chess.KING),
        (chess.BLACK, chess.PAWN),
        (chess.BLACK, chess.KNIGHT),
        (chess.BLACK, chess.BISHOP),
        (chess.BLACK, chess.ROOK),
        (chess.BLACK, chess.QUEEN),
        (chess.BLACK, chess.KING),
    ]

    @staticmethod
    def _encode_pieces(planes: np.ndarray, board: chess.Board, offset: int) -> None:
        """Encode piece planes for a single board into the plane stack."""
        for i, (color, piece_type) in enumerate(BoardEncoder.PIECE_ORDER):
            plane_idx = offset + i
            for square in board.pieces(piece_type, color):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                planes[plane_idx, rank, file] = 1.0

    @staticmethod
    def encode(
        board: chess.Board,
        history: list = None,
        move_count: Optional[int] = None,
    ) -> np.ndarray:
        """Encode a chess board into an 8x8xN tensor (channels last).

        Args:
            board: A python-chess Board object (current position).
            history: Optional list of past boards, oldest->newest (including current).
            move_count: Move count (plies) for normalization.

        Returns:
            A numpy array of shape (8, 8, planes) with float32 values.
        """
        planes = np.zeros((Config.input_planes, 8, 8), dtype=np.float32)

        history_boards = history or [board]
        history_boards = history_boards[-Config.history_length:]

        # Encode current position and history (most recent first)
        for h in range(Config.history_length):
            if h >= len(history_boards):
                break
            hist_board = history_boards[-1 - h]
            BoardEncoder._encode_pieces(planes, hist_board, offset=h * 12)

        offset = 12 * Config.history_length

        # Castling rights
        planes[offset + 0, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[offset + 1, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[offset + 2, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[offset + 3, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

        # Side to move
        planes[offset + 4, :, :] = float(board.turn == chess.WHITE)

        # Repetition flag (threefold claim)
        planes[offset + 5, :, :] = float(board.can_claim_threefold_repetition())

        # Move count plane (normalized)
        if move_count is None:
            move_count = max(0, (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1))
        planes[offset + 6, :, :] = min(float(move_count) / float(Config.max_moves), 1.0)

        # En passant square (optional)
        if Config.include_ep_plane:
            ep_plane = offset + 7
            if board.ep_square is not None:
                rank = chess.square_rank(board.ep_square)
                file = chess.square_file(board.ep_square)
                planes[ep_plane, rank, file] = 1.0

        # Convert to channels-last (8, 8, planes)
        return np.transpose(planes, (1, 2, 0))

    @staticmethod
    def encode_batch(boards: list) -> np.ndarray:
        """Encode multiple boards into a batch (no history)."""
        return np.array([BoardEncoder.encode(b) for b in boards], dtype=np.float32)
