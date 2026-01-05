"""Board encoder: converts chess.Board to 781-dimensional input vector."""

import numpy as np
import chess


class BoardEncoder:
    """Encodes a chess board state into a 781-dimensional flat vector.

    Layout:
        0-63:    White Pawns bitboard
        64-127:  White Knights bitboard
        128-191: White Bishops bitboard
        192-255: White Rooks bitboard
        256-319: White Queens bitboard
        320-383: White King bitboard
        384-447: Black Pawns bitboard
        448-511: Black Knights bitboard
        512-575: Black Bishops bitboard
        576-639: Black Rooks bitboard
        640-703: Black Queens bitboard
        704-767: Black King bitboard
        768:     White kingside castling right
        769:     White queenside castling right
        770:     Black kingside castling right
        771:     Black queenside castling right
        772-779: En passant file (one-hot, 8 values)
        780:     Side to move (1=white, 0=black)
    """

    INPUT_SIZE = 781

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
    def encode(board: chess.Board) -> np.ndarray:
        """Encode a chess board into a 781-dimensional vector.

        Args:
            board: A python-chess Board object.

        Returns:
            A numpy array of shape (781,) with float32 values (0 or 1).
        """
        encoding = np.zeros(781, dtype=np.float32)

        # Encode piece bitboards (12 * 64 = 768 values)
        for i, (color, piece_type) in enumerate(BoardEncoder.PIECE_ORDER):
            bitboard = board.pieces_mask(piece_type, color)
            offset = i * 64
            for square in range(64):
                if bitboard & (1 << square):
                    encoding[offset + square] = 1.0

        # Encode castling rights (4 values at positions 768-771)
        encoding[768] = float(board.has_kingside_castling_rights(chess.WHITE))
        encoding[769] = float(board.has_queenside_castling_rights(chess.WHITE))
        encoding[770] = float(board.has_kingside_castling_rights(chess.BLACK))
        encoding[771] = float(board.has_queenside_castling_rights(chess.BLACK))

        # Encode en passant file (8 values at positions 772-779, one-hot)
        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            encoding[772 + ep_file] = 1.0

        # Encode side to move (1 value at position 780)
        encoding[780] = float(board.turn == chess.WHITE)

        return encoding

    @staticmethod
    def encode_batch(boards: list) -> np.ndarray:
        """Encode multiple boards into a batch.

        Args:
            boards: List of python-chess Board objects.

        Returns:
            A numpy array of shape (len(boards), 781).
        """
        return np.array([BoardEncoder.encode(b) for b in boards], dtype=np.float32)
