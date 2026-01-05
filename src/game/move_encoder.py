"""Move encoder: maps chess moves to indices and vice versa."""

import chess
import numpy as np


class MoveEncoder:
    """Encodes chess moves to indices (0 to NUM_MOVES-1) and decodes back.

    The encoding covers all possible moves in chess:
    - Queen-like moves (sliding in 8 directions, up to 7 squares)
    - Knight moves (8 possible L-shaped moves)
    - Pawn underpromotions (knight, bishop, rook - queen is default)

    This uses a (from_square, move_type) encoding scheme.
    """

    # 8 directions for queen-like moves: N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [
        (0, 1),    # N
        (1, 1),    # NE
        (1, 0),    # E
        (1, -1),   # SE
        (0, -1),   # S
        (-1, -1),  # SW
        (-1, 0),   # W
        (-1, 1),   # NW
    ]

    # Knight move offsets
    KNIGHT_MOVES = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    # Underpromotion pieces (queen promotion is encoded as regular move)
    UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    def __init__(self):
        """Initialize move encoder with lookup tables."""
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build move-to-index and index-to-move lookup tables."""
        self.move_to_idx = {}
        self.idx_to_move = {}
        idx = 0

        # Queen-like moves: 8 directions × 7 distances × 64 from-squares
        for from_sq in range(64):
            from_file = chess.square_file(from_sq)
            from_rank = chess.square_rank(from_sq)

            for dir_idx, (df, dr) in enumerate(self.DIRECTIONS):
                for distance in range(1, 8):
                    to_file = from_file + df * distance
                    to_rank = from_rank + dr * distance

                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        to_sq = chess.square(to_file, to_rank)
                        move = chess.Move(from_sq, to_sq)
                        if move not in self.move_to_idx:
                            self.move_to_idx[move] = idx
                            self.idx_to_move[idx] = move
                            idx += 1

        # Knight moves: 8 moves × 64 from-squares
        for from_sq in range(64):
            from_file = chess.square_file(from_sq)
            from_rank = chess.square_rank(from_sq)

            for df, dr in self.KNIGHT_MOVES:
                to_file = from_file + df
                to_rank = from_rank + dr

                if 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_sq = chess.square(to_file, to_rank)
                    move = chess.Move(from_sq, to_sq)
                    if move not in self.move_to_idx:
                        self.move_to_idx[move] = idx
                        self.idx_to_move[idx] = move
                        idx += 1

        # Underpromotions: from 7th/2nd rank pawn moves to 8th/1st rank
        # White promotions (rank 6 to rank 7)
        for from_file in range(8):
            from_sq = chess.square(from_file, 6)  # 7th rank (0-indexed: 6)

            # Forward promotion
            to_sq = chess.square(from_file, 7)
            for promo in self.UNDERPROMOTIONS:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if move not in self.move_to_idx:
                    self.move_to_idx[move] = idx
                    self.idx_to_move[idx] = move
                    idx += 1

            # Capture promotions (left and right)
            for df in [-1, 1]:
                to_file = from_file + df
                if 0 <= to_file < 8:
                    to_sq = chess.square(to_file, 7)
                    for promo in self.UNDERPROMOTIONS:
                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        if move not in self.move_to_idx:
                            self.move_to_idx[move] = idx
                            self.idx_to_move[idx] = move
                            idx += 1

        # Black promotions (rank 1 to rank 0)
        for from_file in range(8):
            from_sq = chess.square(from_file, 1)  # 2nd rank (0-indexed: 1)

            # Forward promotion
            to_sq = chess.square(from_file, 0)
            for promo in self.UNDERPROMOTIONS:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if move not in self.move_to_idx:
                    self.move_to_idx[move] = idx
                    self.idx_to_move[idx] = move
                    idx += 1

            # Capture promotions (left and right)
            for df in [-1, 1]:
                to_file = from_file + df
                if 0 <= to_file < 8:
                    to_sq = chess.square(to_file, 0)
                    for promo in self.UNDERPROMOTIONS:
                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        if move not in self.move_to_idx:
                            self.move_to_idx[move] = idx
                            self.idx_to_move[idx] = move
                            idx += 1

        self.num_moves = idx

    @property
    def policy_size(self) -> int:
        """Return the total number of possible moves."""
        return self.num_moves

    def encode(self, move: chess.Move) -> int:
        """Encode a chess move to an index.

        Args:
            move: A python-chess Move object.

        Returns:
            Index in range [0, num_moves).

        Raises:
            KeyError: If the move is not in the encoding.
        """
        # Handle queen promotions as regular moves (no promotion flag)
        if move.promotion == chess.QUEEN:
            move = chess.Move(move.from_square, move.to_square)
        return self.move_to_idx[move]

    def decode(self, idx: int) -> chess.Move:
        """Decode an index to a chess move.

        Args:
            idx: Index in range [0, num_moves).

        Returns:
            A python-chess Move object.
        """
        return self.idx_to_move[idx]

    def encode_legal_moves(self, board: chess.Board) -> np.ndarray:
        """Create a mask of legal moves for the current position.

        Args:
            board: A python-chess Board object.

        Returns:
            A numpy array of shape (num_moves,) with 1s for legal moves, 0s otherwise.
        """
        mask = np.zeros(self.num_moves, dtype=np.float32)
        for move in board.legal_moves:
            try:
                idx = self.encode(move)
                mask[idx] = 1.0
            except KeyError:
                # Move not in encoding (shouldn't happen with correct encoding)
                pass
        return mask

    def get_legal_move_indices(self, board: chess.Board) -> list:
        """Get indices of all legal moves for the current position.

        Args:
            board: A python-chess Board object.

        Returns:
            List of indices for legal moves.
        """
        indices = []
        for move in board.legal_moves:
            try:
                idx = self.encode(move)
                indices.append(idx)
            except KeyError:
                pass
        return indices


# Global instance for convenience
_encoder = None


def get_move_encoder() -> MoveEncoder:
    """Get the global MoveEncoder instance (singleton)."""
    global _encoder
    if _encoder is None:
        _encoder = MoveEncoder()
    return _encoder
