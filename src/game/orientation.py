"""Canonical (side-to-move) orientation helpers."""

import chess


def canonical_board(board: chess.Board) -> chess.Board:
    """Return the board from the side-to-move's perspective.

    python-chess Board.mirror() swaps colors and flips vertically, so the
    side-to-move is always White on the canonical board.
    """
    return board if board.turn == chess.WHITE else board.mirror()


def to_canonical_move(move: chess.Move, turn: bool) -> chess.Move:
    """Map a real move to the canonical frame (mirror squares if Black to move)."""
    if turn == chess.WHITE:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def from_canonical_move(move: chess.Move, turn: bool) -> chess.Move:
    """Map a canonical-frame move back to the real frame (mirror is its own inverse)."""
    return to_canonical_move(move, turn)
