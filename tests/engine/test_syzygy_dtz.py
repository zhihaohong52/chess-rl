import chess

from src.engine.syzygy_tb import SyzygyTablebase

KRK = "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"  # white K+R vs k, white to move
# white K+Q vs k + pawn; the only capture (Qxd4) is a winning zeroing move.
KQKP = "8/8/8/3k4/3p4/8/3QK3/8 w - - 0 1"


class DTZFake:
    """Maps a child FEN -> (wdl, dtz) from the child's (opponent's) POV.

    Unlisted positions return `default` (a draw by default), so a test only needs
    to populate the moves it cares about.
    """

    def __init__(self, table=None, default=(0, 0)):
        self.table = table or {}
        self.default = default

    def get_wdl(self, board, default=None):
        return self.table.get(board.fen(), self.default)[0]

    def get_dtz(self, board, default=None):
        return self.table.get(board.fen(), self.default)[1]

    def close(self):
        pass


def _child_fen(board, move):
    board.push(move)
    fen = board.fen()
    board.pop()
    return fen


def test_rank_key_orders_win_over_draw_then_by_dtz():
    k = SyzygyTablebase._rank_key
    assert k(2, -5) > k(0, 100)        # a win beats a draw regardless of dtz
    assert k(2, -2) > k(2, -10)        # faster mate (dtz closer to 0) preferred
    assert k(-2, 30) > k(-2, 5)        # longest resistance preferred when losing


def test_rank_key_zeroing_breaks_dtz_but_not_wdl():
    k = SyzygyTablebase._rank_key
    # Among equal-WDL wins, a winning zeroing move beats a quiet move even when the
    # quiet move has a DTZ closer to 0 (DTZ is not comparable across a zeroing
    # boundary — that is exactly the trap the middle term removes).
    assert k(2, -30, win_zeroing=1) > k(2, -2, win_zeroing=0)
    # But zeroing never rescues a worse WDL: a quiet win still beats a drawing capture.
    assert k(2, -50, win_zeroing=0) > k(0, 0, win_zeroing=0)


def test_best_dtz_prefers_winning_move():
    board = chess.Board(KRK)
    moves = list(board.legal_moves)
    win_move = moves[0]
    # child of win_move: opponent is lost (wdl -2 from their POV)
    tb = SyzygyTablebase(_tb=DTZFake({_child_fen(board, win_move): (-2, -4)}))
    assert tb.best_dtz_move(board) == win_move


def test_best_dtz_breaks_ties_toward_faster_mate():
    board = chess.Board(KRK)
    moves = list(board.legal_moves)
    fast, slow = moves[0], moves[1]
    tb = SyzygyTablebase(_tb=DTZFake({
        _child_fen(board, fast): (-2, -2),    # win, mate in fewer
        _child_fen(board, slow): (-2, -20),   # win, mate in many
    }))
    assert tb.best_dtz_move(board) == fast


def test_best_dtz_prefers_winning_zeroing_move_over_quiet_faster_dtz():
    # In KQKP the only zeroing move is Qxd4 (capturing the lone pawn); white has no
    # pawns, so every other legal move is quiet. We rig the fake so the *quiet* move
    # has the DTZ closest to 0 (what the old key chased) while the capture's DTZ is
    # far from 0 — yet the capture must win, because zeroing resets the 50-move
    # counter and guarantees progress toward the pawnless mate.
    board = chess.Board(KQKP)
    capture = next(m for m in board.legal_moves if board.is_zeroing(m))
    quiet = next(m for m in board.legal_moves if not board.is_zeroing(m))
    tb = SyzygyTablebase(_tb=DTZFake({
        _child_fen(board, capture): (-2, -30),  # win, but DTZ far from 0
        _child_fen(board, quiet): (-2, -2),      # win, DTZ closest to 0 (old trap)
    }))
    assert tb.best_dtz_move(board) == capture


def test_best_dtz_zeroing_never_overrides_wdl():
    # The zeroing bonus only applies to *winning* moves. A drawing/losing capture
    # must never be chosen over a quiet move that holds the win.
    board = chess.Board(KQKP)
    capture = next(m for m in board.legal_moves if board.is_zeroing(m))
    quiet = next(m for m in board.legal_moves if not board.is_zeroing(m))
    tb = SyzygyTablebase(_tb=DTZFake({
        _child_fen(board, capture): (2, 10),     # child wins -> we LOSE by capturing
        _child_fen(board, quiet): (-2, -50),     # quiet move keeps the win
    }))
    assert tb.best_dtz_move(board) == quiet


def test_best_dtz_none_out_of_scope_without_probing():
    fake = DTZFake(default=(None, None))
    tb = SyzygyTablebase(max_pieces=5, _tb=fake)
    assert tb.best_dtz_move(chess.Board()) is None  # 32 men, gated


def test_best_dtz_none_when_table_incomplete():
    tb = SyzygyTablebase(_tb=DTZFake(default=(None, None)))
    assert tb.best_dtz_move(chess.Board(KRK)) is None
