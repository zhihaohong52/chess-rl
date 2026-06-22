import chess

from src.engine.syzygy_tb import SyzygyTablebase

KRK = "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"  # white K+R vs k, white to move


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


def test_best_dtz_none_out_of_scope_without_probing():
    fake = DTZFake(default=(None, None))
    tb = SyzygyTablebase(max_pieces=5, _tb=fake)
    assert tb.best_dtz_move(chess.Board()) is None  # 32 men, gated


def test_best_dtz_none_when_table_incomplete():
    tb = SyzygyTablebase(_tb=DTZFake(default=(None, None)))
    assert tb.best_dtz_move(chess.Board(KRK)) is None
