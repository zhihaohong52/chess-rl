import chess

from src.eval.arena import play_match
from src.model.presets import build_model
from src.game.move_encoder import get_move_encoder
import scripts.arena_eval as ae


def _first_legal(board):
    return next(iter(board.legal_moves), None)


def test_self_play_scores_half_over_even_games():
    """A deterministic mover vs itself nets exactly 0.5 over an even game count
    (color alternation makes the fixed game's result cancel). This guards the
    head-to-head scoring the --vs mode relies on."""
    res = play_match(_first_legal, _first_legal, num_games=4, max_moves=40)
    assert res.total == 4
    assert res.score == 0.5


def test_build_raw_mover_returns_legal_move():
    net, cfg = build_model("baseline-v1")
    me = get_move_encoder()
    mover = ae.build_raw_mover(net, "cpu", me)
    move = mover(chess.Board())
    assert move in chess.Board().legal_moves
