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


def test_random_opening_is_non_terminal():
    import random
    board = ae._random_opening(random.Random(0), plies=8)
    assert not board.is_game_over()


def test_head_to_head_openings_symmetry_is_half():
    # Identical deterministic engines: each random opening is played by both
    # colours, so the per-pair scores cancel to exactly 0.5 overall.
    w, d, l = ae.head_to_head_openings(_first_legal, _first_legal, games=20,
                                       max_moves=40, seed=1, opening_plies=6)
    total = w + d + l
    assert total == 20
    assert (w + 0.5 * d) / total == 0.5
