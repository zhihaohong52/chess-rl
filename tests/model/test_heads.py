import chess
import numpy as np
import torch
from src.model.heads import build_policy_index_map, PolicyHead, ValueHead, MovesLeftHead
from src.game.move_encoder import get_move_encoder


def test_policy_index_map_ranges():
    fi, ti, pr = build_policy_index_map()
    P = get_move_encoder().policy_size
    assert fi.shape == (P,) and ti.shape == (P,) and pr.shape == (P,)
    assert fi.min() >= 0 and fi.max() < 64
    assert ti.min() >= 0 and ti.max() < 64
    assert set(np.unique(pr)).issubset({0, 1, 2, 3})


def test_policy_head_shape_and_finite():
    P = get_move_encoder().policy_size
    head = PolicyHead(256)
    sq = torch.randn(3, 64, 256)
    logits = head(sq)
    assert tuple(logits.shape) == (3, P)
    assert torch.isfinite(logits).all()


def test_value_and_movesleft_shapes():
    cls = torch.randn(3, 256)
    assert tuple(ValueHead(256)(cls).shape) == (3, 3)
    ml = MovesLeftHead(256)(cls)
    assert tuple(ml.shape) == (3, 1)
    assert (ml >= 0).all()
