import math
import torch

from src.eval.metrics_core import (
    policy_topk_match, policy_cross_entropy, legal_mass,
    wdl_cross_entropy, value_sign_acc, draw_calibration, value_ece,
)


def test_policy_topk_match_top1_and_top3():
    # target's best move is index 2; model ranks index 2 third-highest
    logits = torch.tensor([[3.0, 2.0, 1.5, 0.0]])
    target = torch.zeros(1, 4); target[0, 2] = 1.0
    assert policy_topk_match(logits, target, k=1) == 0.0
    assert policy_topk_match(logits, target, k=3) == 1.0


def test_policy_cross_entropy_matches_manual():
    logits = torch.tensor([[0.0, 0.0]])         # softmax -> [0.5, 0.5]
    target = torch.tensor([[1.0, 0.0]])
    ce = policy_cross_entropy(logits, target)
    assert abs(ce - math.log(2)) < 1e-5


def test_legal_mass_sums_softmax_over_supported_moves():
    logits = torch.tensor([[10.0, 10.0, -10.0]])  # mass ~ on idx 0,1
    target = torch.tensor([[0.5, 0.5, 0.0]])      # legal support = {0,1}
    assert legal_mass(logits, target) > 0.99


def test_wdl_cross_entropy_and_sign_acc():
    wdl_logits = torch.tensor([[5.0, 0.0, -5.0]])  # predicts Win
    wdl_target = torch.tensor([[1.0, 0.0, 0.0]])
    assert wdl_cross_entropy(wdl_logits, wdl_target) < 0.05
    assert value_sign_acc(wdl_logits, wdl_target) == 1.0


def test_draw_calibration_gap_small_when_matched():
    wdl_logits = torch.tensor([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])  # ~all draw
    wdl_target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])    # all draw
    assert draw_calibration(wdl_logits, wdl_target) < 0.01


def test_draw_calibration_gap_large_when_mismatched():
    wdl_logits = torch.tensor([[0.0, 10.0, 0.0]])  # predicts draw
    wdl_target = torch.tensor([[1.0, 0.0, 0.0]])   # actually win
    assert draw_calibration(wdl_logits, wdl_target) > 0.99


def test_value_ece_zero_when_perfectly_calibrated():
    # predicted expected score == actual in every bin -> ECE 0
    s = torch.linspace(0.0, 1.0, 200)
    assert value_ece(s, s.clone(), n_bins=10) < 1e-6


def test_value_ece_recovers_constant_offset():
    # predictions overconfident by a flat 0.2 -> ECE 0.2
    pred = torch.linspace(0.0, 0.8, 200)
    target = pred - 0.2
    assert abs(value_ece(pred, target, n_bins=10) - 0.2) < 1e-5


def test_value_ece_large_when_miscalibrated():
    # predicts 0.5 everywhere, outcomes all wins -> gap 0.5
    pred = torch.full((100,), 0.5)
    target = torch.ones(100)
    assert abs(value_ece(pred, target, n_bins=10) - 0.5) < 1e-5


def test_value_ece_handles_empty_input():
    assert value_ece(torch.empty(0), torch.empty(0)) == 0.0
