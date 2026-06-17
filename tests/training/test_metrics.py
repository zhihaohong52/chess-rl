import torch
from src.training.distill_metrics import value_sign_accuracy, top1_move_match


def test_value_sign_accuracy():
    wdl_logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])  # win, loss
    wdl_target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert value_sign_accuracy(wdl_logits, wdl_target) == 1.0


def test_top1_move_match():
    pol = torch.tensor([[0.1, 9.0, 0.2], [3.0, 0.0, 0.0]])
    tgt = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert top1_move_match(pol, tgt) == 1.0
