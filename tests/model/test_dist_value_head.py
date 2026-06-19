import torch
from src.model.heads import DistributionalValueHead


def test_dist_value_head_shape():
    head = DistributionalValueHead(256, 64)
    out = head(torch.randn(4, 256))
    assert out.shape == (4, 64)
