import torch
from torch import nn
from src.model.smolgen import Smolgen


def test_smolgen_output_shape():
    shared_out = nn.Linear(128, 64 * 64, bias=False)
    sm = Smolgen(d_model=256, n_heads=8, compress=32, hidden=128, gen=128, shared_out=shared_out)
    x = torch.randn(2, 64, 256)
    bias = sm(x)
    assert tuple(bias.shape) == (2, 8, 64, 64)
    assert torch.isfinite(bias).all()
