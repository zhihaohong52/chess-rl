import torch
from src.model.transformer import SwiGLUFFN, ChessTransformer
from src.model.presets import resolve_config


def test_swiglu_shape_and_param_parity():
    d, d_ff = 256, 1024
    sw = SwiGLUFFN(d, d_ff)
    x = torch.randn(2, 5, d)
    assert sw(x).shape == (2, 5, d)
    mlp_params = 2 * d * d_ff + d_ff + d
    sw_params = sum(p.numel() for p in sw.parameters())
    assert abs(sw_params - mlp_params) / mlp_params < 0.10


def test_swiglu_preset_builds():
    cfg = resolve_config("baseline-v1")
    cfg.ffn_type = "swiglu"
    net = ChessTransformer(cfg)
    sq = torch.zeros(2, 64, dtype=torch.long); sf = torch.zeros(2, 18)
    pol, val, ml = net(sq, sf)
    assert pol.shape[0] == 2
