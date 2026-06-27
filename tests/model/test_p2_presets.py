import torch
from src.model.presets import PRESETS, build_model


def test_p2_presets_registered():
    for name in ("p2-value", "p2-value-swiglu", "p2-value-swiglu-drop"):
        assert name in PRESETS


def test_p2_value_builds_hlgauss():
    net, cfg = build_model("p2-value")
    assert net.value_head_type == "hlgauss"
    sq = torch.zeros(2, 64, dtype=torch.long); sf = torch.zeros(2, 18)
    _, val, _ = net(sq, sf)
    assert val.shape == (2, cfg.value_buckets)


def test_p2_swiglu_drop_flags():
    _, cfg = build_model("p2-value-swiglu-drop")
    assert cfg.ffn_type == "swiglu"
    assert cfg.transformer_dropout > 0.0
    assert cfg.value_head_type == "hlgauss"
