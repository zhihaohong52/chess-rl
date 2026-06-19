import torch
from src.model.presets import build_model, resolve_config
from src.model.transformer import ChessTransformer


def test_baseline_wdl_unchanged():
    net, _ = build_model("baseline-v1")
    assert net.value_head_type == "wdl"
    sq = torch.zeros(2, 64, dtype=torch.long)
    sf = torch.zeros(2, 18, dtype=torch.float32)
    pol, val, ml = net(sq, sf)
    assert val.shape == (2, 3)


def test_hlgauss_value_output_shape():
    cfg = resolve_config("baseline-v1")
    cfg.value_head_type = "hlgauss"
    cfg.value_buckets = 64
    net = ChessTransformer(cfg)
    assert net.value_head_type == "hlgauss" and net.value_buckets == 64
    sq = torch.zeros(2, 64, dtype=torch.long)
    sf = torch.zeros(2, 18, dtype=torch.float32)
    _, val, _ = net(sq, sf)
    assert val.shape == (2, 64)
