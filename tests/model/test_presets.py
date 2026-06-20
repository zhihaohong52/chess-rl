import pytest
from src.model.presets import PRESETS, resolve_config, build_model

BASELINE_PARAMS = 10_255_879


def test_baseline_v1_is_registered():
    assert "baseline-v1" in PRESETS


def test_resolve_config_applies_overrides():
    cfg = resolve_config("baseline-v1")
    assert cfg.d_model == 256
    assert cfg.n_layers == 8
    assert cfg.transformer_dropout == 0.0


def test_build_baseline_v1_param_count_is_locked():
    net, cfg = build_model("baseline-v1")
    n = sum(p.numel() for p in net.parameters())
    assert n == BASELINE_PARAMS, f"baseline-v1 drifted: {n} != {BASELINE_PARAMS}"


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_config("does-not-exist")


def test_override_is_applied(monkeypatch):
    monkeypatch.setitem(PRESETS, "tmp-test", {"d_model": 999, "n_layers": 3})
    cfg = resolve_config("tmp-test")
    assert cfg.d_model == 999          # override wins over Config default
    assert cfg.n_layers == 3
    assert cfg.n_heads == 8            # untouched Config default still present


def test_p3_80m_is_registered_and_locked():
    assert "p3-80m" in PRESETS
    net, cfg = build_model("p3-80m")
    n = sum(p.numel() for p in net.parameters())
    assert n == 76_982_212, f"p3-80m drifted: {n} != 76982212"
    assert cfg.d_model == 512
    assert cfg.n_layers == 16
    assert cfg.n_heads == 16
    assert cfg.d_ff == 3072
    assert cfg.ffn_type == "swiglu"
    assert cfg.value_head_type == "hlgauss"
    assert cfg.value_buckets == 64
    assert cfg.transformer_dropout == 0.05
