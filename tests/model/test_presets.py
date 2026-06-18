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
    with pytest.raises(KeyError):
        resolve_config("does-not-exist")
