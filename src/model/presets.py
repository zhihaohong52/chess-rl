"""Named model presets layered over Config, plus a model builder.

A preset is a dict of architecture-knob overrides applied on top of the base
Config. `baseline-v1` is the empty-override preset: it reproduces the current
model exactly (10,255,879 params). Future architecture variants (Phase 2) are
added here as new presets; the model stays config-driven and the default forward
contract is unchanged.
"""

from types import SimpleNamespace

from config import Config
from src.model.transformer import ChessTransformer

# name -> dict of Config attribute overrides
PRESETS = {
    "baseline-v1": {},
    "p2-value": {"value_head_type": "hlgauss", "value_buckets": 64},
    "p2-value-swiglu": {"value_head_type": "hlgauss", "value_buckets": 64,
                        "ffn_type": "swiglu"},
    "p2-value-swiglu-drop": {"value_head_type": "hlgauss", "value_buckets": 64,
                             "ffn_type": "swiglu", "transformer_dropout": 0.05},
    "p3-80m": {"d_model": 512, "n_layers": 16, "n_heads": 16, "d_ff": 3072,
               "ffn_type": "swiglu", "value_head_type": "hlgauss",
               "value_buckets": 64, "transformer_dropout": 0.05},
}


def resolve_config(preset_name: str) -> SimpleNamespace:
    """Return a config namespace = Config defaults with the preset's overrides."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; known: {sorted(PRESETS)}"
        )
    # NOTE: Config has derived fields (extra_planes, input_planes, input_shape)
    # computed at class-definition time. Overriding their source fields
    # (board_size, include_ep_plane, history_length) in a preset will NOT recompute
    # the derived fields; override the derived fields explicitly if a preset needs them.
    base = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
    base.update(PRESETS[preset_name])
    return SimpleNamespace(**base)


def build_model(preset_name: str) -> tuple["ChessTransformer", SimpleNamespace]:
    """Build a ChessTransformer for the named preset.

    Returns (net, resolved_config).
    """
    cfg = resolve_config(preset_name)
    net = ChessTransformer(cfg)
    return net, cfg
