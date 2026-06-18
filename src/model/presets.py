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
}


def resolve_config(preset_name: str) -> SimpleNamespace:
    """Return a config namespace = Config defaults with the preset's overrides."""
    if preset_name not in PRESETS:
        raise KeyError(
            f"unknown preset {preset_name!r}; known: {sorted(PRESETS)}"
        )
    base = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
    base.update(PRESETS[preset_name])
    return SimpleNamespace(**base)


def build_model(preset_name: str):
    """Build a ChessTransformer for the named preset.

    Returns (net, resolved_config).
    """
    cfg = resolve_config(preset_name)
    net = ChessTransformer(cfg)
    return net, cfg
