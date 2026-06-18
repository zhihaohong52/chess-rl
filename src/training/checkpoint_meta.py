"""JSON sidecar for checkpoints: records how a model must be evaluated.

The sidecar lives next to the .pt (best.pt -> best.json) and carries at least
{preset, objective}. The eval routing helper reads it so a model is always
searched with the matching evaluator (policy vs action_value).
"""

import json
import os
import subprocess


def sidecar_path(ckpt_path: str) -> str:
    return os.path.splitext(ckpt_path)[0] + ".json"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_sidecar(ckpt_path: str, meta: dict) -> str:
    """Write <ckpt>.json with meta (git_sha auto-added if absent). Returns path."""
    data = dict(meta)
    data.setdefault("git_sha", _git_sha())
    path = sidecar_path(ckpt_path)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


def read_sidecar(ckpt_path: str):
    """Return the sidecar dict, or None if it does not exist."""
    path = sidecar_path(ckpt_path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
