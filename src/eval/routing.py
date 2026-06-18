"""Load a checkpoint for evaluation with the CORRECT evaluator.

Reads the checkpoint sidecar (preset, objective) and builds the matching
TransformerEvaluator. Prevents objective mismatches (e.g. searching a policy
model with action_value semantics). If no sidecar exists, the caller MUST pass
objective and preset explicitly.
"""

import torch

from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.training.checkpoint_meta import read_sidecar


def load_for_eval(ckpt_path, objective=None, preset=None, device=None,
                  use_fp16=False):
    """Return (net, evaluator) ready for gates.

    objective/preset override the sidecar; if there is no sidecar they are
    required.
    """
    meta = read_sidecar(ckpt_path) or {}
    preset = preset or meta.get("preset")
    objective = objective or meta.get("objective")
    if preset is None or objective is None:
        raise ValueError(
            f"no sidecar for {ckpt_path} and preset/objective not given; "
            "pass preset= and objective= explicitly"
        )
    net, _cfg = build_model(preset)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    evaluator = TransformerEvaluator(net, use_fp16=use_fp16, device=device,
                                     objective=objective)
    return net, evaluator
