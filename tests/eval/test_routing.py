import pytest
import torch

from src.model.presets import build_model
from src.training.checkpoint_meta import write_sidecar
from src.eval.routing import load_for_eval


def _save_baseline(tmp_path, objective):
    net, _ = build_model("baseline-v1")
    ckpt = str(tmp_path / "m.pt")
    torch.save(net.state_dict(), ckpt)
    write_sidecar(ckpt, {"preset": "baseline-v1", "objective": objective})
    return ckpt


def test_routing_uses_sidecar_objective(tmp_path):
    ckpt = _save_baseline(tmp_path, "action_value")
    net, ev = load_for_eval(ckpt, device="cpu")
    assert ev.objective == "action_value"


def test_routing_missing_sidecar_requires_explicit(tmp_path):
    net, _ = build_model("baseline-v1")
    ckpt = str(tmp_path / "bare.pt")
    torch.save(net.state_dict(), ckpt)
    with pytest.raises(ValueError):
        load_for_eval(ckpt, device="cpu")  # no sidecar, no explicit args
    # explicit override works
    net2, ev2 = load_for_eval(ckpt, objective="policy", preset="baseline-v1",
                              device="cpu")
    assert ev2.objective == "policy"
