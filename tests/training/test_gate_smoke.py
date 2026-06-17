from scripts.gate_smoke import run_smoke_gate


def test_smoke_gate_passes_on_synthetic_batch():
    ok, info = run_smoke_gate(steps=30)
    assert ok is True
    assert info["final_loss"] < info["initial_loss"]
