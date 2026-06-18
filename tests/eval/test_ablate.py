import torch

from src.model.presets import build_model
from src.training.checkpoint_meta import write_sidecar
from scripts.ablate import run_row, to_markdown, to_csv


def _baseline_ckpt(tmp_path):
    net, _ = build_model("baseline-v1")
    ckpt = str(tmp_path / "m.pt")
    torch.save(net.state_dict(), ckpt)
    write_sidecar(ckpt, {"preset": "baseline-v1", "objective": "policy"})
    return ckpt


def test_run_row_has_params_and_metric(tmp_path):
    ckpt = _baseline_ckpt(tmp_path)
    row = run_row(ckpt, device="cpu",
                  gate_kwargs=dict(puzzle_counts=(20,), mcts_puzzles=0,
                                   dense_val=False, throughput_batch=8,
                                   throughput_iters=2))
    assert row["preset"] == "baseline-v1"
    assert row["params"] == 10_255_879
    assert "raw_top1@20" in row


def test_table_formatters_roundtrip():
    rows = [{"preset": "baseline-v1", "params": 10255879, "raw_top1@20": 0.25}]
    md = to_markdown(rows)
    csv = to_csv(rows)
    assert "baseline-v1" in md and "params" in md
    assert "preset,params,raw_top1@20" in csv
