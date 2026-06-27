import torch

from src.model.presets import build_model
from src.model.evaluator import TransformerEvaluator
from src.eval.gates import raw_top1, mate_in_one_acc, throughput, run_gates
from src.eval.fixtures import load_gate_puzzles, mate_in_one_positions


def test_raw_top1_runs_on_small_puzzle_set():
    net, _ = build_model("baseline-v1")
    pz = load_gate_puzzles(20)
    acc = raw_top1(net, "cpu", pz)
    assert 0.0 <= acc <= 1.0


def test_mate_in_one_acc_in_range():
    net, _ = build_model("baseline-v1")
    acc = mate_in_one_acc(net, "cpu", mate_in_one_positions())
    assert 0.0 <= acc <= 1.0


def test_throughput_reports_positive_latency():
    net, _ = build_model("baseline-v1")
    out = throughput(net, "cpu", batch=8, iters=2)
    assert out["batch_latency_ms"] > 0.0


def test_run_gates_smoke_returns_expected_keys():
    net, _ = build_model("baseline-v1")
    ev = TransformerEvaluator(net, device="cpu", objective="policy")
    out = run_gates(net, ev, "cpu",
                    puzzle_counts=(20,), mcts_puzzles=0, dense_val=False,
                    throughput_batch=8, throughput_iters=2)
    assert "raw_top1@20" in out
    assert "mate_in_1" in out
    assert "batch_latency_ms" in out


def test_gates_return_zero_on_empty_inputs():
    net, _ = build_model("baseline-v1")
    assert raw_top1(net, "cpu", []) == 0.0
    assert mate_in_one_acc(net, "cpu", []) == 0.0
