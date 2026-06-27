import os
import chess
import pytest
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.model.evaluator import TransformerEvaluator
from src.eval.gates import policy_value_metrics
from src.data.dataset import make_dataloader
from src.eval.fixtures import ensure_dense_val
from src.game.move_encoder import get_move_encoder


def _hlgauss_net():
    cfg = resolve_config("baseline-v1")
    cfg.value_head_type = "hlgauss"; cfg.value_buckets = 64
    return ChessTransformer(cfg)


def test_evaluator_hlgauss_value_in_range():
    net = _hlgauss_net()
    ev = TransformerEvaluator(net, device="cpu", objective="policy")
    policy, value = ev.evaluate(chess.Board())
    assert -1.0 <= value <= 1.0
    assert abs(sum(policy.values()) - 1.0) < 1e-4


def test_gates_policy_value_metrics_hlgauss_smoke():
    if not os.path.exists("data/test/action_value_data.bag"):
        pytest.skip("test bag absent")
    net = _hlgauss_net()
    path = ensure_dense_val()
    P = get_move_encoder().policy_size
    loader = make_dataloader([path], batch_size=64, policy_size=P, shuffle=False)
    m = policy_value_metrics(net, "cpu", loader, max_batches=1)
    assert "value_sign_acc" in m and 0.0 <= m["value_sign_acc"] <= 1.0


def test_evaluator_hlgauss_value_matches_expected_value():
    import torch
    from src.model.value_dist import expected_value
    net = _hlgauss_net()
    ev = TransformerEvaluator(net, device="cpu", objective="policy")
    board = chess.Board()
    # raw value logits from the net
    from src.game.token_encoder import encode_position
    sq, sf = encode_position(board, 0)
    sq_t = torch.tensor(sq[None], dtype=torch.long)
    sf_t = torch.tensor(sf[None], dtype=torch.float32)
    with torch.no_grad():
        _, val_logits, _ = net.predict_batch(sq_t, sf_t)
    vhat = float(expected_value(val_logits)[0])
    expected = 2.0 * vhat - 1.0
    _, value = ev.evaluate(board)
    assert abs(value - expected) < 1e-4, f"{value} != {expected} (double-softmax?)"
