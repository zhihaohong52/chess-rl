import numpy as np, torch, chess
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder
from src.training.distill_trainer import DistillTrainer


def _batch():
    P = get_move_encoder().policy_size
    boards = [chess.Board(), chess.Board()]
    sq, sf = encode_batch(boards, [0, 0])
    pol = np.zeros((2, P), dtype=np.float32); pol[:, 0] = 1.0
    wdl = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], dtype=np.float32)
    ml = np.array([[40.0], [40.0]], dtype=np.float32)
    inputs = (torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32))
    targets = (torch.tensor(pol), torch.tensor(wdl), torch.tensor(ml))
    return inputs, targets


def test_trainer_hlgauss_step_and_eval():
    cfg = resolve_config("baseline-v1")
    cfg.value_head_type = "hlgauss"; cfg.value_buckets = 64
    net = ChessTransformer(cfg)
    trainer = DistillTrainer(net, cfg, device="cpu")
    inputs, targets = _batch()
    loss, parts = trainer.train_step(inputs, targets)
    assert np.isfinite(loss) and np.isfinite(parts["value"])
    metrics = trainer.evaluate([(inputs, targets)], max_batches=1)
    assert np.isfinite(metrics["val_value_sign_acc"])


def test_trainer_evaluate_reports_calibration_metrics():
    # value calibration is first-class: evaluate() always returns these keys for
    # both head types, so the training loop prints them live every val step.
    for vht in ("hlgauss", "wdl"):
        cfg = resolve_config("baseline-v1")
        cfg.value_head_type = vht
        if vht == "hlgauss":
            cfg.value_buckets = 64
        net = ChessTransformer(cfg)
        trainer = DistillTrainer(net, cfg, device="cpu")
        inputs, targets = _batch()
        m = trainer.evaluate([(inputs, targets)], max_batches=1)
        for k in ("val_wdl_ce", "val_draw_cal", "val_ece"):
            assert k in m, f"{k} missing for {vht}"
            assert np.isfinite(m[k]), f"{k} not finite for {vht}"
        assert 0.0 <= m["val_ece"] <= 1.0
