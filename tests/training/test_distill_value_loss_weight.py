import scripts.distill as distill


def _parse(extra):
    # baseline args + whatever the test wants to add
    return distill.build_parser().parse_args(
        ["--train", "x.npz", "--preset", "p2-value-swiglu-drop", *extra]
    )


def test_value_loss_weight_flag_overrides_default():
    cfg = distill.resolve_run_config(_parse(["--value-loss-weight", "0.3"]))
    assert cfg.value_loss_weight == 0.3


def test_value_loss_weight_absent_keeps_config_default():
    # preset doesn't set it, so it must fall back to Config's 1.0 default
    cfg = distill.resolve_run_config(_parse([]))
    assert cfg.value_loss_weight == 1.0


def test_resolve_run_config_preserves_schedule_overrides():
    # guard the refactor didn't drop the existing lr/warmup/steps/ema plumbing
    cfg = distill.resolve_run_config(
        _parse(["--lr", "1e-3", "--warmup", "7", "--steps", "55", "--ema-decay", "0.99"])
    )
    assert cfg.distill_lr == 1e-3
    assert cfg.distill_warmup_steps == 7
    assert cfg.distill_total_steps == 55
    assert cfg.ema_decay == 0.99
