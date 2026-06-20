"""Supervised distillation trainer for ChessTransformer (PyTorch)."""

import contextlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.distill_losses import total_loss
from src.training.distill_metrics import value_sign_accuracy, top1_move_match
from src.training.checkpoint_meta import write_sidecar


def _make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_frac: float = 0.05):
    """LambdaLR: linear warmup then cosine decay to min_lr_frac * base_lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _default_device():
    """Best available device: CUDA (box) > MPS (local M1) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DistillTrainer:
    def __init__(self, net, cfg, mixed_precision: bool = False, device=None):
        if device is None:
            device = _default_device()
        self.device = device
        self.net = net.to(device)
        self.cfg = cfg
        self.mixed_precision = mixed_precision

        self.opt = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.distill_lr,
            weight_decay=1e-4,
        )
        self.scheduler = _make_scheduler(
            self.opt,
            warmup_steps=cfg.distill_warmup_steps,
            total_steps=cfg.distill_total_steps,
        )

        self.ema_decay = getattr(cfg, "ema_decay", 0.0)
        self._ema = None
        if self.ema_decay > 0:
            self._ema = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def _to_device(self, inputs, targets):
        sq, sf = inputs
        pol_t, wdl_t, ml_t = targets
        sq = sq.to(self.device)
        sf = sf.to(self.device)
        pol_t = pol_t.to(self.device)
        wdl_t = wdl_t.to(self.device)
        ml_t = ml_t.to(self.device)
        return (sq, sf), (pol_t, wdl_t, ml_t)

    def _update_ema(self):
        if self._ema is None:
            return
        d = self.ema_decay
        for k, v in self.net.state_dict().items():
            if v.dtype.is_floating_point:
                self._ema[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self._ema[k].copy_(v)

    def train_step(self, inputs, targets):
        (sq, sf), (pol_t, wdl_t, ml_t) = self._to_device(inputs, targets)
        self.net.train()
        self.opt.zero_grad()
        use_amp = self.mixed_precision and self.device == "cuda"
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp else contextlib.nullcontext()
        )
        with amp_ctx:
            pol, wdl, ml = self.net(sq, sf)
            loss, parts = total_loss(
                pol, wdl, ml, pol_t, wdl_t, ml_t,
                value_weight=getattr(self.cfg, "value_loss_weight", 1.0),
                value_head_type=getattr(self.net, "value_head_type", "wdl"),
                value_buckets=getattr(self.net, "value_buckets", 64),
                value_sigma_frac=getattr(self.cfg, "value_sigma_frac", 0.75),
            )
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self._update_ema()
        self.scheduler.step()
        return float(loss.detach()), {k: float(v.detach()) for k, v in parts.items()}

    # ------------------------------------------------------------------
    # Action-value objective (pointwise ChessBench train data)
    #   Interpret sigmoid(policy_logits[a]) as predicted Q(s,a) = win%.
    #   Supervise only the single sampled action per example via BCE.
    # ------------------------------------------------------------------
    def _av_to_device(self, inputs, targets):
        sq, sf = inputs
        action_idx, win = targets
        return ((sq.to(self.device), sf.to(self.device)),
                (action_idx.to(self.device), win.to(self.device)))

    def train_step_av(self, inputs, targets):
        (sq, sf), (action_idx, win) = self._av_to_device(inputs, targets)
        self.net.train()
        self.opt.zero_grad()
        pol, _wdl, _ml = self.net(sq, sf)
        logit_a = pol.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logit_a, win)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()
        return float(loss.detach())

    def evaluate_av(self, val_loader, max_batches: int = 50):
        self.net.eval()
        bce, mae, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= max_batches:
                    break
                (sq, sf), (action_idx, win) = self._av_to_device(inputs, targets)
                pol, _wdl, _ml = self.net(sq, sf)
                logit_a = pol.gather(1, action_idx.unsqueeze(1)).squeeze(1)
                bce += float(F.binary_cross_entropy_with_logits(logit_a, win))
                mae += float((torch.sigmoid(logit_a) - win).abs().mean())
                n += 1
        n = max(n, 1)
        return {"val_av_bce": bce / n, "val_av_mae": mae / n}

    def fit_av(self, train_loader, steps: int, val_loader=None, val_every: int = 500,
               ckpt_dir: str = "checkpoints/distill_av", log_every: int = 100, meta=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        best = float("inf")
        it = iter(train_loader)
        for step in range(1, steps + 1):
            try:
                inputs, targets = next(it)
            except StopIteration:
                it = iter(train_loader)
                inputs, targets = next(it)
            loss = self.train_step_av(inputs, targets)
            if step % log_every == 0:
                print(f"step {step} av_bce {loss:.4f}", flush=True)
            if val_loader is not None and step % val_every == 0:
                metrics = self.evaluate_av(val_loader)
                print(f"step {step} loss {loss:.4f} {metrics}", flush=True)
                if metrics["val_av_bce"] < best:
                    best = metrics["val_av_bce"]
                    self._save_ckpt(ckpt_dir, "best.pt", "action_value", meta)
        self._save_ckpt(ckpt_dir, "last.pt", "action_value", meta)
        return best

    def _save_ckpt(self, ckpt_dir, name, objective, meta):
        path = os.path.join(ckpt_dir, name)
        torch.save(self.net.state_dict(), path)
        if meta is not None:
            full = dict(meta)
            full["objective"] = objective
            write_sidecar(path, full)

    def _save_ema(self, ckpt_dir, name, objective, meta, val_loader):
        live = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(self._ema)
        try:
            ema_metrics = self.evaluate(val_loader)
            best = getattr(self, "_ema_best", float("inf"))
            if ema_metrics["val_policy_loss"] < best:
                self._ema_best = ema_metrics["val_policy_loss"]
                self._save_ckpt(ckpt_dir, name, objective, meta)
        finally:
            self.net.load_state_dict(live)

    def evaluate(self, val_loader, max_batches: int = 50):
        self.net.eval()
        ps, vs, ts, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= max_batches:
                    break
                (sq, sf), (pol_t, wdl_t, ml_t) = self._to_device(inputs, targets)
                pol, wdl, ml = self.net(sq, sf)
                _, parts = total_loss(
                    pol, wdl, ml, pol_t, wdl_t, ml_t,
                    value_weight=getattr(self.cfg, "value_loss_weight", 1.0),
                    value_head_type=getattr(self.net, "value_head_type", "wdl"),
                    value_buckets=getattr(self.net, "value_buckets", 64),
                    value_sigma_frac=getattr(self.cfg, "value_sigma_frac", 0.75),
                )
                ps += float(parts["policy"])
                if getattr(self.net, "value_head_type", "wdl") == "hlgauss":
                    from src.model.value_dist import expected_value
                    vhat = expected_value(wdl)
                    v_tgt = wdl_t[:, 0] + 0.5 * wdl_t[:, 1]
                    vs += float(((vhat > 0.5) == (v_tgt > 0.5)).float().mean())
                else:
                    vs += value_sign_accuracy(wdl, wdl_t)
                ts += top1_move_match(pol, pol_t)
                n += 1
        n = max(n, 1)
        return {
            "val_policy_loss": ps / n,
            "val_value_sign_acc": vs / n,
            "val_top1": ts / n,
        }

    def fit(self, train_loader, steps: int, val_loader=None, val_every: int = 1000,
            ckpt_dir: str = "checkpoints/distill", meta=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        best = float("inf")
        it = iter(train_loader)
        for step in range(1, steps + 1):
            try:
                inputs, targets = next(it)
            except StopIteration:
                it = iter(train_loader)
                inputs, targets = next(it)
            loss, parts = self.train_step(inputs, targets)
            if val_loader is not None and step % val_every == 0:
                metrics = self.evaluate(val_loader)
                print(f"step {step} loss {loss:.4f} {metrics}")
                if metrics["val_policy_loss"] < best:
                    best = metrics["val_policy_loss"]
                    self._save_ckpt(ckpt_dir, "best.pt", "policy", meta)
                if self._ema is not None:
                    self._save_ema(ckpt_dir, "best_ema.pt", "policy", meta, val_loader)
        self._save_ckpt(ckpt_dir, "last.pt", "policy", meta)
        return best
