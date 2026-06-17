"""Supervised distillation trainer for ChessTransformer (PyTorch)."""

import math
import os

import torch
import torch.nn as nn

from src.training.distill_losses import total_loss
from src.training.distill_metrics import value_sign_accuracy, top1_move_match


def _make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_frac: float = 0.05):
    """LambdaLR: linear warmup then cosine decay to min_lr_frac * base_lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DistillTrainer:
    def __init__(self, net, cfg, mixed_precision: bool = False, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
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

    def _to_device(self, inputs, targets):
        sq, sf = inputs
        pol_t, wdl_t, ml_t = targets
        sq = sq.to(self.device)
        sf = sf.to(self.device)
        pol_t = pol_t.to(self.device)
        wdl_t = wdl_t.to(self.device)
        ml_t = ml_t.to(self.device)
        return (sq, sf), (pol_t, wdl_t, ml_t)

    def train_step(self, inputs, targets):
        (sq, sf), (pol_t, wdl_t, ml_t) = self._to_device(inputs, targets)
        self.net.train()
        self.opt.zero_grad()
        pol, wdl, ml = self.net(sq, sf)
        loss, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()
        return float(loss.detach()), {k: float(v.detach()) for k, v in parts.items()}

    def evaluate(self, val_loader, max_batches: int = 50):
        self.net.eval()
        ps, vs, ts, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= max_batches:
                    break
                (sq, sf), (pol_t, wdl_t, ml_t) = self._to_device(inputs, targets)
                pol, wdl, ml = self.net(sq, sf)
                _, parts = total_loss(pol, wdl, ml, pol_t, wdl_t, ml_t)
                ps += float(parts["policy"])
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
            ckpt_dir: str = "checkpoints/distill"):
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
                    torch.save(self.net.state_dict(), os.path.join(ckpt_dir, "best.pt"))
        torch.save(self.net.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        return best
