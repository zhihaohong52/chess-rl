"""Smolgen: learned position-dependent additive attention bias (per head). PyTorch."""

import torch
import torch.nn as nn


class Smolgen(nn.Module):
    """From token embeddings [B,64,d] produce a per-head bias [B,n_heads,64,64].

    The final ``gen -> 64*64`` projection (``shared_out``) is shared across all
    encoder layers to amortize parameters.
    """

    def __init__(self, d_model, n_heads, compress, hidden, gen, shared_out):
        super().__init__()
        self.n_heads = n_heads
        self.gen = gen
        self.compress = nn.Linear(d_model, compress, bias=False)
        self.dense1 = nn.Linear(64 * compress, hidden)
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(hidden)
        self.dense2 = nn.Linear(hidden, n_heads * gen)
        self.shared_out = shared_out  # shared nn.Linear(gen, 64*64, bias=False)

    def forward(self, x):  # x: [B, 64, d]
        b = x.shape[0]
        c = self.compress(x).reshape(b, -1)        # [B, 64*compress]
        h = self.ln(self.act(self.dense1(c)))      # [B, hidden]
        g = self.dense2(h).reshape(b * self.n_heads, self.gen)
        bias = self.shared_out(g)                  # [B*n_heads, 64*64]
        return bias.reshape(b, self.n_heads, 64, 64)
