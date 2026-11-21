from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import _metal


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None, *, causal: bool = False) -> Tensor:
        """Attend over ``x[B, N, D]``; ``causal`` declares ``mask`` lower-triangular.

        A causal call without gradients on an Apple GPU runs as one kernel that never
        reads the mask; elsewhere ``mask`` is applied as given.
        """
        B, N, D = x.shape
        qkv = self.qkv(x)
        if causal and _metal.attention_supported(qkv, self.d_head):
            return self.out(_metal.causal_attention(qkv, self.n_heads))
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # (B, H, N, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None if mask is None else mask.unsqueeze(0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)
