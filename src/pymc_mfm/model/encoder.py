"""Causal cell-grid encoder with per-window media and control representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pymc_mfm.data.features import (
    CONTROL_FEATURE_BLOCKS,
    MEDIA_FEATURE_BLOCKS,
    MOMENT_FEATURE_BLOCKS,
)
from pymc_mfm.model.components import Encoder
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.primitives import MultiHeadAttention, RMSNorm, SwiGLU

# value, spend mean, spend std, is_query, is_revenue, week; then is_control.
N_CELL_BASE_FEATURES = 6
_CELL_INPUT_WIDTH = N_CELL_BASE_FEATURES + 1 + sum(width for _, width in MOMENT_FEATURE_BLOCKS)


class BiAttentionBlock(nn.Module):
    """Attention over weeks per entity, over entities per week, then a shared FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.time_norm = RMSNorm(d_model)
        self.time_attn = MultiHeadAttention(d_model, n_heads)
        self.chan_norm = RMSNorm(d_model)
        self.chan_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor,
        entity_active: Tensor | None = None,
    ) -> Tensor:
        b, t, c, d = x.shape
        if entity_active is None:
            entity_active = torch.ones(b, c, dtype=torch.bool, device=x.device)
        if entity_active.shape != (b, c):
            raise ValueError(
                f"entity_active must have shape {(b, c)}, got {tuple(entity_active.shape)}"
            )
        active = entity_active.to(dtype=torch.bool)
        active_grid = active[:, None, :, None].to(x.dtype)
        x = x * active_grid
        xt = x.permute(0, 2, 1, 3).reshape(b * c, t, d)
        xt = xt + self.time_attn(self.time_norm(xt), time_mask)
        x = xt.reshape(b, c, t, d).permute(0, 2, 1, 3) * active_grid

        xc = x.reshape(b * t, c, d)
        normed = self.chan_norm(xc)
        qkv = self.chan_attn.qkv(normed).reshape(
            b * t,
            c,
            3,
            self.chan_attn.n_heads,
            self.chan_attn.d_head,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key_active = active[:, None, :].expand(b, t, c).reshape(b * t, 1, 1, c)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_active,
        )
        attended = attended.transpose(1, 2).reshape(b * t, c, d)
        xc = xc + self.chan_attn.out(attended)
        x = xc.reshape(b, t, c, d) * active_grid
        return (x + self.ffn(self.ffn_norm(x))) * active_grid


@dataclass(frozen=True)
class SplitRepresentations:
    """One window's encoder output, in the order the flow head consumes it."""

    weekly: Tensor  # (B, T, D)
    channels: Tensor  # (B, T, C, D)
    controls: Tensor  # (B, T, M, D)


@dataclass(frozen=True)
class TimelineRepresentations:
    """Two window readouts of one timeline, not paired posterior draws."""

    context: SplitRepresentations
    query: SplitRepresentations


class BiAttentionEncoder(nn.Module, Encoder):
    """One token per week and media/control entity, plus a revenue cell.

    The shared projection and entity attention are permutation-equivariant.
    Query revenue is zero-valued, so lower-triangular attention over the whole
    context/query timeline can include earlier query weeks without targets.
    All five context feature blocks are part of the fixed 31-column layout.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.input_proj = nn.Linear(_CELL_INPUT_WIDTH, config.d_model)
        self.blocks = nn.ModuleList(
            [
                BiAttentionBlock(config.d_model, config.n_heads, config.d_ff)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)

    def _cell_grid(self, batch: dict[str, Tensor]) -> tuple[Tensor, int, int, int, Tensor]:
        ctx_spend = batch["ctx_spend_z"]
        qry_spend = batch["qry_spend_z"]
        b, n_ctx, c = ctx_spend.shape
        n_q = qry_spend.shape[1]
        t = n_ctx + n_q
        device, dtype = ctx_spend.device, ctx_spend.dtype

        spend = torch.cat([ctx_spend, qry_spend], dim=1)
        controls = torch.cat([batch["ctx_controls_z"], batch["qry_controls_z"]], dim=1)
        n_controls = controls.shape[2]
        spend = torch.cat([spend, controls], dim=2)
        revenue = torch.cat(
            [batch["ctx_revenue_z"], torch.zeros(b, n_q, 1, device=device, dtype=dtype)],
            dim=1,
        )
        value = torch.cat([spend, revenue], dim=2)
        n_cells = c + n_controls + 1

        mean = batch["spend_mean"].reshape(b, 1, c).expand(b, t, c)
        std = batch["spend_std"].reshape(b, 1, c).expand(b, t, c)
        zeros_rest = torch.zeros(b, t, n_controls + 1, device=device, dtype=dtype)
        mean = torch.cat([mean, zeros_rest], dim=2)
        std = torch.cat([std, zeros_rest], dim=2)

        is_query = torch.zeros(b, t, 1, device=device, dtype=dtype)
        is_query[:, n_ctx:] = 1.0
        is_query = is_query.expand(b, t, n_cells)

        is_revenue = torch.zeros(b, t, n_cells, device=device, dtype=dtype)
        is_revenue[:, :, -1] = 1.0

        week = torch.cat([batch["ctx_week"], batch["qry_week"]], dim=1)
        week = week.expand(b, t, n_cells)

        is_control = torch.zeros(b, t, n_cells, device=device, dtype=dtype)
        is_control[:, :, c : c + n_controls] = 1.0
        cells = torch.stack([value, mean, std, is_query, is_revenue, week, is_control], dim=-1)
        for key, width in MEDIA_FEATURE_BLOCKS:
            grid = torch.zeros(b, t, n_cells, width, device=device, dtype=dtype)
            grid[:, :, :c, :] = batch[key].to(dtype).unsqueeze(1).expand(b, t, c, width)
            cells = torch.cat([cells, grid], dim=-1)
        for key, width in CONTROL_FEATURE_BLOCKS:
            grid = torch.zeros(b, t, n_cells, width, device=device, dtype=dtype)
            grid[:, :, c : c + n_controls, :] = (
                batch[key].to(dtype).unsqueeze(1).expand(b, t, n_controls, width)
            )
            cells = torch.cat([cells, grid], dim=-1)

        active_c = batch.get("active_c_mask")
        if active_c is None:
            active_c = torch.ones(b, c, device=device, dtype=dtype)
        active_c = active_c.to(device=device)
        if active_c.shape != (b, c):
            raise ValueError(f"active_c_mask must have shape {(b, c)}, got {tuple(active_c.shape)}")
        if ((active_c != 0) & (active_c != 1)).any():
            raise ValueError("active_c_mask must be binary")
        if (active_c.sum(dim=-1) == 0).any():
            raise ValueError("encoder batch contains a task with no active media channels")
        active_m = batch.get("active_m_mask")
        if active_m is None:
            active_m = torch.ones(b, n_controls, device=device, dtype=dtype)
        active_m = active_m.to(device=device)
        if active_m.shape != (b, n_controls):
            raise ValueError(
                f"active_m_mask must have shape {(b, n_controls)}, got {tuple(active_m.shape)}"
            )
        if ((active_m != 0) & (active_m != 1)).any():
            raise ValueError("active_m_mask must be binary")
        if n_controls > 0 and (active_m.sum(dim=-1) == 0).any():
            raise ValueError("encoder batch contains a task with no active controls")
        active_parts = [active_c, active_m, torch.ones(b, 1, device=device, dtype=dtype)]
        entity_active = torch.cat(active_parts, dim=-1).to(torch.bool)
        cells = cells * entity_active[:, None, :, None].to(cells.dtype)
        return cells, n_ctx, n_q, c, entity_active

    def assert_batch_alignment(self, batch: dict[str, Tensor]) -> int:
        """Require the five feature blocks and the checkpoint's fixed input width."""
        b, _, n_channels = batch["ctx_spend_z"].shape
        n_controls = batch["ctx_controls_z"].shape[2]
        encoded_width = N_CELL_BASE_FEATURES + 1
        for blocks, n_entities in (
            (MEDIA_FEATURE_BLOCKS, n_channels),
            (CONTROL_FEATURE_BLOCKS, n_controls),
        ):
            for key, width in blocks:
                feature = batch.get(key)
                if feature is None:
                    raise ValueError(f"encoder batch omits required feature block {key!r}")
                if feature.shape != (b, n_entities, width):
                    raise ValueError(
                        f"{key} must have shape {(b, n_entities, width)}, "
                        f"got {tuple(feature.shape)}"
                    )
                encoded_width += width
        actual = int(self.input_proj.in_features)
        if actual != encoded_width:
            raise ValueError(
                f"encoder input width {actual} does not match the width the batch "
                f"encodes ({encoded_width}); the cell layout moved under this checkpoint"
            )
        return encoded_width

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        weekly_repr, channel_repr, _ = self.forward_with_controls(batch)
        return weekly_repr, channel_repr

    def forward_with_controls(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Query weekly, channel and control readouts for the conditional heads."""
        query = self.forward_timeline(batch).query
        return query.weekly, query.channels, query.controls

    def forward_timeline(self, batch: dict[str, Tensor]) -> TimelineRepresentations:
        """Read both windows from one causal backbone pass."""
        self.assert_batch_alignment(batch)
        cells, n_ctx, n_q, n_real, entity_active = self._cell_grid(batch)
        x = self.input_proj(cells)
        n_weeks = n_ctx + n_q
        time_mask = torch.tril(torch.ones(n_weeks, n_weeks, dtype=torch.bool)).to(x.device)
        for block in self.blocks:
            x = block(x, time_mask, entity_active)
        x = self.norm(x)
        return TimelineRepresentations(
            context=self._split(x[:, :n_ctx], n_real),
            query=self._split(x[:, n_ctx:], n_real),
        )

    def _split(self, cells: Tensor, n_real: int) -> SplitRepresentations:
        return SplitRepresentations(
            weekly=cells[:, :, -1],
            channels=cells[:, :, :n_real],
            controls=cells[:, :, n_real:-1],
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def assert_encoder_batch_alignment(encoder: BiAttentionEncoder, batch: dict[str, Tensor]) -> int:
    """Check the canonical feature layout at a batch-building boundary."""
    return encoder.assert_batch_alignment(batch)
