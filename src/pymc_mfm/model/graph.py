"""Complete graph geometry, equivariant structure prediction, and conditioning."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

STRUCT_EDGE_KEYS: tuple[str, ...] = ("cy", "dc", "dz", "db", "zb", "zc", "cc", "zz")
STRUCT_MASK_KEYS: tuple[str, ...] = ("active_c_mask", "active_m_mask", "active_j_mask")
SQUARE_EDGE_TYPES: tuple[str, ...] = ("cc", "zz")

NODE_CHANNEL, NODE_CONTROL, NODE_DEMAND, NODE_BASELINE, NODE_SALES = 0, 1, 2, 3, 4
N_NODE_TYPES = 5

EDGE_TYPE_NODES: dict[str, tuple[int, int]] = {
    "cy": (NODE_CHANNEL, NODE_SALES),
    "dc": (NODE_DEMAND, NODE_CHANNEL),
    "db": (NODE_DEMAND, NODE_BASELINE),
    "zb": (NODE_CONTROL, NODE_BASELINE),
    "dz": (NODE_DEMAND, NODE_CONTROL),
    "zc": (NODE_CONTROL, NODE_CHANNEL),
    "cc": (NODE_CHANNEL, NODE_CHANNEL),
    "zz": (NODE_CONTROL, NODE_CONTROL),
}


class EdgeSlotLayout:
    """Flat slot layout over the complete eight-family C-DAG edge set.

    The fixed order is ``[cy (K) | dc (J*K) | dz (J*M) | db (J) | zb (M)
    | zc (M*K) | cc (K*(K-1)) | zz (M*(M-1))]``. Blocks are row-major,
    excluding the structurally-zero diagonals of cc and zz.

    Per-slot endpoint indices address a node bank ordered
    ``[channels | controls | demand | baseline | sales]``.
    """

    def __init__(self, n_channels: int, n_controls: int, n_demand: int):
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if n_controls < 0 or n_demand < 0:
            raise ValueError("n_controls and n_demand must be >= 0")
        self.K = int(n_channels)
        self.M = int(n_controls)
        self.J = int(n_demand)
        K, M, J = self.K, self.M, self.J
        # Node-bank offsets: [channels | controls | demand | baseline | sales].
        ch, ct, de = 0, K, K + M
        self._baseline = K + M + J
        self._sales = K + M + J + 1
        self.n_nodes = K + M + J + 2

        slots: list[tuple[str, int, int]] = []  # (edge_type, src_node, dst_node)
        slots += [("cy", ch + k, self._sales) for k in range(K)]
        slots += [("dc", de + j, ch + k) for j in range(J) for k in range(K)]
        slots += [("dz", de + j, ct + m) for j in range(J) for m in range(M)]
        slots += [("db", de + j, self._baseline) for j in range(J)]
        slots += [("zb", ct + m, self._baseline) for m in range(M)]
        slots += [("zc", ct + m, ch + k) for m in range(M) for k in range(K)]
        slots += [("cc", ch + i, ch + k) for i in range(K) for k in range(K) if i != k]
        slots += [("zz", ct + i, ct + m) for i in range(M) for m in range(M) if i != m]

        self.n_slots = len(slots)
        self.slot_edge_types: tuple[str, ...] = tuple(et for et, _, _ in slots)
        self.slot_src_index = np.asarray([s for _, s, _ in slots], dtype=np.int64)
        self.slot_dst_index = np.asarray([d for _, _, d in slots], dtype=np.int64)
        self.slot_src_type = np.asarray(
            [EDGE_TYPE_NODES[et][0] for et, _, _ in slots], dtype=np.int64
        )
        self.slot_dst_type = np.asarray(
            [EDGE_TYPE_NODES[et][1] for et, _, _ in slots], dtype=np.int64
        )

        self.slices: dict[str, slice] = {}
        start = 0
        for et, width in (
            ("cy", K),
            ("dc", J * K),
            ("dz", J * M),
            ("db", J),
            ("zb", M),
            ("zc", M * K),
            ("cc", K * (K - 1)),
            ("zz", M * (M - 1)),
        ):
            self.slices[et] = slice(start, start + width)
            start += width
        assert start == self.n_slots

        # One gather packs full blocks; one scatter restores zero diagonals.
        self._block_shapes: dict[str, tuple[int, ...]] = {
            "cy": (K,),
            "dc": (J, K),
            "dz": (J, M),
            "db": (J,),
            "zb": (M,),
            "zc": (M, K),
            "cc": (K, K),
            "zz": (M, M),
        }
        gather: list[int] = []
        offset = 0
        for et in STRUCT_EDGE_KEYS:
            size = int(np.prod(self._block_shapes[et])) if self._block_shapes[et] else 0
            if et in SQUARE_EDGE_TYPES:
                n = self._block_shapes[et][0]
                gather += [offset + i * n + k for i in range(n) for k in range(n) if i != k]
            else:
                gather += list(range(offset, offset + size))
            offset += size
        self._full_size = offset
        self._gather_index = np.asarray(gather, dtype=np.int64)

    def pack(self, blocks: dict[str, Tensor | np.ndarray]) -> Tensor | np.ndarray:
        """Pack all eight blocks to ``(..., n_slots)``, dropping square diagonals."""
        missing = [et for et in STRUCT_EDGE_KEYS if et not in blocks]
        if missing:
            raise KeyError(f"pack needs all edge-type blocks; missing {missing}")
        parts = []
        torch_mode = isinstance(blocks["cy"], Tensor)
        for et in STRUCT_EDGE_KEYS:
            block = blocks[et]
            shape = self._block_shapes[et]
            lead = block.shape[: block.ndim - len(shape)]
            if tuple(block.shape[block.ndim - len(shape) :]) != shape:
                raise ValueError(
                    f"block {et!r} must end with shape {shape}, got {tuple(block.shape)}"
                )
            parts.append(block.reshape(*lead, -1))
        if torch_mode:
            full = torch.cat(parts, dim=-1)
            idx = torch.as_tensor(self._gather_index, device=full.device)
            return full.index_select(-1, idx)
        full = np.concatenate(parts, axis=-1)
        return full[..., self._gather_index]

    def unpack(self, flat: Tensor | np.ndarray) -> dict[str, Tensor | np.ndarray]:
        """Restore full blocks from ``(..., n_slots)``, with zero square diagonals."""
        if flat.shape[-1] != self.n_slots:
            raise ValueError(f"expected {self.n_slots} slots in the last dim, got {flat.shape}")
        lead = flat.shape[:-1]
        if isinstance(flat, Tensor):
            full = flat.new_zeros(*lead, self._full_size)
            idx = torch.as_tensor(self._gather_index, device=flat.device)
            full.index_copy_(-1, idx, flat)
        else:
            full = np.zeros((*lead, self._full_size), dtype=flat.dtype)
            full[..., self._gather_index] = flat
        out: dict[str, Tensor | np.ndarray] = {}
        offset = 0
        for et in STRUCT_EDGE_KEYS:
            shape = self._block_shapes[et]
            size = int(np.prod(shape)) if shape else 0
            out[et] = full[..., offset : offset + size].reshape(*lead, *shape)
            offset += size
        return out

    def active_slot_mask(
        self,
        active_c: Tensor | np.ndarray,
        active_m: Tensor | np.ndarray,
        active_j: Tensor | np.ndarray,
    ) -> Tensor | np.ndarray:
        """Mark slots whose endpoints are active; baseline and sales always are."""
        torch_mode = isinstance(active_c, Tensor)
        if torch_mode:
            ones = torch.ones(*active_c.shape[:-1], 2, device=active_c.device, dtype=active_c.dtype)
            bank = torch.cat([active_c, active_m, active_j, ones], dim=-1)
        else:
            active_c = np.asarray(active_c)
            ones = np.ones((*active_c.shape[:-1], 2), dtype=active_c.dtype)
            bank = np.concatenate([active_c, np.asarray(active_m), np.asarray(active_j), ones], -1)
        if bank.shape[-1] != self.n_nodes:
            raise ValueError(
                f"active masks imply {bank.shape[-1]} nodes but the layout has {self.n_nodes} "
                f"(K={self.K}, M={self.M}, J={self.J})"
            )
        return bank[..., self.slot_src_index] * bank[..., self.slot_dst_index]


def structure_active_masks(batch: dict[str, Tensor]) -> dict[str, Tensor] | None:
    """Per-edge activity from independently optional per-node masks.

    Missing node-family masks mean that family is fully active. When no
    activity mask is present, fixed-size live samplers need no explicit mask.
    """
    if not any(key in batch for key in STRUCT_MASK_KEYS):
        return None
    c = batch["active_c_mask"] if "active_c_mask" in batch else torch.ones_like(batch["g_cy"])
    m = batch["active_m_mask"] if "active_m_mask" in batch else torch.ones_like(batch["g_zb"])
    j = batch["active_j_mask"] if "active_j_mask" in batch else torch.ones_like(batch["g_db"])
    return {
        "cy": c,
        "dc": j.unsqueeze(-1) * c.unsqueeze(-2),
        "dz": j.unsqueeze(-1) * m.unsqueeze(-2),
        "db": j,
        "zb": m,
        "zc": m.unsqueeze(-1) * c.unsqueeze(-2),
        "cc": c.unsqueeze(-1) * c.unsqueeze(-2),
        "zz": m.unsqueeze(-1) * m.unsqueeze(-2),
    }


@lru_cache(maxsize=32)
def _offdiag_mask_cached(n: int, device: torch.device) -> Tensor:
    return ~torch.eye(n, dtype=torch.bool, device=device)


def _offdiag_mask(n: int, device=None) -> Tensor:
    """Cache structural off-diagonal masks by size and device."""
    return _offdiag_mask_cached(int(n), torch.device("cpu") if device is None else device)


@lru_cache(maxsize=32)
def _conditioner_index(
    n_channels: int, n_controls: int, n_demand: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cache node types and slot endpoints/families: each fresh tensor is a device copy."""
    layout = EdgeSlotLayout(n_channels, n_controls, n_demand)
    node_types = torch.tensor(
        [NODE_CHANNEL] * layout.K
        + [NODE_CONTROL] * layout.M
        + [NODE_DEMAND] * layout.J
        + [NODE_BASELINE, NODE_SALES],
        device=device,
    )
    src = torch.as_tensor(layout.slot_src_index, device=device)
    dst = torch.as_tensor(layout.slot_dst_index, device=device)
    edge_types = torch.tensor(
        [STRUCT_EDGE_KEYS.index(family) for family in layout.slot_edge_types], device=device
    )
    return node_types, src, dst, edge_types


class PairwiseBilinearEdge(nn.Module):
    """Shared bilinear edge scorer, equivariant over both endpoint axes."""

    def __init__(self, d_src: int, d_dst: int):
        super().__init__()
        self.bilinear = nn.Linear(d_src, d_dst, bias=False)
        self.src_lin = nn.Linear(d_src, 1)
        self.dst_lin = nn.Linear(d_dst, 1, bias=False)
        self.scale = float(d_dst) ** -0.5

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
        """Map ``(B, S, d_src)`` and ``(B, D, d_dst)`` to ``(B, S, D)`` logits."""
        interaction = torch.einsum("bsd,btd->bst", self.bilinear(src), dst) * self.scale
        return interaction + self.src_lin(src) + self.dst_lin(dst).transpose(-1, -2)


class StructureHead(nn.Module):
    """Eight-family edge logits equivariant over channel and control slots."""

    def __init__(
        self,
        d_channel: int,
        d_global: int,
        n_demand: int,
        n_controls: int,
    ):
        super().__init__()
        if n_demand < 0 or n_controls < 0:
            raise ValueError("n_demand and n_controls must be >= 0")
        self.n_demand = int(n_demand)
        self.n_controls = int(n_controls)
        if self.n_controls == 0:
            raise ValueError("StructureHead requires n_controls > 0")
        # Shared across channels -> permutation-equivariant over the K axis.
        self.proj_cy = nn.Linear(d_channel, 1)
        if n_demand > 0:
            self.proj_dc = nn.Linear(d_channel, n_demand)
            self.proj_db = nn.Linear(d_global, n_demand)
        self.proj_zb = nn.Linear(d_channel, 1)
        self.proj_cc = PairwiseBilinearEdge(d_channel, d_channel)
        self.proj_zc = PairwiseBilinearEdge(d_channel, d_channel)
        self.proj_zz = PairwiseBilinearEdge(d_channel, d_channel)
        if n_demand > 0:
            self.proj_dz = nn.Linear(d_channel, n_demand)

    def forward(
        self,
        channel_repr: Tensor,
        global_repr: Tensor,
        control_repr: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict edges from channel, pooled task, and required control states.

        Inputs have shapes ``(B, K, d_channel)``, ``(B, d_global)``, and
        ``(B, M, d_channel)``. The cc/zz diagonals are excluded from the loss
        and zeroed in :meth:`predictive_probs`.
        """
        out: dict[str, Tensor] = {"cy": self.proj_cy(channel_repr).squeeze(-1)}
        if self.n_demand > 0:
            out["dc"] = self.proj_dc(channel_repr).transpose(-1, -2)
            out["db"] = self.proj_db(global_repr)
        if control_repr is None:
            raise ValueError("StructureHead requires control_repr")
        out["zb"] = self.proj_zb(control_repr).squeeze(-1)
        out["cc"] = self.proj_cc(channel_repr, channel_repr)
        out["zc"] = self.proj_zc(control_repr, channel_repr)
        out["zz"] = self.proj_zz(control_repr, control_repr)
        if self.n_demand > 0:
            out["dz"] = self.proj_dz(control_repr).transpose(-1, -2)
        return out

    def loss(
        self,
        edge_logits: dict[str, Tensor],
        g_true: dict[str, Tensor],
        active_mask: dict[str, Tensor] | None = None,
        edge_weights: dict[str, float | Tensor] | None = None,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Equal-family BCE over the present, non-empty edge families.

        Each family is normalized within each task by that task's active-slot
        count, then averaged over tasks carrying the family. Declared family
        weights combine those means without letting geometry or padding
        change their relative weights. Families with no active slots or an
        explicit zero weight do not enter the weighted average.
        """
        unknown_weights = set(edge_weights or ()) - set(STRUCT_EDGE_KEYS)
        if unknown_weights:
            raise ValueError(f"unknown structure edge_weights: {sorted(unknown_weights)}")

        weighted_family_means: list[Tensor] = []
        active_family_weights: list[Tensor] = []
        family_trace: dict[str, dict[str, object]] = {}
        for edge_type in STRUCT_EDGE_KEYS:
            if edge_type not in edge_logits:
                continue
            if edge_type not in g_true:
                raise KeyError(f"missing structure target for edge family {edge_type!r}")
            logits = edge_logits[edge_type]
            target = g_true[edge_type].to(logits.dtype)
            if target.shape != logits.shape:
                raise ValueError(
                    f"structure target {edge_type!r} has shape {tuple(target.shape)}, "
                    f"expected {tuple(logits.shape)}"
                )
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            mask = torch.ones_like(bce)
            if active_mask is not None and edge_type in active_mask:
                mask = mask * active_mask[edge_type].to(device=logits.device, dtype=logits.dtype)
            if edge_type in SQUARE_EDGE_TYPES:
                mask = mask * _offdiag_mask(logits.shape[-1], logits.device).to(logits.dtype)

            # Score each task on its own slots before averaging tasks together.
            reduce_dims = tuple(range(1, bce.ndim))
            task_count = mask.sum(dim=reduce_dims)
            task_numerator = (bce * mask).sum(dim=reduce_dims)
            task_active = (task_count > 0.0).to(bce.dtype)
            count = task_count.sum()
            per_task_mean = task_numerator / task_count.clamp_min(1.0)
            numerator = (per_task_mean * task_active).sum()
            denominator = task_active.sum()
            raw_weight = (edge_weights or {}).get(edge_type, 1.0)
            if isinstance(raw_weight, Tensor) and raw_weight.numel() != 1:
                raise ValueError(
                    f"structure family weight {edge_type!r} must be scalar, "
                    f"got shape {tuple(raw_weight.shape)}"
                )
            family_weight_value = float(raw_weight)
            if not math.isfinite(family_weight_value) or family_weight_value < 0.0:
                raise ValueError(
                    f"structure family weight {edge_type!r} must be finite and nonnegative, "
                    f"got {family_weight_value!r}"
                )
            family_weight = logits.new_tensor(family_weight_value)
            active_count = float(count.detach().item())
            inactive_reason = None
            if active_count == 0.0:
                inactive_reason = "no_active_slots"
            elif family_weight_value == 0.0:
                inactive_reason = "declared_zero_family_weight"
            else:
                family_mean = numerator / denominator
                weighted_family_means.append(family_weight * family_mean)
                active_family_weights.append(family_weight)
            family_trace[edge_type] = {
                "numerator": numerator,
                "denominator": denominator,
                "active_slot_count": count,
                "raw_family_weight": family_weight,
                "effective_family_weight": (
                    family_weight if inactive_reason is None else logits.new_zeros(())
                ),
                "mask": mask,
                "inactive_reason": inactive_reason,
            }

        if not weighted_family_means:
            if not edge_logits:
                raise ValueError("edge_logits has no known edge types")
            raise ValueError("structure BCE has no active family with positive weight")
        weight_sum = torch.stack(active_family_weights).sum()
        combined_numerator = torch.stack(weighted_family_means).sum()
        loss = combined_numerator / weight_sum
        if not return_trace:
            return loss
        return loss, {
            "normalization": "per_task_slot_mean_then_active_task_mean_then_weighted_family_mean",
            "combined_numerator": combined_numerator,
            "combined_denominator": weight_sum,
            "active_family_count": loss.new_tensor(float(len(weighted_family_means))),
            "families": family_trace,
        }

    def predictive_probs(self, edge_logits: dict[str, Tensor]) -> dict[str, Tensor]:
        out = {et: torch.sigmoid(v) for et, v in edge_logits.items()}
        for et in SQUARE_EDGE_TYPES:
            if et in out:
                out[et] = out[et] * _offdiag_mask(out[et].shape[-1], out[et].device).to(
                    out[et].dtype
                )
        return out


class GraphConditioner(nn.Module):
    """Typed incident-edge messages shared across channel and control slots.

    All eight edge families feed their incident nodes and the pooled weekly
    graph representation. Demand nodes share a type embedding rather than
    learning an arbitrary ordering of latent demand slots.
    """

    def __init__(self, d_model: int, d_channel: int):
        super().__init__()
        self.weekly_node = nn.Linear(d_model, d_channel)
        self.node_type = nn.Embedding(N_NODE_TYPES, d_channel)
        self.edge_type = nn.Embedding(len(STRUCT_EDGE_KEYS), d_channel)
        self.message = nn.Sequential(
            nn.Linear(3 * d_channel, d_channel),
            nn.SiLU(),
            nn.Linear(d_channel, 2 * d_channel),
        )
        self.node_update = nn.Sequential(
            nn.Linear(3 * d_channel, d_channel),
            nn.SiLU(),
            nn.Linear(d_channel, d_channel),
        )
        self.weekly_update = nn.Linear(d_channel, d_model)

    def forward(
        self,
        weekly: Tensor,
        channels: Tensor,
        controls: Tensor,
        graph: Tensor,
        layout: EdgeSlotLayout,
        active_nodes: Tensor,
        live_slots: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Condition one batch of windows on validated binary ``graph[B,L]``.

        ``live_slots`` optionally gives the slots present in some graph (see below), so
        that a GPU ``graph`` is not read back here.
        """
        b, t, _ = weekly.shape
        h = channels.shape[-1]
        device = weekly.device
        node_types, src, dst, edge_types = _conditioner_index(layout.K, layout.M, layout.J, device)
        latent = self.weekly_node(weekly).unsqueeze(-2).expand(b, t, layout.J + 2, h)
        nodes = torch.cat([channels, controls, latent], dim=-2)
        mask = active_nodes[:, None, :, None].to(nodes.dtype)
        nodes = (nodes + self.node_type(node_types)) * mask
        # A slot absent from every graph sends zero messages: skip it exactly.
        live = live_slots
        if live is None:
            live = torch.nonzero((graph != 0).any(dim=0)).squeeze(-1)
        live_src, live_dst = src[live], dst[live]
        messages = self.message(
            torch.cat(
                [
                    nodes.index_select(-2, live_src),
                    nodes.index_select(-2, live_dst),
                    self.edge_type(edge_types[live]).expand(b, t, -1, -1),
                ],
                dim=-1,
            )
        ) * graph[:, None, live, None].to(nodes.dtype)
        incoming = nodes.new_zeros(nodes.shape).index_add(-2, live_dst, messages[..., :h])
        outgoing = nodes.new_zeros(nodes.shape).index_add(-2, live_src, messages[..., h:])
        # Normalize by possible active incident edges, not present-edge degree:
        # otherwise graphs differing only in multiplicity would lose that signal.
        active = active_nodes.to(nodes.dtype)
        possible = active.index_select(-1, src) * active.index_select(-1, dst)
        in_count = active.new_zeros(active.shape).index_add(-1, dst, possible).clamp_min(1)
        out_count = active.new_zeros(active.shape).index_add(-1, src, possible).clamp_min(1)
        update = (
            self.node_update(
                torch.cat(
                    [
                        nodes,
                        incoming / in_count[:, None, :, None],
                        outgoing / out_count[:, None, :, None],
                    ],
                    dim=-1,
                )
            )
            * mask
        )
        pooled = update.sum(dim=-2) / active.sum(dim=-1)[:, None, None].clamp_min(1)
        conditioned = (update + pooled.unsqueeze(-2)) * mask
        return (
            weekly + self.weekly_update(pooled),
            channels + conditioned[..., : layout.K, :],
            controls + conditioned[..., layout.K : layout.K + layout.M, :],
        )


def binary_graphs_from_flow(g_samples: Tensor, threshold: Tensor | float = 0.0) -> Tensor:
    """Threshold the query-week mean to produce one complete graph per draw.

    ``g_samples`` is ``(S, ..., Tq, n_slots)`` with signed edge columns repeated
    across query weeks. The threshold broadcasts against the trailing slot
    axis; the default zero threshold is the strict sign rule.
    """
    return (g_samples.mean(dim=-2) > threshold).to(g_samples.dtype)


def empirical_full_graph_mode(graphs: Tensor) -> Tensor:
    """Per-batch empirical full-vector mode; lexicographically first breaks ties.

    ``graphs[S,B,L]`` must already be binary. Unlike thresholding edge marginals,
    every returned graph was actually drawn from the joint router.
    """
    modes = []
    for batch_graphs in graphs.unbind(dim=1):
        unique, counts = torch.unique(batch_graphs, dim=0, sorted=True, return_counts=True)
        modes.append(unique[counts.argmax()])
    return torch.stack(modes)
