"""Trainable graph-conditional Marketing Foundational Model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from . import _metal
from .compiled import CompiledVelocity, SamplingStatus
from .config import LossWeights, ModelConfig
from .encoder import BiAttentionEncoder, SplitRepresentations
from .flow import (
    _FLOW_SAMPLE_CHUNK,
    _FLOW_SAMPLE_ROWS,
    DEFAULT_N_SAMPLES,
    JointFlowHead,
    _draw_host_noise,
    _edge_slot_layout,
    _map_batches,
    _noise_device,
    _NoisePrefetch,
    _NoiseRequest,
    _random_seed,
    _sample_count,
    _sample_rows,
    _sampling_generator,
    _to_device,
)
from .graph import (
    STRUCT_EDGE_KEYS,
    STRUCT_MASK_KEYS,
    EdgeSlotLayout,
    GraphConditioner,
    StructureHead,
    binary_graphs_from_flow,
    empirical_full_graph_mode,
    structure_active_masks,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from xarray import DataTree

    from pymc_mfm.inference.curves import ResponseCurves
    from pymc_mfm.inference.dataframe import _FittedContext, _PredictionSnapshot
    from pymc_mfm.inference.result import DecompositionResult
    from pymc_mfm.plotting.charts import Plots


def _move_together(tensors: Sequence[Tensor], device: torch.device) -> list[Tensor]:
    """``tensors`` on ``device``, moved with one copy when they share a dtype.

    Each blocking copy waits for the GPU; a host-to-MPS copy does not wait (see ``_to_device``).
    """
    if all(tensor.device == device for tensor in tensors):
        return list(tensors)
    if len({tensor.dtype for tensor in tensors}) > 1:
        return [_to_device(tensor, device) for tensor in tensors]
    flat = _to_device(torch.cat([tensor.reshape(-1) for tensor in tensors]), device)
    parts = flat.split([tensor.numel() for tensor in tensors])
    return [part.view(tensor.shape) for part, tensor in zip(parts, tensors, strict=True)]


def _distinct_task_graphs(graphs: Tensor, n_tasks: int) -> tuple[Tensor, Tensor]:
    """``torch.unique(keys, dim=0, return_inverse=True)`` of the ``[task, graph]`` keys.

    ``graphs[S*B, L]`` are binary host draws, task-minor. Each key is compared as its
    big-endian task index and packed graph bits, which order as the keys do, so the
    distinct keys, their order and the inverse match ``torch.unique`` at a fraction
    of its cost.
    """
    count = graphs.shape[0]
    task = torch.arange(n_tasks).repeat(count // n_tasks)
    keys = np.concatenate(
        [
            task.numpy().astype(">u4").view(np.uint8).reshape(count, 4),
            np.packbits((graphs > 0.5).numpy(), axis=-1),
        ],
        axis=1,
    )
    _, first, inverse = np.unique(
        keys.view(np.dtype((np.void, keys.shape[1]))).ravel(),
        return_index=True,
        return_inverse=True,
    )
    first = torch.from_numpy(first)
    distinct = torch.cat([task[first].unsqueeze(-1).to(graphs.dtype), graphs[first]], dim=-1)
    return distinct, torch.from_numpy(inverse.reshape(-1))


class MFM(nn.Module):
    """One shared encoder, a joint graph router and a graph-conditioned decoder.

    The encoder reads observed context and query covariates. Training supervises
    both windows' joint router marginals, both graph-conditioned decomposition
    marginals, and equal-family auxiliary structure probabilities. Context and
    query samples remain separate marginals, never paired full-horizon draws.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = ModelConfig() if config is None else config
        config = self.config
        self.encoder = BiAttentionEncoder(config)
        self.head = JointFlowHead(config, graph_router=True)
        self.struct_head = StructureHead(
            d_channel=config.d_model,
            d_global=config.d_model,
            n_demand=config.n_demand,
            n_controls=config.n_controls,
        )
        # Preserve construction order: the conditioner and decoder are fresh
        # modules initialized after the encoder, router and auxiliary head.
        self.graph_conditioner = GraphConditioner(config.d_model, config.d_model)
        self.conditional_head = JointFlowHead(config, graph_router=False)
        self._inference_context: _FittedContext | None = None
        self._prediction_snapshot: _PredictionSnapshot | None = None
        self._idata: DataTree | None = None
        self._inference_revision = 0

    def structure_logits(
        self,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor,
    ) -> dict[str, Tensor]:
        """Read edge logits from query-week-pooled entity representations."""
        return self.struct_head(
            channel_repr.mean(dim=1),
            weekly_repr.mean(dim=1),
            control_repr.mean(dim=1),
        )

    def _structure_loss(
        self,
        batch: dict[str, Tensor],
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor,
        *,
        weights: LossWeights,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        g_true = {
            edge_type: batch[f"g_{edge_type}"]
            for edge_type in STRUCT_EDGE_KEYS
            if f"g_{edge_type}" in batch
        }
        if not g_true:
            raise ValueError("structure loss requires g_* targets in the training batch")
        all_logits = self.structure_logits(weekly_repr, channel_repr, control_repr)
        g_true = {
            edge_type: target for edge_type, target in g_true.items() if edge_type in all_logits
        }
        if not g_true:
            raise ValueError("this geometry has no non-empty structure heads")
        logits = {edge_type: all_logits[edge_type] for edge_type in g_true}
        all_active_masks = structure_active_masks(batch)
        active_mask = (
            {
                edge_type: all_active_masks[edge_type]
                for edge_type in g_true
                if edge_type in all_active_masks
            }
            if all_active_masks is not None
            else None
        )
        bce_result = self.struct_head.loss(
            logits,
            g_true,
            active_mask,
            edge_weights={"cy": weights.struct_cy_weight},
            return_trace=return_trace,
        )
        if return_trace:
            bce_loss, bce_trace = bce_result
        else:
            bce_loss = bce_result
        loss = weights.struct_weight * bce_loss
        if not return_trace:
            return loss
        return loss, {
            "objective": "per_family_structure_bce",
            "weight": weights.struct_weight,
            "unweighted_loss": bce_loss,
            "weighted_numerator": weights.struct_weight * bce_trace["combined_numerator"],
            "weighted_denominator": bce_trace["combined_denominator"],
            "bce": bce_trace,
        }

    def _joint_graph_layout(self, batch: dict[str, Tensor]) -> EdgeSlotLayout:
        """Resolve the supervised graph layout; never used by inference."""
        missing = [edge_type for edge_type in STRUCT_EDGE_KEYS if f"g_{edge_type}" not in batch]
        if missing:
            raise ValueError(f"joint graph training requires all g_* targets; missing {missing}")
        return _edge_slot_layout(
            int(batch["g_cy"].shape[-1]),
            int(batch["g_zb"].shape[-1]),
            int(batch["g_db"].shape[-1]),
        )

    def _flow_targets(self, batch: dict[str, Tensor], key: str = "targets_z") -> Tensor:
        """Append the signed complete graph, repeated across the selected window."""
        targets_z = batch[key]
        layout = self._joint_graph_layout(batch)
        flat = layout.pack({edge_type: batch[f"g_{edge_type}"] for edge_type in STRUCT_EDGE_KEYS})
        g = 2.0 * flat.to(targets_z.dtype) - 1.0
        g_weekly = g.unsqueeze(1).expand(-1, targets_z.shape[1], -1)
        return torch.cat([targets_z, g_weekly], dim=-1)

    def _flow_state_mask(
        self, batch: dict[str, Tensor], host_masks: Mapping[str, Tensor] | None = None
    ) -> Tensor:
        """Mask decomposition entities and graph slots using only node activity.

        The decomposition prefix is ``[1 | active_c | active_m | 1,1,1,1]``;
        a graph slot is active exactly when both incident nodes are active. Given
        ``host_masks``, host copies of the batch's activity masks, the mask is built on
        the host.
        """
        missing = [name for name in ("active_c_mask", "active_m_mask") if name not in batch]
        if missing:
            raise ValueError(f"decomposition flow requires activity masks {missing}")
        active_c = batch["active_c_mask"]
        active_m = batch["active_m_mask"]
        if active_c.ndim != 2 or active_m.ndim != 2 or active_c.shape[0] != active_m.shape[0]:
            raise ValueError("decomposition activity masks must be rank-2 with a shared batch axis")
        if host_masks is not None:
            active_c, active_m = host_masks["active_c_mask"], host_masks["active_m_mask"]
        # Validate on a host copy: the masks are tiny, and each check of a GPU tensor would
        # wait for the device.
        host = torch.cat([active_c, active_m], dim=-1).cpu()
        for name, family, active in (
            ("active_c_mask", "media channels", host[:, : active_c.shape[-1]]),
            ("active_m_mask", "controls", host[:, active_c.shape[-1] :]),
        ):
            if not torch.isfinite(active).all() or ((active != 0) & (active != 1)).any():
                raise ValueError(f"{name} must be finite and binary")
            if active.shape[-1] > 0 and (active.sum(dim=-1) == 0).any():
                raise ValueError(f"decomposition flow has a task with no active {family}")

        n_channels = int(active_c.shape[-1])
        n_controls = int(active_m.shape[-1])
        fixed = torch.ones(
            active_c.shape[0],
            1 + self.head.n_fixed,
            device=active_c.device,
            dtype=active_c.dtype,
        )
        prefix = torch.cat([fixed[:, :1], active_c, active_m, fixed[:, 1:]], dim=-1)
        if "active_j_mask" not in batch:
            raise ValueError("joint graph flow requires active_j_mask")
        active_j = batch["active_j_mask"]
        if active_j.ndim != 2 or active_j.shape[0] != active_c.shape[0]:
            raise ValueError("active_j_mask must be rank-2 with the shared batch axis")
        if host_masks is not None:
            active_j = host_masks["active_j_mask"]
        host_j = active_j.cpu()
        if not torch.isfinite(host_j).all() or ((host_j != 0) & (host_j != 1)).any():
            raise ValueError("active_j_mask must be finite and binary")
        if host_j.shape[-1] > 0 and (host_j.sum(dim=-1) == 0).any():
            raise ValueError("joint graph has a task with no active demand entities")
        layout = _edge_slot_layout(n_channels, n_controls, int(active_j.shape[-1]))
        slot_mask = layout.active_slot_mask(active_c, active_m, active_j)
        return torch.cat([prefix, slot_mask], dim=-1).unsqueeze(1)

    def _graph_conditional_geometry(
        self,
        batch: dict[str, Tensor],
        representations: SplitRepresentations,
        host_masks: Mapping[str, Tensor] | None = None,
    ) -> tuple[EdgeSlotLayout, Tensor, Tensor]:
        """Resolve geometry from observations and padding masks, never graph truth.

        With ``host_masks`` (see :meth:`_flow_state_mask`) the state mask is on the host.
        """
        weekly, channels, controls = (
            representations.weekly,
            representations.channels,
            representations.controls,
        )
        if weekly.ndim != 3 or channels.shape[:2] != weekly.shape[:2]:
            raise ValueError("graph conditioning requires a shared [B,T] representation grid")
        if controls.shape[:2] != weekly.shape[:2] or min(weekly.shape[:2]) < 1:
            raise ValueError("graph conditioning requires nonempty aligned batch/window axes")
        layout = _edge_slot_layout(channels.shape[-2], controls.shape[-2], self.head.n_demand)
        for key, width in (
            ("active_c_mask", layout.K),
            ("active_m_mask", layout.M),
            ("active_j_mask", layout.J),
        ):
            if key not in batch or batch[key].shape != (weekly.shape[0], width):
                raise ValueError(f"{key} must have shape {(weekly.shape[0], width)}")
        state_mask = self._flow_state_mask(batch, host_masks)
        active_nodes = torch.cat(
            [
                batch["active_c_mask"],
                batch["active_m_mask"],
                batch["active_j_mask"],
                weekly.new_ones(weekly.shape[0], 2),
            ],
            dim=-1,
        )
        return layout, state_mask, active_nodes

    @staticmethod
    def _validate_conditional_graph(
        graph: Tensor,
        layout: EdgeSlotLayout,
        state_mask: Tensor,
        *,
        n_samples: int | None = None,
    ) -> Tensor:
        """Validate binary values and every draw's padding before dtype conversion."""
        if not isinstance(graph, Tensor):
            raise ValueError("graph must be a binary tensor")
        shape = (state_mask.shape[0], layout.n_slots)
        allowed = (shape,) if n_samples is None else (shape, (n_samples, *shape))
        if tuple(graph.shape) not in allowed:
            raise ValueError(f"graph shape must be one of {allowed}, got {tuple(graph.shape)}")
        # A non-finite value is neither 0 nor 1.
        if graph.is_complex() or ((graph != 0) & (graph != 1)).any():
            raise ValueError("graph must contain only finite binary values 0 or 1")
        graph = graph.to(device=state_mask.device, dtype=state_mask.dtype)
        active_slots = state_mask[:, 0, -layout.n_slots :]
        if ((graph != 0) & (active_slots == 0)).any():
            raise ValueError("graph has live edges incident to padded inactive nodes")
        if n_samples is not None and graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(n_samples, -1, -1)
        return graph

    def _condition_on_graph(
        self,
        representations: SplitRepresentations,
        graph: Tensor,
        layout: EdgeSlotLayout,
        active_nodes: Tensor,
        live_slots: Tensor | None = None,
    ) -> SplitRepresentations:
        return SplitRepresentations(
            *self.graph_conditioner(
                representations.weekly,
                representations.channels,
                representations.controls,
                graph,
                layout,
                active_nodes,
                live_slots,
            )
        )

    def _conditioned_state_mask(
        self, mask: Tensor, graph: Tensor, layout: EdgeSlotLayout
    ) -> Tensor:
        """Transport only coordinates allowed by the supplied conditioning graph."""
        mask = mask.clone()
        n_dec = mask.shape[-1]
        for gate, contribution_slice, edge_slice in self.head.zero_atom_gate_slices(
            layout.K, layout.M
        ):
            live = graph[..., edge_slice.start - n_dec : edge_slice.stop - n_dec] > 0.5
            if gate.reduction == "any":
                live = live.any(dim=-1, keepdim=True)
            mask[..., contribution_slice].masked_fill_(~live.unsqueeze(-2), 0.0)
        return mask

    def loss_terms(
        self,
        batch: dict[str, Tensor],
        *,
        weights: LossWeights = LossWeights(),
        return_trace: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, object]]:
        """Execute every active training objective once with the recipe's ``weights``.

        ``struct_loss`` already includes ``weights.struct_weight``. ``task_loss``
        combines the query router loss, context router loss, and both conditional
        losses as documented on :class:`LossWeights`. The optional trace records
        the values and weights actually used, with the structure head's
        per-family normalization details.
        """
        if "context_targets_z" not in batch:
            raise ValueError("training requires context_targets_z and targets_z")
        timeline = self.encoder.forward_timeline(batch)
        context_repr, query_repr = timeline.context, timeline.query
        weekly_repr, channel_repr, control_repr = (
            query_repr.weekly,
            query_repr.channels,
            query_repr.controls,
        )
        flow_loss = self.head.loss(
            weekly_repr,
            channel_repr,
            self._flow_targets(batch),
            control_repr=control_repr,
            state_mask=self._flow_state_mask(batch),
        )
        # Each window is normalized by its own active-coordinate count. The
        # longer context window cannot outvote the query through its length.
        context_loss = self.head.loss(
            context_repr.weekly,
            context_repr.channels,
            self._flow_targets(batch, key="context_targets_z"),
            control_repr=context_repr.controls,
            state_mask=self._flow_state_mask(batch),
            graph_loss_weight=weights.context_graph_weight,
        )
        task_loss = flow_loss + weights.context_decomposition_weight * context_loss

        layout, conditional_mask, active_nodes = self._graph_conditional_geometry(batch, query_repr)
        # Generating graphs are training targets only. Inference never executes
        # this lookup and instead validates an explicit graph or a router draw.
        self._joint_graph_layout(batch)
        true_graph = self._validate_conditional_graph(
            layout.pack({edge_type: batch[f"g_{edge_type}"] for edge_type in STRUCT_EDGE_KEYS}),
            layout,
            conditional_mask,
        )
        n_dec = self.conditional_head.state_dim(layout.K, layout.M)
        conditional_mask = self._conditioned_state_mask(
            conditional_mask[..., :n_dec], true_graph, layout
        )
        conditional_repr = self._condition_on_graph(query_repr, true_graph, layout, active_nodes)
        conditional_loss = self.conditional_head.loss(
            conditional_repr.weekly,
            conditional_repr.channels,
            batch["targets_z"],
            control_repr=conditional_repr.controls,
            state_mask=conditional_mask,
        )
        task_loss = task_loss + weights.graph_conditional_weight * conditional_loss
        conditional_context_repr = self._condition_on_graph(
            context_repr, true_graph, layout, active_nodes
        )
        conditional_context_loss = self.conditional_head.loss(
            conditional_context_repr.weekly,
            conditional_context_repr.channels,
            batch["context_targets_z"],
            control_repr=conditional_context_repr.controls,
            state_mask=conditional_mask,
        )
        task_loss = task_loss + (
            weights.graph_conditional_weight
            * weights.context_decomposition_weight
            * conditional_context_loss
        )
        struct_result = self._structure_loss(
            batch,
            weekly_repr,
            channel_repr,
            control_repr,
            weights=weights,
            return_trace=return_trace,
        )
        if return_trace:
            struct_loss, struct_trace = struct_result
        else:
            struct_loss = struct_result
        total = task_loss + struct_loss
        terms = {
            "flow_loss": flow_loss,
            "context_flow_loss": context_loss,
            "graph_conditional_flow_loss": conditional_loss,
            "graph_conditional_context_flow_loss": conditional_context_loss,
            "task_loss": task_loss,
            "struct_loss": struct_loss,
            "loss": total,
        }
        if not return_trace:
            return terms
        trace = {
            "executed_objectives": [
                "flow_matching",
                "context_flow_matching",
                "graph_conditional_flow_matching",
                "graph_conditional_context_flow_matching",
                "per_family_structure_bce",
            ],
            "structure": struct_trace,
            "context_graph_weight": weights.context_graph_weight,
            "loss_reconstruction": {
                **terms,
                "context_decomposition_weight": weights.context_decomposition_weight,
                "graph_conditional_weight": weights.graph_conditional_weight,
                "graph_conditional_context_weight": (
                    weights.graph_conditional_weight * weights.context_decomposition_weight
                ),
            },
        }
        return terms, trace

    def loss(self, batch: dict[str, Tensor], *, weights: LossWeights = LossWeights()) -> Tensor:
        """Return the complete scalar training objective."""
        return self.loss_terms(batch, weights=weights)["loss"]

    def _add_context(
        self,
        data: DataFrame,
        *,
        date_column: str,
        target_column: str,
        channel_columns: Sequence[str],
        control_columns: Sequence[str],
    ) -> _FittedContext:
        """Prepare validated history without publishing it as model state."""
        from pymc_mfm.inference.dataframe import _FittedContext

        return _FittedContext.from_data(
            data,
            model_config=self.config,
            date_column=date_column,
            target_column=target_column,
            channel_columns=channel_columns,
            control_columns=control_columns,
        )

    def input_usage(
        self,
        context: DataFrame,
        query: DataFrame | None = None,
        *,
        date_column: str,
        target_column: str,
        channel_columns: Sequence[str],
        control_columns: Sequence[str],
    ) -> dict[str, object]:
        """Inspect explicit context/query input volume without sampling or storing state.

        Each window reports steps and allocated/represented time-entity tokens.
        With no query, report context only; otherwise also report the query and
        full-request totals. Inputs follow inference's role, cadence and numeric
        validation. Counts do not measure model capacity, memory or runtime.
        """
        from pymc_mfm.inference._usage import _input_usage

        fitted = self._add_context(
            context,
            date_column=date_column,
            target_column=target_column,
            channel_columns=channel_columns,
            control_columns=control_columns,
        )
        if query is None:
            return _input_usage(
                context_steps=fitted.spend.shape[0],
                channel_slots=fitted.spend.shape[1],
                control_slots=fitted.controls.shape[1],
                represented_channels=len(fitted.channel_columns),
                represented_controls=int(fitted.control_mask.sum()),
            )

        from pymc_mfm.inference.api import _model_batch, observation_batch

        observations, _ = fitted.observations(query)
        batch = observation_batch(**observations, model_config=self.config, device="cpu")
        batch = _model_batch(batch, device="cpu", dtype=next(self.parameters()).dtype)
        return _input_usage(
            context_steps=batch["ctx_spend_z"].shape[1],
            query_steps=batch["qry_spend_z"].shape[1],
            channel_slots=batch["ctx_spend_z"].shape[2],
            control_slots=batch["ctx_controls_z"].shape[2],
            represented_channels=int(batch["active_c_mask"][0].count_nonzero()),
            represented_controls=int(batch["active_m_mask"][0].count_nonzero()),
        )

    def inference(
        self,
        context: DataFrame,
        query: DataFrame,
        *,
        date_column: str,
        target_column: str,
        channel_columns: Sequence[str],
        control_columns: Sequence[str],
        n_samples: int = DEFAULT_N_SAMPLES,
        mode: str = "model_average",
        graph=None,
        router_samples: int | None = None,
        units: str = "outcome",
        provenance: Mapping[str, object] | None = None,
        return_idata: bool = True,
        random_seed: int | None = None,
    ) -> DataTree | None:
        """Compute and store both marginals for an explicit history/future pair.

        Context dates must have a regular cadence, with positive observed outcomes,
        nonnegative media covariates and at least one media and control covariate.
        Query dates start at the next context period without gaps. Inputs are
        sorted and snapshotted; predictions never append themselves to history.
        Query outcomes, when supplied, are kept only for analysis and charts.

        Store a complete ``xarray.DataTree`` in :attr:`idata` on every successful
        call. Return that exact tree by default; ``return_idata=False`` returns
        ``None`` without skipping sampling, conversion or storage. Both windows
        are separate conditional marginals, not paired full-horizon trajectories,
        and their draws exclude observation noise.

        A successful rerun replaces the entire analysis and drops response curves.
        Failed calls preserve the last successful context, draws, tree and curves.
        Runtime analysis state is never included in ``save_pretrained``.

        ``random_seed`` (0 to 2**31-1) replays the same random numbers: the same seed and
        inputs give the same draws. The outputs agree up to floating-point round-off. They can
        differ slightly across devices, thread counts, and compiled or eager runs.
        On a CPU with the same thread count, a repeat gives the same result. A seeded call
        uses its own generator and does not draw from or seed PyTorch's global random state,
        so other models and ``torch`` calls cannot disturb it (``compile_sampling`` states
        one limit). Both windows share one stream: context continues where query ended.
        With ``random_seed=None`` they share PyTorch's default stream instead, so
        ``torch.manual_seed`` before ``inference`` still sets the draws and another
        unseeded call continues the stream. ``idata`` records an explicit seed.
        For deliberately different seeds, call ``decompose`` separately for each
        window. Neither ``predict`` nor ``decompose`` records the seed.
        """
        from pymc_mfm.inference._datatree import _require_arviz, prediction_datatree

        _require_arviz()
        random_seed = _random_seed(random_seed)
        # Use the same generator for both windows, including the default CPU stream
        # used for base noise on CPU, CUDA and MPS.
        device = next(self.parameters()).device
        if random_seed is None and _noise_device(device).type == "cpu":
            generator = torch.default_generator
        else:
            generator = _sampling_generator(random_seed, device)
        fitted = self._add_context(
            context,
            date_column=date_column,
            target_column=target_column,
            channel_columns=channel_columns,
            control_columns=control_columns,
        )
        prediction = self._add_query(
            fitted,
            query,
            n_samples=n_samples,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
            units=units,
            provenance=provenance,
            generator=generator,
        )
        context_result = self._sample_context(prediction, generator=generator)
        revision = self._inference_revision + 1
        tree = prediction_datatree(
            context_result, prediction, revision=revision, random_seed=random_seed
        )
        # Publish only the completed run: no candidate state leaks on failure.
        self._inference_context = fitted
        self._prediction_snapshot = prediction
        self._idata = tree
        self._inference_revision = revision
        return tree if return_idata else None

    def predict(
        self,
        query: DataFrame,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        mode: str = "model_average",
        graph=None,
        router_samples: int | None = None,
        units: str = "outcome",
        provenance: Mapping[str, object] | None = None,
        random_seed: int | None = None,
    ) -> DecompositionResult:
        """Sample query only using the last complete inference's history and roles.

        Return independent decomposition draws without replacing ``idata``,
        ``query_result``, model charts or response curves. Query dates begin
        immediately after the stored history, not after previous predictions.
        Sampling options use their normal defaults, not the previous run's.
        ``random_seed`` works as in ``inference``. The returned result does not record the
        seed; keep it yourself if you save the result.
        """
        generator = _sampling_generator(random_seed, next(self.parameters()).device)
        context = self._inference_context
        if context is None:
            raise RuntimeError("call inference(context, query, ...) before predict(query)")
        result, _, _, _ = self._sample_query(
            context,
            query,
            n_samples=n_samples,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
            units=units,
            provenance=provenance,
            generator=generator,
        )
        return result

    def _add_query(
        self,
        context: _FittedContext,
        query: DataFrame,
        *,
        n_samples: int,
        mode: str,
        graph,
        router_samples: int | None,
        units: str,
        provenance: Mapping[str, object] | None,
        generator: torch.Generator | None,
    ) -> _PredictionSnapshot:
        """Prepare a sampled query snapshot without publishing it as model state."""
        from pymc_mfm.inference.dataframe import _PredictionSnapshot

        result, observations, time, annotations = self._sample_query(
            context,
            query,
            n_samples=n_samples,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
            units=units,
            provenance=provenance,
            generator=generator,
        )
        return _PredictionSnapshot(
            fitted_context=context,
            observations=observations,
            time=time,
            query=result,
            # Derived result metadata is not caller provenance to replay.
            annotations=deepcopy(annotations),
            n_samples=result.n_samples,
            mode=result.mode,
            router_samples=(
                result.provenance["inference_config"]["router_samples"]
                if result.mode == "map"
                else None
            ),
            units=result.units,
            query_outcome=context.query_outcome(query),
        )

    def _sample_query(
        self,
        context: _FittedContext,
        query: DataFrame,
        *,
        n_samples: int,
        mode: str,
        graph,
        router_samples: int | None,
        units: str,
        provenance: Mapping[str, object] | None,
        generator: torch.Generator | None,
    ) -> tuple[DecompositionResult, dict[str, np.ndarray], np.ndarray, dict[str, object] | None]:
        """Compute query against an explicit candidate context without publishing it."""
        from pymc_mfm.inference._digest import json_bytes
        from pymc_mfm.inference.api import _decompose
        from pymc_mfm.inference.result import _require_json_text, _require_text

        observations, time = context.observations(query)
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        annotations = deepcopy(dict(provenance)) if provenance is not None else None
        if annotations is not None:
            json_bytes(annotations)
            _require_json_text(annotations, "provenance")
            state_sha256 = annotations.get("state_sha256")
            if state_sha256 is not None and not isinstance(state_sha256, str):
                raise ValueError("provenance state_sha256 must be text")
            for key in ("state_sha256", "data_sha256", "graph_source"):
                if "\x00" in str(annotations.get(key, "")):
                    raise ValueError(f"provenance {key} must not contain NUL characters")
        if not isinstance(units, str) or not units:
            raise ValueError("units must be a nonempty label")
        _require_text(units, "units")
        if "\x00" in units:
            raise ValueError("units must not contain NUL characters")
        units = str(units)  # a str subclass, such as np.str_, cannot be an HDF5 attribute
        result = _decompose(
            self,
            observations,
            n_samples=n_samples,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
            channel_names=context.channel_columns,
            control_names=context.control_names,
            time=time,
            units=units,
            provenance=annotations,
            generator=generator,
        )
        return result, observations, time, annotations

    def _sample_context(
        self, prediction: _PredictionSnapshot, *, generator: torch.Generator | None
    ) -> DecompositionResult:
        """Sample history for this candidate query without publishing either marginal."""
        from pymc_mfm.inference.api import _decompose

        query = prediction.query
        return _decompose(
            self,
            prediction.observations,
            n_samples=prediction.n_samples,
            window="context",
            mode=prediction.mode,
            graph=query.graphs[:, None, :] if prediction.mode == "fixed" else None,
            router_samples=prediction.router_samples,
            channel_names=query.channel_names,
            control_names=query.control_names,
            time=prediction.time,
            units=prediction.units,
            provenance=prediction.annotations,
            generator=generator,
        )

    @property
    def idata(self) -> DataTree | None:
        """The stored complete inference and curves, or ``None``; never samples or converts."""
        return self._idata

    @property
    def query_result(self) -> DecompositionResult | None:
        """The exact query draws of the last complete inference, never an independent predict."""
        prediction = self._prediction_snapshot
        return None if prediction is None else prediction.query

    @property
    def plot(self) -> Plots:
        """Sampling-free charts of the last complete inference and its response curves.

        Each method returns a matplotlib or plotly figure and imports its backend
        only when drawing. Independent ``predict`` calls do not change these charts.
        """
        from pymc_mfm.plotting.charts import Plots

        return Plots._live(lambda: self.idata)

    def response_curves(
        self,
        *,
        levels: Sequence[float] | None = None,
        channels: Sequence[str] | None = None,
        weeks: int | None = None,
        n_samples: int | None = None,
        random_seed: int | None = None,
    ) -> ResponseCurves:
        """Each media covariate's contribution as its level in the first forecast weeks scales.

        Needs a successful ``inference``. For each media covariate in ``channels``
        (default: every channel column) and each multiplier in ``levels`` (at least two;
        finite, nonnegative, sorted and deduplicated; default 0, 0.25, ..., 2), that
        covariate in the first ``weeks`` forecast weeks (default: all of them) is
        multiplied by the level and those weeks are decomposed again; each draw sums the
        covariate's contribution over them. The runs see the history and a plan of just
        those weeks, every other input as given to ``inference``, as ``inference`` on that
        shorter plan would: the model reads every week of a plan, so with ``weeks`` below
        the forecast length level 1 is that shorter plan, not the original inference.
        Level 1, when listed, is one unchanged-input run that every covariate shares, and
        a covariate that is zero in those weeks reuses that run at every level (scaling
        zero changes no input); every other level runs once per covariate. Each run
        draws ``n_samples`` (default: the inference's) with the inference's mode, router
        samples, units and provenance; a ``mode="fixed"`` inference reuses its graphs,
        one per draw, so ``n_samples`` must equal its draw count.

        ``random_seed`` (0 to 2**31-1) replays the same random numbers: all runs of the
        sweep draw from one stream that the seed starts. The outputs agree up to floating-point
        round-off. They can differ slightly across devices, thread counts, and
        compiled or eager runs. On a CPU with the same thread count, a repeat gives the
        same result. A seeded call does not draw from or seed
        PyTorch's global random state. ``idata`` records the seed. With
        ``random_seed=None`` the runs draw from PyTorch's global generator and advance
        it. Levels are sampled separately, so draws are not paired across levels and
        intervals are pointwise.
        These are conditional contributions, not causal lift, and exclude observation
        noise. The result is kept in ``idata`` and for ``plot.response_curves`` until the
        next successful ``inference``. Independent ``predict`` and failed calls preserve it.
        """
        from pymc_mfm.inference._datatree import attach_response_curves
        from pymc_mfm.inference.curves import response_curves

        random_seed = _random_seed(random_seed)
        curves = response_curves(
            self,
            levels=levels,
            channels=channels,
            weeks=weeks,
            n_samples=n_samples,
            random_seed=random_seed,
        )
        tree = attach_response_curves(
            self._idata, curves, self._prediction_snapshot, random_seed=random_seed
        )
        self._idata = tree
        return curves

    def compile_sampling(self, mode: str = "default", *, dynamic: bool | None = True) -> MFM:
        """Compile the flow velocity that sampling evaluates, for faster repeated calls.

        Opt-in and per model; sampling stays eager until this is called. The first call
        at each new input shape compiles (tens of seconds on a GPU), so warm the model up
        with representative inputs before timing or serving. Compiled draws follow the same
        distribution as eager draws but are not bit-identical to them. With a fixed seed, a
        call replays the same random numbers, whether or not it compiles. While a compilation
        runs, PyTorch saves and then restores its global random state, which can undo global
        draws that another thread makes in that time. Warm the model up before you use other
        PyTorch random work in other threads.
        ``mode`` is a :func:`torch.compile` mode: ``"default"``, ``"reduce-overhead"`` or
        ``"max-autotune-no-cudagraphs"``. A compiler failure warns and returns this model to
        eager sampling; :attr:`sampling_status` reports it. Returns the model.
        """
        compiled = CompiledVelocity(mode, dynamic=dynamic)
        self.head._compiled_velocity = compiled
        self.conditional_head._compiled_velocity = compiled
        return self

    def disable_compiled_sampling(self) -> MFM:
        """Return sampling to eager execution; returns the model."""
        self.head._compiled_velocity = None
        self.conditional_head._compiled_velocity = None
        return self

    @property
    def sampling_status(self) -> SamplingStatus:
        """Whether sampling is compiled, with its compile count and any eager fallback."""
        compiled = self.head._compiled_velocity
        return SamplingStatus(compiled=False) if compiled is None else compiled.status()

    @torch.no_grad()
    def predict_batch(
        self,
        batch: dict[str, Tensor],
        n_samples: int = DEFAULT_N_SAMPLES,
        *,
        return_structure: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Draw the query joint-router marginal in normalized decomposition space.

        Samples have shape ``[S,B,Tquery,Ddec]``. ``return_structure`` also returns
        the signed graph flow coordinates ``[S,B,Tquery,L]`` that gate each draw.
        No decomposition targets or generating graphs are read.
        """
        weekly, channels, controls = self.encoder.forward_with_controls(batch)
        return self.sample(
            batch,
            weekly,
            channels,
            n_samples=n_samples,
            return_structure=return_structure,
            control_repr=controls,
        )

    @torch.no_grad()
    def sample_disentangled(
        self,
        batch: dict[str, Tensor],
        n_samples: int = DEFAULT_N_SAMPLES,
        *,
        mode: str = "model_average",
        graph: Tensor | None = None,
        router_samples: int | None = None,
        window: str = "query",
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw the learned decomposition posterior conditioned on complete graphs.

        Returns normalized decomposition ``[S,B,T,Ddec]`` and its exact binary
        conditioning graphs ``[S,B,L]``. Model averaging draws one query-router
        graph per fresh conditional draw. MAP selects the empirical complete-
        graph mode, breaking ties lexicographically; it never thresholds edge
        marginals. ``router_samples`` sets the MAP selection bank size.

        Fixed graphs must have shape ``[B,L]`` or exactly ``[S,B,L]``. Graph-
        forbidden coordinates remain zero throughout transport and decoding.
        Context and query calls are separate marginals, not aligned draws.
        ``random_seed`` seeds a generator that belongs to this call: sampling draws use
        that generator rather than PyTorch's global state. If enabled, :meth:`compile_sampling`
        can save and restore the global state during compilation. With ``random_seed=None``
        sampling draws from PyTorch's global generator and advances it. A call-owned
        ``generator`` from :func:`_sampling_generator` continues one stream over several
        calls; pass either it or ``random_seed``, not both. A request that fails a check
        draws nothing and starts no noise prefetch.
        """
        if mode not in ("model_average", "map", "fixed"):
            raise ValueError("mode must be 'model_average', 'map', or 'fixed'")
        if window not in ("query", "context"):
            raise ValueError("window must be 'query' or 'context'")
        if mode == "fixed" and graph is None:
            raise ValueError("mode='fixed' requires an explicit binary graph")
        if mode != "fixed" and graph is not None:
            raise ValueError("graph is only accepted with mode='fixed'")
        if mode != "map" and router_samples is not None:
            raise ValueError("router_samples is only accepted with mode='map'")

        s = _sample_count(n_samples)
        router_count = _sample_count(
            s if router_samples is None else router_samples, "router_samples"
        )
        query = batch["qry_spend_z"]
        if generator is None:
            generator = _sampling_generator(random_seed, query.device)
        elif random_seed is not None:
            raise ValueError("pass random_seed or generator, not both")
        # On a GPU the CPU generator draws the base noise (see _noise_device). A worker thread
        # draws the router's and then the decoder's chunks, in the inline order, while the
        # GPU transports. Both requests wait for the geometry checks, so a rejected request
        # draws nothing. The router's chunks follow the query window's shape.
        prefetch = None
        if mode != "fixed" and _noise_device(query.device) != query.device:
            prefetch = _NoisePrefetch(pin=query.device.type == "cuda")
        # On an NVIDIA GPU the sampling kernels build in background threads while the GPU
        # encodes (see _cuda.prefetch); compiled sampling runs PyTorch operations instead.
        if query.device.type == "cuda" and self.head._compiled_velocity is None:
            _metal.prepare(query.device, (self.head, self.conditional_head))
        # On a GPU the host copies the activity masks before the encoder is queued; the host
        # then decides the live coordinates and graph masks without waiting for the GPU.
        host_masks = None
        if query.device.type in ("mps", "cuda") and all(key in batch for key in STRUCT_MASK_KEYS):
            host_masks = dict(
                zip(
                    STRUCT_MASK_KEYS,
                    _move_together([batch[key] for key in STRUCT_MASK_KEYS], torch.device("cpu")),
                    strict=True,
                )
            )
        try:
            if window == "context":
                timeline = self.encoder.forward_timeline(batch)
                query_repr, selected_repr = timeline.query, timeline.context
            else:
                query_repr = SplitRepresentations(*self.encoder.forward_with_controls(batch))
                selected_repr = query_repr
            layout, state_mask, active_nodes = self._graph_conditional_geometry(
                batch, selected_repr, host_masks
            )
            router_noise = None
            if prefetch is not None:
                router_noise = prefetch.request(
                    self.head.noise_shapes(
                        tuple(query.shape[:2]),
                        query.shape[-1],
                        batch["qry_controls_z"].shape[-1],
                        router_count,
                    ),
                    next(self.parameters()).dtype,
                    generator,
                )
            head = self.conditional_head
            n_dec = head.state_dim(layout.K, layout.M)
            b, t = selected_repr.weekly.shape[:2]
            weekly = selected_repr.weekly
            # Conditional base noise is drawn once per chunk of _FLOW_SAMPLE_CHUNK draws, in
            # draw order; seeded draws depend on this chunking, not on the batches below.
            noise_shapes = [
                (1, (min(start + _FLOW_SAMPLE_CHUNK, s) - start) * b, t, n_dec)
                for start in range(0, s, _FLOW_SAMPLE_CHUNK)
            ]
            decoder_noise = None
            if prefetch is not None:
                decoder_noise = prefetch.request(noise_shapes, weekly.dtype, generator)

            def draw_noise() -> tuple[Tensor, Tensor]:
                device = weekly.device
                if _noise_device(device) == device:
                    noise = [
                        head._draw_noise(
                            shape, device=device, dtype=weekly.dtype, generator=generator
                        )
                        for shape in noise_shapes
                    ]
                    return (
                        torch.cat([white for white, _ in noise], dim=1),
                        torch.cat([z for _, z in noise], dim=1),
                    )
                if decoder_noise is None:
                    uniform = _metal.uniform_normals()
                    noise = [
                        _draw_host_noise(shape, weekly.dtype, uniform, None, generator)
                        for shape in noise_shapes
                    ]
                else:
                    noise = list(decoder_noise)
                return tuple(
                    _metal.to_device(parts, device).view(1, s * b, *parts[0].shape[2:])
                    for parts in ([white for white, _ in noise], [z for _, z in noise])
                )

            if mode == "fixed":
                graphs = self._validate_conditional_graph(graph, layout, state_mask, n_samples=s)
                white, latents = draw_noise()
            else:
                _, graph_flow = self.sample(
                    batch,
                    query_repr.weekly,
                    query_repr.channels,
                    n_samples=router_count,
                    return_structure=True,
                    control_repr=query_repr.controls,
                    structure_only=True,
                    noise=router_noise,
                    state_mask=state_mask,
                    generator=generator,
                )
                graphs = binary_graphs_from_flow(graph_flow)
                # Taken before the graphs are read, so a GPU router runs while the CPU draws.
                white, latents = draw_noise()
        finally:
            if prefetch is not None:
                prefetch.close()

        # Conditioning depends only on a draw's task and graph: condition each distinct
        # (task, graph) pair once (per call or per transport batch, see below), then index it
        # for every draw that shares it. The host checks router graphs and finds the distinct
        # pairs and their masks: on a GPU, a copy each way instead of a round trip per step.
        device = weekly.device
        host_state_mask = state_mask.cpu()
        if mode != "fixed":
            graphs = self._validate_conditional_graph(
                graphs.cpu(), layout, host_state_mask, n_samples=router_count
            )
            if mode == "map":
                graphs = empirical_full_graph_mode(graphs).unsqueeze(0).expand(s, -1, -1)
        distinct, inverse = _distinct_task_graphs(graphs.cpu().reshape(s * b, layout.n_slots), b)
        tasks = distinct[:, 0].long()
        # The masks stay on the host, where transport decides its live coordinates.
        masks = self._conditioned_state_mask(
            host_state_mask[tasks][..., :n_dec], distinct[:, 1:], layout
        )
        live_width = int((masks != 0).reshape(-1, n_dec).any(dim=0).sum())
        # Draws per transport batch, from the decision the decoder's transports make: the
        # fused GPU kernels take larger batches than PyTorch operations (see _sample_rows).
        batch_draws = max(
            1,
            _sample_rows(head._uses_kernels(device, weekly.dtype, masks.shape))
            // (b * t * live_width),
        )
        if device.type == "cpu":
            # On the CPU each transport batch conditions its own distinct pairs in its
            # worker: a model drawing many distinct graphs conditions them concurrently,
            # and every product keeps the shape, and so the round-off, it had before.
            flat_graphs = graphs.reshape(s * b, layout.n_slots)

            def decode(start: int, stop: int) -> Tensor:
                n = stop - start
                rows = slice(start * b, stop * b)
                pairs, index = _distinct_task_graphs(flat_graphs[rows], b)
                pair_tasks = pairs[:, 0].long()
                conditioned = self._condition_on_graph(
                    SplitRepresentations(
                        weekly[pair_tasks],
                        selected_repr.channels[pair_tasks],
                        selected_repr.controls[pair_tasks],
                    ),
                    pairs[:, 1:],
                    layout,
                    active_nodes[pair_tasks],
                )
                weekly_rows = conditioned.weekly[index]
                x, trajectory_latent = head._mix_noise(
                    white[:, rows], latents[:, rows], weekly_rows
                )
                return head._transport(
                    x,
                    trajectory_latent,
                    weekly_rows,
                    conditioned.channels[index],
                    conditioned.controls[index],
                    masks[inverse[rows]],
                    apply_zero_atom=True,
                ).reshape(n, b, t, n_dec)

        else:
            # The GPU kernels read conditioned representations once per distinct pair. Pairs
            # that fit one transport batch are conditioned once per call; otherwise (a diffuse
            # graph posterior draws nearly one graph per draw) each batch conditions its own,
            # so conditioned representations never outgrow a batch.
            host_graphs = graphs.cpu().reshape(s * b, layout.n_slots)
            (graphs,) = _move_together((graphs,), device)
            # Pairs per graph-conditioner call: as many as a CPU transport batch holds, which
            # bounds the conditioner's intermediates (per pair, week and live slot).
            pair_chunk = max(1, _FLOW_SAMPLE_ROWS // (t * live_width))

            def condition(pairs: Tensor) -> tuple[SplitRepresentations, Tensor]:
                """Condition the distinct host ``pairs`` on the device; also their host masks."""
                pair_tasks = pairs[:, 0].long()
                pair_masks = self._conditioned_state_mask(
                    host_state_mask[pair_tasks][..., :n_dec], pairs[:, 1:], layout
                )
                live_slots = torch.nonzero((pairs[:, 1:] != 0).any(dim=0)).squeeze(-1)
                (pairs,) = _move_together((pairs,), device)
                pair_tasks, live_slots = _move_together((pair_tasks, live_slots), device)
                parts = [
                    self._condition_on_graph(
                        SplitRepresentations(
                            weekly[part_tasks],
                            selected_repr.channels[part_tasks],
                            selected_repr.controls[part_tasks],
                        ),
                        part_pairs[:, 1:],
                        layout,
                        active_nodes[part_tasks],
                        live_slots,
                    )
                    for part_tasks, part_pairs in zip(
                        pair_tasks.split(pair_chunk), pairs.split(pair_chunk), strict=True
                    )
                ]
                if len(parts) == 1:
                    return parts[0], pair_masks
                conditioned = SplitRepresentations(
                    torch.cat([part.weekly for part in parts]),
                    torch.cat([part.channels for part in parts]),
                    torch.cat([part.controls for part in parts]),
                )
                return conditioned, pair_masks

            if distinct.shape[0] <= batch_draws * b:
                every_pair = condition(distinct)
                (inverse,) = _move_together((inverse,), device)

                def conditioned_rows(rows: slice) -> tuple[SplitRepresentations, Tensor, Tensor]:
                    return (*every_pair, inverse[rows])

            else:

                def conditioned_rows(rows: slice) -> tuple[SplitRepresentations, Tensor, Tensor]:
                    pairs, index = _distinct_task_graphs(host_graphs[rows], b)
                    (index,) = _move_together((index,), device)
                    return (*condition(pairs), index)

            def decode(start: int, stop: int) -> Tensor:
                n = stop - start
                rows = slice(start * b, stop * b)
                conditioned, pair_masks, index = conditioned_rows(rows)
                x, trajectory_latent = head._mix_noise(
                    white[:, rows], latents[:, rows], conditioned.weekly[index]
                )
                return head._transport(
                    x,
                    trajectory_latent,
                    conditioned.weekly,
                    conditioned.channels,
                    conditioned.controls,
                    pair_masks,
                    apply_zero_atom=True,
                    representation_index=index,
                ).reshape(n, b, t, n_dec)

        sampled = torch.cat(_map_batches(decode, s, batch_draws, weekly.device), dim=0)
        for gate, contribution_slice, edge_slice in self.head.zero_atom_gate_slices(
            layout.K, layout.M
        ):
            live = graphs[..., edge_slice.start - n_dec : edge_slice.stop - n_dec] > 0.5
            if gate.reduction == "any":
                live = live.any(dim=-1, keepdim=True)
            sampled[..., contribution_slice].masked_fill_(~live.unsqueeze(-2), 0.0)
        return sampled, graphs

    @torch.no_grad()
    def sample_context(
        self,
        batch: dict[str, Tensor],
        n_samples: int = DEFAULT_N_SAMPLES,
        return_structure: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Draw the trained context joint marginal without reading graph truth.

        The result uses ``Tcontext`` in place of ``Tquery`` in :meth:`predict_batch`.
        It is not paired with a separately sampled query-window marginal.
        """
        timeline = self.encoder.forward_timeline(batch)
        return self.sample(
            batch,
            timeline.context.weekly,
            timeline.context.channels,
            n_samples=n_samples,
            return_structure=return_structure,
            control_repr=timeline.context.controls,
        )

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Tensor],
        weekly_repr: Tensor,
        channel_repr: Tensor,
        n_samples: int = DEFAULT_N_SAMPLES,
        return_structure: bool = False,
        control_repr: Tensor | None = None,
        *,
        structure_only: bool = False,
        noise: _NoiseRequest | None = None,
        state_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor | tuple[Tensor | None, Tensor]:
        """Draw a router marginal from already-encoded window representations.

        ``structure_only=True`` (with ``return_structure=True``) returns
        ``(None, g_samples)`` and skips the fixed confounder and indirect totals and
        other work that only the decomposition coordinates read: no graph coordinate
        depends on them, so the graph draws are unchanged. ``noise`` optionally supplies
        the router's prefetched base-noise chunks, and ``state_mask`` this batch's
        :meth:`_flow_state_mask` when the caller already has it. ``generator`` supplies
        the base noise in place of PyTorch's global generator.
        """
        if structure_only and not return_structure:
            raise ValueError("structure_only requires return_structure=True")
        c = channel_repr.shape[-2]
        m = control_repr.shape[-2] if control_repr is not None else self.head.n_controls
        n_decomp = 1 + c + m + self.head.n_fixed
        if state_mask is None:
            state_mask = self._flow_state_mask(batch)
        if structure_only:
            state_mask = state_mask.clone()
            state_mask[..., 1 + c + m : n_decomp] = 0.0
        flow_samples = self.head.sample(
            weekly_repr,
            channel_repr,
            n_samples=n_samples,
            control_repr=control_repr,
            state_mask=state_mask,
            noise=noise,
            graph_only=structure_only,
            generator=generator,
        )
        g_samples = flow_samples[..., n_decomp:]
        if structure_only:
            return None, g_samples
        flow_samples = flow_samples[..., :n_decomp]
        return (flow_samples, g_samples) if return_structure else flow_samples


def objective_trace_to_jsonable(value: Any) -> Any:
    """Detach an executed objective trace without changing its structure."""
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, dict):
        return {str(key): objective_trace_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [objective_trace_to_jsonable(item) for item in value]
    return value
