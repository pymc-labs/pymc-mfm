"""Trainable graph-conditional Marketing Foundation Model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self

import torch
from torch import Tensor, nn

from .compiled import CompiledVelocity, SamplingStatus
from .config import LossWeights, ModelConfig
from .encoder import BiAttentionEncoder, SplitRepresentations
from .flow import (
    _FLOW_SAMPLE_CHUNK,
    DEFAULT_N_SAMPLES,
    JointFlowHead,
    _edge_slot_layout,
    _map_batches,
    _noise_device,
    _NoisePrefetch,
    _NoiseRequest,
    _sample_count,
    _sample_rows,
)
from .graph import (
    STRUCT_EDGE_KEYS,
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

    from pymc_mfm.inference.dataframe import _FittedContext, _PredictionSnapshot
    from pymc_mfm.inference.result import DecompositionResult


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
        self._context_decomposition: DecompositionResult | None = None
        self._idata: DataTree | None = None

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

    def _flow_state_mask(self, batch: dict[str, Tensor]) -> Tensor:
        """Mask decomposition entities and graph slots using only node activity.

        The decomposition prefix is ``[1 | active_c | active_m | 1,1,1,1]``;
        a graph slot is active exactly when both incident nodes are active.
        """
        missing = [name for name in ("active_c_mask", "active_m_mask") if name not in batch]
        if missing:
            raise ValueError(f"decomposition flow requires activity masks {missing}")
        active_c = batch["active_c_mask"]
        active_m = batch["active_m_mask"]
        if active_c.ndim != 2 or active_m.ndim != 2 or active_c.shape[0] != active_m.shape[0]:
            raise ValueError("decomposition activity masks must be rank-2 with a shared batch axis")
        for name, family, active in (
            ("active_c_mask", "media channels", active_c),
            ("active_m_mask", "controls", active_m),
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
        if not torch.isfinite(active_j).all() or ((active_j != 0) & (active_j != 1)).any():
            raise ValueError("active_j_mask must be finite and binary")
        if active_j.shape[-1] > 0 and (active_j.sum(dim=-1) == 0).any():
            raise ValueError("joint graph has a task with no active demand entities")
        layout = _edge_slot_layout(n_channels, n_controls, int(active_j.shape[-1]))
        slot_mask = layout.active_slot_mask(active_c, active_m, active_j)
        return torch.cat([prefix, slot_mask], dim=-1).unsqueeze(1)

    def _graph_conditional_geometry(
        self, batch: dict[str, Tensor], representations: SplitRepresentations
    ) -> tuple[EdgeSlotLayout, Tensor, Tensor]:
        """Resolve geometry from observations and padding masks, never graph truth."""
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
        state_mask = self._flow_state_mask(batch)
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
        if (
            graph.is_complex()
            or not torch.isfinite(graph).all()
            or ((graph != 0) & (graph != 1)).any()
        ):
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
    ) -> SplitRepresentations:
        return SplitRepresentations(
            *self.graph_conditioner(
                representations.weekly,
                representations.channels,
                representations.controls,
                graph,
                layout,
                active_nodes,
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

    def add_context(
        self,
        data: DataFrame,
        *,
        date_column: str,
        target_column: str,
        channel_columns: Sequence[str],
        control_columns: Sequence[str],
    ) -> Self:
        """Replace observed history and column roles without training the weights.

        Dates are sorted and must follow a regular cadence. History needs at
        least two rows, positive outcomes, nonnegative spend and at least one
        channel and control. Controls are padded to this model's capacity.
        A successful call replaces context and invalidates prediction/export
        state; a failed call preserves the last successful state.
        The stored history is independent of later changes to ``data``.
        """
        from pymc_mfm.inference.dataframe import _FittedContext

        context = _FittedContext.from_data(
            data,
            model_config=self.config,
            date_column=date_column,
            target_column=target_column,
            channel_columns=channel_columns,
            control_columns=control_columns,
        )
        self._inference_context = context
        self._prediction_snapshot = None
        self._context_decomposition = None
        self._idata = None
        return self

    def inference(
        self,
        data: DataFrame,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        mode: str = "model_average",
        graph=None,
        router_samples: int | None = None,
        units: str = "outcome",
        provenance: Mapping[str, object] | None = None,
    ) -> DecompositionResult:
        """Infer future conditional components using history loaded by ``add_context``.

        Supply the configured date, channel and control columns, starting at
        the next context period without gaps. Dates are sorted; query targets
        and unrelated columns are ignored. Every call reuses the same context
        and context-only scales, without appending predictions to history.
        Returned draws use original outcome units and exclude observation noise.
        A successful call retains its inputs and query draws for ``build_idata``,
        without sampling context or converting to a DataTree. Failed calls
        preserve the last successful prediction and export. Runtime snapshots
        are never included in ``save_pretrained``.
        """
        from pymc_mfm.inference.api import decompose
        from pymc_mfm.inference.dataframe import _PredictionSnapshot

        context = self._inference_context
        if context is None:
            raise RuntimeError("call add_context(history, ...) before inference(future_data)")
        observations, time = context.observations(data)
        annotations = deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else provenance
        query = decompose(
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
        )
        prediction = _PredictionSnapshot(
            fitted_context=context,
            observations=observations,
            time=time,
            query=query,
            # Keep caller annotations separate from the public result metadata.
            # Derived metadata is not valid provenance to replay on loaded models.
            annotations=deepcopy(annotations),
            n_samples=query.n_samples,
            mode=query.mode,
            router_samples=(
                query.provenance["inference_config"]["router_samples"]
                if query.mode == "map"
                else None
            ),
            units=query.units,
        )
        self._prediction_snapshot = prediction
        self._context_decomposition = None
        self._idata = None
        return query

    @property
    def idata(self) -> DataTree | None:
        """The explicitly stored current export, or ``None``; never samples or converts."""
        return self._idata

    def build_idata(self, *, store: bool = False) -> DataTree:
        """Export context and query subtrees for the latest successful prediction.

        A successful ``inference`` after the current ``add_context`` is required. The
        first build samples the missing context marginal and caches those draws;
        query is never resampled. Context and query remain separate marginals,
        not paired full-horizon trajectories.

        By default return a fresh DataTree without changing ``idata``. With
        ``store=True``, store and return the same object, reusing a current stored
        export if available. Successful add_context/inference calls drop these references
        without rewriting any earlier caller-held tree. Return-only builds still
        retain inference state, but not their export container.

        The optional ``pymc-mfm[arviz]`` dependencies are checked before context
        sampling. Array conversion and context inference happen only on demand.
        """
        prediction = self._prediction_snapshot
        if prediction is None:
            raise RuntimeError(
                "call inference(future_data) successfully after add_context(history, ...) "
                "before build_idata()"
            )
        if store and self._idata is not None:
            return self._idata

        from pymc_mfm.inference._datatree import _require_arviz, prediction_datatree
        from pymc_mfm.inference.api import decompose

        _require_arviz()
        context = self._context_decomposition
        if context is None:
            context = decompose(
                self,
                prediction.observations,
                n_samples=prediction.n_samples,
                window="context",
                mode=prediction.mode,
                graph=prediction.query.graphs[:, None, :] if prediction.mode == "fixed" else None,
                router_samples=prediction.router_samples,
                channel_names=prediction.query.channel_names,
                control_names=prediction.query.control_names,
                time=prediction.time,
                units=prediction.units,
                provenance=prediction.annotations,
            )
            # Cache inference before conversion: a failed export must not redraw
            # successfully sampled context on the next build.
            self._context_decomposition = context
        tree = prediction_datatree(
            context, prediction.query, observed=prediction.fitted_context.outcome
        )
        if store:
            self._idata = tree
        return tree

    def compile_sampling(self, mode: str = "default", *, dynamic: bool | None = True) -> MFM:
        """Compile the flow velocity that sampling evaluates, for faster repeated calls.

        Opt-in and per model; sampling stays eager until this is called. The first call
        at each new input shape compiles (tens of seconds on a GPU), so warm the model up
        with representative inputs before timing or serving. Compiled draws follow the same
        distribution as eager draws but are not bit-identical to them. With a fixed seed,
        serialized calls on one device repeat their draws, whether or not a call compiles.
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
        # On MPS the CPU generator draws the base noise (see _noise_device). A worker thread
        # draws the router's and then the decoder's chunks, in the inline order, while the
        # GPU encodes and transports. The router's chunks follow the query window's shape.
        query = batch["qry_spend_z"]
        prefetch = None
        if mode != "fixed" and _noise_device(query.device) != query.device:
            prefetch = _NoisePrefetch()
        try:
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
                )
            if window == "context":
                timeline = self.encoder.forward_timeline(batch)
                query_repr, selected_repr = timeline.query, timeline.context
            else:
                query_repr = SplitRepresentations(*self.encoder.forward_with_controls(batch))
                selected_repr = query_repr
            layout, state_mask, active_nodes = self._graph_conditional_geometry(
                batch, selected_repr
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
                decoder_noise = prefetch.request(noise_shapes, weekly.dtype)

            def draw_noise() -> tuple[Tensor, Tensor]:
                if decoder_noise is None:
                    device = _noise_device(weekly.device)
                    noise = [
                        head._draw_noise(shape, device=device, dtype=weekly.dtype)
                        for shape in noise_shapes
                    ]
                else:
                    noise = list(decoder_noise)
                return (
                    torch.cat([white for white, _ in noise], dim=1).to(weekly.device),
                    torch.cat([z for _, z in noise], dim=1).to(weekly.device),
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
                )
                # Taken before the graphs are read, so a GPU router runs while the CPU draws.
                white, latents = draw_noise()
                graphs = self._validate_conditional_graph(
                    binary_graphs_from_flow(graph_flow), layout, state_mask, n_samples=router_count
                )
                if mode == "map":
                    graphs = empirical_full_graph_mode(graphs).unsqueeze(0).expand(s, -1, -1)
        finally:
            if prefetch is not None:
                prefetch.close()

        # Conditioning depends only on a draw's task and graph: condition each distinct
        # (task, graph) pair once, then index it for every draw that shares it. The keys
        # are few and small, so the CPU finds the distinct pairs.
        flat_graphs = graphs.reshape(s * b, layout.n_slots)
        task = torch.arange(b, device=flat_graphs.device).repeat(s)
        keys = torch.cat([task.unsqueeze(-1).to(flat_graphs.dtype), flat_graphs], dim=-1)
        distinct, inverse = torch.unique(keys.cpu(), dim=0, return_inverse=True)
        distinct, inverse = distinct.to(keys.device), inverse.to(keys.device)
        tasks = distinct[:, 0].long()
        masks = self._conditioned_state_mask(
            state_mask[tasks][..., :n_dec], distinct[:, 1:], layout
        )
        live_width = int((masks != 0).reshape(-1, n_dec).any(dim=0).sum())
        conditioned = self._condition_on_graph(
            SplitRepresentations(
                weekly[tasks], selected_repr.channels[tasks], selected_repr.controls[tasks]
            ),
            distinct[:, 1:],
            layout,
            active_nodes[tasks],
        )

        def decode(start: int, stop: int) -> Tensor:
            n = stop - start
            rows = slice(start * b, stop * b)
            index = inverse[rows]
            x, trajectory_latent = head._mix_noise(
                white[:, rows], latents[:, rows], conditioned.weekly[index]
            )
            return head._transport(
                x,
                trajectory_latent,
                conditioned.weekly,
                conditioned.channels,
                conditioned.controls,
                masks,
                apply_zero_atom=True,
                representation_index=index,
            ).reshape(n, b, t, n_dec)

        sampled = torch.cat(
            _map_batches(
                decode,
                s,
                max(1, _sample_rows(weekly.device) // (b * t * live_width)),
                weekly.device,
            ),
            dim=0,
        )
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
    ) -> Tensor | tuple[Tensor | None, Tensor]:
        """Draw a router marginal from already-encoded window representations.

        ``structure_only=True`` (with ``return_structure=True``) returns
        ``(None, g_samples)`` and skips the fixed confounder and indirect totals:
        they feed no other coordinate, so the graph draws are unchanged. ``noise``
        optionally supplies the router's prefetched base-noise chunks.
        """
        if structure_only and not return_structure:
            raise ValueError("structure_only requires return_structure=True")
        c = channel_repr.shape[-2]
        m = control_repr.shape[-2] if control_repr is not None else self.head.n_controls
        n_decomp = 1 + c + m + self.head.n_fixed
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
