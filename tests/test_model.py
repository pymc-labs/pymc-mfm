"""Contracts at risk when pruning the research implementation."""

import pytest
import torch

from pymc_mktfm.model.graph import EdgeSlotLayout, empirical_full_graph_mode


def test_complete_objective_trains_every_active_component(model, batch):
    torch.manual_seed(31)
    loss = model.loss(batch)
    loss.backward()
    for group in ("encoder", "head", "struct_head", "graph_conditioner", "conditional_head"):
        gradients = [parameter.grad for parameter in getattr(model, group).parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    for head in (model.head, model.conditional_head):
        assert head.trajectory_state_proj.weight.grad[:, 1].abs().sum() > 0
        assert head.trajectory_loading_net[0].weight.grad.abs().sum() > 0


def test_single_week_windows_have_finite_training_gradients(model, batch):
    for key in batch:
        if key.startswith(("ctx_", "qry_")) or key in ("targets_z", "context_targets_z"):
            batch[key] = batch[key][:, :1]
    model.loss(batch).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_context_truth_changes_training_but_never_inference(model, batch):
    changed = dict(batch, context_targets_z=batch["context_targets_z"] + 10)
    torch.manual_seed(32)
    before = model.loss(batch)
    torch.manual_seed(32)
    after = model.loss(changed)
    assert not torch.equal(before, after)
    poisoned = {
        key: torch.full_like(value, float("nan"))
        if key.startswith("g_") or "targets" in key
        else value
        for key, value in batch.items()
    }
    torch.manual_seed(33)
    expected, graph = model.sample_disentangled(batch, 3)
    torch.manual_seed(33)
    actual, poisoned_graph = model.sample_disentangled(poisoned, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(poisoned_graph, graph, rtol=0, atol=0)


@pytest.mark.parametrize("window,weeks", [("query", 3), ("context", 5)])
def test_graph_support_keeps_draw_and_batch_alignment_across_chunks(model, batch, window, weeks):
    c, m, j = 3, 2, 2
    layout = EdgeSlotLayout(c, m, j)
    graph = layout.pack({key: batch[f"g_{key}"] for key in layout.slices})
    graphs = graph.unsqueeze(0).repeat(19, 1, 1)
    # Independent column/gate contract, not the implementation's gate table.
    gates = (
        ("cy", slice(1, 1 + c), False),
        ("zb", slice(1 + c, 1 + c + m), False),
        ("db", slice(1 + c + m, 2 + c + m), True),
        ("cc", slice(2 + c + m, 3 + c + m), True),
        ("zc", slice(3 + c + m, 4 + c + m), True),
        ("dc", slice(4 + c + m, 5 + c + m), True),
    )
    for draw in range(19):
        for row in range(2):
            family = gates[(draw + row) % len(gates)][0]
            graphs[draw, row, layout.slices[family]] = 0
    torch.manual_seed(34)
    samples, returned = model.sample_disentangled(
        batch, 19, mode="fixed", graph=graphs, window=window
    )
    assert samples.shape == (19, 2, weeks, 1 + c + m + 4)
    assert torch.equal(returned, graphs)
    for family, columns, aggregate in gates:
        live = graphs[..., layout.slices[family]].bool()
        if aggregate:
            live = live.any(dim=-1, keepdim=True)
        values = samples[..., columns]
        gated = values.masked_select(~live.unsqueeze(-2).expand_as(values))
        assert torch.equal(gated, torch.zeros_like(gated))
    changed_graph = graphs.clone()
    changed_graph[17, 1, layout.slices["cy"].start] = 1
    torch.manual_seed(34)
    changed, changed_returned = model.sample_disentangled(
        batch, 19, mode="fixed", graph=changed_graph, window=window
    )
    assert torch.equal(changed_returned, changed_graph)
    assert not torch.allclose(changed[17, 1, :, 0], samples[17, 1, :, 0])


def test_graph_mode_selects_complete_graph_not_independent_edge_majorities():
    graphs = torch.tensor([[1, 1, 0]] * 3 + [[1, 0, 1]] * 2 + [[0, 1, 1]] * 2).unsqueeze(1)
    torch.testing.assert_close(empirical_full_graph_mode(graphs), torch.tensor([[1, 1, 0]]))


@pytest.mark.parametrize("value", [float("nan"), 1 + 1e-9])
def test_fixed_graph_validates_original_precision_before_casting(model, batch, value):
    layout = EdgeSlotLayout(3, 2, 2)
    graph = torch.full((2, layout.n_slots), value, dtype=torch.float64)
    with pytest.raises(ValueError):
        model.sample_disentangled(batch, 3, mode="fixed", graph=graph)


SAMPLING_ENTRY_POINTS = ("predict_batch", "sample_context", "sample_disentangled", "sample", "head")


def _draws(model, batch, entry, **count):
    """Leading draws of one public sampling entry point, forwarding only the count."""
    if entry == "predict_batch":
        return model.predict_batch(batch, **count)
    if entry == "sample_context":
        return model.sample_context(batch, **count)
    if entry == "sample_disentangled":
        return model.sample_disentangled(batch, **count)[0]
    weekly, channels, controls = model.encoder.forward_with_controls(batch)
    if entry == "sample":
        return model.sample(batch, weekly, channels, control_repr=controls, **count)
    return model.head.sample(weekly, channels, control_repr=controls, **count)


@pytest.mark.parametrize("n_samples", [True, 2.5, 0])
@pytest.mark.parametrize("entry", SAMPLING_ENTRY_POINTS)
def test_draw_counts_are_never_coerced(model, batch, entry, n_samples):
    with pytest.raises(ValueError):
        _draws(model, batch, entry, n_samples=n_samples)


@pytest.mark.parametrize("router_samples", [True, 2.5, 0])
def test_map_router_bank_size_is_never_coerced(model, batch, router_samples):
    with pytest.raises(ValueError):
        model.sample_disentangled(batch, 3, mode="map", router_samples=router_samples)
