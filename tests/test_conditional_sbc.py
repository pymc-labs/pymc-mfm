"""Independent-cell conditional live-effect SBC and saved-evidence regression checks."""

import json
from dataclasses import replace

import numpy as np
import pytest

from pymc_mktfm.diagnostics import CalibrationReport, conditional_sbc_report, render_report
from pymc_mktfm.inference.result import DecompositionResult

PROVENANCE = {
    "independent_worlds": True,
    "prior": "independent Bernoulli gates; lognormal media and signed normal control",
    "data_sha256": "a" * 64,
    "synthetic_training_evaluation": True,
}


def _result(channel, control, channel_gate, control_gate, *, mode="model_average"):
    n = len(channel)
    components = np.zeros((n, 2, 3))
    components[:, :, 1] = channel[:, None] / 2
    components[:, :, 2] = control[:, None] / 2
    return DecompositionResult(
        components=components,
        graphs=np.column_stack((channel_gate, control_gate)),
        component_names=("baseline", "channel:media", "control:signed"),
        component_families=("baseline", "channel", "control"),
        component_mask=np.ones(3, dtype=bool),
        edge_names=("cy:channel:media->outcome", "zb:control:signed->outcome"),
        edge_mask=np.ones(2, dtype=bool),
        channel_names=("media",),
        control_names=("signed",),
        time=np.arange(2),
        window="context",
        mode=mode,
        units="outcome",
        provenance={
            "data_sha256": "b" * 64,
            "edge_families": ["cy", "zb"],
            "edge_endpoints": [["channel:media", "outcome"], ["control:signed", "outcome"]],
        },
        context_length=2,
    )


def _record(rng, *, shrink=False):
    true_gate = rng.binomial(1, 0.75, size=2)
    truth_total = np.array([rng.lognormal(), rng.normal()]) * true_gate
    gates = rng.binomial(1, 0.75, size=(64, 2))
    media = rng.lognormal(size=64) * gates[:, 0]
    control = rng.normal(size=64) * gates[:, 1]
    if shrink:
        for values, on in ((media, gates[:, 0] == 1), (control, gates[:, 1] == 1)):
            if on.any():
                center = values[on].mean()
                values[on] = center + 0.05 * (values[on] - center)
    result = _result(media, control, gates[:, 0], gates[:, 1])
    truth = np.zeros((2, 3))
    truth[:, 1:] = truth_total / 2
    return result, truth, true_gate, np.zeros(2)


def _card(records, **kwargs):
    return conditional_sbc_report(
        records,
        world_ids=[f"world-{i}" for i in range(len(records))],
        cell_ids=[("evaluation", i) for i in range(len(records))],
        provenance=PROVENANCE,
        simulations=100,
        **kwargs,
    )


def test_true_conditional_posterior_does_not_systematically_reject_but_shrinkage_does():
    pytest.importorskip("scipy")
    null_rejections = 0
    for seed in range(20):
        rng = np.random.default_rng(seed + 20)
        records = [_record(rng) for _ in range(100)]
        card = _card(records, seed=seed)
        null_rejections += bool(card["family_wise"]["rejecting_quantities"])
        assert card["families"]["control"]["conditional"]["n_ranks"] > 40
        assert card["families"]["channel"]["model_average"]["n_ranks"] == 100
    assert null_rejections <= 3

    reference = _record(np.random.default_rng(7))
    squeezed = _record(np.random.default_rng(7), shrink=True)
    for slot in (0, 1):
        on = reference[0].graphs[:, slot] == 1
        if on.any():
            np.testing.assert_allclose(
                reference[0].components[on, :, slot + 1].sum(axis=1).mean(),
                squeezed[0].components[on, :, slot + 1].sum(axis=1).mean(),
            )
    rng = np.random.default_rng(29)
    shrunk = [_record(rng, shrink=True) for _ in range(150)]
    card = _card(shrunk, seed=29)
    assert "control/conditional" in card["family_wise"]["rejecting_quantities"]
    assert "channel/conditional" in card["family_wise"]["rejecting_quantities"]


def test_family_picks_entity_before_truth_gate_eligibility():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(103)
    records = []
    for _ in range(50):
        true_a, true_b = rng.binomial(1, 0.5, size=2)
        gates = rng.binomial(1, 0.5, size=(32, 2))
        components = np.zeros((32, 2, 3))
        components[:, :, 1] = (gates[:, 0] * (1 + 2 * gates[:, 1]))[:, None] / 2
        components[:, :, 2] = gates[:, 1, None]
        truth = np.zeros((2, 3))
        truth[:, 1] = true_a * (1 + 2 * true_b) / 2
        truth[:, 2] = true_b
        result = DecompositionResult(
            components=components,
            graphs=gates,
            component_names=("baseline", "channel:a", "channel:b"),
            component_families=("baseline", "channel", "channel"),
            component_mask=np.ones(3),
            edge_names=("cy:channel:a->outcome", "cy:channel:b->outcome"),
            edge_mask=np.ones(2),
            channel_names=("a", "b"),
            control_names=(),
            time=np.arange(2),
            window="context",
            mode="model_average",
            units="outcome",
            provenance={
                "edge_families": ["cy", "cy"],
                "edge_endpoints": [["channel:a", "outcome"], ["channel:b", "outcome"]],
            },
            context_length=2,
        )
        records.append((result, truth, np.array([true_a, true_b]), np.zeros(2)))
    family = _card(records, seed=1)["families"]["channel"]
    assert family["model_average"]["n_ranks"] == 50
    assert len(set(family["model_average"]["world_ids"])) == 50
    assert len({tuple(cell) for cell in family["model_average"]["cell_ids"]}) == 50
    assert all(entity["model_average"]["n_ranks"] == 50 for entity in family["entities"].values())
    average = dict(
        zip(
            family["model_average"]["world_ids"],
            family["model_average"]["entity_names"],
            strict=True,
        )
    )
    assert family["conditional"]["n_ranks"] > 10
    assert len(set(family["conditional"]["world_ids"])) == family["conditional"]["n_ranks"]
    assert all(
        average[world] == entity
        for world, entity in zip(
            family["conditional"]["world_ids"], family["conditional"]["entity_names"], strict=True
        )
    )


def test_named_edge_gate_lookup_preserves_hand_computed_signed_ranks():
    pytest.importorskip("scipy")
    result = _result(
        np.array([0.0, 0.0, 1.0, 2.0]),
        np.array([-1.0, 0.0, 0.0, 0.0]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
    )
    result = replace(
        result,
        graphs=result.graphs[:, ::-1],
        edge_names=result.edge_names[::-1],
        provenance={
            **result.provenance,
            "edge_families": ["zb", "cy"],
            "edge_endpoints": [["control:signed", "outcome"], ["channel:media", "outcome"]],
        },
    )
    truth = np.zeros((2, 3))
    truth[:, 1:] = [0.25, -0.25]
    record = (result, truth, np.ones(2), np.zeros(2))
    card = _card([record, record])
    assert card["families"]["channel"]["model_average"]["ranks"] == [2, 2]
    assert card["families"]["channel"]["conditional"]["ranks"] == [0, 0]
    assert card["families"]["control"]["conditional"]["ranks"] == [1, 1]


def test_mixed_gate_on_counts_pool_pit_and_keep_exact_full_subsets():
    pytest.importorskip("scipy")
    records = []
    for count in (2, 3, *([4] * 30)):
        gates = np.array([1] * count + [0] * (4 - count))
        draws = np.array([1, 2, 3, 4]) * gates
        result = _result(draws, np.zeros(4), gates, np.zeros(4))
        truth = np.zeros((2, 3))
        truth[:, 1] = 0.25
        records.append((result, truth, np.array([1, 0]), np.zeros(2)))
    card = _card(records)
    stats = card["families"]["channel"]["conditional"]
    assert stats["n_on"] == [2, 3] + [4] * 30
    assert stats["ranks"] == [0] * 32
    assert all(0 <= pit < 1 / (count + 1) for pit, count in zip(stats["pit"], stats["n_on"]))
    exact = stats["exact_full_draw_count"]["4"]
    assert exact["n_ranks"] == 30
    assert card["family_wise"]["alpha_per_quantity"] == pytest.approx(0.05 / 14)
    assert stats["ecdf_upper"][50] == pytest.approx(
        0.5 + np.sqrt(np.log(2 / (0.05 / 14)) / (2 * 32))
    )
    from scipy.stats import binom

    tail = card["family_wise"]["alpha_per_quantity"] / (2 * 4)
    assert exact["ecdf_upper"][0] == pytest.approx(binom.ppf(1 - tail, 30, 1 / 5) / 30)
    assert exact["ecdf_band_reject"] == 1
    assert (
        "channel/conditional/exact_full_draw_count/4" in card["family_wise"]["rejecting_quantities"]
    )


def test_signed_control_truths_and_cell_dependence_are_not_filtered():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(41)
    records = []
    for _ in range(40):
        gates = np.ones(12, dtype=int)
        result = _result(np.exp(rng.normal(size=12)), rng.normal(size=12) - 1, gates, gates)
        truth = np.zeros((2, 3))
        truth[:, 1:] = [0.5, -0.5]
        records.append((result, truth, np.ones(2), np.zeros(2)))
    card = _card(records, seed=2)
    control = card["families"]["control"]["conditional"]
    assert control["n_ranks"] == 40
    assert control["exact_full_draw_count"]["12"]["n_ranks"] == 40
    assert card["families"]["control"]["skip_reasons"]["dead_truth_edge"] == 0

    duplicate = conditional_sbc_report(
        records,
        world_ids=[f"world-{i}" for i in range(40)],
        cell_ids=[("evaluation", i // 2) for i in range(40)],
        provenance=PROVENANCE,
        simulations=100,
    )
    assert duplicate["world_ids"] == [f"world-{i}" for i in range(0, 40, 2)]
    assert duplicate["cell_ids"] == [["evaluation", i] for i in range(20)]
    assert duplicate["families"]["control"]["conditional"]["ranks"] == control["ranks"][::2]


def test_skips_provenance_and_window_contracts():
    pytest.importorskip("scipy")
    gates = np.array([0, 0, 0, 1])
    result = _result(gates, np.ones(4), gates, np.ones(4))
    truth = np.zeros((2, 3))
    truth[:, 1:] = [0.5, -0.5]
    record = (result, truth, np.array([1, 0]), np.zeros(2))
    card = _card([record, record])
    assert card["families"]["channel"]["conditional"] is None
    assert card["families"]["channel"]["skip_reasons"]["fewer_than_two_edge_on_draws"] == 2
    assert card["families"]["control"]["skip_reasons"]["dead_truth_edge"] == 2
    other = replace(result, mode="map")
    card = _card([(other, truth, np.ones(2), np.zeros(2))] * 2)
    assert card["families"]["control"]["conditional"] is None
    assert card["families"]["control"]["model_average"] is None
    assert card["families"]["control"]["skip_reasons"]["not_model_average"] == 2
    assert card["families"]["control"]["status"] == "not_applicable"
    padded = _result(np.zeros(4), np.ones(4), np.zeros(4), np.ones(4))
    padded = replace(
        padded,
        component_mask=np.array([1, 0, 1]),
        edge_mask=np.array([0, 1]),
    )
    card = _card([(padded, truth * np.array([1, 0, 1]), np.array([0, 1]), np.zeros(2))] * 2)
    assert card["families"]["channel"]["skip_reasons"]["padded"] == 2
    assert card["families"]["channel"]["entities"]["channel:media"]["skip_reasons"]["padded"] == 2
    map_padded = replace(padded, mode="map")
    card = _card([(map_padded, truth * np.array([1, 0, 1]), np.array([0, 1]), np.zeros(2))] * 2)
    assert card["families"]["channel"]["status"] == "not_applicable"
    assert card["families"]["channel"]["entities"]["channel:media"]["status"] == "not_applicable"
    assert card["families"]["channel"]["skip_reasons"]["padded"] == 2
    with pytest.raises(ValueError, match="independent_worlds"):
        conditional_sbc_report(
            [record, record],
            world_ids=["a", "b"],
            cell_ids=[("evaluation", 1), ("evaluation", 2)],
            provenance={},
        )
    with pytest.raises(ValueError, match="prior description and data_sha256"):
        conditional_sbc_report(
            [record, record],
            world_ids=["a", "b"],
            cell_ids=[("evaluation", 1), ("evaluation", 2)],
            provenance={**PROVENANCE, "data_sha256": "not-a-digest"},
        )
    with pytest.raises(ValueError, match="uniquely identify"):
        conditional_sbc_report(
            [record, record],
            world_ids=["a", "a"],
            cell_ids=[("evaluation", 1), ("evaluation", 2)],
            provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="two independent generator cells"):
        conditional_sbc_report(
            [record, record],
            world_ids=["a", "b"],
            cell_ids=[("evaluation", 1), ("evaluation", 1)],
            provenance=PROVENANCE,
        )
    normalized = conditional_sbc_report(
        [record, record],
        world_ids=["a", "b"],
        cell_ids=[("evaluation", np.int64(1)), ("evaluation", np.int64(2))],
        provenance={**PROVENANCE, "epoch": np.int64(1)},
        simulations=100,
    )
    assert json.loads(json.dumps(normalized, allow_nan=False))["cell_ids"] == [
        ["evaluation", 1],
        ["evaluation", 2],
    ]
    with pytest.raises(ValueError, match="strict JSON"):
        conditional_sbc_report(
            [record, record],
            world_ids=["a", "b"],
            cell_ids=[("evaluation", 1), ("evaluation", 2)],
            provenance={**PROVENANCE, "description": float("nan")},
        )
    with pytest.raises(ValueError, match="edge_families"):
        missing = replace(result, provenance={})
        _card([(missing, truth, np.ones(2), np.zeros(2)), record])
    with pytest.raises(ValueError, match="cannot be pooled"):
        query = replace(result, window="query")
        _card([(query, truth, np.ones(2), np.zeros(2)), record])


def test_saved_card_replay_needs_no_model_or_rank_recomputation(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(21)
    card = _card([_record(rng) for _ in range(20)])
    json.dumps(card, allow_nan=False)
    from pymc_mktfm.diagnostics import calibration

    manifest = CalibrationReport(card).report(tmp_path / "card")
    saved = json.loads((tmp_path / "card" / "calibration.json").read_text())
    assert saved["families"]["channel"]["conditional"]["n_ranks"] > 0
    assert manifest["metrics"]["calibration/channel/conditional/n_ranks"] > 0
    assert (
        manifest["metrics"]["calibration/control/entities/control:signed/model_average/n_ranks"] > 0
    )
    assert manifest["statuses"]["calibration/channel/conditional"]["status"] == (
        "independent_cells_declared"
    )
    for kind in ("ranks", "ecdf", "coverage"):
        figure = CalibrationReport(card).plot.card(kind)
        assert any(axis.lines or axis.patches or axis.collections for axis in figure.axes)
        figure.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("replay cannot rescore or simulate")

    monkeypatch.setattr(calibration, "rank_uniformity_stats", forbidden)
    monkeypatch.setattr(calibration, "conditional_sbc_report", forbidden)
    replayed = render_report(tmp_path / "card", tmp_path / "replay")
    assert replayed["metrics"] == manifest["metrics"]
    assert replayed["statuses"] == manifest["statuses"]
