"""Native date coordinates survive inference, persisted evidence and its consumers."""

import io
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus, needs_arviz
from test_comparisons import omission_pair, record, reference
from test_diagnostics import make_result
from test_inference import observed

from pymc_mfm.diagnostics import build_report, compute_diagnostics, render_report
from pymc_mfm.diagnostics.comparisons import (
    comparison_summary,
    matched_omission,
    reference_diagnostics,
    score_record,
)
from pymc_mfm.diagnostics.core import json_safe
from pymc_mfm.inference.api import decompose
from pymc_mfm.inference.result import DecompositionResult

ISO_DATES = [
    "2024-01-01T12:00:00.123456789",
    "2024-01-03T12:00:00.123456789",
    "2024-01-07T12:00:00.123456789",
]
DATES = np.array(ISO_DATES, dtype="datetime64[ns]")


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(DATES, id="nanoseconds"),
        pytest.param(DATES.astype("datetime64[D]"), id="days"),
        pytest.param(
            np.array(["2024-01", "2024-03", "2024-07"], dtype="datetime64[M]"), id="months"
        ),
        pytest.param(DATES.astype(DATES.dtype.newbyteorder(">")), id="big-endian-nanoseconds"),
    ],
)
def test_saved_dates_retain_exact_values_dtype_and_read_only_snapshot(tmp_path, labels):
    supplied = labels.copy()
    result = replace(make_result(), time=supplied)
    supplied[0] = np.datetime64("2000-01-01")
    path = result.save(tmp_path / "result")
    loaded = DecompositionResult.load(path)
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["time"], labels)
        assert arrays["time"].dtype == labels.dtype
    for actual in (result, loaded):
        np.testing.assert_array_equal(actual.time, labels)
        assert actual.time.dtype == labels.dtype
        assert not actual.time.flags.writeable
    np.testing.assert_array_equal(loaded.components, result.components)
    np.testing.assert_array_equal(loaded.graphs, result.graphs)
    assert loaded.provenance == result.provenance
    assert json.loads((path / "result.json").read_text())["format_version"] == 2


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(np.array([DATES[0], np.datetime64("NaT"), DATES[2]]), id="NaT"),
        pytest.param(DATES[[0, 0, 2]], id="duplicate"),
        pytest.param(DATES[::-1], id="decreasing"),
        pytest.param(DATES[:2], id="wrong-length"),
        pytest.param(DATES[:, None], id="matrix"),
        pytest.param(np.array(ISO_DATES), id="unparsed-strings"),
        pytest.param(DATES.astype(object), id="object-coordinates"),
        pytest.param(np.arange(3).astype("timedelta64[D]"), id="timedeltas"),
    ],
)
def test_result_rejects_invalid_or_non_native_date_coordinates(labels):
    with pytest.raises(ValueError, match="time"):
        replace(make_result(), time=labels)


@pytest.mark.parametrize("window", ["context", "query"])
def test_decompose_dates_select_the_original_window_without_changing_draws(
    model, synthetic_corpus, window
):
    inputs = observed(synthetic_corpus)
    dates = DATES[0] + np.arange(len(inputs["spend"])) * np.timedelta64(7, "D")
    torch.manual_seed(82)
    numeric = decompose(model, inputs, time=np.arange(len(dates)), n_samples=3, window=window)
    torch.manual_seed(82)
    dated = decompose(model, inputs, time=dates, n_samples=3, window=window)
    expected = (
        dates[: dated.context_length] if window == "context" else dates[dated.context_length :]
    )
    np.testing.assert_array_equal(dated.time, expected)
    assert dated.time.dtype == dates.dtype
    np.testing.assert_array_equal(dated.components, numeric.components)
    np.testing.assert_array_equal(dated.graphs, numeric.graphs)
    assert dated.provenance == numeric.provenance


@pytest.mark.parametrize(
    "mutation", ["NaT", "duplicate", "decreasing", "short", "matrix", "strings"]
)
def test_invalid_full_horizon_dates_fail_before_sampling_or_rng_consumption(
    model, synthetic_corpus, monkeypatch, mutation
):
    dates = DATES[0] + np.arange(12) * np.timedelta64(7, "D")
    if mutation == "NaT":
        dates[0] = np.datetime64("NaT")  # Even a context label must be valid for a query result.
    elif mutation == "duplicate":
        dates[1] = dates[0]
    elif mutation == "decreasing":
        dates = dates[::-1]
    elif mutation == "short":
        dates = dates[:-1]
    elif mutation == "matrix":
        dates = dates[:, None]
    else:
        dates = dates.astype(str)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid time coordinates reached the sampler")

    monkeypatch.setattr(model, "sample_disentangled", forbidden)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="time"):
        decompose(model, observed(synthetic_corpus), time=dates, n_samples=3)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("labels", [DATES, DATES.astype(">M8[ns]")], ids=["native", "big-endian"])
def test_diagnostic_json_uses_iso_dates_at_array_and_scalar_boundaries(labels):
    result = replace(make_result(), time=labels)
    summary = compute_diagnostics(result)
    assert json.loads(json.dumps(summary))["metadata"]["time"] == ISO_DATES
    assert json_safe({"dates": labels, "date": labels[0], "scalar": np.array(labels[1])}) == {
        "dates": ISO_DATES,
        "date": ISO_DATES[0],
        "scalar": ISO_DATES[1],
    }
    for value in (np.datetime64("NaT"), np.array(["NaT"], dtype="datetime64[ns]")):
        with pytest.raises(ValueError, match="NaT"):
            json_safe(value)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([4, 7, 12], dtype=np.int64),
        np.array([2**63 + 1, 2**63 + 3, 2**63 + 7], dtype=np.uint64),
        np.array([0.5, 1.5, 3.5], dtype=np.float16),
        np.array([False, True]),
    ],
)
def test_numeric_result_labels_remain_numeric_in_saved_results_and_json(tmp_path, labels):
    original = make_result()
    result = replace(original, components=original.components[:, : len(labels)], time=labels)
    loaded = DecompositionResult.load(result.save(tmp_path / "result"))
    np.testing.assert_array_equal(loaded.time, labels)
    assert loaded.time.dtype == labels.dtype
    assert compute_diagnostics(loaded)["metadata"]["time"] == labels.tolist()


@pytest.mark.parametrize(
    "panel",
    [
        "channel_contributions",
        "components",
        "target_recovery",
        "cumulative",
        "graph_uncertainty",
    ],
)
def test_date_time_series_render_on_a_date_axis_instead_of_string_categories(panel):
    pytest.importorskip("matplotlib")
    from matplotlib.dates import AutoDateLocator, date2num

    from pymc_mfm.plotting.panels import plot_panel

    result = replace(make_result(), time=DATES)
    truth = result.components.mean(axis=0)
    summary = compute_diagnostics(
        result, truth=truth, graph_truth=[1, 0, 1], observed=truth.sum(axis=1) + 0.25
    )
    # The same JSON-loaded values are used by report rerendering.
    figure = plot_panel(panel, json.loads(json.dumps(summary)))
    try:
        image = io.BytesIO()
        figure.savefig(image, format="png")
        assert image.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
        axis = figure.axes[0]
        assert isinstance(axis.xaxis.get_major_locator(), AutoDateLocator)
        np.testing.assert_allclose(axis.lines[0].get_xdata(orig=False), date2num(DATES))
        boundary = next(line for line in axis.lines if line.get_label() == "query begins")
        expected = date2num(DATES)[0] - np.median(np.diff(date2num(DATES))) / 2
        np.testing.assert_allclose(boundary.get_xdata(), [expected, expected])
    finally:
        figure.clear()


@pytest.mark.parametrize("window", ["context", "query"])
@pytest.mark.parametrize("single", [False, True], ids=["multiple-dates", "single-date"])
def test_date_boundaries_follow_the_window_and_use_one_day_for_a_single_point(window, single):
    pytest.importorskip("matplotlib")
    from matplotlib.dates import date2num

    original = make_result()
    labels = DATES[:1] if single else DATES
    result = replace(
        original,
        time=labels,
        components=original.components[:, : len(labels)],
        window=window,
        context_length=len(labels) if window == "context" else original.context_length,
    )
    figure = result.plot.components()
    try:
        positions = date2num(labels)
        spacing = np.median(np.diff(positions)) if len(labels) > 1 else 1.0
        expected = (
            positions[-1] + spacing / 2 if window == "context" else positions[0] - spacing / 2
        )
        label = "context ends" if window == "context" else "query begins"
        line = next(line for line in figure.axes[0].lines if line.get_label() == label)
        np.testing.assert_allclose(line.get_xdata(), [expected, expected])
        figure.canvas.draw()
    finally:
        figure.clear()


def test_date_report_persists_iso_evidence_and_rerenders_without_rescoring(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from pymc_mfm.diagnostics import report as reporting

    result = replace(make_result(), time=DATES)
    truth = result.components.mean(axis=0)
    first = build_report(
        result,
        tmp_path / "first",
        truth=truth,
        graph_truth=[1, 0, 1],
        observed=truth.sum(axis=1) + [0.1, -0.2, 0.3],
        reference=reference(result),
    )
    assert first["status"] == "complete"
    computed = json.loads((tmp_path / "first" / "diagnostics.json").read_text())
    evidence = json.loads((tmp_path / "first" / "evidence.json").read_text())
    assert computed["metadata"]["time"] == evidence["reference"]["time"] == ISO_DATES

    def forbidden(*args, **kwargs):
        raise AssertionError("rendering attempted to rescore saved evidence")

    monkeypatch.setattr(reporting, "compute_diagnostics", forbidden)
    second = render_report(tmp_path / "first", tmp_path / "second")
    assert second["status"] == "complete"
    assert second["metrics"] == first["metrics"]
    for panel in (
        "target_recovery",
        "components",
        "cumulative",
        "graph_uncertainty",
        "reference_comparison",
    ):
        assert second["panels"][panel]["render_status"] == "ok"
    loaded = DecompositionResult.load(tmp_path / "second" / "result")
    np.testing.assert_array_equal(loaded.time, DATES)
    assert loaded.time.dtype == DATES.dtype


def test_failed_scoring_keeps_iso_dates_in_the_report_metadata(tmp_path):
    result = replace(make_result(), time=DATES)
    manifest = build_report(result, tmp_path / "failed", truth=np.zeros((2, 8)), strict=False)
    assert manifest["status"] == "failed"
    computed = json.loads((tmp_path / "failed" / "diagnostics.json").read_text())
    assert computed["metadata"]["time"] == ISO_DATES


def test_comparison_and_reference_json_dates_render_on_native_date_axes():
    pytest.importorskip("matplotlib")
    from matplotlib.dates import AutoDateLocator, date2num

    from pymc_mfm.plotting.comparisons import plot_reference_comparison, plot_world_comparison

    original = record()
    item = replace(original, result=replace(original.result, time=DATES[:2]))
    assert score_record(item)["time"] == ISO_DATES[:2]
    summary = json.loads(json.dumps(comparison_summary([item], bootstrap_samples=20)))
    reference_summary = json.loads(
        json.dumps(reference_diagnostics(item.result, reference(item.result)))
    )
    assert reference_summary["arrays"]["time"] == ISO_DATES[:2]
    figures = [
        plot_world_comparison(summary, case=item.case, world=item.world),
        plot_reference_comparison(reference_summary),
    ]
    try:
        for figure in figures:
            figure.canvas.draw()
            axis = figure.axes[0]
            assert isinstance(axis.xaxis.get_major_locator(), AutoDateLocator)
            np.testing.assert_allclose(axis.lines[0].get_xdata(orig=False), date2num(DATES[:2]))
    finally:
        for figure in figures:
            figure.clear()


def test_matched_omission_preserves_full_world_date_labels_and_checks_them_exactly():
    before, after = omission_pair()
    dates = DATES[0] + np.arange(4) * np.timedelta64(1, "D")
    dated = [
        replace(
            item,
            result=replace(item.result, time=dates[2:]),
            world_evidence={**item.world_evidence, "time": dates},
        )
        for item in (before, after)
    ]
    assert matched_omission(*dated) == matched_omission(before, after)
    changed = dates.copy()
    changed[0] += np.timedelta64(1, "ns")
    mismatched = replace(dated[1], world_evidence={**dated[1].world_evidence, "time": changed})
    with pytest.raises(ValueError, match="unmatched original world evidence: time"):
        matched_omission(dated[0], mismatched)


def test_omission_lookup_keeps_the_full_world_in_its_original_date_resolution():
    before, after = omission_pair()
    dates = np.array(
        ["1500-01-01", "1501-01-01", "2024-01-01", "2024-01-02"], dtype="datetime64[D]"
    )
    dated = [
        replace(
            item,
            result=replace(item.result, time=dates[2:].astype("datetime64[ns]")),
            world_evidence={**item.world_evidence, "time": dates},
        )
        for item in (before, after)
    ]
    assert matched_omission(*dated) == matched_omission(before, after)


@needs_arviz
def test_datatree_exports_native_dates_on_each_aligned_evidence_group(tmp_path):
    result = replace(make_result(), time=DATES)
    loaded = DecompositionResult.load(result.save(tmp_path / "result"))
    truth = loaded.components.mean(axis=0)
    observed_values = truth.sum(axis=1) + 0.25
    tree = loaded.to_datatree(truth=truth, graph_truth=[1, 0, 1], observed=observed_values)
    for group in ("posterior", "truth", "observed_data"):
        assert tree[group]["time"].dtype.kind == "M"
        np.testing.assert_array_equal(tree[group]["time"].values, DATES)
    for index, label in enumerate(DATES):
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(time=label).values[0],
            loaded.components[:, index, loaded.component_mask],
        )
        assert tree["observed_data"]["outcome"].sel(time=label).item() == observed_values[index]


@needs_arviz
def test_datatree_normalizes_scaled_date_units_without_relabelling_rows():
    labels = np.array([1, 2, 3], dtype="datetime64[2D]")
    result = replace(make_result(), time=labels)
    tree = result.to_datatree()
    np.testing.assert_array_equal(
        tree["posterior"]["time"].values.astype("datetime64[D]"), labels.astype("datetime64[D]")
    )
    for index, label in enumerate(labels.astype("datetime64[D]")):
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(time=label).values[0],
            result.components[:, index, result.component_mask],
        )


@needs_arviz
def test_datatree_rejects_subnanosecond_labels_instead_of_collapsing_distinct_rows():
    result = replace(make_result(), time=np.array([1, 2, 1001], dtype="datetime64[ps]"))
    with pytest.raises(ValueError, match="lossless"):
        result.to_datatree()


@needs_arviz
def test_archive_datatree_unions_mixed_date_units_without_integer_ticks(model, tmp_path):
    from test_archive import capture, identity_for, write_archive

    corpora = [make_synthetic_corpus(n_weeks=9, context_lengths=(6,))]
    identity = identity_for(model, corpora, [0], [(0, 8), (0, 9)], n_samples=3)
    captured = []
    for result, truth, graph_truth, observed_values in capture(model, corpora, identity):
        if result.window == "query":
            labels = (
                np.array(["2024-01-01", "2024-01-03", "2024-01-07"], dtype="datetime64[D]")
                if result.provenance["row"] == 8
                else np.array(
                    ["2024-01-03", "2024-01-05T00:00:00.000000009", "2024-01-07"],
                    dtype="datetime64[ns]",
                )
            )
            result = replace(result, time=labels)
        captured.append((result, truth, graph_truth, observed_values))
    archive = write_archive(tmp_path / "archive", identity, captured)
    tree = archive.to_datatree("query")
    expected = np.array(
        ["2024-01-01", "2024-01-03", "2024-01-05T00:00:00.000000009", "2024-01-07"],
        dtype="datetime64[ns]",
    )
    assert tree["posterior"]["time"].dtype.kind == "M"
    np.testing.assert_array_equal(tree["posterior"]["time"].values, expected)
    for entry in archive.entries:
        if entry.window != "query":
            continue
        result, truth, _, observed_values = archive.record(entry.index)
        selection = {"world": entry.world_id, "time": result.time}
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(selection).values[0],
            result.components[..., result.component_mask],
        )
        np.testing.assert_array_equal(
            tree["observed_data"]["outcome"].sel(selection), observed_values
        )
    # Separate marginal windows may still use different kinds of labels.
    assert archive.to_datatree("context")["posterior"]["time"].dtype.kind in "iuf"

    mixed = [
        (replace(result, time=np.arange(3)), *evidence)
        if result.window == "query" and result.provenance["row"] == 9
        else (result, *evidence)
        for result, *evidence in captured
    ]
    with pytest.raises(ValueError, match="cannot mix numeric and datetime64"):
        write_archive(tmp_path / "mixed", identity, mixed).to_datatree("query")

    unrepresentable = [
        (
            replace(
                result,
                time=np.array(["3000-01-01", "3000-01-03", "3000-01-07"], dtype="datetime64[D]"),
            ),
            *evidence,
        )
        if result.window == "query" and result.provenance["row"] == 8
        else (result, *evidence)
        for result, *evidence in captured
    ]
    with pytest.raises(ValueError, match="lossless common time unit"):
        write_archive(tmp_path / "unrepresentable", identity, unrepresentable).to_datatree("query")


def test_comparisons_accept_equivalent_date_units_but_reject_overflowed_alignment():
    original = record()
    dates = np.array(["3000-01-01", "3000-01-03"], dtype="datetime64[D]")
    first = replace(original, result=replace(original.result, time=dates))
    second = replace(
        first, method="B", result=replace(first.result, time=dates.astype("datetime64[s]"))
    )
    summary = comparison_summary([first, second], method_pairs=[("A", "B")], bootstrap_samples=20)
    assert summary["paired"]
    overflowed = replace(second, result=replace(second.result, time=dates.astype("datetime64[ns]")))
    with pytest.raises(ValueError, match="identical time coordinates"):
        comparison_summary([first, overflowed], method_pairs=[("A", "B")], bootstrap_samples=20)


def test_rounded_numeric_coordinates_cannot_align_comparisons_or_reference_evidence():
    original = record()
    dates = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    item = replace(original, result=replace(original.result, time=dates))
    rounded = np.array([2**53, 2**53 + 4], dtype=np.float64)
    other = replace(item, method="B", result=replace(item.result, time=rounded))
    with pytest.raises(ValueError):
        comparison_summary([item, other], method_pairs=[("A", "B")], bootstrap_samples=20)
    external = reference(item.result)
    external["time"] = rounded
    with pytest.raises(ValueError):
        reference_diagnostics(item.result, external)


@pytest.mark.parametrize("year", ["0001", "0000", "10000"])
def test_calendar_boundary_and_outside_calendar_dates_render_with_exact_iso_labels(year):
    pytest.importorskip("matplotlib")
    from matplotlib.dates import date2num

    dates = np.array([f"{year}-01-01", f"{year}-01-03", f"{year}-01-07"], dtype="datetime64[D]")
    result = replace(make_result(), time=dates)
    figure = result.plot.components()
    try:
        figure.canvas.draw()
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(orig=False), date2num(dates))
        assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == dates.astype(
            str
        ).tolist()
    finally:
        figure.clear()


@needs_arviz
def test_archive_selects_one_lossless_resolution_for_modern_and_far_future_worlds(model, tmp_path):
    from test_archive import capture, identity_for, write_archive

    corpora = [make_synthetic_corpus(n_weeks=9, context_lengths=(6,))]
    identity = identity_for(model, corpora, [0], [(0, 8), (0, 9)], n_samples=3)
    captured = []
    for result, truth, graph_truth, observed_values in capture(model, corpora, identity):
        if result.window == "query":
            year = "2024" if result.provenance["row"] == 8 else "3000"
            labels = np.array(
                [f"{year}-01-01", f"{year}-01-03", f"{year}-01-07"], dtype="datetime64[D]"
            )
            result = replace(result, time=labels)
        captured.append((result, truth, graph_truth, observed_values))
    archive = write_archive(tmp_path / "archive", identity, captured)
    tree = archive.to_datatree("query")
    assert np.datetime_as_string(tree["posterior"]["time"].values, unit="D").tolist() == [
        "2024-01-01",
        "2024-01-03",
        "2024-01-07",
        "3000-01-01",
        "3000-01-03",
        "3000-01-07",
    ]
    for entry in archive.entries:
        if entry.window != "query":
            continue
        result, _, _, observed_values = archive.record(entry.index)
        selection = {"world": entry.world_id, "time": result.time}
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(selection).values[0],
            result.components[..., result.component_mask],
        )
        np.testing.assert_array_equal(
            tree["observed_data"]["outcome"].sel(selection), observed_values
        )


def test_unsigned_reference_dates_remain_aligned_after_json_normalization_and_report(tmp_path):
    pytest.importorskip("matplotlib")
    labels = np.array([1, 2**63 + 1], dtype=np.uint64)
    original = record().result
    result = replace(original, time=labels)
    external = reference(result)
    normalized = json.loads(json.dumps(json_safe(external)))
    assert reference_diagnostics(result, normalized) == reference_diagnostics(result, external)
    manifest = build_report(result, tmp_path / "report", reference=external)
    assert manifest["status"] == "complete"
    computed = json.loads((tmp_path / "report" / "diagnostics.json").read_text())
    evidence = json.loads((tmp_path / "report" / "evidence.json").read_text())
    assert computed["metadata"]["time"] == labels.tolist()
    assert evidence["reference"]["time"] == labels.tolist()
