"""Saved results bind arrays and metadata while retaining supported older evidence."""

import copy
import hashlib
import json
import pickle
import tracemalloc
from dataclasses import fields, replace

import numpy as np
import pytest
from test_diagnostics import make_result
from test_inference import observed

from pymc_mfm.inference._digest import json_bytes
from pymc_mfm.inference.api import decompose
from pymc_mfm.inference.result import DecompositionResult


def read_payload(path):
    return json.loads((path / "result.json").read_text())


def write_payload(path, payload):
    (path / "result.json").write_text(json.dumps(payload))


def single_cell_result():
    """Every array has one element, so dataclass equality is defined for it."""
    return DecompositionResult(
        components=np.array([[[5.0]]]),
        graphs=np.array([[1]]),
        component_names=("baseline",),
        component_families=("baseline",),
        component_mask=np.array([True]),
        edge_names=("db:latent:0->outcome",),
        edge_mask=np.array([True]),
        channel_names=(),
        control_names=(),
        time=np.array([3]),
        window="query",
        mode="model_average",
        units="outcome",
        provenance={},
        context_length=3,
    )


def test_round_trip_preserves_every_field(tmp_path):
    result = make_result()
    replay = DecompositionResult.load(result.save(tmp_path / "saved"))
    for item in fields(DecompositionResult):
        if not item.init:
            continue
        expected, actual = getattr(result, item.name), getattr(replay, item.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
        else:
            assert actual == expected


@pytest.mark.parametrize(
    ("edit", "edited"),
    [
        pytest.param(
            lambda metadata: metadata.update(units="EUR"),
            lambda result: result.units == "EUR",
            id="units",
        ),
        pytest.param(
            lambda metadata: metadata["provenance"]["edge_endpoints"][0].__setitem__(0, "x"),
            lambda result: result.provenance["edge_endpoints"][0] == ["x", "sales"],
            id="provenance-leaf",
        ),
    ],
)
def test_edited_metadata_is_rejected_by_its_checksum(tmp_path, edit, edited):
    path = make_result().save(tmp_path / "saved")
    payload = read_payload(path)
    edit(payload["metadata"])
    write_payload(path, payload)
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        DecompositionResult.load(path)
    # With a matching checksum the same edit is otherwise valid evidence: only the
    # checksum stands between an edited result.json and silently relabeled draws.
    payload["metadata_sha256"] = hashlib.sha256(json_bytes(payload["metadata"])).hexdigest()
    write_payload(path, payload)
    assert edited(DecompositionResult.load(path))


def test_version_one_results_are_rejected_by_version(tmp_path):
    path = make_result().save(tmp_path / "saved")
    payload = read_payload(path)
    legacy = {key: payload[key] for key in ("arrays_sha256", "metadata")}
    write_payload(path, {"format_version": 1, **legacy})
    with pytest.raises(ValueError, match="format_version 1"):
        DecompositionResult.load(path)


def test_version_two_results_without_input_usage_preserve_original_evidence(
    model, synthetic_corpus, tmp_path
):
    result = decompose(
        model,
        observed(synthetic_corpus, row=2),
        n_samples=3,
        provenance={"scenario": "before input accounting"},
    )
    path = result.save(tmp_path / "older-result")
    payload = read_payload(path)
    assert payload["format_version"] == 2
    assert payload["metadata"]["provenance"].pop("input_usage") == result.provenance["input_usage"]
    payload["metadata_sha256"] = hashlib.sha256(json_bytes(payload["metadata"])).hexdigest()
    write_payload(path, payload)

    replay = DecompositionResult.load(path)
    assert "input_usage" not in replay.provenance
    assert replay.provenance == {
        key: value for key, value in result.provenance.items() if key != "input_usage"
    }
    for name in ("components", "graphs", "time", "component_mask", "edge_mask"):
        np.testing.assert_array_equal(getattr(replay, name), getattr(result, name))
    assert replay.context_length == result.context_length
    assert replay.window == result.window


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param({1: "top level"}, id="top-level"),
        pytest.param({"nested": {"inner": {2: 3}}}, id="nested-mapping"),
        pytest.param({"records": [{"fine": 1}, {3: 4}]}, id="mapping-in-list"),
    ],
)
def test_provenance_keys_must_be_strings_at_every_depth(annotation):
    result = make_result()
    with pytest.raises(ValueError, match="keys must be strings"):
        replace(result, provenance={**result.provenance, **annotation})


@pytest.mark.parametrize(
    ("field", "labels"),
    [
        pytest.param("units", {"units": "rev\ud800enue"}, id="units"),
        pytest.param("channel_names", {"channel_names": ("paid", "pad\udcffding")}, id="name"),
        pytest.param("provenance/note", {"note": "x\ud800"}, id="provenance-value"),
        pytest.param("provenance/nested key", {"nested": {"k\ud800": 1}}, id="provenance-key"),
    ],
)
def test_text_that_cannot_encode_as_utf8_is_rejected_naming_its_field(field, labels):
    # A lone surrogate, as surrogateescape-decoded bytes leave, breaks every figure and report.
    result = make_result()
    if field.startswith("provenance"):
        labels = {"provenance": {**result.provenance, **labels}}
    with pytest.raises(ValueError, match=field):
        replace(result, **labels)


def nested(depth, leaf):
    """``leaf`` inside ``depth`` nested lists."""
    for _ in range(depth):
        leaf = [leaf]
    return leaf


def test_deep_provenance_saves_and_is_checked_at_every_depth(tmp_path):
    result = make_result()
    deep = replace(result, provenance={**result.provenance, "note": nested(3000, "fine")})
    assert DecompositionResult.load(deep.save(tmp_path / "saved")).provenance == deep.provenance
    with pytest.raises(ValueError, match="provenance/note/0/0"):
        replace(result, provenance={**result.provenance, "note": nested(3000, "x\ud800")})
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError):
        replace(result, provenance={**result.provenance, "loop": circular})


def test_loading_streams_the_arrays_file_instead_of_buffering_it(tmp_path):
    base = make_result()
    draws, weeks = 250, 104
    rng = np.random.default_rng(0)
    result = replace(
        base,
        components=rng.normal(size=(draws, weeks, len(base.component_mask))) * base.component_mask,
        graphs=np.zeros((draws, base.graphs.shape[1])),
        time=np.arange(weeks) + base.context_length,
    )
    path = result.save(tmp_path / "saved")
    tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    tracemalloc.reset_peak()
    before = tracemalloc.get_traced_memory()[0]
    try:
        loaded = DecompositionResult.load(path)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if not tracing:
            tracemalloc.stop()
    np.testing.assert_array_equal(loaded.components, result.components)
    # The decoded draws and the result's own read-only copy take about two arrays; holding
    # the whole (barely compressible) arrays.npz as well would take almost a third.
    assert peak < 2.6 * result.components.nbytes


def test_cached_diagnostics_are_not_identity_or_saved_evidence(tmp_path):
    result = single_cell_result()
    twin = replace(result)
    cached = result.diagnostics()
    assert result.diagnostics() is cached
    assert result == twin
    assert "_cache" not in repr(result)
    assert twin.diagnostics() is not cached
    relabeled = replace(result, units="EUR")
    assert relabeled.diagnostics().summary["metadata"]["units"] == "EUR"
    path = result.save(tmp_path / "saved")
    assert "_cache" not in read_payload(path)["metadata"]
    assert DecompositionResult.load(path) == twin


def test_pickles_and_copies_leave_the_diagnostics_cache_behind():
    result = single_cell_result()
    twin = replace(result)
    cached = result.diagnostics()
    assert pickle.dumps(result) == pickle.dumps(twin)
    for restored in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
        assert restored == result
        assert restored.diagnostics() is not cached
        assert restored.diagnostics() is restored.diagnostics()
        # Restored draws stay read-only like the originals.
        with pytest.raises(ValueError):
            restored.components[0, 0, 0] = 1.0
