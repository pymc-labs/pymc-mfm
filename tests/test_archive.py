"""Evaluation archives: replay without the model, refusals, atomic publishing and tampering."""

import copy
import errno
import gc
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import tracemalloc
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus, needs_arviz
from test_diagnostics import make_result
from test_inference import observed
from test_result_persistence import nested

from pymc_mktfm.data.observations import observation_identity
from pymc_mktfm.diagnostics import (
    WINDOW_TOTAL_FAMILIES,
    ArchiveIntegrityError,
    EvaluationArchive,
    conditional_sbc_report,
    family_window_totals,
    result_effect_rows,
    sbc_report,
    strata_summary,
)
from pymc_mktfm.diagnostics import archive as archive_module
from pymc_mktfm.inference._digest import digest_bytes, json_bytes
from pymc_mktfm.inference.api import decompose
from pymc_mktfm.inference.result import DecompositionResult
from pymc_mktfm.model.config import ModelConfig
from pymc_mktfm.model.flow import FLOW_STEPS_SAMPLE
from pymc_mktfm.model.model import MFM
from pymc_mktfm.training import evaluation as training_evaluation
from pymc_mktfm.training._provenance import corpus_identity
from pymc_mktfm.training.evaluation import evaluate_rows
from pymc_mktfm.training.stream import batches_for_rows

TINY = {
    "d_model": 8,
    "n_layers": 1,
    "n_heads": 2,
    "d_ff": 16,
    "flow_hidden": 16,
    "n_controls": 2,
    "n_demand": 2,
}
#: Shard seed and prior hash of the corpus made with each generator seed.
SHARD = {0: 530001, 1: 530002, 2: 530003}
PRIOR = {seed: hashlib.sha256(f"prior-{seed}".encode()).hexdigest() for seed in SHARD}
#: Both corpora, held-out and training rows, both context lengths: (0, 8), (0, 9) and
#: (0, 0) have context 6, the others 9. (0, 8)/(0, 9), (1, 2)/(1, 3) and (1, 10)/(1, 11)
#: share a cell; cell 5 exists in both corpora.
ROWS = [(1, 10), (0, 8), (1, 2), (0, 11), (0, 9), (1, 3), (0, 0), (1, 11)]
WINDOWS = ("query", "context")
ARCHIVES = ("archived", "chunked")
#: Another configuration's digest, which no capture of the tiny model records.
OTHER_CONFIG = digest_bytes(json_bytes(ModelConfig(**{**TINY, "d_model": 16}).to_dict()))


class Built(NamedTuple):
    archive: EvaluationArchive
    captured: list[tuple[Any, ...]]
    identity: dict[str, Any]


class Context(NamedTuple):
    model: MFM
    corpora: list[dict[str, Any]]
    captured: list[tuple[Any, ...]]


def identity_for(
    model,
    corpora,
    generator_seeds,
    rows,
    *,
    n_samples=8,
    seed=5,
    worlds_per_call=None,
    leg="evaluation",
    heads=("joint", "conditional"),
    priors=None,
    null_content=False,
):
    """A hand-built archive identity, keyed by corpus content (no training builder)."""
    priors = {**PRIOR, **(priors or {})}
    records = []
    for corpus, generator_seed in zip(corpora, generator_seeds, strict=True):
        inputs = corpus_identity(corpus)
        prior = priors[generator_seed]
        records.append(
            {
                "shard_seed": SHARD[generator_seed],
                "n_worlds": int(corpus["is_val"].shape[0]),
                "actual_inputs_sha256": digest_bytes(json_bytes(inputs)),
                "content_sha256": None if null_content else inputs["content_sha256"],
                "source_sha256": None,
                "data_origin": "unknown" if prior is None else "synthetic",
                "prior_sha256": prior,
            }
        )
    worlds = {}
    for batch, group in batches_for_rows(corpora, rows, device="cpu"):
        for b, (corpus_index, row) in enumerate(group):
            worlds[(corpus_index, row)] = {
                "corpus_index": corpus_index,
                "row": row,
                "cell_id": int(corpora[corpus_index]["cell_id"][row]),
                "context_length": int(batch["ctx_spend_z"].shape[1]),
                "data_sha256": observation_identity(batch, world=b),
            }
    size = len(rows) if worlds_per_call is None else worlds_per_call
    calls = [
        {"worlds": list(range(start, min(start + size, len(rows)))), "seed": seed + k}
        for k, start in enumerate(range(0, len(rows), size))
    ]
    return {
        "model": {
            "state_sha256": "1" * 64,
            "config_sha256": digest_bytes(json_bytes(model.config.to_dict())),
            "checkpoint_sha256": "2" * 64,
            "bundle_sha256": None,
            "profile": None,
            "source_step": None,
        },
        "corpora": records,
        "evaluation": {
            "worlds": [worlds[key] for key in rows],
            "calls": calls,
            "leg": leg,
            "heads": list(heads),
            "n_samples": n_samples,
            "flow_steps": FLOW_STEPS_SAMPLE,
            "mode": "model_average",
            "units": "outcome",
            "device": "cpu",
        },
    }


def capture(model, corpora, identity, writer=None):
    """Run the declared calls; keep frozen copies of every sink tuple, in sink order."""
    evaluation = identity["evaluation"]
    worlds = evaluation["worlds"]
    captured = []

    def tee(result, truth, graph_truth, observations):
        frozen = []
        for array in (truth, graph_truth, observations):
            array = np.array(array, copy=True)
            array.setflags(write=False)
            frozen.append(array)
        captured.append((result, *frozen))
        if writer is not None:
            writer.add(result, truth, graph_truth, observations)

    for call in evaluation["calls"]:
        evaluate_rows(
            model,
            corpora,
            [(worlds[w]["corpus_index"], worlds[w]["row"]) for w in call["worlds"]],
            n_samples=evaluation["n_samples"],
            seed=call["seed"],
            device="cpu",
            leg=evaluation["leg"],
            heads=tuple(evaluation["heads"]),
            result_sink=tee,
        )
    return captured


def write_archive(path, identity, captured):
    """An archive written from already captured tuples, loaded back."""
    with EvaluationArchive.create(path, identity) as writer:
        for item in captured:
            writer.add(*item)
    return EvaluationArchive.load(writer.path)


def where(result):
    return result.provenance["corpus_index"], result.provenance["row"], result.window


def label_of(corpus_index):
    """The archive label of a corpus of the main fixture (generator seed = corpus index)."""
    return f"shard:{SHARD[corpus_index]}"


def world_id_of(result):
    corpus_index, row, _ = where(result)
    return f"{label_of(corpus_index)}/row:{row}"


def cell_of(result, corpora):
    corpus_index, row, _ = where(result)
    return label_of(corpus_index), int(corpora[corpus_index]["cell_id"][row])


def assert_same_result(actual, expected):
    for item in fields(DecompositionResult):
        if not item.init:
            continue
        left, right = getattr(actual, item.name), getattr(expected, item.name)
        if isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right)
            assert left.dtype == right.dtype, item.name
        else:
            assert left == right, item.name


def canonical(report):
    return json.dumps(report, sort_keys=True)


def copy_archive(source, destination):
    shutil.copytree(source, destination, symlinks=True)
    return destination


def entry_dir(root, index):
    return root / "entries" / f"{index:04d}"


def read_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def write_manifest(root, manifest):
    text = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (root / "manifest.json").write_text(text)


def edit_manifest(root, edit):
    manifest = read_manifest(root)
    edit(manifest)
    write_manifest(root, manifest)


def seal(root):
    """Re-sign the manifest's self-digest only; the digest formula is computed here."""
    manifest = read_manifest(root)
    body = {key: value for key, value in manifest.items() if key != "archive_sha256"}
    text = json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n"
    manifest["archive_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    write_manifest(root, manifest)


def resign(root):
    """Re-sign every listed file record from the current files, then the self-digest."""
    manifest = read_manifest(root)
    for index, item in enumerate(manifest["entries"]):
        for name in item["files"]:
            data = (entry_dir(root, index) / name).read_bytes()
            item["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    write_manifest(root, manifest)
    seal(root)


def flip(path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(bytes(data))


def swap_dirs(root, first, second):
    a, b = root / "entries" / first, root / "entries" / second
    parked = root / "entries" / ".parked"
    a.rename(parked)
    b.rename(a)
    parked.rename(b)


def rewrite_npz(path, change):
    with np.load(path) as stored:
        arrays = dict(stored)
    change(arrays)
    np.savez_compressed(path, **arrays)


def edit_result(root, index, edit):
    """Edit an entry's result metadata and re-sign its metadata_sha256 as save does."""
    path = entry_dir(root, index) / "result.json"
    payload = json.loads(path.read_text())
    edit(payload["metadata"])
    payload["metadata_sha256"] = hashlib.sha256(json_bytes(payload["metadata"])).hexdigest()
    path.write_text(json.dumps(payload, indent=2))


def edit_payload(root, index, edit):
    """Edit an entry's top-level result.json payload (not its metadata), re-signing nothing."""
    path = entry_dir(root, index) / "result.json"
    payload = json.loads(path.read_text())
    edit(payload)
    path.write_text(json.dumps(payload, indent=2))


def symlink_to_copy(path, outside):
    """Replace ``path`` by a symbolic link to a byte-identical copy outside the archive."""
    duplicate = outside / f"copy-of-{path.name}"
    if path.is_dir():
        shutil.copytree(path, duplicate)
        shutil.rmtree(path)
    else:
        shutil.copy2(path, duplicate)
        path.unlink()
    path.symlink_to(duplicate, target_is_directory=duplicate.is_dir())


def peak_traced(fn):
    """The exception ``fn`` raises (``None`` if none) and its peak traced allocation in bytes."""
    tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    tracemalloc.reset_peak()
    before = tracemalloc.get_traced_memory()[0]
    try:
        error = None
        try:
            fn()
        except Exception as exc:
            error = exc
        return error, tracemalloc.get_traced_memory()[1] - before
    finally:
        if not tracing:
            tracemalloc.stop()


def bounded(fn, timeout=10):
    """Call ``fn`` in a daemon thread, failing if it still runs after ``timeout`` seconds."""
    outcome = []

    def run():
        try:
            fn()
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"the read blocked for {timeout} s"
    if outcome:
        raise outcome[0]


def staging_left(parent, name):
    return [path.name for path in parent.iterdir() if path.name.startswith(f".{name}-")]


def hidden(parent):
    return [path.name for path in parent.iterdir() if path.name.startswith(".")]


def fill(writer, captured):
    for item in captured:
        writer.add(*item)


@pytest.fixture(scope="module")
def world():
    """The tiny model and the two corpora (generator seeds 0 and 1) of every archive here."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        model = MFM(ModelConfig(**TINY)).eval()
    return model, [make_synthetic_corpus(seed=0), make_synthetic_corpus(seed=1)]


@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    """``archived``: one call over ROWS; ``chunked``: calls of three worlds, seeds 5, 6, 7."""
    model, corpora = world
    root = tmp_path_factory.mktemp("archives")
    archives = {}
    with torch.random.fork_rng(devices=[]):
        for name, worlds_per_call in zip(ARCHIVES, (None, 3), strict=True):
            identity = identity_for(model, corpora, [0, 1], ROWS, worlds_per_call=worlds_per_call)
            with EvaluationArchive.create(root / name, identity) as writer:
                captured = capture(model, corpora, identity, writer)
            archives[name] = Built(EvaluationArchive.load(writer.path), captured, identity)
    return archives


@pytest.fixture(scope="module")
def mixed(world, tmp_path_factory):
    """One archive of two geometries: three channels over 12 weeks, five over 14 weeks.

    Corpus 0 rows 8 and 0, then rows 0 and 2 of generator seed 2's five-channel corpus,
    whose channel_3 no three-channel world has a slot for.
    """
    model, corpora = world
    sources = [corpora[0], make_synthetic_corpus(seed=2, k=5, n_weeks=14, context_lengths=(7,))]
    identity = identity_for(model, sources, [0, 2], [(0, 8), (1, 0), (0, 0), (1, 2)])
    root = tmp_path_factory.mktemp("mixed")
    with torch.random.fork_rng(devices=[]):
        with EvaluationArchive.create(root / "archive", identity) as writer:
            capture(model, sources, identity, writer)
    return EvaluationArchive.load(writer.path)


@pytest.mark.parametrize("name", ARCHIVES)
def test_loaded_records_equal_the_in_memory_captures_in_archive_order(built, world, name):
    archive, captured, identity = built[name]
    _, corpora = world
    assert len(archive) == len(captured) == 2 * len(ROWS)
    evaluation = identity["evaluation"]
    worlds = evaluation["worlds"]
    # The sink order of the declared calls is the archive order rule.
    rule = [
        (worlds[w]["corpus_index"], worlds[w]["row"], window)
        for call in evaluation["calls"]
        for w in sorted(call["worlds"], key=lambda member: worlds[member]["context_length"])
        for window in WINDOWS
    ]
    assert [where(result) for result, *_ in captured] == rule
    assert [(e.corpus_index, e.row, e.window) for e in archive.entries] == rule
    for entry, record, expected in zip(archive.entries, archive, captured, strict=True):
        result, *evidence = record
        assert_same_result(result, expected[0])
        for actual, wanted in zip(evidence, expected[1:], strict=True):
            np.testing.assert_array_equal(actual, wanted)
            assert actual.dtype == np.float64 and not actual.flags.writeable
        call = evaluation["calls"][entry.call]
        assert entry.world in call["worlds"]
        assert entry.seed == call["seed"] == result.provenance["seed"]
        assert (
            entry.context_length == result.context_length == worlds[entry.world]["context_length"]
        )
        assert entry.shard_seed == SHARD[entry.corpus_index]
        assert entry.world_id == f"{label_of(entry.corpus_index)}/row:{entry.row}"
        assert entry.cell == cell_of(result, corpora)
        assert entry.path == archive.path / "entries" / f"{entry.index:04d}"
    query = [item for item in captured if item[0].window == "query"]
    loaded = list(archive.records("query"))
    assert len(loaded) == len(query)
    for (result, truth, *_), (expected, wanted, *_) in zip(loaded, query, strict=True):
        assert where(result) == where(expected)
        np.testing.assert_array_equal(result.components, expected.components)
        np.testing.assert_array_equal(truth, wanted)
    detached = archive.identity
    detached["evaluation"]["worlds"].clear()
    assert archive.identity == identity


def test_strata_rows_from_an_archive_equal_those_of_the_in_memory_captures(built):
    archive, captured, _ = built["archived"]
    for window in WINDOWS:
        expected = [
            {**row, "world": world_id_of(result)}
            for result, truth, graph_truth, _ in captured
            if result.window == window
            for row in result_effect_rows(result, truth, graph_truth)
        ]
        rows = archive.effect_rows(window)
        assert rows == expected
        assert {row["world"] for row in rows} == {world_id_of(item[0]) for item in captured}
        assert canonical(strata_summary(rows)) == canonical(strata_summary(expected))


@pytest.mark.parametrize("name", ARCHIVES)
def test_sbc_reports_from_an_archive_equal_those_from_the_in_memory_captures(built, world, name):
    pytest.importorskip("scipy")
    _, corpora = world
    archive, captured, _ = built[name]
    for window in WINDOWS:
        records = [item for item in captured if item[0].window == window]
        ids = [world_id_of(item[0]) for item in records]
        cells = [cell_of(item[0], corpora) for item in records]
        first = [cells.index(cell) for cell in dict.fromkeys(cells)]
        inputs = archive.sbc_inputs(window)
        totals = [family_window_totals(records[i][0], records[i][1]) for i in first]
        expected = sbc_report(
            np.stack([draws for draws, _ in totals]),
            np.stack([truth for _, truth in totals]),
            world_ids=[ids[i] for i in first],
            quantity_names=list(WINDOW_TOTAL_FAMILIES),
            provenance=inputs["provenance"],
            simulations=100,
        )
        assert canonical(sbc_report(**inputs, simulations=100)) == canonical(expected)
        # Every record of the window: the report's own one-world-per-cell rule agrees.
        conditional = archive.conditional_sbc_inputs(window)
        expected = conditional_sbc_report(
            records,
            world_ids=ids,
            cell_ids=cells,
            provenance=conditional["provenance"],
            simulations=100,
        )
        actual = conditional_sbc_report(**conditional, simulations=100)
        assert canonical(actual) == canonical(expected)


@pytest.mark.parametrize("name", ARCHIVES)
def test_independent_entries_take_the_first_world_of_each_cell_in_archive_order(built, world, name):
    _, corpora = world
    archive, captured, _ = built[name]
    for window in WINDOWS:
        seen, expected = set(), []
        for result, *_ in captured:
            if result.window == window and cell_of(result, corpora) not in seen:
                seen.add(cell_of(result, corpora))
                expected.append(world_id_of(result))
        selected = archive.independent_entries(window)
        assert [entry.world_id for entry in selected] == expected
        assert all(entry.window == window for entry in selected)
        # Cell 5 of each shard stays its own cell; repeated cells keep their first world.
        assert {"shard:530001/row:11", "shard:530002/row:10"} <= set(expected)
        assert {"shard:530001/row:9", "shard:530002/row:3", "shard:530002/row:11"}.isdisjoint(
            expected
        )
        inputs = archive.sbc_inputs(window)
        assert inputs["samples"].shape == (len(expected), 8, len(WINDOW_TOTAL_FAMILIES))
        assert inputs["truth"].shape == (len(expected), len(WINDOW_TOTAL_FAMILIES))
        assert inputs["world_ids"] == expected
        assert inputs["quantity_names"] == list(WINDOW_TOTAL_FAMILIES)


def test_family_window_totals_sum_each_active_family_over_the_window():
    result = make_result()
    truth = np.arange(24, dtype=np.float64).reshape(3, 8)  # the padded channel column 2 is not 0
    draws, totals = family_window_totals(result, truth)
    # make_result draws per sample: baseline 10 + s + t, the channel 0/0/6/6/6, the control
    # summing to 0, latent s / 4 and each of the three indirect components t / 10.
    expected = np.array(
        [
            [33, 0, 0, 0.0, 0.9],
            [36, 0, 0, 0.75, 0.9],
            [39, 6, 0, 1.5, 0.9],
            [42, 6, 0, 2.25, 0.9],
            [45, 6, 0, 3.0, 0.9],
        ]
    )
    np.testing.assert_allclose(draws, expected, rtol=0, atol=1e-12)
    # Truth columns 0, 1 (not the padded 2), 3, 4 and 5 + 6 + 7, summed over three weeks.
    np.testing.assert_allclose(totals, [24, 27, 33, 36, 126], rtol=0, atol=1e-12)
    assert draws.dtype == totals.dtype == np.float64
    with pytest.raises(ValueError, match="align"):
        family_window_totals(result, truth[:-1])


def test_sbc_provenance_names_the_referenced_dataset_model_and_prior(built, world, tmp_path):
    model, corpora = world
    expected_prior = {
        "name": "verified_generation_priors",
        "corpus_prior_hashes": {"shard:530001": PRIOR[0], "shard:530002": PRIOR[1]},
    }
    for name, seeds in zip(ARCHIVES, ([5], [5, 6, 7]), strict=True):
        archive, _, identity = built[name]
        digests = sorted(record["actual_inputs_sha256"] for record in identity["corpora"])
        dataset = hashlib.sha256((json.dumps(digests, indent=2) + "\n").encode()).hexdigest()
        body = {k: v for k, v in read_manifest(archive.path).items() if k != "archive_sha256"}
        manifest_text = json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n"
        for window in WINDOWS:
            for inputs in (archive.sbc_inputs(window), archive.conditional_sbc_inputs(window)):
                provenance = inputs["provenance"]
                assert provenance["independent_worlds"] is True
                assert provenance["data_sha256"] == archive.dataset_sha256 == dataset
                assert (
                    provenance["archive_sha256"]
                    == archive.archive_sha256
                    == hashlib.sha256(manifest_text.encode()).hexdigest()
                )
                assert provenance["state_sha256"] == "1" * 64
                assert provenance["checkpoint_sha256"] == "2" * 64
                assert provenance["n_samples"] == 8
                assert provenance["prior"] == archive.prior == expected_prior
                assert provenance["synthetic_training_evaluation"] is True
                assert (
                    provenance["window"],
                    provenance["mode"],
                    provenance["units"],
                    provenance["data_split"],
                    provenance["head"],
                ) == (window, "model_average", "outcome", "evaluation", "conditional")
                assert provenance["evaluation_seeds"] == seeds
    archive, captured, identity = built["archived"]

    # The corpus order changes the identity, not the dataset, the prior, the ids or the cells.
    reordered_rows = [(1 - corpus, row) for corpus, row in ROWS]
    reordered = identity_for(model, corpora[::-1], [1, 0], reordered_rows)
    with EvaluationArchive.create(tmp_path / "reordered", reordered) as writer:
        capture(model, corpora[::-1], reordered, writer)
    other = EvaluationArchive.load(writer.path)
    assert other.identity["corpora"] != archive.identity["corpora"]
    assert other.dataset_sha256 == archive.dataset_sha256
    assert other.prior == archive.prior
    assert {e.world_id for e in other.entries} == {e.world_id for e in archive.entries}
    assert {e.cell for e in other.entries} == {e.cell for e in archive.entries}

    # Only the input digest can carry a change to a row no archived world uses.
    rows = [(0, 8), (1, 10)]
    changed = {**corpora[0], "outcome_raw": corpora[0]["outcome_raw"].copy()}
    changed["outcome_raw"][3, 0] += 1.0
    datasets = []
    for name, sources in (("before", corpora), ("after", [changed, corpora[1]])):
        declared = identity_for(model, sources, [0, 1], rows, null_content=True)
        with EvaluationArchive.create(tmp_path / name, declared) as writer:
            capture(model, sources, declared, writer)
        datasets.append(EvaluationArchive.load(writer.path).dataset_sha256)
    assert datasets[0] != datasets[1]

    # A declared corpus no world comes from changes neither the prior nor the dataset.
    third = identity_for(
        model, [*corpora, make_synthetic_corpus(seed=2)], [0, 1, 2], ROWS, priors={2: None}
    )
    unreferenced = write_archive(tmp_path / "unreferenced", third, captured)
    assert unreferenced.prior == archive.prior is not None
    assert unreferenced.synthetic is True
    assert unreferenced.dataset_sha256 == archive.dataset_sha256

    # Without a verified prior for a referenced corpus, SBC needs an explicit description.
    unknown = write_archive(
        tmp_path / "unknown", identity_for(model, corpora, [0, 1], ROWS, priors={1: None}), captured
    )
    assert unknown.prior is None and unknown.synthetic is False
    for convert in (unknown.sbc_inputs, unknown.conditional_sbc_inputs):
        with pytest.raises(ValueError, match="shard:530002"):
            convert("query")
        provenance = convert("query", prior="described")["provenance"]
        assert provenance["prior"] == "described"
        assert provenance["synthetic_training_evaluation"] is False


def label_grids(records, times, components):
    """Draws ``[1, S, world, time, component]``, truth and observations of ``records`` by label.

    Each value sits at its world's time and component labels; every other cell is NaN.
    """
    draws = np.full((1, records[0][0].n_samples, len(records), len(times), len(components)), np.nan)
    truths = np.full(draws.shape[2:], np.nan)
    outcomes = np.full(draws.shape[2:4], np.nan)
    for w, (result, truth, _, observations) in enumerate(records):
        for t, label in enumerate(result.time.tolist()):
            row = times.index(label)
            outcomes[w, row] = observations[t]
            for d in np.flatnonzero(result.component_mask):
                column = components.index(result.component_names[d])
                draws[0, :, w, row, column] = result.components[:, t, d]
                truths[w, row, column] = truth[t, d]
    return draws, truths, outcomes


@needs_arviz
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.array([False]), np.array([True])),
        (np.array([True]), np.array([2])),
        (np.array([True]), np.array([2.5])),
        (np.array([True]), np.array([1])),
        (np.array([1]), np.array([True])),
        (np.array([False]), np.array([0])),
        (np.array([0]), np.array([False])),
    ],
    ids=[
        "both-boolean",
        "boolean-integer",
        "boolean-float",
        "shared-label",
        "shared-reversed",
        "shared-zero",
        "shared-zero-reversed",
    ],
)
def test_original_boolean_and_numeric_archive_labels_select_their_evidence_rows(
    world, tmp_path, left, right
):
    from arviz_stats import summary

    model, _ = world
    corpora = [make_synthetic_corpus(n_weeks=7, context_lengths=(6,))]
    identity = identity_for(model, corpora, [0], [(0, 8), (0, 9)], n_samples=5)
    captured = []
    for result, truth, graph_truth, observations in capture(model, corpora, identity):
        if result.window == "query":
            result = replace(result, time=left if result.provenance["row"] == 8 else right)
        captured.append((result, truth, graph_truth, observations))
    archive = write_archive(tmp_path / "archive", identity, captured)
    tree = archive.to_datatree("query")
    for entry in archive.entries:
        if entry.window != "query":
            continue
        result, truth, _, observations = archive.record(entry.index)
        selection = {"world": entry.world_id, "time": result.time.tolist()[0]}
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(selection).values[0],
            result.components[:, 0][:, result.component_mask],
        )
        np.testing.assert_array_equal(
            tree["truth"]["components"].sel(selection).values, truth[0, result.component_mask]
        )
        assert tree["observed_data"]["outcome"].sel(selection).item() == observations[0]
        if result.time.dtype.kind != "b":
            vector = {"world": entry.world_id, "time": result.time}
            np.testing.assert_array_equal(
                tree["posterior"]["components"].sel(vector).values[0],
                result.components[..., result.component_mask],
            )
            np.testing.assert_array_equal(
                tree["truth"]["components"].sel(vector).values, truth[:, result.component_mask]
            )
            np.testing.assert_array_equal(
                tree["observed_data"]["outcome"].sel(vector).values, observations
            )
    sizes = tree["posterior"].sizes
    assert len(summary(tree, kind="stats")) == (
        sizes["world"] * sizes["time"] * sizes["component"] + sizes["world_edge"]
    )


@needs_arviz
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("boolean_first", [True, False])
def test_shared_two_label_numeric_vectors_select_rows_in_either_world_order(
    world, tmp_path, window, boolean_first
):
    model, _ = world
    corpora = [make_synthetic_corpus(n_weeks=4, context_lengths=(2,))]
    identity = identity_for(model, corpora, [0], [(0, 8), (0, 9)], n_samples=5)
    captured = []
    for result, truth, graph_truth, observations in capture(model, corpora, identity):
        boolean_world = (result.provenance["row"] == 8) == boolean_first
        labels = np.array([False, True]) if boolean_world else np.array([0, 1])
        captured.append((replace(result, time=labels), truth, graph_truth, observations))
    archive = write_archive(tmp_path / "archive", identity, captured)
    tree = archive.to_datatree(window)
    for entry in archive.entries:
        if entry.window != window:
            continue
        result, truth, _, observations = archive.record(entry.index)
        if result.time.dtype.kind == "b":
            continue  # Boolean arrays are masks in xarray, not vectors of label selectors.
        selection = {"world": entry.world_id, "time": result.time}
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(selection).values[0],
            result.components[..., result.component_mask],
        )
        np.testing.assert_array_equal(
            tree["truth"]["components"].sel(selection).values, truth[:, result.component_mask]
        )
        np.testing.assert_array_equal(
            tree["observed_data"]["outcome"].sel(selection).values, observations
        )


@needs_arviz
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.arange(6, dtype=np.int64) + 2**53, np.arange(6) + 0.25),
        (np.arange(6, dtype=np.int64) * 4 + 2**53 + 1, np.arange(6) + 0.25),
        (np.arange(6, dtype=np.uint64) + 2**63, np.arange(6) + 0.25),
        (np.arange(6, dtype=np.int64) - 6, np.arange(6, dtype=np.uint64) + 2**63),
        (np.arange(6, dtype=np.int64) * 2 + 2**53, np.arange(6) + 0.25),
    ],
    ids=[
        "integer-float-collision",
        "rounding-only",
        "unsigned-float",
        "signed-unsigned",
        "lossless",
    ],
)
def test_an_archive_preserves_exact_mixed_numeric_time_labels_and_every_row(
    world, tmp_path, left, right
):
    from arviz_stats import summary

    model, corpora = world
    identity = identity_for(model, corpora[:1], [0], [(0, 8), (0, 9)], n_samples=5)
    captured = []
    for result, truth, graph_truth, observations in capture(model, corpora[:1], identity):
        if result.window == "query":
            labels = left if result.provenance["row"] == 8 else right
            assert result.time.shape == labels.shape
            result = replace(result, time=labels)
        captured.append((result, truth, graph_truth, observations))
    archive = write_archive(tmp_path / "archive", identity, captured)
    records = list(archive.records("query"))
    times = sorted({label for result, *_ in records for label in result.time.tolist()})
    tree = archive.to_datatree("query")
    posterior = tree["posterior"]
    # Python numeric equality detects rounding that NumPy cross-dtype comparison can hide.
    assert posterior["time"].values.tolist() == times
    components = posterior["component"].values.tolist()
    draws, truths, outcomes = label_grids(records, times, components)
    np.testing.assert_array_equal(posterior["components"].values, draws)
    np.testing.assert_array_equal(tree["truth"]["components"].values, truths)
    np.testing.assert_array_equal(tree["observed_data"]["outcome"].values, outcomes)
    sizes = posterior.sizes
    assert len(summary(tree, kind="stats")) == (
        sizes["world"] * len(times) * len(components) + sizes["world_edge"]
    )


@needs_arviz
@pytest.mark.parametrize(
    ("name", "introduced"),
    [("archived", set()), ("mixed", {"channel:channel_3"})],
    ids=["one geometry", "mixed geometries"],
)
def test_an_archive_tree_places_every_world_value_at_its_labels(built, mixed, name, introduced):
    from arviz_stats import summary

    archive = {"archived": built["archived"].archive, "mixed": mixed}[name]
    for window in WINDOWS:
        entries = [entry for entry in archive.entries if entry.window == window]
        records = [archive.record(entry.index) for entry in entries]
        results = [result for result, *_ in records]
        active = [
            {label for label, keep in zip(r.component_names, r.component_mask) if keep}
            for r in results
        ]
        times = sorted(set().union(*(result.time.tolist() for result in results)))
        widest = max(results, key=lambda result: len(result.component_names))
        # Preconditions: the first world pads a component a later world has, a later world
        # may bring a label the first has no slot for, and a world lacks some time labels.
        first, later = set(results[0].component_names), set().union(*active[1:])
        assert (first - active[0]) & later
        assert later - first == introduced
        assert any(len(result.time) < len(times) for result in results)
        assert all(set(r.component_names) <= set(widest.component_names) for r in results)

        tree = archive.to_datatree(window)
        posterior, constants = tree["posterior"], tree["constant_data"]
        assert tree.attrs["window"] == posterior.attrs["window"] == window
        assert posterior["world"].values.tolist() == [entry.world_id for entry in entries]
        assert posterior["chain"].values.tolist() == [0]
        assert posterior["draw"].values.tolist() == list(
            range(archive.identity["evaluation"]["n_samples"])
        )
        assert posterior["time"].values.tolist() == times
        # The labels active in any world, in slot order: that of the widest world's slots.
        union = set().union(*active)
        components = posterior["component"].values.tolist()
        assert components == [label for label in widest.component_names if label in union]
        families = {
            label: family
            for result in results
            for label, family in zip(result.component_names, result.component_families)
        }
        assert constants["component_family"].values.tolist() == [families[c] for c in components]
        draws, truths, outcomes = label_grids(records, times, components)
        assert posterior["components"].dims == ("chain", "draw", "world", "time", "component")
        np.testing.assert_array_equal(posterior["components"].values, draws)
        assert tree["truth"]["components"].dims == ("world", "time", "component")
        assert tree["observed_data"]["outcome"].dims == ("world", "time")
        np.testing.assert_array_equal(tree["truth"]["components"].values, truths)
        np.testing.assert_array_equal(tree["observed_data"]["outcome"].values, outcomes)

        # One (world, edge) pair per active edge, world by world in archive order.
        pairs = [
            (entry.world_id, result, graph_truth, slot)
            for entry, (result, _, graph_truth, _) in zip(entries, records, strict=True)
            for slot in np.flatnonzero(result.edge_mask)
        ]
        assert posterior["world_edge"].values.tolist() == [
            f"{world_id}|{result.edge_names[slot]}" for world_id, result, _, slot in pairs
        ]
        np.testing.assert_array_equal(
            posterior["graph"].values[0],
            np.stack([result.graphs[:, slot] for _, result, _, slot in pairs], axis=1),
        )
        assert tree["truth"]["graph"].dims == ("world_edge",)
        np.testing.assert_array_equal(
            tree["truth"]["graph"].values, [graph_truth[slot] for _, _, graph_truth, slot in pairs]
        )
        keys = ("edge_world", "edge_name", "edge_family", "edge_source", "edge_target")
        assert list(zip(*(constants[key].values.tolist() for key in keys), strict=True)) == [
            (
                world_id,
                result.edge_names[slot],
                result.provenance["edge_families"][slot],
                *result.provenance["edge_endpoints"][slot],
            )
            for world_id, result, _, slot in pairs
        ]

        # Per-world constants follow the world axis, not the declared world order.
        keys = ("corpus_index", "row", "cell_id", "context_length", "call", "seed", "data_sha256")
        assert {key: constants[key].values.tolist() for key in keys} == {
            key: [getattr(entry, key) for entry in entries] for key in keys
        }
        assert constants["cell"].values.tolist() == [
            f"{label}/cell:{cell_id}" for label, cell_id in (entry.cell for entry in entries)
        ]
        assert constants["sales_scale"].values.tolist() == [
            result.provenance["sales_scale"] for result in results
        ]
        assert constants["metadata_sha256"].values.tolist() == [
            json.loads((entry.path / "result.json").read_text())["metadata_sha256"]
            for entry in entries
        ]
        independent = constants["world"].values[constants["independent"].values].tolist()
        assert independent == [entry.world_id for entry in archive.independent_entries(window)]

        # A summary row per cell and pair; NaN exactly where a world lacks the cell.
        table = summary(tree, kind="stats")
        assert len(table) == draws[0, 0].size + len(pairs)
        assert table["mean"].isna().sum() == np.isnan(draws[0, 0]).sum()


def _families_swapped(captured):
    """Captures whose second world's query names channel_0 a control, control_0 a channel."""
    result, *evidence = captured[2]
    names, families = result.component_names, list(result.component_families)
    channel, control = names.index("channel:channel_0"), names.index("control:control_0")
    families[channel], families[control] = families[control], families[channel]
    edited = replace(result, component_families=tuple(families))
    return [*captured[:2], (edited, *evidence), *captured[3:]]


def _components_padded(captured):
    """Captures with every component padded: zero draws and truth."""
    return [
        (
            replace(
                result,
                components=np.zeros_like(result.components),
                component_mask=np.zeros_like(result.component_mask),
            ),
            np.zeros_like(truth),
            graph_truth,
            observations,
        )
        for result, truth, graph_truth, observations in captured
    ]


def _edges_padded(captured):
    """Captures with every edge padded: zero graphs and graph truth."""
    return [
        (
            replace(
                result,
                graphs=np.zeros_like(result.graphs),
                edge_mask=np.zeros_like(result.edge_mask),
            ),
            truth,
            np.zeros_like(graph_truth),
            observations,
        )
        for result, truth, graph_truth, observations in captured
    ]


#: Hand-edited captures whose query window the export refuses: (edit, message).
REFUSED_WINDOWS = {
    "a label of two families": (
        _families_swapped,
        "component 'channel:channel_0' is 'channel' in one world and 'control' in "
        "shard:530001/row:9",
    ),
    "no active component": (
        _components_padded,
        "DataTree export needs at least one active component",
    ),
}


@needs_arviz
@pytest.mark.parametrize("case", REFUSED_WINDOWS)
def test_an_archive_tree_refuses_a_window_it_cannot_label(built, tmp_path, case):
    edit, message = REFUSED_WINDOWS[case]
    archived = built["archived"]
    archive = write_archive(tmp_path / "archive", archived.identity, edit(archived.captured))
    with pytest.raises(ValueError, match=re.escape(message)):
        archive.to_datatree("query")


@needs_arviz
def test_an_archive_window_without_active_edges_exports_no_pair_axis(built, tmp_path):
    from arviz_stats import summary

    archived = built["archived"]
    captured = _edges_padded(archived.captured)
    tree = write_archive(tmp_path / "archive", archived.identity, captured).to_datatree("query")
    assert all("world_edge" not in tree[group].dims for group in tree.children)
    assert set(tree["posterior"].data_vars) == set(tree["truth"].data_vars) == {"components"}
    assert not [name for name in tree["constant_data"].data_vars if name.startswith("edge_")]
    sizes = tree["posterior"].sizes
    assert len(summary(tree, kind="stats")) == sizes["world"] * sizes["time"] * sizes["component"]


@needs_arviz
@pytest.mark.parametrize("first", [2**63 - 1, -2], ids=["straddling 2**63", "negative"])
def test_archive_tree_calls_and_seeds_equal_the_declared_calls_exactly(world, tmp_path, first):
    model, corpora = world
    rows = [(0, 8), (1, 10)]  # one world per call: calls 0 and 1, seeds first and first + 1
    identity = identity_for(model, corpora, [0, 1], rows, seed=first, worlds_per_call=1)
    with EvaluationArchive.create(tmp_path / "archive", identity) as writer:
        capture(model, corpora, identity, writer)
    archive = EvaluationArchive.load(writer.path)
    entries = [entry for entry in archive.entries if entry.window == "query"]
    constants = archive.to_datatree("query")["constant_data"]
    seeds = constants["seed"].values.tolist()
    assert seeds == [entry.seed for entry in entries] == [first, first + 1]
    assert constants["call"].values.tolist() == [entry.call for entry in entries] == [0, 1]


@needs_arviz
def test_archive_tree_attributes_name_the_archive_and_only_the_recorded_model_keys(built, tmp_path):
    archived = built["archived"]
    identity = copy.deepcopy(archived.identity)
    identity["model"].update(
        checkpoint_sha256=None, bundle_sha256="3" * 64, profile="tiny", source_step=7
    )
    bundled = write_archive(tmp_path / "bundled", identity, archived.captured)
    optional = ("checkpoint_sha256", "bundle_sha256", "profile", "source_step")
    for archive, recorded in (
        (archived.archive, {"checkpoint_sha256": "2" * 64}),
        (bundled, {"bundle_sha256": "3" * 64, "profile": "tiny", "source_step": 7}),
    ):
        expected = {
            "window": "context",
            "mode": "model_average",
            "units": "outcome",
            "n_samples": 8,
            "leg": "evaluation",
            "archive_sha256": archive.archive_sha256,
            "dataset_sha256": archive.dataset_sha256,
            "state_sha256": "1" * 64,
            "config_sha256": identity["model"]["config_sha256"],
            "mcmc_diagnostics": "not_applicable",
        }
        tree = archive.to_datatree("context")
        for attrs in (tree.attrs, tree["posterior"].attrs):
            assert expected.items() <= attrs.items()
            # Model keys present exactly when recorded: an unrecorded key is absent, never None.
            assert {key: attrs[key] for key in optional if key in attrs} == recorded


@pytest.mark.parametrize(
    "hidden",
    [("xarray", "arviz_base"), ("arviz_base",)],
    ids=["base install", "xarray without arviz-base"],
)
def test_an_archive_export_without_the_arviz_extra_names_the_extra(built, monkeypatch, hidden):
    # Import the export module anew with the extra hidden: its own imports must not need it.
    monkeypatch.delitem(sys.modules, "pymc_mktfm.inference._datatree", raising=False)
    monkeypatch.setattr(sys.modules["pymc_mktfm.inference"], "_datatree", None, raising=False)
    for name in hidden:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=re.escape("pymc-mktfm[arviz]")):
        built["archived"].archive.to_datatree("query")


def _first(ctx, writer):
    writer.add(*ctx.captured[0])


def _observation_only(ctx, writer):
    result = decompose(ctx.model, observed(ctx.corpora[0], 8), n_samples=8)
    writer.add(result, *ctx.captured[0][1:])


#: The capture keys of corpus 0 row 8 in the first declared call, as a forgery claims them.
CLAIM = {
    "capture": "existing conditional evaluator samples; no additional inference",
    "data_split": "evaluation",
    "corpus_index": 0,
    "row": 8,
    "world_id": "corpus:0/row:8",
    "seed": 5,
}


def _forged(row, **options):
    """Attempt: ``decompose`` of corpus 0 row ``row``'s inputs, claiming to capture row 8.

    Row 8's own inputs match the declared world's ``data_sha256``, so only the
    ``decompose`` ``options`` can tell the forgery apart.
    """

    def attempt(ctx, writer):
        inputs = observed(ctx.corpora[0], row)
        result = decompose(ctx.model, inputs, n_samples=8, provenance=CLAIM, **options)
        writer.add(result, *ctx.captured[0][1:])

    return attempt


def _provenance(*removed, **changes):
    """Attempt: the first capture with provenance keys ``removed`` and ``changes`` applied."""

    def attempt(ctx, writer):
        result, *evidence = ctx.captured[0]
        kept = {key: value for key, value in result.provenance.items() if key not in removed}
        writer.add(replace(result, provenance={**kept, **changes}), *evidence)

    return attempt


def _edited(key, change):
    """Attempt: the first capture with the list provenance ``key`` changed by ``change``."""

    def attempt(ctx, writer):
        result, *evidence = ctx.captured[0]
        provenance = {**result.provenance, key: change(result.provenance[key])}
        writer.add(replace(result, provenance=provenance), *evidence)

    return attempt


def _with_evidence(edit):
    """Attempt: the first capture with copies of its evidence changed by ``edit``."""

    def attempt(ctx, writer):
        result, *arrays = ctx.captured[0]
        evidence = dict(zip(("truth", "graph_truth", "observed"), arrays, strict=True))
        evidence = {name: array.copy() for name, array in evidence.items()}
        edit(result, evidence)
        writer.add(result, evidence["truth"], evidence["graph_truth"], evidence["observed"])

    return attempt


def _short_truth(result, evidence):
    evidence["truth"] = evidence["truth"][:-1]


def _fractional_edge(result, evidence):
    evidence["graph_truth"][np.flatnonzero(result.edge_mask)[0]] = 0.5


def _inactive_edge(result, evidence):
    evidence["graph_truth"][np.flatnonzero(~result.edge_mask)[0]] = 1.0


def _padded_truth(result, evidence):
    evidence["truth"][:, np.flatnonzero(~result.component_mask)[0]] = 1.0


def _nan_truth(result, evidence):
    evidence["truth"][0, np.flatnonzero(result.component_mask)[0]] = np.nan


def _complex_truth(result, evidence):
    evidence["truth"] = evidence["truth"] + 0j


def _duplicate(ctx, writer):
    writer.add(*ctx.captured[0])
    writer.add(*ctx.captured[0])


def _out_of_order(ctx, writer):
    rows = list(ROWS)
    first, second = rows.index((0, 8)), rows.index((0, 9))
    rows[first], rows[second] = rows[second], rows[first]
    evaluate_rows(
        ctx.model,
        ctx.corpora,
        rows,
        n_samples=8,
        seed=5,
        device="cpu",
        leg="evaluation",
        result_sink=writer.add,
    )


def _not_a_result(ctx, writer):
    writer.add("x", *ctx.captured[0][1:])


def _world_one(identity, **changes):
    """Edit the declared (0, 8): world 1, the first world of archive order."""
    identity["evaluation"]["worlds"][1].update(changes)


def _differs(name, actual, expected):
    return re.escape(f"{name} {actual!r} differs from the archive's {expected!r}")


#: name: (identity edit, attempt, error, message pattern). The observation-only refusal
#: stands alone; the others name the capture.
REFUSALS = {
    "observation-only": (
        None,
        _observation_only,
        ValueError,
        "^"
        + re.escape(
            "only evaluator captures of declared corpus worlds enter an evaluation archive"
        ),
    ),
    "forged inputs": (
        None,
        _forged(9),
        ValueError,
        "data_sha256 '[0-9a-f]{64}' differs from the archive's",
    ),
    "forged map draws": (
        None,
        _forged(8, mode="map"),
        ValueError,
        _differs("mode", "map", "model_average"),
    ),
    "forged units": (
        None,
        _forged(8, units="revenue"),
        ValueError,
        _differs("units", "revenue", "outcome"),
    ),
    "undeclared world": (
        None,
        _provenance(row=1, world_id="corpus:0/row:1"),
        ValueError,
        "corpus 0 row 1 is not a declared world",
    ),
    "joint head": (
        None,
        _provenance(head="joint"),
        ValueError,
        _differs("head", "joint", "conditional"),
    ),
    "world id": (
        None,
        _provenance(world_id="corpus:0/row:9"),
        ValueError,
        _differs("world_id", "corpus:0/row:9", "corpus:0/row:8"),
    ),
    "float seed": (None, _provenance(seed=5.0), ValueError, _differs("seed", 5.0, 5)),
    "call seed": (
        lambda identity: identity["evaluation"]["calls"][0].update(seed=6),
        _first,
        ValueError,
        "^" + re.escape("cannot archive corpus 0 row 8 query: ") + _differs("seed", 5, 6),
    ),
    "n_samples": (
        lambda identity: identity["evaluation"].update(n_samples=9),
        _first,
        ValueError,
        _differs("n_samples", 8, 9),
    ),
    "leg": (
        lambda identity: identity["evaluation"].update(leg="holdout"),
        _first,
        ValueError,
        _differs("data_split", "evaluation", "holdout"),
    ),
    "declared inputs": (
        lambda identity: _world_one(identity, data_sha256="0" * 64),
        _first,
        ValueError,
        "data_sha256 '[0-9a-f]{64}' differs from the archive's '0{64}'",
    ),
    "context length": (
        lambda identity: _world_one(identity, context_length=5),
        _first,
        ValueError,
        _differs("context_length", 6, 5),
    ),
    "model configuration": (
        lambda identity: identity["model"].update(config_sha256=OTHER_CONFIG),
        _first,
        ValueError,
        "model configuration digest",
    ),
    "duplicate": (None, _duplicate, ValueError, "already added"),
    "out of order": (None, _out_of_order, ValueError, "declared world order"),
    "truth shape": (None, _with_evidence(_short_truth), ValueError, "truth has shape"),
    "fractional graph truth": (
        None,
        _with_evidence(_fractional_edge),
        ValueError,
        "graph_truth must be binary",
    ),
    "edge outside the support": (
        None,
        _with_evidence(_inactive_edge),
        ValueError,
        "graph_truth has edges outside the active support",
    ),
    "padded truth": (
        None,
        _with_evidence(_padded_truth),
        ValueError,
        "truth is nonzero on padded components",
    ),
    "non-finite truth": (None, _with_evidence(_nan_truth), ValueError, "truth must be finite"),
    "complex truth": (
        None,
        _with_evidence(_complex_truth),
        ValueError,
        "truth must be a real array",
    ),
    **{
        f"sales_scale {label}": (
            None,
            _provenance(*removed, **changes),
            ValueError,
            "sales_scale must be a finite positive number",
        )
        for label, removed, changes in (
            ("missing", ("sales_scale",), {}),
            ("zero", (), {"sales_scale": 0}),
            ("boolean", (), {"sales_scale": True}),
            ("beyond float64", (), {"sales_scale": 10**309}),
        )
    },
    "short edge_families": (
        None,
        _edited("edge_families", lambda families: families[:-1]),
        ValueError,
        "edge_families must name one family per edge slot",
    ),
    "empty edge family": (
        None,
        _edited("edge_families", lambda families: ["", *families[1:]]),
        ValueError,
        "edge_families must name one family per edge slot",
    ),
    "short edge_endpoints": (
        None,
        _edited("edge_endpoints", lambda pairs: pairs[:-1]),
        ValueError,
        re.escape("edge_endpoints must hold one [source, target] pair per edge slot"),
    ),
    "edge endpoint triple": (
        None,
        _edited("edge_endpoints", lambda pairs: [[*pairs[0], "x"], *pairs[1:]]),
        ValueError,
        re.escape("edge_endpoints must hold one [source, target] pair per edge slot"),
    ),
    "not a result": (None, _not_a_result, TypeError, "takes a DecompositionResult, got str"),
}
#: Refusals that follow captures the writer accepted.
AFTER_ADDING = ("duplicate", "out of order")


@pytest.mark.parametrize("case", REFUSALS)
def test_the_writer_refuses_results_that_do_not_belong_to_the_archive(built, world, tmp_path, case):
    model, corpora = world
    archived = built["archived"]
    edit, attempt, error, pattern = REFUSALS[case]
    identity = copy.deepcopy(archived.identity)
    if edit is not None:
        edit(identity)
    writer = EvaluationArchive.create(tmp_path / "archive", identity)
    with pytest.raises(error, match=pattern):
        attempt(Context(model, corpora, archived.captured), writer)
    if edit is None and case not in AFTER_ADDING:
        # A refusal changes nothing: the same writer still takes every declared window.
        fill(writer, archived.captured)
        assert len(EvaluationArchive.load(writer.close())) == 2 * len(ROWS)
    else:
        writer.abort()
        assert list(tmp_path.iterdir()) == []


def test_archives_publish_atomically_and_never_partially(built, tmp_path, monkeypatch):
    identity, captured = built["archived"].identity, built["archived"].captured
    original_savez = np.savez_compressed

    def full_disk(file, *args, **kwargs):
        if Path(file).name == "truth.npz":
            raise OSError(errno.ENOSPC, "No space left on device")
        return original_savez(file, *args, **kwargs)

    # An incomplete close publishes nothing and fails the writer.
    writer = EvaluationArchive.create(tmp_path / "incomplete", identity)
    fill(writer, captured[:3])
    with pytest.raises(ValueError, match=re.escape("13 of 16 world windows were never added")):
        writer.close()
    assert not (tmp_path / "incomplete").exists() and not staging_left(tmp_path, "incomplete")
    with pytest.raises(ValueError, match="failed"):
        writer.add(*captured[3])
    with pytest.raises(ValueError, match="failed"):
        writer.close()

    # An exception inside the block aborts.
    with pytest.raises(RuntimeError, match="interrupted evaluation"):
        with EvaluationArchive.create(tmp_path / "raised", identity) as writer:
            writer.add(*captured[0])
            raise RuntimeError("interrupted evaluation")
    assert not (tmp_path / "raised").exists() and not staging_left(tmp_path, "raised")
    with pytest.raises(ValueError, match="aborted"):
        writer.close()

    # A failing write removes the staged entries and fails the writer.
    writer = EvaluationArchive.create(tmp_path / "full", identity)
    writer.add(*captured[0])
    with monkeypatch.context() as patch:
        patch.setattr(np, "savez_compressed", full_disk)
        with pytest.raises(OSError, match="No space left"):
            writer.add(*captured[1])
    for call in (lambda: writer.add(*captured[1]), writer.close):
        with pytest.raises(ValueError, match="failed"):
            call()
    assert not (tmp_path / "full").exists() and not staging_left(tmp_path, "full")

    # A result the strict reader would refuse never enters: the writer reads each entry back.
    writer = EvaluationArchive.create(tmp_path / "deep", identity)
    result, *evidence = captured[0]
    deep = replace(result, provenance={**result.provenance, "note": nested(1500, "x")})
    with pytest.raises(ValueError, match="would not pass the archive reader"):
        writer.add(deep, *evidence)
    with pytest.raises(ValueError, match="failed"):
        writer.close()
    assert not (tmp_path / "deep").exists() and not staging_left(tmp_path, "deep")

    # An interrupt while publishing leaves neither the target nor the staging directory.
    writer = EvaluationArchive.create(tmp_path / "interrupted", identity)
    fill(writer, captured)

    def interrupt(source, destination):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            writer.close()
    assert not (tmp_path / "interrupted").exists()
    assert not staging_left(tmp_path, "interrupted")

    # A target filled by someone else before close is refused and left alone.
    taken = tmp_path / "taken"
    writer = EvaluationArchive.create(taken, identity)
    fill(writer, captured)
    taken.mkdir()
    (taken / "neighbour.txt").write_text("kept")
    with pytest.raises(FileExistsError, match="new or empty"):
        writer.close()
    assert [path.name for path in taken.iterdir()] == ["neighbour.txt"]
    assert (taken / "neighbour.txt").read_text() == "kept"
    assert not staging_left(tmp_path, "taken")

    # States: close is idempotent, abort never deletes a published archive, and a
    # clean exit after a swallowed failure raises.
    writer = EvaluationArchive.create(tmp_path / "states", identity)
    fill(writer, captured)
    assert writer.archive_sha256 is None
    path = writer.close()
    assert writer.close() == path == writer.path
    writer.abort()
    published = EvaluationArchive.load(path)
    assert writer.archive_sha256 == published.archive_sha256 and len(published) == 16
    with pytest.raises(ValueError, match="failed; nothing was published"):
        with EvaluationArchive.create(tmp_path / "swallowed", identity) as writer:
            with monkeypatch.context() as patch:
                patch.setattr(np, "savez_compressed", full_disk)
                try:
                    writer.add(*captured[0])
                except OSError:
                    pass
    assert not (tmp_path / "swallowed").exists() and not staging_left(tmp_path, "swallowed")


def test_a_writer_that_is_never_closed_leaves_no_staging(built, tmp_path):
    identity, captured = built["archived"].identity, built["archived"].captured
    # Dropped mid-evaluation, outside any with block: collection removes the staged worlds.
    writer = EvaluationArchive.create(tmp_path / "dropped", identity)
    fill(writer, captured[:3])
    assert len(hidden(tmp_path)) == 1
    del writer
    gc.collect()
    assert list(tmp_path.iterdir()) == []
    # A published archive outlives its writer.
    writer = EvaluationArchive.create(tmp_path / "kept", identity)
    fill(writer, captured)
    path = writer.close()
    del writer
    gc.collect()
    assert len(EvaluationArchive.load(path)) == 2 * len(ROWS)
    # Still open when the interpreter exits: the staging directory goes with the process.
    target = tmp_path / "at-exit" / "archive"
    code = (
        "import json, os, sys\n"
        "from pymc_mktfm.diagnostics import EvaluationArchive\n"
        "writer = EvaluationArchive.create(sys.argv[1], json.load(sys.stdin))\n"
        "print(len(os.listdir(os.path.dirname(sys.argv[1]))))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(target)],
        input=json.dumps(identity),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1"]
    assert list(target.parent.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="this platform cannot fork")
def test_a_forked_child_never_removes_its_parents_staging(built, tmp_path):
    code = (
        "import os, sys\n"
        "from pymc_mktfm.diagnostics import EvaluationArchive\n"
        "source = EvaluationArchive.load(sys.argv[1])\n"
        "records = list(source)\n"
        "writer = EvaluationArchive.create(sys.argv[2], source.identity)\n"
        "for record in records[:4]:\n"
        "    writer.add(*record)\n"
        "if os.fork() == 0:\n"
        "    sys.exit(0)  # through interpreter shutdown: the inherited finalizer runs here\n"
        "_, status = os.wait()\n"
        "for record in records[4:]:\n"
        "    writer.add(*record)\n"
        "print(os.waitstatus_to_exitcode(status), len(EvaluationArchive.load(writer.close())))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(built["archived"].archive.path), str(tmp_path / "a")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["0", str(2 * len(ROWS))]
    assert hidden(tmp_path) == []


@pytest.mark.parametrize(
    ("before", "change", "problem"),
    [
        pytest.param(4, shutil.rmtree, "(entries/0000: missing)", id="staging removed"),
        pytest.param(
            16,
            lambda staging: shutil.rmtree(staging / "entries" / "0003"),
            "(entries/0003: missing)",
            id="entry removed",
        ),
        pytest.param(
            16,
            lambda staging: flip(staging / "entries" / "0002" / "truth.npz"),
            "(entries/0002/truth.npz: SHA-256 or size differs from manifest.json)",
            id="file changed",
        ),
        pytest.param(
            16,
            lambda staging: (staging / "entries" / "0016").mkdir(),
            "(entries/0016: is not listed in manifest.json)",
            id="extra entry",
        ),
    ],
)
def test_close_refuses_staged_entries_that_changed_after_they_were_added(
    built, tmp_path, before, change, problem
):
    captured = built["archived"].captured
    writer = EvaluationArchive.create(tmp_path / "archive", built["archived"].identity)
    fill(writer, captured[:before])
    (staging,) = (tmp_path / name for name in staging_left(tmp_path, "archive"))
    change(staging)
    fill(writer, captured[before:])  # a removed staging directory is recreated by later adds
    with pytest.raises(ValueError, match=re.escape(problem)) as caught:
        writer.close()
    assert "cannot be published: its staged files changed after add recorded them" in str(
        caught.value
    )
    with pytest.raises(ValueError, match="is failed; nothing was published"):
        writer.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("code", "fragment"),
    [(errno.ENOTEMPTY, "new or empty"), (errno.EXDEV, "mount point"), (errno.EBUSY, "mount point")],
    ids=["ENOTEMPTY", "EXDEV", "EBUSY"],
)
def test_a_refused_publish_names_why_the_target_is_unusable(
    built, tmp_path, monkeypatch, code, fragment
):
    writer = EvaluationArchive.create(tmp_path / "archive", built["archived"].identity)
    fill(writer, built["archived"].captured)

    def refuse(source, destination):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(FileExistsError, match=fragment):
        writer.close()
    assert list(tmp_path.iterdir()) == []


def test_unusable_targets_are_refused_before_anything_is_written(built, tmp_path, monkeypatch):
    identity = built["archived"].identity

    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "neighbour.txt").write_text("kept")
    with pytest.raises(FileExistsError, match="new or empty"):
        EvaluationArchive.create(taken, identity)
    assert [path.name for path in taken.iterdir()] == ["neighbour.txt"]
    assert (taken / "neighbour.txt").read_text() == "kept"

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    spellings = [".", ".."]
    if os.path.exists(str(cwd).swapcase()):  # a case-insensitive volume
        spellings.append(str(cwd).swapcase())
    for spelling in spellings:
        with pytest.raises(FileExistsError, match="current directory or one of its parents"):
            EvaluationArchive.create(spelling, identity)
    assert list(cwd.iterdir()) == []

    mount = tmp_path / "mount"
    mount.mkdir()
    resolved = Path(os.path.realpath(mount))
    ismount = os.path.ismount
    with monkeypatch.context() as patch:
        patch.setattr(os.path, "ismount", lambda path: Path(path) == resolved or ismount(path))
        with pytest.raises(FileExistsError, match="mount point"):
            EvaluationArchive.create(mount, identity)
    assert list(mount.iterdir()) == [] and not staging_left(tmp_path, "mount")

    empty = tmp_path / "e"
    empty.mkdir()
    (tmp_path / "tlink").symlink_to(empty, target_is_directory=True)
    for spelling in (tmp_path / "tlink", f"{tmp_path / 'tlink'}/"):
        with pytest.raises(FileExistsError, match="new or empty"):
            EvaluationArchive.create(spelling, identity)
    assert list(empty.iterdir()) == []
    assert not staging_left(tmp_path, "tlink") and not staging_left(tmp_path, "e")

    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "dangling").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    with pytest.raises(FileExistsError, match="new or empty"):
        EvaluationArchive.create(folder / "dangling" / "arch", identity)
    assert not (tmp_path / "nowhere").exists()
    assert [path.name for path in folder.iterdir()] == ["dangling"]

    beyond = tmp_path / "beyond"
    beyond.mkdir()
    with pytest.raises(FileExistsError, match="new or empty"):
        EvaluationArchive.create(beyond / ("x" * 256), identity)  # one byte over NAME_MAX
    assert list(beyond.iterdir()) == []


def test_a_name_the_filesystem_cannot_create_is_refused_before_inference(built, tmp_path):
    name = "run\udcff"  # bytes that are not UTF-8: UTF-8-only filesystems such as APFS refuse it
    try:
        (tmp_path / name).mkdir()
    except OSError:
        pass
    else:
        pytest.skip("this filesystem creates names that are not UTF-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        EvaluationArchive.create(tmp_path / name, built["archived"].identity)
    assert list(tmp_path.iterdir()) == []


def test_a_name_a_filesystem_refuses_is_refused_on_any_platform(built, tmp_path, monkeypatch):
    # Linux filesystems take any byte name; vfat, exFAT and SMB targets refuse some, as here.
    name = "run:1"
    real_mkdir = os.mkdir

    def refusing(path, *args, **kwargs):
        if Path(os.fsdecode(path)).name == name:
            raise OSError(errno.EINVAL, "Invalid argument", os.fsdecode(path))
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", refusing)
    if real_mkdir in os.supports_dir_fd:  # keep the dir_fd branch the platform really takes
        monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {refusing})
    with pytest.raises(FileExistsError, match=re.escape("new or empty directory")):
        EvaluationArchive.create(tmp_path / name, built["archived"].identity)
    assert list(tmp_path.iterdir()) == []


def nest(start, total):
    """A new directory at ``start`` or below whose path is exactly ``total`` bytes long."""
    path = start
    while (room := total - len(os.fsencode(path)) - 1) > 200:
        path = path / ("d" * min(200, room - 2))
    path = path / ("e" * room)
    path.mkdir(parents=True)
    return path


@pytest.mark.skipif(not hasattr(os, "pathconf"), reason="this platform reports no path limit")
def test_archive_files_fit_the_filesystem_path_limit(built, tmp_path, monkeypatch):
    identity, captured = built["archived"].identity, built["archived"].captured
    base = Path(os.path.realpath(tmp_path))
    limit = os.pathconf(base, "PC_PATH_MAX")
    last = f"{2 * len(ROWS) - 1:04d}"
    # Under a short name the staged files are the deepest; under a long one, the published.
    for name in ("abcde", "n" * 240):
        extra = max(
            len(f"/{name}/entries/{last}/result.json"),
            len(f"/.{name[:64]}-xxxxxxxx/entries/.result-xxxxxxxx/result/result.json"),
        )
        # The deepest file one byte below the limit: the OS writes and reads every file.
        fits = nest(base / f"fits-{len(name)}", limit - 1 - extra)
        writer = EvaluationArchive.create(fits / name, identity)
        fill(writer, captured)
        assert len(EvaluationArchive.load(writer.close())) == 2 * len(ROWS)
        deeper = nest(base / f"deeper-{len(name)}", limit - extra)
        with pytest.raises(FileExistsError, match="too long for its filesystem"):
            EvaluationArchive.create(deeper / name, identity)
        assert list(deeper.iterdir()) == []
    # So deep that the staging directory beside the target cannot be created at all.
    tight = nest(base / "tight", limit - len("/.abcde-xxxxxxxx") + 4)
    with pytest.raises(FileExistsError, match=re.escape("new or empty directory")):
        EvaluationArchive.create(tight / "abcde", identity)
    assert list(tight.iterdir()) == []
    real_pathconf = os.pathconf

    def answers_only_at_the_root(path, key):  # macOS exFAT and FAT32 volumes
        if os.fspath(path) != os.sep:
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_pathconf(path, key)

    monkeypatch.setattr(os, "pathconf", answers_only_at_the_root)
    with pytest.raises(FileExistsError, match="too long for its filesystem"):
        EvaluationArchive.create(deeper / name, identity)
    monkeypatch.delattr(os, "pathconf")  # no known limit: nothing to refuse
    EvaluationArchive.create(base / "unbounded", identity).abort()
    assert not (base / "unbounded").exists()


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="permission bits bind only an unprivileged POSIX user",
)
def test_a_parent_the_staging_directory_cannot_be_created_in_is_refused(built, tmp_path):
    parent = tmp_path / "read-only"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        with pytest.raises(FileExistsError, match="new or empty directory") as caught:
            EvaluationArchive.create(parent / "archive", built["archived"].identity)
        assert "Permission denied" in str(caught.value)
        assert list(parent.iterdir()) == []
    finally:
        parent.chmod(0o755)


def test_targets_resolve_where_the_os_resolves_them(built, tmp_path, monkeypatch):
    identity, captured = built["archived"].identity, built["archived"].captured

    if sys.platform != "win32":  # a deleted working directory
        target = tmp_path / "out"
        target.mkdir()
        gone = tmp_path / "gone"
        gone.mkdir()
        monkeypatch.chdir(gone)
        gone.rmdir()
        writer = EvaluationArchive.create(target, identity)
        fill(writer, captured)
        assert writer.close() == Path(os.path.realpath(target))
        assert len(EvaluationArchive.load(target)) == 16

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    writer = EvaluationArchive.create("relative", identity)
    monkeypatch.chdir(second)
    fill(writer, captured)
    assert writer.close() == Path(os.path.realpath(first / "relative"))
    assert (first / "relative" / "manifest.json").is_file()
    assert list(second.iterdir()) == []

    folder = tmp_path / "d"
    folder.mkdir()
    real = tmp_path / "outside" / "real"
    real.mkdir(parents=True)
    (folder / "link").symlink_to(real, target_is_directory=True)
    out_dir = folder / "link" / ".." / "arch"  # ".." after the link resolves physically
    writer = EvaluationArchive.create(out_dir, identity)
    fill(writer, captured)
    expected = Path(os.path.realpath(tmp_path / "outside")) / "arch"
    assert writer.close() == writer.path == expected == EvaluationArchive.load(out_dir).path
    assert [path.name for path in folder.iterdir()] == ["link"]

    name = "a" * 250  # a legal name; the hidden staging name beside it must stay legal too
    writer = EvaluationArchive.create(tmp_path / name, identity)
    fill(writer, captured)
    assert writer.close() == Path(os.path.realpath(tmp_path)) / name
    assert len(EvaluationArchive.load(tmp_path / name)) == 2 * len(ROWS)
    assert hidden(tmp_path) == []


def _seeds(identity, seeds):
    identity["evaluation"]["calls"] = [
        {"worlds": [0, 1, 2, 3], "seed": seeds[0]},
        {"worlds": [4, 5, 6, 7], "seed": seeds[1]},
    ]


def _set(*path, **changes):
    """Edit: update the identity block at ``path`` (keys and list indices) with ``changes``."""

    def edit(identity):
        block = identity
        for key in path:
            block = block[key]
        block.update(changes)

    return edit


IDENTITY_ERRORS = [
    (
        "state digest",
        _set("model", state_sha256="1" * 63),
        "identity.model.state_sha256 must be a full lowercase SHA-256",
    ),
    (
        "config digest",
        _set("model", config_sha256="A" * 64),
        "identity.model.config_sha256 must be a full lowercase SHA-256",
    ),
    (
        "checkpoint digest",
        _set("model", checkpoint_sha256="zz"),
        "identity.model.checkpoint_sha256 must be a full lowercase SHA-256 or null",
    ),
    (
        "bundle digest",
        _set("model", bundle_sha256="A" * 64),
        "identity.model.bundle_sha256 must be a full lowercase SHA-256 or null",
    ),
    (
        "profile",
        _set("model", profile="two words"),
        "identity.model.profile must be a simple 1-64 character alphanumeric slug",
    ),
    (
        "negative source step",
        _set("model", source_step=-1),
        "identity.model.source_step must be an integer >= 0 or null",
    ),
    ("no corpora", _set(corpora=[]), "identity.corpora must be a nonempty list"),
    (
        "negative shard seed",
        _set("corpora", 0, shard_seed=-1),
        "identity.corpora[0].shard_seed must be an integer >= 0 or null",
    ),
    (
        "duplicate shard seeds",
        _set("corpora", 1, shard_seed=SHARD[0]),
        f"identity.corpora[1].shard_seed {SHARD[0]} repeats identity.corpora[0]",
    ),
    (
        "text world count",
        _set("corpora", 0, n_worlds="x"),
        "identity.corpora[0].n_worlds must be a positive integer",
    ),
    (
        "duplicate inputs",
        lambda identity: identity["corpora"][1].update(
            actual_inputs_sha256=identity["corpora"][0]["actual_inputs_sha256"]
        ),
        "identity.corpora[1].actual_inputs_sha256 repeats identity.corpora[0]",
    ),
    *(
        (
            f"corpus {key}",
            _set("corpora", 0, **{key: "zz"}),
            f"identity.corpora[0].{key} must be a full lowercase SHA-256 or null",
        )
        for key in ("content_sha256", "source_sha256", "prior_sha256")
    ),
    (
        "data origin",
        _set("corpora", 0, data_origin="real"),
        "identity.corpora[0].data_origin must be 'synthetic' or 'unknown'",
    ),
    (
        "synthetic without prior",
        _set("corpora", 1, prior_sha256=None),
        "identity.corpora[1].data_origin 'synthetic' needs a verified prior_sha256",
    ),
    (
        "no worlds",
        _set("evaluation", worlds=[]),
        "identity.evaluation.worlds must be a nonempty list",
    ),
    (
        "undeclared corpus",
        _set("evaluation", "worlds", 0, corpus_index=99),
        "identity.evaluation.worlds[0].corpus_index must index identity.corpora",
    ),
    (
        "row out of range",
        _set("evaluation", "worlds", 0, row=12),
        "identity.evaluation.worlds[0].row must lie in [0, 12) of corpus 1",
    ),
    (
        "text row",
        _set("evaluation", "worlds", 0, row="x"),
        "identity.evaluation.worlds[0].row must lie in [0, 12) of corpus 1",
    ),
    (
        "text cell id",
        _set("evaluation", "worlds", 0, cell_id="x"),
        "identity.evaluation.worlds[0].cell_id must be an integer",
    ),
    (
        "context length",
        _set("evaluation", "worlds", 0, context_length=0),
        "identity.evaluation.worlds[0].context_length must be a positive integer",
    ),
    (
        "text context length",
        _set("evaluation", "worlds", 0, context_length="x"),
        "identity.evaluation.worlds[0].context_length must be a positive integer",
    ),
    (
        "world inputs digest",
        _set("evaluation", "worlds", 0, data_sha256="zz"),
        "identity.evaluation.worlds[0].data_sha256 must be a full lowercase SHA-256",
    ),
    (
        "repeated world",
        lambda identity: identity["evaluation"]["worlds"].__setitem__(
            1, dict(identity["evaluation"]["worlds"][0])
        ),
        "identity.evaluation.worlds[1] repeats identity.evaluation.worlds[0]",
    ),
    (
        "joint head only",
        _set("evaluation", heads=["joint"]),
        "identity.evaluation.heads must list distinct heads",
    ),
    (
        "heads out of order",
        _set("evaluation", heads=["conditional", "joint"]),
        "identity.evaluation.heads must list distinct heads",
    ),
    (
        "extra key",
        _set("evaluation", extra=1),
        "identity.evaluation must contain exactly",
    ),
    (
        "numpy integer",
        _set("evaluation", n_samples=np.int64(8)),
        "archive identity must hold only JSON values",
    ),
    (
        "no draws",
        _set("evaluation", n_samples=0),
        "identity.evaluation.n_samples must be a positive integer",
    ),
    (
        "float draws",
        _set("evaluation", n_samples=8.0),
        "identity.evaluation.n_samples must be a positive integer",
    ),
    (
        "float flow steps",
        _set("evaluation", flow_steps=1.5),
        "identity.evaluation.flow_steps must be a positive integer",
    ),
    (
        "no flow steps",
        _set("evaluation", flow_steps=0),
        "identity.evaluation.flow_steps must be a positive integer",
    ),
    (
        "map mode",
        _set("evaluation", mode="map"),
        "identity.evaluation.mode must be 'model_average'",
    ),
    *(
        (
            f"empty {key}",
            _set("evaluation", **{key: ""}),
            f"identity.evaluation.{key} must be a nonempty string",
        )
        for key in ("leg", "units", "device")
    ),
    *(
        (
            f"{key} not Unicode",
            _set("evaluation", **{key: "holdout\udcff"}),
            f"identity.evaluation.{key} is not valid Unicode text",
        )
        for key in ("leg", "units", "device")
    ),
    ("no calls", _set("evaluation", calls=[]), "identity.evaluation.calls must be a nonempty list"),
    (
        "calls miss a world",
        lambda identity: identity["evaluation"]["calls"][0]["worlds"].pop(),
        "identity.evaluation.calls must split worlds 0..7 into consecutive blocks in order",
    ),
    (
        "calls out of order",
        _set(
            "evaluation",
            calls=[
                {"worlds": [3, 4, 5], "seed": 5},
                {"worlds": [0, 1, 2], "seed": 6},
                {"worlds": [6, 7], "seed": 7},
            ],
        ),
        "identity.evaluation.calls must split worlds 0..7 into consecutive blocks in order",
    ),
    (
        "empty call",
        lambda identity: identity["evaluation"]["calls"].append({"worlds": [], "seed": 6}),
        "identity.evaluation.calls[1].worlds must be a nonempty list of world indices",
    ),
    (
        "float call member",
        lambda identity: identity["evaluation"]["calls"][0]["worlds"].__setitem__(1, 1.0),
        "identity.evaluation.calls[0].worlds must be a nonempty list of world indices",
    ),
    (
        "float seed",
        _set("evaluation", "calls", 0, seed=5.0),
        "identity.evaluation.calls[0].seed must be an integer",
    ),
    (
        "seeds equal modulo 2**32",
        lambda identity: _seeds(identity, [5, 5 + 2**32]),
        "identity.evaluation.calls[1].seed 4294967301 must be calls[0].seed + 1",
    ),
    (
        "seed gap",
        lambda identity: _seeds(identity, [5, 7]),
        "identity.evaluation.calls[1].seed 7 must be calls[0].seed + 1",
    ),
    (
        "first seed beyond torch",
        _set("evaluation", "calls", 0, seed=2**64),
        "identity.evaluation.calls[0].seed 18446744073709551616 is outside the torch seed range",
    ),
    (
        "last seed beyond torch",
        lambda identity: _seeds(identity, [2**64 - 1, 2**64]),
        "identity.evaluation.calls[1].seed 18446744073709551616 is outside the torch seed range",
    ),
]


@pytest.mark.parametrize(
    ("edit", "fragment"),
    [pytest.param(edit, text, id=name) for name, edit, text in IDENTITY_ERRORS],
)
def test_create_validates_the_identity_and_names_the_field(built, tmp_path, edit, fragment):
    identity = copy.deepcopy(built["archived"].identity)
    edit(identity)
    with pytest.raises(ValueError, match=re.escape(fragment)):
        EvaluationArchive.create(tmp_path / "archive", identity)
    assert list(tmp_path.iterdir()) == []


def test_create_accepts_identities_up_to_the_declared_limits(built, tmp_path):
    archived = built["archived"]
    # Shard seeds have no upper bound; world ids carry them in full.
    large = copy.deepcopy(archived.identity)
    large["corpora"][0]["shard_seed"] = 2**63
    archive = write_archive(tmp_path / "large-seed", large, archived.captured)
    assert archive.entries[0].world_id == f"shard:{2**63}/row:8"

    def declared(n_worlds, n_corpora=1):
        corpus = {**archived.identity["corpora"][0], "n_worlds": 5000, "shard_seed": None}
        worlds = [
            {
                "corpus_index": 0,
                "row": row,
                "cell_id": row,
                "context_length": 6,
                "data_sha256": "0" * 64,
            }
            for row in range(n_worlds)
        ]
        return {
            "model": archived.identity["model"],
            "corpora": [
                {**corpus, "actual_inputs_sha256": hashlib.sha256(b"%d" % index).hexdigest()}
                for index in range(n_corpora)
            ],
            "evaluation": {
                **archived.identity["evaluation"],
                "worlds": worlds,
                "calls": [{"worlds": list(range(n_worlds)), "seed": 5}],
            },
        }

    for name, identity in (
        ("worlds-limit", declared(4096)),
        ("corpora-limit", declared(1, n_corpora=4096)),
    ):
        writer = EvaluationArchive.create(tmp_path / name, identity)
        assert writer.path == Path(os.path.realpath(tmp_path)) / name
        writer.abort()
    with pytest.raises(
        ValueError, match=re.escape("holds 4097 worlds; an archive declares at most 4096")
    ):
        EvaluationArchive.create(tmp_path / "worlds", declared(4097))
    with pytest.raises(
        ValueError, match=re.escape("holds 4097 corpora; an archive declares at most 4096")
    ):
        EvaluationArchive.create(tmp_path / "corpora", declared(1, n_corpora=4097))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["large-seed"]


def _self_consistent_rewrite(root, outside):
    folder = entry_dir(root, 5)
    original = DecompositionResult.load(folder).components
    rewrite_npz(folder / "arrays.npz", lambda arrays: arrays.update(components=2 * original))
    payload = json.loads((folder / "result.json").read_text())
    payload["arrays_sha256"] = hashlib.sha256((folder / "arrays.npz").read_bytes()).hexdigest()
    (folder / "result.json").write_text(json.dumps(payload, indent=2))
    # The result directory alone accepts the rewrite; only the archive's records catch it.
    np.testing.assert_array_equal(DecompositionResult.load(folder).components, 2 * original)


def _truth_record_resigned_alone(root, outside):
    path = entry_dir(root, 2) / "truth.npz"
    rewrite_npz(path, lambda arrays: arrays.update(observed=arrays["observed"] + 1))
    manifest = read_manifest(root)
    data = path.read_bytes()
    manifest["entries"][2]["files"]["truth.npz"] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    write_manifest(root, manifest)


def _duplicate_manifest_key(root, outside):
    text = (root / "manifest.json").read_text()
    (root / "manifest.json").write_text(text.replace("{", '{"kind": "evaluation_archive", ', 1))


def _context_length_of_first_world(manifest, value):
    for world in manifest["identity"]["evaluation"]["worlds"]:
        if (world["corpus_index"], world["row"]) == (0, 8):
            world["context_length"] = value


def _swap_manifest_worlds(manifest):
    entries = manifest["entries"]
    entries[0]["world"], entries[2]["world"] = entries[2]["world"], entries[0]["world"]


def _shift_call_seeds(manifest):
    for call in manifest["identity"]["evaluation"]["calls"]:
        call["seed"] += 1


def _then_resign(mutate):
    def mutation(root, outside):
        mutate(root, outside)
        resign(root)

    return mutation


def _resealed(edit):
    """Mutation: ``edit`` the manifest, keeping its file records, and re-sign its self-digest."""

    def mutation(root, outside):
        edit_manifest(root, edit)
        seal(root)

    return mutation


def _result_resigned(edit):
    """Mutation: ``edit`` the metadata of ``entries/0000`` (corpus 0 row 8 query), re-signed."""
    return _then_resign(lambda root, outside: edit_result(root, 0, edit))


def _truth_resigned(change):
    """Mutation: rewrite the arrays of ``entries/0002/truth.npz`` with ``change``, re-signed."""
    return _then_resign(lambda root, outside: rewrite_npz(entry_dir(root, 2) / "truth.npz", change))


def _nan_stored_truth(arrays):
    arrays["truth"][0, 0] = np.nan


def _edit_text(path, old, new):
    path.write_text(path.read_text().replace(old, new))


DIFFERS = "SHA-256 or size differs from manifest.json"
TAMPERING = [
    (
        "flipped arrays byte",
        lambda root, outside: flip(entry_dir(root, 1) / "arrays.npz"),
        "entries/0001/arrays.npz",
        DIFFERS,
    ),
    (
        "flipped truth byte",
        lambda root, outside: flip(entry_dir(root, 2) / "truth.npz"),
        "entries/0002/truth.npz",
        DIFFERS,
    ),
    (
        "edited result text",
        lambda root, outside: _edit_text(
            entry_dir(root, 0) / "result.json", '"outcome"', '"revenue"'
        ),
        "entries/0000/result.json",
        DIFFERS,
    ),
    ("self-consistent rewrite", _self_consistent_rewrite, "entries/0005/result.json", DIFFERS),
    (
        "re-signed result of another format",
        _then_resign(
            lambda root, outside: edit_payload(root, 0, lambda p: p.update(format_version=3))
        ),
        "entries/0000/result.json",
        "not a saved result: unsupported result format_version 3",
    ),
    (
        "re-signed arrays the result does not record",
        _then_resign(
            lambda root, outside: rewrite_npz(
                entry_dir(root, 1) / "arrays.npz",
                lambda arrays: arrays.update(components=2 * arrays["components"]),
            )
        ),
        "entries/0001",
        "saved result does not load: ValueError: result array checksum mismatch",
    ),
    (
        "edited n_samples",
        lambda root, outside: edit_manifest(
            root, lambda m: m["identity"]["evaluation"].update(n_samples=9)
        ),
        "manifest.json",
        "archive_sha256 does not match the manifest contents",
    ),
    (
        "truth record re-signed alone",
        _truth_record_resigned_alone,
        "manifest.json",
        "archive_sha256 does not match the manifest contents",
    ),
    (
        "deleted truth",
        lambda root, outside: (entry_dir(root, 3) / "truth.npz").unlink(),
        "entries/0003/truth.npz",
        "missing",
    ),
    (
        "extra entry directory",
        lambda root, outside: (root / "entries" / "8000").mkdir(),
        "entries/8000",
        "is not listed in manifest.json",
    ),
    (
        "extra root file",
        lambda root, outside: (root / "extra.txt").write_text("x"),
        "extra.txt",
        "is not listed in manifest.json",
    ),
    (
        "extra entry file",
        lambda root, outside: np.save(entry_dir(root, 0) / "extra.npy", np.zeros(1)),
        "entries/0000/extra.npy",
        "is not listed in manifest.json",
    ),
    (
        "symlinked truth",
        lambda root, outside: symlink_to_copy(entry_dir(root, 4) / "truth.npz", outside),
        "entries/0004/truth.npz",
        "is a symbolic link",
    ),
    (
        "symlinked entry directory",
        lambda root, outside: symlink_to_copy(entry_dir(root, 4), outside),
        "entries/0004",
        "is a symbolic link",
    ),
    (
        "symlinked entries",
        lambda root, outside: symlink_to_copy(root / "entries", outside),
        "entries",
        "is a symbolic link",
    ),
    ("duplicate manifest key", _duplicate_manifest_key, "manifest.json", "duplicate keys"),
    (
        "deeply nested manifest",
        lambda root, outside: (root / "manifest.json").write_text(
            json.dumps({"a": nested(2000, 0)})
        ),
        "manifest.json",
        "not strict JSON",
    ),
    (
        "truncated truth",
        _then_resign(
            lambda root, outside: rewrite_npz(
                entry_dir(root, 2) / "truth.npz",
                lambda arrays: arrays.update(truth=arrays["truth"][:-1]),
            )
        ),
        "entries/0002/truth.npz",
        "truth has shape",
    ),
    (
        "call seeds shifted",
        _then_resign(lambda root, outside: edit_manifest(root, _shift_call_seeds)),
        "entries/0000/result.json",
        "seed 5 differs from the archive's 6",
    ),
    (
        "entry removed",
        _then_resign(lambda root, outside: edit_manifest(root, lambda m: m["entries"].pop())),
        "manifest.json",
        "entries hold 15 world windows; the identity declares 16",
    ),
    (
        "format version 2",
        _then_resign(
            lambda root, outside: edit_manifest(root, lambda m: m.update(format_version=2))
        ),
        "manifest.json",
        "format_version 2",
    ),
    (
        "world swap",
        _then_resign(lambda root, outside: swap_dirs(root, "0000", "0002")),
        "entries/0000/result.json",
        "manifest.json entries[0] declares corpus 0 row 8",
    ),
    (
        "window swap",
        _then_resign(lambda root, outside: swap_dirs(root, "0000", "0001")),
        "entries/0000/result.json",
        "holds the context window; manifest.json entries[0] declares query",
    ),
    (
        "manifest world swap",
        _then_resign(lambda root, outside: edit_manifest(root, _swap_manifest_worlds)),
        "manifest.json",
        "the declared calls capture world 1 query there",
    ),
    (
        "lowered context length",
        _then_resign(
            lambda root, outside: edit_manifest(
                root, lambda m: _context_length_of_first_world(m, 5)
            )
        ),
        "entries/0000/result.json",
        "context_length 6 differs from the archive's 5",
    ),
    (
        "other model configuration",
        _then_resign(
            lambda root, outside: edit_manifest(
                root, lambda m: m["identity"]["model"].update(config_sha256=OTHER_CONFIG)
            )
        ),
        "entries/0000/result.json",
        "model configuration digest",
    ),
    (
        "deeply nested result",
        _then_resign(
            lambda root, outside: (entry_dir(root, 0) / "result.json").write_text(
                json.dumps({"a": nested(2000, 0)})
            )
        ),
        "entries/0000/result.json",
        "not strict JSON",
    ),
    (
        "edge_families not a list",
        _then_resign(
            lambda root, outside: edit_result(
                root, 0, lambda m: m["provenance"].update(edge_families=3)
            )
        ),
        "entries/0000/result.json",
        "edge_families must name one family per edge slot",
    ),
    (
        "sales_scale removed",
        _then_resign(
            lambda root, outside: edit_result(root, 0, lambda m: m["provenance"].pop("sales_scale"))
        ),
        "entries/0000/result.json",
        "sales_scale must be a finite positive number",
    ),
    (
        "joint head",
        _result_resigned(lambda m: m["provenance"].update(head="joint")),
        "entries/0000/result.json",
        "head 'joint' differs from the archive's 'conditional'",
    ),
    (
        "revenue units",
        _result_resigned(lambda m: m.update(units="revenue")),
        "entries/0000/result.json",
        "units 'revenue' differs from the archive's 'outcome'",
    ),
    (
        "other world id",
        _result_resigned(lambda m: m["provenance"].update(world_id="corpus:0/row:9")),
        "entries/0000/result.json",
        "world_id 'corpus:0/row:9' differs from the archive's 'corpus:0/row:8'",
    ),
    (
        "non-finite stored truth",
        _truth_resigned(_nan_stored_truth),
        "entries/0002/truth.npz",
        "truth must be finite",
    ),
    (
        "float32 stored truth",
        _truth_resigned(lambda arrays: arrays.update(truth=arrays["truth"].astype(np.float32))),
        "entries/0002/truth.npz",
        "truth must be float64, got float32",
    ),
    (
        "extra stored evidence",
        _truth_resigned(lambda arrays: arrays.update(extra=np.zeros(1))),
        "entries/0002/truth.npz",
        "must hold exactly ['graph_truth', 'observed', 'truth']",
    ),
    (
        "other kind",
        _resealed(_set(kind="other")),
        "manifest.json",
        "kind must be 'evaluation_archive'",
    ),
    (
        "extra manifest key",
        _resealed(_set(note="x")),
        "manifest.json",
        "manifest must contain exactly",
    ),
    (
        "undeclared corpus",
        _resealed(_set("identity", "evaluation", "worlds", 0, corpus_index=99)),
        "manifest.json",
        "identity.evaluation.worlds[0].corpus_index must index identity.corpora",
    ),
    (
        "text world count",
        _resealed(_set("identity", "corpora", 0, n_worlds="x")),
        "manifest.json",
        "identity.corpora[0].n_worlds must be a positive integer",
    ),
    (
        "entries not a list",
        _resealed(_set(entries={})),
        "manifest.json",
        "entries must be a list",
    ),
    (
        "undeclared entry world",
        _resealed(_set("entries", 0, world=99)),
        "manifest.json",
        "entries[0].world must index identity.evaluation.worlds",
    ),
    (
        "unknown window",
        _resealed(_set("entries", 0, window="both")),
        "manifest.json",
        "entries[0].window must be 'query' or 'context'",
    ),
    (
        "files without truth.npz",
        _resealed(lambda m: m["entries"][0]["files"].pop("truth.npz")),
        "manifest.json",
        "entries[0].files must contain exactly",
    ),
    (
        "extra file record",
        _resealed(_set("entries", 0, "files", **{"extra.npy": {"sha256": "0" * 64, "size": 1}})),
        "manifest.json",
        "entries[0].files must contain exactly",
    ),
    (
        "invalid file digest",
        _resealed(_set("entries", 0, "files", "truth.npz", sha256="zz")),
        "manifest.json",
        "entries[0].files['truth.npz'].sha256 must be a full lowercase SHA-256",
    ),
    (
        "empty file size",
        _resealed(_set("entries", 0, "files", "truth.npz", size=0)),
        "manifest.json",
        "entries[0].files['truth.npz'].size must be a positive integer",
    ),
    (
        "text file size",
        _resealed(_set("entries", 0, "files", "truth.npz", size="x")),
        "manifest.json",
        "entries[0].files['truth.npz'].size must be a positive integer",
    ),
    (
        "boolean format version",
        _resealed(_set(format_version=True)),
        "manifest.json",
        "unsupported evaluation archive format_version True; this release reads version 1",
    ),
]


@pytest.mark.parametrize("name", ARCHIVES)
@pytest.mark.parametrize(
    ("mutate", "expected", "fragment"),
    [pytest.param(mutate, path, text, id=case) for case, mutate, path, text in TAMPERING],
)
def test_tampering_is_detected_at_load_and_names_the_offending_path(
    built, tmp_path, name, mutate, expected, fragment
):
    root = copy_archive(built[name].archive.path, tmp_path / "archive")
    outside = tmp_path / "outside"
    outside.mkdir()
    mutate(root, outside)
    with pytest.raises(ArchiveIntegrityError) as caught:
        EvaluationArchive.load(root)
    error = caught.value
    assert error.path == expected
    assert error.archive == Path(os.path.realpath(root))
    assert expected in str(error) and fragment in str(error)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is ArchiveIntegrityError
    assert (restored.path, str(restored)) == (error.path, str(error))


def test_hidden_files_and_a_symlinked_archive_root_still_load(built, tmp_path):
    source = built["archived"]
    root = copy_archive(source.archive.path, tmp_path / "archive")
    for folder in (root, root / "entries", entry_dir(root, 3)):
        (folder / ".DS_Store").write_bytes(b"\0\1")
    archive = EvaluationArchive.load(root)
    assert archive.archive_sha256 == source.archive.archive_sha256
    np.testing.assert_array_equal(archive.record(3)[1], source.captured[3][1])
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    linked = EvaluationArchive.load(link)
    assert linked.path == Path(os.path.realpath(root)) != link
    assert all(entry.path.parent == linked.path / "entries" for entry in linked.entries)
    assert [entry.world_id for entry in linked.entries] == [
        entry.world_id for entry in source.archive.entries
    ]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="permission bits bind only an unprivileged POSIX user",
)
def test_an_archive_that_cannot_be_searched_is_unreadable_not_missing(built, tmp_path):
    source = built["archived"].archive.path
    root = copy_archive(source, tmp_path / "archive")
    nested = copy_archive(source, tmp_path / "parent" / "archive")
    # The root without search permission: its name is visible, nothing inside it is. A
    # parent without any permission: the root itself cannot even be looked up.
    for folder, mode, archive, relative in (
        (root, 0o600, root, "manifest.json"),
        (nested.parent, 0o000, nested, "."),
    ):
        folder.chmod(mode)
        try:
            with pytest.raises(ArchiveIntegrityError) as caught:
                EvaluationArchive.load(archive)
        finally:
            folder.chmod(0o700)
        assert caught.value.path == relative
        assert "unreadable: PermissionError" in str(caught.value)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="evaluation archive manifest not found"):
        EvaluationArchive.load(tmp_path / "empty")


def test_load_needs_an_archive_directory(built, tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    paths = (
        built["archived"].archive.path / "manifest.json",  # a file
        built["archived"].archive.path / "manifest.json" / "archive",  # below a file
        tmp_path / "missing",
        loop,
    )
    for path in paths:
        with pytest.raises(FileNotFoundError, match="not an evaluation archive directory"):
            EvaluationArchive.load(path)


@pytest.mark.parametrize(
    "method",
    (
        "records",
        "independent_entries",
        "effect_rows",
        "sbc_inputs",
        "conditional_sbc_inputs",
        "to_datatree",
    ),
)
def test_reader_methods_take_only_a_known_window(built, method):
    with pytest.raises(ValueError, match=re.escape("window must be 'query' or 'context'")):
        getattr(built["archived"].archive, method)("Query")


def _replace_by_fifo(path, outside):
    if not hasattr(os, "mkfifo"):
        pytest.skip("this platform has no FIFOs")
    path.unlink()
    os.mkfifo(path)


def _sparse(path, outside):
    os.truncate(path, 1 << 30)


SUBSTITUTES = {"fifo": _replace_by_fifo, "symlink": symlink_to_copy, "sparse": _sparse}


@pytest.mark.parametrize(
    ("substitute", "problem"),
    [
        ("fifo", "is not a regular file"),
        ("symlink", "is a symbolic link"),
        ("sparse", "exceeds the 8 MiB format limit"),
    ],
)
def test_a_substituted_manifest_is_refused_before_it_is_read(built, tmp_path, substitute, problem):
    root = copy_archive(built["archived"].archive.path, tmp_path / "archive")
    SUBSTITUTES[substitute](root / "manifest.json", tmp_path)
    error, peak = peak_traced(lambda: bounded(lambda: EvaluationArchive.load(root)))
    assert isinstance(error, ArchiveIntegrityError) and error.path == "manifest.json"
    assert problem in str(error)
    assert peak < 64 << 20


def test_files_changed_after_load_fail_on_read_before_rows_are_returned(built, tmp_path):
    source = built["archived"]
    root = copy_archive(source.archive.path, tmp_path / "archive")
    archive = EvaluationArchive.load(root)
    changed = archive.independent_entries("query")[1]
    flip(changed.path / "truth.npz")
    reads = [
        lambda: archive.effect_rows("query"),
        lambda: archive.sbc_inputs("query"),
        lambda: archive.conditional_sbc_inputs("query"),
        lambda: list(archive.records()),
        lambda: list(archive),
    ]
    if all(importlib.util.find_spec(name) is not None for name in ("xarray", "arviz_base")):
        reads.append(lambda: archive.to_datatree("query"))
    for read in reads:
        with pytest.raises(ArchiveIntegrityError, match=re.escape(DIFFERS)) as caught:
            read()
        assert caught.value.path == f"entries/{changed.index:04d}/truth.npz"
    context = archive.effect_rows("context")
    assert {row["world"] for row in context} == {world_id_of(item[0]) for item in source.captured}
    np.testing.assert_array_equal(archive.record(0)[1], source.captured[0][1])


@pytest.mark.parametrize(
    ("substitute", "index", "relative", "problem"),
    [
        ("fifo", 4, "entries/0004/truth.npz", "is not a regular file"),
        ("symlink", 4, "entries/0004/truth.npz", "is a symbolic link"),
        ("sparse", 4, "entries/0004/truth.npz", DIFFERS),
        ("directory", 4, "entries/0004", "is a symbolic link"),
        ("entries", 0, "entries", "is a symbolic link"),
    ],
)
def test_an_entry_substituted_after_load_is_refused_before_it_is_read(
    built, tmp_path, substitute, index, relative, problem
):
    root = copy_archive(built["archived"].archive.path, tmp_path / "archive")
    archive = EvaluationArchive.load(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    if substitute in SUBSTITUTES:
        SUBSTITUTES[substitute](entry_dir(root, 4) / "truth.npz", outside)
    else:
        symlink_to_copy(root / relative, outside)
    error, peak = peak_traced(lambda: bounded(lambda: archive.record(index)))
    assert isinstance(error, ArchiveIntegrityError)
    assert error.path == relative and problem in str(error)
    assert peak < 64 << 20


def test_reports_from_an_archive_never_call_the_model(built, monkeypatch):
    path = built["archived"].archive.path

    def reports(archive):
        out = {
            "records": [where(result) for result, *_ in archive.records()],
            "strata": canonical(strata_summary(archive.effect_rows("query"))),
            "sbc": archive.sbc_inputs("context"),
            "conditional": archive.conditional_sbc_inputs("context"),
        }
        if importlib.util.find_spec("scipy") is not None:
            out["sbc_card"] = canonical(sbc_report(**out["sbc"], simulations=100))
            card = conditional_sbc_report(**out["conditional"], simulations=100)
            out["conditional_card"] = canonical(card)
        return out

    expected = reports(EvaluationArchive.load(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("an archive report called the model")

    for name in (
        "predict_batch",
        "sample_disentangled",
        "sample_context",
        "sample",
        "loss_terms",
        "structure_logits",
    ):
        monkeypatch.setattr(MFM, name, forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr(training_evaluation, "evaluate_rows", forbidden)
    actual = reports(EvaluationArchive.load(path))
    assert actual.keys() == expected.keys()
    for key in ("records", "strata", "sbc_card", "conditional_card"):
        assert actual.get(key) == expected.get(key), key
    np.testing.assert_array_equal(actual["sbc"]["samples"], expected["sbc"]["samples"])
    assert actual["conditional"]["cell_ids"] == expected["conditional"]["cell_ids"]


def test_diagnostics_verifies_and_reads_archives_without_importing_training(built):
    code = (
        "import sys\n"
        "from pymc_mktfm.diagnostics import EvaluationArchive, family_window_totals\n"
        "archive = EvaluationArchive.load(sys.argv[1])\n"
        "records = list(archive.records())\n"
        "result, truth, _, _ = archive.record(0)\n"
        "draws, totals = family_window_totals(result, truth)\n"
        "selected = archive.independent_entries('query')\n"
        "inputs = archive.sbc_inputs('query')\n"
        "conditional = archive.conditional_sbc_inputs('query')\n"
        "assert len(records) == len(archive) == 16\n"
        "assert inputs['world_ids'] == conditional['world_ids']\n"
        "assert inputs['world_ids'] == [entry.world_id for entry in selected]\n"
        "assert draws.shape == (result.n_samples, 5) and totals.shape == (5,)\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('pymc_mktfm.training'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(built["archived"].archive.path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sbc_inputs_need_two_independent_cells(world, tmp_path):
    model, corpora = world
    identity = identity_for(model, corpora, [0, 1], [(0, 8), (0, 9)])
    with EvaluationArchive.create(tmp_path / "one-cell", identity) as writer:
        capture(model, corpora, identity, writer)
    archive = EvaluationArchive.load(writer.path)
    for convert in (archive.sbc_inputs, archive.conditional_sbc_inputs):
        with pytest.raises(ValueError, match="at least two independent generator cells") as caught:
            convert("query")
        assert "the archive has 1 across 2 worlds" in str(caught.value)
    rows = archive.effect_rows("query")
    assert {row["world"] for row in rows} == {"shard:530001/row:8", "shard:530001/row:9"}


@pytest.mark.parametrize("name", ["truth.npz", "arrays.npz"])
def test_a_file_replaced_while_an_entry_is_read_is_never_decoded(
    built, tmp_path, monkeypatch, name
):
    source = built["archived"]
    archive = EvaluationArchive.load(copy_archive(source.archive.path, tmp_path / "archive"))
    result, truth, graph_truth, observations = source.captured[2]
    target = archive.path / "entries" / "0002" / name
    if name == "truth.npz":
        other = {"truth": truth + 1, "graph_truth": graph_truth, "observed": observations}
    else:
        other = {
            "components": result.components * 2,
            "graphs": result.graphs,
            "component_mask": result.component_mask,
            "edge_mask": result.edge_mask,
            "time": result.time,
        }
    read = archive_module._read_bounded
    fired = []

    def replace_after_reading(path, *args, **kwargs):
        data = read(path, *args, **kwargs)
        if Path(path) == target and not fired:
            fired.append(path)
            np.savez_compressed(target, **other)
        return data

    monkeypatch.setattr(archive_module, "_read_bounded", replace_after_reading)
    loaded, loaded_truth, _, _ = archive.record(2)
    assert fired
    np.testing.assert_array_equal(loaded_truth, truth)
    np.testing.assert_array_equal(loaded.components, result.components)
    with pytest.raises(ArchiveIntegrityError, match="differs from manifest.json") as caught:
        archive.record(2)
    assert caught.value.path == f"entries/0002/{name}"
