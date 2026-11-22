"""The observation boundary uses no synthetic labels and preserves result identity."""

import copy
import inspect
import json
import subprocess
import sys
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES

from pymc_mfm.data.corpus import corpus_batch
from pymc_mfm.inference.api import decompose, observation_batch
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.inference.result import DecompositionResult, result_from_batch
from pymc_mfm.model.graph import EdgeSlotLayout


def observed(corpus, row=0):
    context = int(corpus["support_mask"][row].sum())
    return {
        "spend": corpus["treatment_raw"][row],
        "controls": corpus["covariates"][row],
        "context_outcome": corpus["outcome_raw"][row, :context],
        "channel_mask": corpus["treatment_active_mask"][row],
        "control_mask": corpus["covariate_active_mask"][row],
        "latent_mask": corpus["latent_active_mask"][row],
    }


def test_observed_representation_matches_training_without_targets(model, synthetic_corpus):
    inputs = observed(synthetic_corpus)
    actual = observation_batch(**inputs, model_config=model.config)
    training = corpus_batch(synthetic_corpus, [0])
    assert not any(name.startswith("g_") or "targets" in name for name in actual)
    for name, value in actual.items():
        torch.testing.assert_close(value, training[name], rtol=0, atol=0)
    graph = torch.zeros(1, EdgeSlotLayout(3, 2, 2).n_slots)
    torch.manual_seed(92)
    expected, expected_graph = model.sample_disentangled(training, 5, mode="fixed", graph=graph)
    torch.manual_seed(92)
    result = decompose(model, inputs, n_samples=5, mode="fixed", graph=graph)
    np.testing.assert_allclose(
        result.components, expected[:, 0].numpy() * float(training["sales_scale"][0]), rtol=1e-7
    )
    np.testing.assert_array_equal(result.graphs, expected_graph[:, 0].numpy())
    assert result.provenance["graph_source"] == "supplied_hypothesis"
    assert result.components.shape[1] == len(inputs["spend"]) - len(inputs["context_outcome"])


def test_result_replay_retains_labels_and_rejects_corruption(model, synthetic_corpus, tmp_path):
    inputs = {
        **observed(synthetic_corpus, row=2),
        "channel_mask": np.array([1, 1, 0]),
        "control_mask": np.array([1, 1]),
    }
    result = decompose(
        model,
        inputs,
        n_samples=4,
        channel_names=("paid social", "search", "padding"),
        control_names=("price", "holiday"),
        units="USD",
    )
    assert result.provenance["input_usage"] == {
        "context": {"steps": 9, "allocated_tokens": 54, "represented_tokens": 45},
        "query": {"steps": 3, "allocated_tokens": 18, "represented_tokens": 15},
        "total": {"allocated_tokens": 72, "represented_tokens": 60},
    }
    path = result.save(tmp_path / "evidence")
    replay = DecompositionResult.load(path)
    np.testing.assert_array_equal(replay.components, result.components)
    np.testing.assert_array_equal(replay.graphs, result.graphs)
    assert replay.component_names == result.component_names
    assert replay.provenance == result.provenance
    assert replay.units == "USD"
    with pytest.raises(FileExistsError):
        result.save(path)
    with (path / "arrays.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        DecompositionResult.load(path)


def test_invalid_alignment_padding_and_binary_graphs_are_rejected(model, synthetic_corpus):
    result = decompose(model, observed(synthetic_corpus), n_samples=4)
    with pytest.raises(ValueError) as caught:
        decompose(model, pd.DataFrame({"spend": [1.0]}), n_samples=4)
    assert str(caught.value) == (
        "observations must be a mapping of arrays. For pandas input, use "
        "model.inference(context, query, date_column=..., target_column=..., "
        "channel_columns=..., control_columns=...)"
    )
    with pytest.raises(ValueError, match="aligned"):
        replace(result, graphs=result.graphs[:-1])
    bad_graph = result.graphs.copy()
    bad_graph[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        replace(result, graphs=bad_graph)
    bad_components = result.components.copy()
    bad_components[..., ~result.component_mask] = 1
    with pytest.raises(ValueError, match="padded"):
        replace(result, components=bad_components)
    with pytest.raises(ValueError, match="time"):
        replace(result, time=result.time[::-1])


def test_observation_input_rejects_silent_padding_and_invalid_domain(model, synthetic_corpus):
    inputs = observed(synthetic_corpus)
    spend = inputs["spend"].copy()
    spend[0, -1] = 1
    with pytest.raises(ValueError, match="padded"):
        observation_batch(**{**inputs, "spend": spend}, model_config=model.config)
    outcome = inputs["context_outcome"].copy()
    outcome[0] = 0
    with pytest.raises(ValueError, match="positive"):
        observation_batch(**{**inputs, "context_outcome": outcome}, model_config=model.config)
    with pytest.raises(ValueError, match="control width"):
        observation_batch(
            **{**inputs, "controls": inputs["controls"][:, :1]}, model_config=model.config
        )


@pytest.mark.parametrize(
    ("name", "column", "value"),
    [("spend", 0, -1e-50), ("spend", -1, 1e-50), ("controls", -1, 1e-50)],
)
def test_raw_domain_and_padding_are_validated_before_rounding(
    model, synthetic_corpus, name, column, value
):
    inputs = observed(synthetic_corpus)
    values = inputs[name].astype(np.float64)
    if name == "controls":
        inputs["control_mask"] = np.array([1, 0])
        values[:, -1] = 0
    values[0, column] = value
    with pytest.raises(ValueError):
        observation_batch(**{**inputs, name: values}, model_config=model.config)


@pytest.mark.parametrize(
    "mutation",
    [
        "not_mapping",
        "missing_spend",
        "non_tensor",
        "scalar_spend",
        "query_width",
        "missing_controls",
        "scale_shape",
        "mask_shape",
        "missing_feature",
        "unknown_scalar",
        "unknown_vector",
    ],
)
def test_encoded_observations_validate_structure_before_indexing(model, synthetic_corpus, mutation):
    encoded = observation_batch(**observed(synthetic_corpus), model_config=model.config)
    if mutation == "not_mapping":
        encoded = list(encoded.items())
    elif mutation == "missing_spend":
        del encoded["ctx_spend_z"]
    elif mutation == "non_tensor":
        encoded["ctx_spend_z"] = encoded["ctx_spend_z"].tolist()
    elif mutation == "scalar_spend":
        encoded["ctx_spend_z"] = torch.tensor(1.0)
    elif mutation == "query_width":
        encoded["qry_spend_z"] = encoded["qry_spend_z"][..., :-1]
    elif mutation == "missing_controls":
        del encoded["ctx_controls_z"]
    elif mutation == "scale_shape":
        encoded["sales_scale"] = encoded["sales_scale"].reshape(1, 1)
    elif mutation == "mask_shape":
        encoded["active_j_mask"] = encoded["active_j_mask"][0]
    elif mutation.startswith("unknown_"):
        encoded["note"] = torch.tensor(1.0) if mutation == "unknown_scalar" else torch.ones(1)
    else:
        del encoded["control_effect_moments"]
    with pytest.raises(ValueError):
        decompose(model, encoded, n_samples=3)


@pytest.mark.parametrize("name", ["active_c_mask", "active_m_mask", "active_j_mask"])
def test_encoded_masks_are_validated_before_dtype_conversion(model, synthetic_corpus, name):
    encoded = observation_batch(**observed(synthetic_corpus), model_config=model.config)
    encoded[name] = encoded[name].double()
    encoded[name][0, 0] = 1 + 1e-9
    with pytest.raises(ValueError):
        decompose(model, encoded, n_samples=3)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("qry_spend_z", float("nan")),
        ("response_moments", float("inf")),
        ("ctx_spend_z", 1 + 1j),
        ("sales_scale", -1.0),
        ("qry_controls_z", 1e300),
    ],
)
def test_encoded_values_are_validated_before_sampling(model, synthetic_corpus, name, value):
    encoded = observation_batch(**observed(synthetic_corpus), model_config=model.config)
    dtype = torch.complex128 if isinstance(value, complex) else torch.float64
    encoded[name] = encoded[name].to(dtype)
    encoded[name].reshape(-1)[0] = value
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        decompose(model, encoded, n_samples=3)
    assert torch.equal(torch.get_rng_state(), before)


def test_encoded_observations_preserve_dtype_and_ignore_training_truth(model, synthetic_corpus):
    model = model.double()
    inputs = observed(synthetic_corpus)
    encoded = observation_batch(**inputs, model_config=model.config)
    for name in ("active_c_mask", "active_m_mask", "active_j_mask"):
        encoded[name] = encoded[name].bool()
    encoded["g_cy"] = object()
    encoded["targets_z"] = torch.tensor(float("nan"))
    encoded["context_targets_z"] = None
    torch.manual_seed(93)
    expected = decompose(model, inputs, n_samples=3)
    torch.manual_seed(93)
    actual = decompose(model, encoded, n_samples=3)
    np.testing.assert_array_equal(actual.components, expected.components)
    np.testing.assert_array_equal(actual.graphs, expected.graphs)
    assert actual.provenance["data_sha256"] == expected.provenance["data_sha256"]


def test_input_usage_is_owned_metadata_for_raw_encoded_and_zero_valued_inputs(
    model, synthetic_corpus
):
    inputs = observed(synthetic_corpus, row=2)
    inputs["spend"] = inputs["spend"].copy()
    inputs["controls"] = inputs["controls"].copy()
    inputs["spend"][:, 1:] = 0
    inputs["controls"][:, 1] = 0
    inputs["channel_mask"] = np.array([1, 0, 0])
    inputs["control_mask"] = np.array([1, 0])
    expected_usage = {
        "context": {"steps": 9, "allocated_tokens": 54, "represented_tokens": 27},
        "query": {"steps": 3, "allocated_tokens": 18, "represented_tokens": 9},
        "total": {"allocated_tokens": 72, "represented_tokens": 36},
    }
    annotations = {
        "input_usage": {"context": {"steps": -1}, "total": {"allocated_tokens": -1}},
        "scenario": {"name": "synthetic forecast"},
    }
    original_annotations = copy.deepcopy(annotations)
    torch.manual_seed(95)
    reference = decompose(model, inputs, n_samples=3)
    encoded = observation_batch(**inputs, model_config=model.config)
    for observations in (inputs, encoded):
        torch.manual_seed(95)
        result = decompose(model, observations, n_samples=3, provenance=annotations)
        assert result.provenance["input_usage"] == expected_usage
        assert result.provenance["scenario"] == original_annotations["scenario"]
        assert annotations == original_annotations
        assert result.provenance["data_sha256"] == reference.provenance["data_sha256"]
        np.testing.assert_array_equal(result.components, reference.components)
        np.testing.assert_array_equal(result.graphs, reference.graphs)

    # Identical zero columns are represented when masks are omitted, not padding.
    unmasked = {name: inputs[name] for name in ("spend", "controls", "context_outcome")}
    represented = decompose(model, unmasked, n_samples=2)
    assert represented.provenance["input_usage"] == {
        "context": {"steps": 9, "allocated_tokens": 54, "represented_tokens": 54},
        "query": {"steps": 3, "allocated_tokens": 18, "represented_tokens": 18},
        "total": {"allocated_tokens": 72, "represented_tokens": 72},
    }


def test_input_usage_uses_selected_world_masks_for_both_output_windows(model, batch):
    encoded = {
        **batch,
        "active_c_mask": batch["active_c_mask"].clone(),
        "sales_scale": torch.tensor([100.0, 200.0], dtype=torch.float64),
    }
    encoded["active_c_mask"][0] = torch.tensor([1.0, 0.0, 0.0])
    expected_usage = (
        {
            "context": {"steps": 5, "allocated_tokens": 30, "represented_tokens": 20},
            "query": {"steps": 3, "allocated_tokens": 18, "represented_tokens": 12},
            "total": {"allocated_tokens": 48, "represented_tokens": 32},
        },
        {
            "context": {"steps": 5, "allocated_tokens": 30, "represented_tokens": 25},
            "query": {"steps": 3, "allocated_tokens": 18, "represented_tokens": 15},
            "total": {"allocated_tokens": 48, "represented_tokens": 40},
        },
    )
    query, query_graphs = model.sample_disentangled(encoded, 2)
    context, context_graphs = model.sample_disentangled(
        encoded,
        3,
        window="context",
        mode="fixed",
        graph=torch.zeros(2, EdgeSlotLayout(3, 2, 2).n_slots),
    )
    for window, components, graphs, mode in (
        ("query", query, query_graphs, "model_average"),
        ("context", context, context_graphs, "fixed"),
    ):
        for world, usage in enumerate(expected_usage):
            result = result_from_batch(
                encoded,
                components,
                graphs,
                model_config=model.config,
                world=world,
                window=window,
                mode=mode,
            )
            assert result.provenance["input_usage"] == usage
            assert result.components.shape[1] == (5 if window == "context" else 3)


def test_context_is_a_separate_marginal_and_preserves_training_mode(
    model, synthetic_corpus, tmp_path
):
    model.train()
    inputs = observed(synthetic_corpus)
    time = np.arange(len(inputs["spend"])) + 2020
    result = decompose(model, inputs, n_samples=4, window="context", time=time)
    assert model.training
    assert result.window == "context"
    np.testing.assert_array_equal(result.time, time[: len(inputs["context_outcome"])])
    direct = replace(result, provenance={})
    replay = DecompositionResult.load(direct.save(tmp_path / "direct-context"))
    np.testing.assert_array_equal(replay.components, direct.components)
    np.testing.assert_array_equal(replay.time, time[: len(inputs["context_outcome"])])
    assert replay.provenance == {}
    with pytest.raises(ValueError):
        replace(direct, context_length=direct.context_length + 1)


def test_supplied_graph_is_validated_before_float32_rounding(model, synthetic_corpus):
    graph = np.zeros(EdgeSlotLayout(3, 2, 2).n_slots, dtype=np.float64)
    graph[0] = 1 + 1e-9
    for supplied in (graph, graph.tolist(), torch.from_numpy(graph)):
        with pytest.raises(ValueError, match="binary"):
            decompose(model, observed(synthetic_corpus), n_samples=3, mode="fixed", graph=supplied)


def test_a_results_own_read_only_graph_is_a_valid_fixed_hypothesis(model, synthetic_corpus):
    inputs = observed(synthetic_corpus)
    drawn = decompose(model, inputs, n_samples=3)
    assert not drawn.graphs.flags.writeable
    # PyTorch warns about non-writable arrays once per process unless told otherwise.
    previous = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fixed = decompose(model, inputs, n_samples=3, mode="fixed", graph=drawn.graphs[0])
    finally:
        torch.set_warn_always(previous)
    np.testing.assert_array_equal(fixed.graphs, np.tile(drawn.graphs[0], (3, 1)))


@pytest.mark.parametrize(
    ("mode", "supply_graph", "router_samples"),
    [
        ("fixed", False, None),
        ("model_average", True, None),
        ("map", True, None),
        ("model_average", False, 4),
        ("fixed", True, 4),
    ],
)
def test_graph_arguments_cannot_be_silently_ignored(
    model, synthetic_corpus, mode, supply_graph, router_samples
):
    graph = torch.zeros(EdgeSlotLayout(3, 2, 2).n_slots) if supply_graph else None
    with pytest.raises(ValueError):
        decompose(
            model,
            observed(synthetic_corpus),
            n_samples=3,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
        )


@pytest.mark.parametrize("field", ["mode", "window", "n_samples", "graph_source"])
def test_saved_result_rejects_contradictory_inference_labels(
    model, synthetic_corpus, tmp_path, field
):
    graph = torch.zeros(EdgeSlotLayout(3, 2, 2).n_slots)
    result = decompose(model, observed(synthetic_corpus), n_samples=3, mode="fixed", graph=graph)
    path = result.save(tmp_path / "evidence")
    payload = json.loads((path / "result.json").read_text())
    metadata = payload["metadata"]
    if field == "mode":
        metadata["provenance"]["inference_config"]["mode"] = "map"
    elif field == "window":
        # This world's context/query lengths coincide, so shape cannot expose relabeling.
        metadata["window"] = "context"
    elif field == "n_samples":
        metadata["provenance"]["inference_config"]["n_samples"] += 1
    else:
        metadata["provenance"]["graph_source"] = "joint_router"
    (path / "result.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        DecompositionResult.load(path)


def test_caller_annotations_cannot_replace_loaded_identity(model, synthetic_corpus, tmp_path):
    source = {
        "source_checkpoint_sha256": "a" * 64,
        "training_run_id": "training-17",
        "tracking_local_run_id": "local-17",
        "tracking_run_id": "backend-17",
    }
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "model", provenance=source, profile="A")
    )
    conflicts = {
        "checkpoint_sha256": "f" * 64,
        "source_checkpoint_sha256": "f" * 64,
        "exported_checkpoint_sha256": "f" * 64,
        "bundle_sha256": "f" * 64,
        "profile": "B",
        "training_run_id": "other-training",
        "tracking_local_run_id": "other-local",
        "tracking_run_id": "other-backend",
        "artifact": {"profile": "B"},
    }
    for key, value in conflicts.items():
        with pytest.raises(ValueError):
            decompose(loaded, observed(synthetic_corpus), n_samples=3, provenance={key: value})


def test_identical_and_supplemental_annotations_preserve_loaded_evidence(
    model, synthetic_corpus, tmp_path
):
    source = {"source_checkpoint_sha256": "a" * 64, "training_run_id": "training-17"}
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "model", provenance=source, profile="A")
    )
    annotations = {
        **loaded.provenance,
        "artifact": loaded.artifact_metadata,
        "analysis_label": "observed-quarter",
    }
    torch.manual_seed(94)
    expected = decompose(loaded, observed(synthetic_corpus), n_samples=3)
    torch.manual_seed(94)
    result = decompose(loaded, observed(synthetic_corpus), n_samples=3, provenance=annotations)
    replay = DecompositionResult.load(result.save(tmp_path / "evidence"))
    np.testing.assert_array_equal(replay.components, expected.components)
    np.testing.assert_array_equal(replay.graphs, expected.graphs)
    for key, value in annotations.items():
        assert replay.provenance[key] == value


def test_results_name_but_do_not_copy_the_bundles_recorded_evaluation(
    model, synthetic_corpus, tmp_path
):
    from test_checkpoint import _training_evaluation

    source = {"source_checkpoint_sha256": "a" * 64, "training_evaluation": _training_evaluation()}
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "model", provenance=source, profile="A")
    )
    plain = decompose(loaded, observed(synthetic_corpus), n_samples=3)
    # Echoing the model's own identity back is consistent, not a conflict.
    echoed = decompose(
        loaded,
        observed(synthetic_corpus),
        n_samples=3,
        provenance={**loaded.provenance, "artifact": loaded.artifact_metadata},
    )
    for result in (plain, echoed):
        saved = (result.save(tmp_path / f"result-{id(result)}") / "result.json").read_text()
        assert "training_evaluation" not in saved
        assert result.provenance["source_checkpoint_sha256"] == "a" * 64
        assert result.provenance["bundle_sha256"] == loaded.provenance["bundle_sha256"]


def test_loading_and_sampling_import_no_compiler_stack(model, synthetic_corpus, tmp_path):
    # torch._dynamo and sympy took 0.3-0.7 s to import on an Apple M3 Max and 2-3 s on Linux
    # hosts, more than sampling 1,000 draws; neither loading a bundle nor sampling needs them.
    path = save_pretrained(model, tmp_path / "model")
    np.savez(tmp_path / "observed.npz", **observed(synthetic_corpus))
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from pymc_mfm import from_pretrained\n"
        "from pymc_mfm.inference import decompose\n"
        "model = from_pretrained(sys.argv[1])\n"
        "decompose(model, dict(np.load(sys.argv[2])), n_samples=3)\n"
        "loaded = [name for name in ('torch._dynamo', 'sympy') if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(tmp_path / "observed.npz")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unsigned_decreasing_time_cannot_overflow_validation(model, synthetic_corpus):
    inputs = observed(synthetic_corpus)
    time = np.arange(len(inputs["spend"]), dtype=np.uint64)[::-1]
    with pytest.raises(ValueError, match="increasing"):
        decompose(model, inputs, time=time, n_samples=3)
    result = decompose(model, inputs, n_samples=3)
    with pytest.raises(ValueError, match="increasing"):
        replace(result, time=result.time.astype(np.uint64)[::-1])


def test_modal_result_retains_actual_router_bank_size(model, synthetic_corpus):
    result = decompose(
        model, observed(synthetic_corpus), n_samples=3, mode="map", router_samples=np.int64(7)
    )
    assert result.provenance["inference_config"]["router_samples"] == 7
    assert result.provenance["inference_config"]["n_samples"] == result.n_samples


def test_fractional_draw_count_is_rejected(model, synthetic_corpus):
    with pytest.raises(ValueError):
        decompose(model, observed(synthetic_corpus), n_samples=2.5)


def test_bfloat16_evidence_preserves_sample_values(model, synthetic_corpus):
    model = model.to(dtype=torch.bfloat16)
    inputs = observed(synthetic_corpus)
    batch = {
        key: value if key == "sales_scale" else value.to(dtype=torch.bfloat16)
        for key, value in observation_batch(**inputs, model_config=model.config).items()
    }
    torch.manual_seed(321)
    expected, graphs = model.sample_disentangled(batch, 3)
    torch.manual_seed(321)
    result = decompose(model, inputs, n_samples=3)
    np.testing.assert_array_equal(
        result.components,
        expected[:, 0].float().numpy().astype(np.float64) * float(batch["sales_scale"][0]),
    )
    np.testing.assert_array_equal(result.graphs, graphs[:, 0].float().numpy())


def test_public_dag_has_one_outcome_for_all_direct_families(model, synthetic_corpus):
    result = decompose(model, observed(synthetic_corpus), n_samples=3)
    targets = {
        destination
        for family, (_, destination) in zip(
            result.provenance["edge_families"], result.provenance["edge_endpoints"], strict=True
        )
        if family in ("cy", "db", "zb")
    }
    assert targets == {"outcome"}
    assert result.provenance["node_names"].count("outcome") == 1


@pytest.mark.parametrize("window", ["context", "query"])
@pytest.mark.parametrize("sales_units", [10.0, 1e200, 1e-200])
def test_decomposition_restores_units_without_changing_graphs(
    model, synthetic_corpus, tmp_path, window, sales_units
):
    inputs = observed(synthetic_corpus)
    scaled = {
        **inputs,
        "spend": inputs["spend"].astype(np.float64) * [1e200, 1e-200, 1],
        "controls": inputs["controls"].astype(np.float64) * [1e-190, 1e190],
        "context_outcome": inputs["context_outcome"].astype(np.float64) * sales_units,
    }
    torch.manual_seed(931)
    reference = decompose(model, inputs, window=window, n_samples=5)
    torch.manual_seed(931)
    actual = decompose(model, scaled, window=window, n_samples=5, units="rescaled sales")
    np.testing.assert_allclose(
        actual.components / sales_units, reference.components, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(actual.graphs, reference.graphs)
    assert actual.provenance["sales_scale"] / sales_units == pytest.approx(
        reference.provenance["sales_scale"], rel=1e-14
    )
    replay = DecompositionResult.load(actual.save(tmp_path / "scaled"))
    np.testing.assert_array_equal(replay.components, actual.components)
    assert replay.units == "rescaled sales"


def test_default_draw_count_is_1000_on_every_public_sampling_signature():
    from pymc_mfm.model.flow import DEFAULT_N_SAMPLES, JointFlowHead
    from pymc_mfm.model.model import MFM

    assert DEFAULT_N_SAMPLES == 1000
    for fn in (
        decompose,
        MFM.inference,
        MFM.predict,
        MFM.predict_batch,
        MFM.sample_context,
        MFM.sample_disentangled,
        MFM.sample,
        JointFlowHead.sample,
    ):
        assert inspect.signature(fn).parameters["n_samples"].default == 1000, fn.__qualname__


def test_calls_that_omit_the_draw_count_sample_1000_draws(model, chart_frames):
    history, future = chart_frames
    tree = model.inference(history, future, **CHART_ROLES, random_seed=0)
    assert tree["context/posterior"].sizes["draw"] == 1000
    assert tree["query/posterior"].sizes["draw"] == 1000
    assert model.predict(future, random_seed=0).n_samples == 1000
