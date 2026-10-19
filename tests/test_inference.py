"""The observation boundary uses no synthetic labels and preserves result identity."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from pymc_mfm.data.corpus import corpus_batch
from pymc_mfm.inference.api import decompose, observation_batch
from pymc_mfm.inference.result import DecompositionResult
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
    result = decompose(
        model,
        observed(synthetic_corpus),
        n_samples=4,
        channel_names=("paid social", "search", "padding"),
        control_names=("price", "holiday"),
        units="USD",
    )
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


def test_context_is_a_separate_marginal_and_preserves_training_mode(model, synthetic_corpus):
    model.train()
    inputs = observed(synthetic_corpus)
    time = np.arange(len(inputs["spend"])) + 2020
    result = decompose(model, inputs, n_samples=4, window="context", time=time)
    assert model.training
    assert result.window == "context"
    np.testing.assert_array_equal(result.time, time[: len(inputs["context_outcome"])])
    assert "not aligned" in result.provenance["marginal_windows"]


def test_supplied_graph_is_validated_before_float32_rounding(model, synthetic_corpus):
    graph = np.zeros(EdgeSlotLayout(3, 2, 2).n_slots, dtype=np.float64)
    graph[0] = 1 + 1e-9
    for supplied in (graph, graph.tolist(), torch.from_numpy(graph)):
        with pytest.raises(ValueError, match="binary"):
            decompose(model, observed(synthetic_corpus), n_samples=3, mode="fixed", graph=supplied)


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
    assert len(result.provenance["data_sha256"]) == 64


def test_bfloat16_evidence_preserves_sample_values(model, synthetic_corpus):
    model = model.to(dtype=torch.bfloat16)
    inputs = observed(synthetic_corpus)
    batch = {
        key: value.to(dtype=torch.bfloat16)
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
