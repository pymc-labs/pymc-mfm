"""Generator-free checks of the corpus-to-model representation boundary."""

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from pymc_mktfm.data.corpus import corpus_batch


@pytest.fixture
def corpus():
    time = np.arange(6, dtype=np.float64)
    factor = np.array([1.0, 2.0])
    outcome = 2.0 ** (time + 3)
    spend = np.column_stack((np.expm1(time), [3, 3, 3, 3, 7, 15], np.zeros(6)))
    controls = np.array(
        [[-3, 0, 2], [-1, 0, 2], [1, 0, 2], [3, 0, 2], [5, 0, 3], [100, 0, 15]],
        dtype=np.float64,
    )
    direct = np.column_stack((time + 1, time + 2, np.zeros(6)))
    direct[0, 0] = 6  # Keep observed outcome positive while the intrinsic baseline is negative.
    covariate = np.column_stack((-(time + 1) / 4, np.zeros((6, 2))))
    latent = np.column_stack((np.zeros(6), -(time + 1) / 2, (time + 1) / 4))
    indirect = np.column_stack((-(time + 1), 2 * (time + 1), (time + 1) / 2))
    noise = time + 0.5
    baseline = outcome - sum(part.sum(axis=1) for part in (direct, covariate, latent, indirect))
    baseline -= noise
    # Literal schema-4 order, independently of EdgeSlotLayout.pack.
    graph = np.array(
        [
            1,
            1,
            0,  # cy
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,  # dc: latent -> treatment
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,  # dz: latent -> covariate
            0,
            1,
            1,  # dy
            1,
            0,
            0,  # zy
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,  # zc: covariate -> treatment
            1,
            0,
            0,
            0,
            0,
            0,  # cc: row-major off-diagonal, 0 -> 1
            0,
            1,
            0,
            0,
            0,
            0,  # zz: row-major off-diagonal, 0 -> 2
        ],
        dtype=np.uint8,
    )
    return {
        "diagnostics": {
            "schema_version": 4,
            "edge_types": ["cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"],
        },
        "support_mask": np.tile([1, 1, 1, 1, 0, 0], (2, 1)).astype(np.uint8),
        "treatment_active_mask": np.tile([1, 1, 0], (2, 1)).astype(np.uint8),
        "covariate_active_mask": np.tile([1, 0, 1], (2, 1)).astype(np.uint8),
        "latent_active_mask": np.tile([0, 1, 1], (2, 1)).astype(np.uint8),
        "treatment_raw": np.broadcast_to(spend, (2, 6, 3)).copy(),
        "covariates": np.broadcast_to(controls, (2, 6, 3)).copy(),
        "outcome_raw": factor[:, None] * outcome,
        "baseline_intrinsic": factor[:, None] * baseline,
        "treatment_contribution_raw": factor[:, None, None] * direct,
        "covariate_contribution": factor[:, None, None] * covariate,
        "latent_unobserved_contribution": factor[:, None, None] * latent,
        "indirect_effects_by_source": factor[:, None, None] * indirect,
        "g": np.tile(graph, (2, 1)),
    }


def test_targets_restore_signed_natural_components_without_observation_noise(corpus):
    batch = corpus_batch(corpus)
    # Baseline, three treatments, three covariates, summed latent, cc/zc/dc.
    expected = torch.tensor(
        [
            [-1.5, 6, 2, 0, -0.25, 0, 0, -0.25, -1, 2, 0.5],
            [7.5, 2, 3, 0, -0.5, 0, 0, -0.5, -2, 4, 1],
            [19.5, 3, 4, 0, -0.75, 0, 0, -0.75, -3, 6, 1.5],
            [47.5, 4, 5, 0, -1, 0, 0, -1, -4, 8, 2],
            [107.5, 5, 6, 0, -1.25, 0, 0, -1.25, -5, 10, 2.5],
            [231.5, 6, 7, 0, -1.5, 0, 0, -1.5, -6, 12, 3],
        ],
        dtype=torch.float64,
    )
    factor = torch.tensor([1.0, 2.0])
    targets = torch.cat((batch["context_targets_z"], batch["targets_z"]), dim=1)
    restored = targets * batch["sales_scale"][:, None, None]
    torch.testing.assert_close(restored, factor[:, None, None] * expected, rtol=1e-6, atol=1e-6)
    noise = factor[:, None] * (torch.arange(6) + 0.5)
    outcome = torch.tensor(corpus["outcome_raw"], dtype=torch.float64)
    torch.testing.assert_close(restored.sum(dim=-1), outcome - noise, rtol=1e-6, atol=1e-6)


def test_training_features_and_targets_do_not_depend_on_world_units(corpus):
    reference = corpus_batch(corpus)
    scaled = deepcopy(corpus)
    sales_units = np.array([1e100, 1e-100])
    scaled["treatment_raw"] *= np.array([1e80, 1e-80, 1])
    scaled["covariates"] *= np.array([1e-90, 1, 1e90])
    for key in (
        "outcome_raw",
        "baseline_intrinsic",
        "treatment_contribution_raw",
        "covariate_contribution",
        "latent_unobserved_contribution",
        "indirect_effects_by_source",
    ):
        scaled[key] *= sales_units.reshape(2, *(1 for _ in range(scaled[key].ndim - 1)))
    actual = corpus_batch(scaled)
    for key in reference:
        if key == "sales_scale":
            np.testing.assert_allclose(
                actual[key].numpy() / sales_units, reference[key].numpy(), rtol=1e-14
            )
        else:
            torch.testing.assert_close(actual[key], reference[key], rtol=1e-5, atol=1e-6)


def test_signed_latents_are_normalized_before_reduction_can_overflow(corpus):
    corpus["outcome_raw"].fill(1e308)
    corpus["latent_active_mask"].fill(1)
    corpus["latent_unobserved_contribution"][:] = [1e308, 1e308, -1e308]
    with np.errstate(over="raise", invalid="raise"):
        batch = corpus_batch(corpus)
    latent_column = 1 + corpus["treatment_raw"].shape[-1] + corpus["covariates"].shape[-1]
    for key in ("context_targets_z", "targets_z"):
        torch.testing.assert_close(
            batch[key][..., latent_column], torch.ones_like(batch[key][..., latent_column])
        )


def test_future_observations_and_truth_cannot_change_context_features(corpus):
    original = corpus_batch(corpus)
    changed = deepcopy(corpus)
    changed["outcome_raw"][:, 4:] *= -1e6
    changed["treatment_raw"][:, 4:, :2] *= 2
    changed["covariates"][:, 4:, [0, 2]] += 10
    changed["baseline_intrinsic"] += 100
    changed["treatment_contribution_raw"] *= 7
    changed["covariate_contribution"] *= -3
    changed["latent_unobserved_contribution"] *= 8
    changed["indirect_effects_by_source"] += [10, -20, 30]
    changed["g"][:, 6] = 0
    modified = corpus_batch(changed)
    observations = (
        "ctx_spend_z",
        "ctx_controls_z",
        "ctx_revenue_z",
        "spend_mean",
        "spend_std",
        "ctx_week",
        "qry_week",
        "sales_scale",
        "response_moments",
        "adstock_moments",
        "control_moments",
        "control_location_scale",
        "control_effect_moments",
    )
    for key in observations:
        torch.testing.assert_close(modified[key], original[key], rtol=0, atol=0)
    assert not torch.equal(modified["targets_z"], original["targets_z"])
    assert not torch.equal(modified["context_targets_z"], original["context_targets_z"])
    assert not torch.equal(modified["g_dc"], original["g_dc"])
    assert not torch.equal(modified["qry_spend_z"], original["qry_spend_z"])
    assert not torch.equal(modified["qry_controls_z"], original["qry_controls_z"])


def test_packed_graph_preserves_direction_offdiagonals_and_outcome_edges(corpus):
    batch = corpus_batch(corpus)
    expected = {
        "cy": [1, 1, 0],
        "dc": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "dz": [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        "db": [0, 1, 1],
        "zb": [1, 0, 0],
        "zc": [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
        "cc": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        "zz": [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
    }
    for family, values in expected.items():
        value = torch.tensor(values, dtype=torch.float32)
        torch.testing.assert_close(batch[f"g_{family}"], value.unsqueeze(0).expand(2, *value.shape))


def test_schema_versions_four_and_five_give_the_same_batch(corpus):
    reference = corpus_batch(corpus)
    corpus["diagnostics"]["schema_version"] = 5
    for name, tensor in corpus_batch(corpus).items():
        torch.testing.assert_close(tensor, reference[name])


@pytest.mark.parametrize("version", [3, 6, "4", True, None])
def test_other_schema_versions_are_rejected(corpus, version):
    corpus["diagnostics"]["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        corpus_batch(corpus)


@pytest.mark.parametrize(
    "support",
    [[1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
    ids=["noncontiguous-context", "empty-context", "empty-query"],
)
def test_context_and_query_must_form_nonempty_temporal_windows(corpus, support):
    corpus["support_mask"][:] = support
    with pytest.raises(ValueError):
        corpus_batch(corpus)


def test_mixed_context_lengths_require_selecting_a_compatible_minibatch(corpus):
    corpus["support_mask"][1] = [1, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        corpus_batch(corpus)
    selected = corpus_batch(corpus, rows=[1])
    torch.testing.assert_close(selected["sales_scale"], torch.tensor([60.8], dtype=torch.float64))
    torch.testing.assert_close(
        selected["ctx_revenue_z"], torch.tensor([[[-math.log(2)], [0], [math.log(2)]]])
    )
    torch.testing.assert_close(selected["qry_week"], torch.tensor([[[3 / 6], [4 / 6], [5 / 6]]]))


def test_zero_observed_outcome_is_rejected_instead_of_clipped(corpus):
    corpus["outcome_raw"][0, 1] = 0
    with pytest.raises(ValueError):
        corpus_batch(corpus)


def test_negative_direct_effect_is_rejected_instead_of_training_outside_support(corpus):
    corpus["treatment_contribution_raw"][0, 1, 0] = -1
    with pytest.raises(ValueError):
        corpus_batch(corpus)


@pytest.mark.parametrize(
    ("key", "index", "value"),
    [
        ("treatment_raw", (0, 0, 0), -1e-50),
        ("treatment_raw", (0, 0, 2), 1e-50),
        ("covariates", (0, 0, 1), 1e-50),
    ],
)
def test_raw_domain_and_padding_are_checked_before_precision_narrowing(corpus, key, index, value):
    corpus[key][index] = value
    with pytest.raises(ValueError):
        corpus_batch(corpus)
