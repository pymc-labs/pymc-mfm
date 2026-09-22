"""Prior recipe translation, reconstruction flags and corpus loading identity."""

import dataclasses
import importlib.util
import json
import subprocess
import sys

import pytest
from conftest import make_synthetic_corpus

from pymc_mfm.data.profiles import PROFILE_A, PROFILE_B
from pymc_mfm.data.recipes import (
    FW53_PRIOR,
    load_corpora,
    prior_sidecar_path,
    profile_prior,
    provenance,
    recipe_for,
)

needs_generator = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)

# Key list of pymc-pfn-backups/fw53/d10_r2/fw53_geo_mm/manifest.json["config"]
# (config_hash 63cc128d...4920), in archive order.
ARCHIVED_KEYS = (
    "n_time_steps",
    "n_treatments",
    "n_covariates",
    "n_latent",
    "n_cells",
    "draws_per_cell",
    "val_cell_frac",
    "query_frac",
    "p_long_horizon",
    "spend_cv_floor",
    "l_max",
    "n_treatments_active_range",
    "n_covariates_active_range",
    "n_latent_active_range",
    "adstock_alpha_range",
    "adstock_family_probs",
    "saturation_family_probs",
    "weibull_lam_range",
    "weibull_k_range",
    "dz_base_rate",
    "zc_base_rate",
    "cc_base_rate",
    "zz_base_rate",
    "dc_coeff_range",
    "dz_coeff_range",
    "zc_coeff_range",
    "cc_coeff_range",
    "zz_coeff_range",
    "db_coeff_range",
    "zb_coeff_range",
    "beta_additive_range",
    "rw_mean_range",
    "rw_positive_mean_range",
    "rw_baseline_mean_range",
    "rw_std_sigma",
    "rw_baseline_std_sigma",
    "rw_sales_std_sigma",
    "outcome_std_mode",
    "rw_baseline_std_range",
    "rw_sales_std_range",
    "rw_smoothness_alpha",
    "rw_smoothness_beta",
    "rw_smoothness_max_weeks",
    "confounding_strength_range",
    "rw_channel_std_range",
    "channel_hf_sigma_range",
    "channel_pulse_prob_range",
    "channel_pulse_amp_range",
    "control_hf_sigma_range",
    "control_pulse_prob_range",
    "control_pulse_amp_range",
    "baseline_floor",
    "baseline_floor_scope",
    "adstock_burn_in",
    "n_channel_shocks",
    "channel_shock_length_range",
    "channel_shock_level_range",
    "include_identifiability_labels",
    "prior_conditioning",
    "prior_cond_width_ranges",
    "edge_rate_overrides",
    "edge_budget",
    "min_dead_channels",
)
RENAMES = {
    "adstock_alpha_range": "carryover_alpha_range",
    "adstock_family_probs": "carryover_family_probs",
    "adstock_burn_in": "carryover_burn_in",
    "channel_hf_sigma_range": "treatment_hf_sigma_range",
    "channel_pulse_prob_range": "treatment_pulse_prob_range",
    "channel_pulse_amp_range": "treatment_pulse_amp_range",
    "channel_shock_length_range": "treatment_shock_length_range",
    "channel_shock_level_range": "treatment_shock_level_range",
    "control_hf_sigma_range": "covariate_hf_sigma_range",
    "control_pulse_prob_range": "covariate_pulse_prob_range",
    "control_pulse_amp_range": "covariate_pulse_amp_range",
    "db_coeff_range": "dy_coeff_range",
    "zb_coeff_range": "zy_coeff_range",
    "min_dead_channels": "min_no_direct_effect_treatments",
    "n_channel_shocks": "n_treatment_shocks",
    "rw_channel_std_range": "rw_treatment_std_range",
    "rw_mean_range": "rw_covariate_mean_range",
    "rw_sales_std_range": "rw_outcome_std_range",
    "rw_sales_std_sigma": "rw_outcome_std_sigma",
    "spend_cv_floor": "treatment_cv_floor",
}
CALL_ARGUMENTS = {"n_cells", "draws_per_cell"}


def test_fw53_prior_is_the_archived_config_after_rename():
    assert len(ARCHIVED_KEYS) == 63
    expected = {RENAMES.get(key, key) for key in ARCHIVED_KEYS} - CALL_ARGUMENTS
    assert set(FW53_PRIOR) == expected
    assert set(FW53_PRIOR["edge_budget"]) == {"cy", "dy", "zy", "dc", "dz", "zc", "cc", "zz"}
    assert FW53_PRIOR["edge_budget"]["cy"] == (5, 5)
    assert FW53_PRIOR["edge_budget"]["zy"] == (3, 3)
    assert FW53_PRIOR["baseline_floor_scope"] == "non_treatment"
    assert set(FW53_PRIOR["carryover_family_probs"]) == {"none", "geometric", "weibull"}
    assert set(FW53_PRIOR["saturation_family_probs"]) == {
        "linear",
        "hill",
        "logistic",
        "michaelis_menten",
        "tanh",
        "root",
    }
    ranges = [value for value in FW53_PRIOR.values() if isinstance(value, (tuple, list))]
    assert ranges and all(isinstance(value, tuple) for value in ranges)


def test_profile_b_is_a_flagged_reconstruction_of_a():
    a, b = recipe_for(PROFILE_A), recipe_for(PROFILE_B)
    assert a.reconstructed is False
    assert b.reconstructed is True
    assert "cy410" in b.source
    assert b.payload["edge_budget"]["cy"] == PROFILE_B.live_channels == (4, 10)
    assert {k: v for k, v in b.payload.items() if k != "edge_budget"} == {
        k: v for k, v in a.payload.items() if k != "edge_budget"
    }
    assert {k: v for k, v in b.payload["edge_budget"].items() if k != "cy"} == {
        k: v for k, v in a.payload["edge_budget"].items() if k != "cy"
    }
    with pytest.raises(KeyError):
        recipe_for(dataclasses.replace(PROFILE_A, name="C"))


def test_import_does_not_load_the_generator():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pymc_mfm.data.recipes, sys; assert 'pymc_generator' not in sys.modules",
        ],
        check=True,
    )


@needs_generator
def test_profile_prior_constructs_exactly_the_payload():
    prior = profile_prior(PROFILE_A, n_cells=2, draws_per_cell=1, seed=1)
    assert dataclasses.asdict(prior) == {
        **FW53_PRIOR,
        "n_cells": 2,
        "draws_per_cell": 1,
        "seed": 1,
    }
    reconstructed = profile_prior(PROFILE_B, n_cells=2, draws_per_cell=1, seed=1)
    assert dataclasses.asdict(reconstructed)["edge_budget"]["cy"] == (4, 10)
    overridden = profile_prior(PROFILE_A, n_cells=2, draws_per_cell=1, seed=1, n_time_steps=52)
    assert overridden.n_time_steps == 52
    assert FW53_PRIOR["edge_budget"]["cy"] == (5, 5)


@needs_generator
def test_load_corpora_reports_identity_and_sidecar(tmp_path):
    from pymc_generator import save_corpus

    corpus = make_synthetic_corpus(n_cells=6, draws_per_cell=2, val_cells=2, context_lengths=(6, 9))
    path = tmp_path / "corpus.npz"
    save_corpus(corpus, path)

    (loaded,) = load_corpora([path])
    assert (loaded.n_worlds, loaded.n_train, loaded.n_holdout) == (12, 8, 4)
    assert loaded.context_lengths == (6, 9)
    assert loaded.corpus["diagnostics"]["schema_version"] == 4
    assert len(loaded.sha256) == 64
    assert provenance(loaded)["prior_sidecar"] is None

    prior_sidecar_path(path).write_text(json.dumps({"prior": {"seed": 7}}))
    record = provenance(loaded)
    assert record["prior_sidecar"] == {"prior": {"seed": 7}}
    assert record["context_lengths"] == [6, 9]
    json.dumps(record, allow_nan=False)


def test_synthetic_sharing_permission_requires_hash_bound_origin(tmp_path):
    from pymc_mfm.data.recipes import LoadedCorpus

    path = tmp_path / "corpus.npz"
    loaded = LoadedCorpus(str(path), "a" * 64, 2, 1, 1, (6,), {})
    assert provenance(loaded)["data_origin"] == "unknown"
    sidecar = prior_sidecar_path(path)
    sidecar.write_text(json.dumps({"data_origin": "synthetic", "corpus_sha256": "b" * 64}))
    assert provenance(loaded)["data_origin"] == "unknown"
    sidecar.write_text(json.dumps({"data_origin": "synthetic", "corpus_sha256": loaded.sha256}))
    assert provenance(loaded)["data_origin"] == "synthetic"


@needs_generator
def test_load_corpora_rejects_missing_split_arrays(tmp_path):
    from pymc_generator import save_corpus

    corpus = make_synthetic_corpus()
    del corpus["is_val"]
    path = tmp_path / "no_split.npz"
    save_corpus(corpus, path)
    with pytest.raises(ValueError, match="is_val"):
        load_corpora([path])


@needs_generator
def test_cli_overrides_reach_the_prior(monkeypatch, tmp_path):
    import pymc_mfm.data.__main__ as cli

    captured = {}

    def fake_generate(prior, path):
        captured["prior"] = prior
        captured["path"] = path
        return {"is_val": make_synthetic_corpus()["is_val"]}

    monkeypatch.setattr(cli, "generate_corpus", fake_generate)
    out = tmp_path / "b.npz"
    assert (
        cli.main(
            [
                "generate",
                "--profile",
                "B",
                "--n-cells",
                "3",
                "--draws-per-cell",
                "2",
                "--seed",
                "9",
                "--out",
                str(out),
                "--override",
                'edge_budget={"cy": [6, 6]}',
                "--override",
                "n_time_steps=52",
            ]
        )
        == 0
    )
    prior = captured["prior"]
    assert captured["path"] == str(out)
    assert (prior.n_cells, prior.draws_per_cell, prior.seed) == (3, 2, 9)
    assert prior.edge_budget == {"cy": (6, 6)}
    assert prior.n_time_steps == 52
