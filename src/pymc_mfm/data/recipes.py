"""Canonical corpus recipes and schema-4 corpus helpers.

``PROFILE_A_PRIOR`` is the fixed ``pymc_generator`` configuration for profile A.
Corpus size and random seed are caller arguments rather than recipe fields.
``generate_corpus`` writes one corpus and a JSON sidecar with its exact
construction inputs; ``load_corpora`` validates schema 4 and records corpus
identity for training.

Profile B changes profile A's ``cy`` edge budget to ``(4, 10)`` while retaining
the other prior settings. ``pymc_generator`` is imported lazily, so this module
remains importable without the ``generation`` extra.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.data.profiles import PROFILE_A, PROFILE_B, ArtifactProfile

if TYPE_CHECKING:
    from pymc_generator import SCMPrior

_GENERATION_EXTRA_HINT = (
    "pymc_generator is required for corpus generation and loading; install the "
    "`generation` extra: pip install 'pymc-mfm[generation]'"
)

#: Canonical profile-A SCMPrior payload in public ``pymc_generator`` field
#: names. ``n_cells``, ``draws_per_cell``, and ``seed`` are caller arguments.
PROFILE_A_PRIOR: Mapping[str, Any] = MappingProxyType(
    {
        "n_time_steps": 104,
        "n_treatments": 15,
        "n_covariates": 6,
        "n_latent": 1,
        "val_cell_frac": 0.2,
        "query_frac": 0.25,
        "p_long_horizon": 0.5,
        "treatment_cv_floor": 0.08,
        "l_max": 4,
        "n_treatments_active_range": (6, 15),
        "n_covariates_active_range": (4, 6),
        "n_latent_active_range": (1, 1),
        "carryover_alpha_range": (0.2, 0.8),
        "carryover_family_probs": {"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        "saturation_family_probs": {
            "linear": 0.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 1.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        "weibull_lam_range": (2.0, 8.0),
        "weibull_k_range": (1.5, 4.0),
        "dz_base_rate": 0.3,
        "zc_base_rate": 0.3,
        "cc_base_rate": 0.15,
        "zz_base_rate": 0.1,
        "dc_coeff_range": (0.1, 0.5),
        "dz_coeff_range": (0.1, 0.5),
        "zc_coeff_range": (0.05, 0.3),
        "cc_coeff_range": (0.05, 0.3),
        "zz_coeff_range": (-0.2, 0.2),
        "dy_coeff_range": (0.15, 0.45),
        "zy_coeff_range": (0.1, 0.4),
        "beta_additive_range": (0.5, 2.0),
        "rw_covariate_mean_range": (-1.0, 1.0),
        "rw_positive_mean_range": (0.3, 4.0),
        "rw_baseline_mean_range": (3.0, 8.0),
        "rw_std_sigma": 1.0,
        "rw_baseline_std_sigma": None,
        "rw_outcome_std_sigma": 0.25,
        "outcome_std_mode": "relative",
        "rw_baseline_std_range": (0.0, 0.093),
        "rw_outcome_std_range": (0.01, 0.028),
        "rw_smoothness_alpha": 2.0,
        "rw_smoothness_beta": 2.0,
        "rw_smoothness_max_weeks": 26,
        "confounding_strength_range": (0.0, 0.0),
        "rw_treatment_std_range": (0.15, 0.8),
        "treatment_hf_sigma_range": (0.08, 0.6),
        "treatment_pulse_prob_range": (0.0, 0.25),
        "treatment_pulse_amp_range": (0.4, 2.5),
        "covariate_hf_sigma_range": (0.1, 0.8),
        "covariate_pulse_prob_range": (0.0, 0.25),
        "covariate_pulse_amp_range": (0.5, 3.0),
        "baseline_floor": 0.0,
        "baseline_floor_scope": "non_treatment",
        "carryover_burn_in": 4,
        "n_treatment_shocks": 0,
        "treatment_shock_length_range": (2, 2),
        "treatment_shock_level_range": (0.0, 0.0),
        "include_identifiability_labels": True,
        "prior_conditioning": False,
        "prior_cond_width_ranges": None,
        "edge_rate_overrides": None,
        "edge_budget": {
            "cy": (5, 5),
            "zy": (3, 3),
            "dy": 0,
            "dc": 0,
            "dz": 0,
            "zc": 0,
            "cc": 0,
            "zz": 0,
        },
        "min_no_direct_effect_treatments": 1,
    }
)

#: Shard cell geometry for profile A (1,250 worlds).
PROFILE_A_TRAIN_CELLS: Mapping[str, int] = MappingProxyType({"n_cells": 125, "draws_per_cell": 10})

#: Base seed for the documented profile-A shard sequence.
PROFILE_A_SHARD_BASE_SEED = 530000


@dataclass(frozen=True, slots=True)
class PriorRecipe:
    """A profile's named SCMPrior payload.

    ``payload`` holds every ``SCMPrior`` field except ``n_cells``,
    ``draws_per_cell``, and ``seed``.
    """

    name: str
    profile: ArtifactProfile
    payload: Mapping[str, Any]


_RECIPES: dict[ArtifactProfile, PriorRecipe] = {
    PROFILE_A: PriorRecipe(
        name="A",
        profile=PROFILE_A,
        payload=PROFILE_A_PRIOR,
    ),
    PROFILE_B: PriorRecipe(
        name="B",
        profile=PROFILE_B,
        payload=MappingProxyType(
            {**PROFILE_A_PRIOR, "edge_budget": {**PROFILE_A_PRIOR["edge_budget"], "cy": (4, 10)}}
        ),
    ),
}


def recipe_for(profile: ArtifactProfile) -> PriorRecipe:
    """Return the corpus recipe for a known profile.

    A returns the profile-A configuration. B changes ``edge_budget["cy"]``
    to ``(4, 10)``. Unknown profiles raise ``KeyError``.
    """
    try:
        return _RECIPES[profile]
    except KeyError:
        raise KeyError(f"no prior recipe for profile {profile.name!r}") from None


def profile_prior(
    profile: ArtifactProfile,
    *,
    n_cells: int,
    draws_per_cell: int,
    seed: int,
    **overrides: Any,
) -> SCMPrior:
    """Build a validated ``SCMPrior`` for a profile's recipe.

    Constructs through ``pymc_generator.make_scm_prior`` with the whole recipe
    payload passed explicitly, so none of the preset's texture or family
    defaults survive: the result's fields equal the payload plus the three
    call arguments, then ``overrides`` (any ``SCMPrior`` field) applied last.
    Requires the ``generation`` extra.
    """
    try:
        from pymc_generator import make_scm_prior
    except ImportError as exc:
        raise ImportError(_GENERATION_EXTRA_HINT) from exc

    payload = _fresh_payload(recipe_for(profile).payload)
    payload.update(n_cells=n_cells, draws_per_cell=draws_per_cell, seed=seed)
    payload.update(overrides)
    return make_scm_prior(**payload)


def _fresh_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a recipe payload so the constructed prior never aliases module constants."""
    return {
        key: dict(value) if isinstance(value, Mapping) else value for key, value in payload.items()
    }


def prior_sidecar_path(path: str | Path) -> Path:
    """Path of the JSON prior sidecar written next to a corpus: ``<path>.prior.json``."""
    path = Path(path)
    return path.with_name(path.name + ".prior.json")


def generate_corpus(prior: SCMPrior, path: str | Path) -> dict[str, Any]:
    """Generate one corpus from ``prior``, save it to ``path`` and write its sidecar.

    Runs ``pymc_generator.DataGenerator(prior).generate()`` (validated) and
    ``pymc_generator.save_corpus``; then writes ``prior_sidecar_path(path)`` with
    ``{"prior": asdict(prior), "pymc_generator_version": ..., "generated_at": ISO-8601 UTC}``
    (tuples become JSON lists). Returns the in-memory corpus. Requires the
    ``generation`` extra.
    """
    try:
        import pymc_generator
    except ImportError as exc:
        raise ImportError(_GENERATION_EXTRA_HINT) from exc

    corpus = pymc_generator.DataGenerator(prior).generate()
    pymc_generator.save_corpus(corpus, path)
    with Path(path).open("rb") as handle:
        corpus_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    sidecar = {
        "prior": dataclasses.asdict(prior),
        "pymc_generator_version": pymc_generator.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_origin": "synthetic",
        "corpus_sha256": corpus_digest,
    }
    prior_sidecar_path(path).write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return corpus


@dataclass(frozen=True, slots=True)
class LoadedCorpus:
    """A schema-4 corpus loaded from disk with the identity facts a run records.

    ``sha256`` is over the file bytes. ``n_holdout`` counts ``is_val == 1`` rows,
    ``n_train`` the rest. ``context_lengths`` are the distinct
    ``support_mask.sum(axis=1)`` values, ascending.
    """

    path: str
    sha256: str
    n_worlds: int
    n_train: int
    n_holdout: int
    context_lengths: tuple[int, ...]
    corpus: dict[str, Any]


_REQUIRED_CORPUS_KEYS = ("is_val", "cell_id", "support_mask")


def load_corpora(paths: Sequence[str | Path]) -> list[LoadedCorpus]:
    """Load schema-4 corpora with ``pymc_generator.load_corpus`` in the given order.

    Each corpus must be stamped ``diagnostics["schema_version"] == 4`` and carry
    ``is_val``, ``cell_id`` and ``support_mask``; anything else raises
    ``ValueError`` naming the file. Geometry agreement across corpora is the
    consumer's check, not this loader's. Requires the ``generation`` extra.
    """
    try:
        from pymc_generator import load_corpus
    except ImportError as exc:
        raise ImportError(_GENERATION_EXTRA_HINT) from exc

    loaded = []
    for raw_path in paths:
        path = Path(raw_path)
        corpus = load_corpus(path)
        diagnostics = corpus.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics.get("schema_version") != 4:
            raise ValueError(f"corpus {path}: diagnostics must declare schema_version 4")
        missing = [key for key in _REQUIRED_CORPUS_KEYS if key not in corpus]
        if missing:
            raise ValueError(f"corpus {path}: missing required arrays {missing}")
        is_val = np.asarray(corpus["is_val"])
        with path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        n_holdout = int(np.count_nonzero(is_val == 1))
        lengths = np.unique(np.asarray(corpus["support_mask"]).sum(axis=1))
        loaded.append(
            LoadedCorpus(
                path=str(path),
                sha256=sha256,
                n_worlds=int(is_val.shape[0]),
                n_train=int(is_val.shape[0]) - n_holdout,
                n_holdout=n_holdout,
                context_lengths=tuple(int(length) for length in lengths),
                corpus=corpus,
            )
        )
    return loaded


def provenance(loaded: LoadedCorpus) -> dict[str, Any]:
    """JSON-safe identity record of a loaded corpus for run manifests.

    Contains ``path``, ``sha256``, the three row counts, ``context_lengths`` and
    ``prior_sidecar``: the parsed ``<path>.prior.json`` when present, else ``None``.
    """
    sidecar = prior_sidecar_path(loaded.path)
    sidecar_record = json.loads(sidecar.read_text()) if sidecar.exists() else None
    synthetic = (
        isinstance(sidecar_record, dict)
        and sidecar_record.get("data_origin") == "synthetic"
        and sidecar_record.get("corpus_sha256") == loaded.sha256
    )
    return {
        "path": loaded.path,
        "sha256": loaded.sha256,
        "n_worlds": loaded.n_worlds,
        "n_train": loaded.n_train,
        "n_holdout": loaded.n_holdout,
        "context_lengths": list(loaded.context_lengths),
        "prior_sidecar": sidecar_record,
        "data_origin": "synthetic" if synthetic else "unknown",
    }
