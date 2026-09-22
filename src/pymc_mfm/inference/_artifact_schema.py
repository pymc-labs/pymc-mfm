"""Versioned, non-executable metadata for the one canonical architecture."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from pymc_mfm.data.features import (
    ADSTOCK_LEVEL_LIMIT,
    ADSTOCK_MOMENT_DECAYS,
    ADSTOCK_MOMENT_L_MAX,
    CONTROL_MOMENT_T_DIVISOR,
    MOMENT_FEATURE_BLOCKS,
    RESPONSE_MOMENT_DECAYS,
    RESPONSE_MOMENT_L_MAX,
)
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.flow import (
    _TRAJECTORY_CONSTANT_LOADING_SCALE,
    _TRAJECTORY_FEATURE_VERSION,
    _TRAJECTORY_RANK,
    DECOMPOSITION_FIXED_TARGETS,
    ZERO_ATOM_GATES,
)
from pymc_mfm.model.graph import STRUCT_EDGE_KEYS

FORMAT_VERSION = 1
ARCHITECTURE = "mfm-graph-conditional-v1"
BUNDLE_FILES = ("config.json", "model.safetensors", "manifest.json")
COMPATIBILITY = {
    "python": ">=3.13",
    "pymc-mfm": ">=0.0.1.dev0,<0.0.2",
    "numpy": ">=2.1,<3",
    "torch": ">=2.6,<3",
    "safetensors": ">=0.6.2,<1",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
PROFILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SECRET_KEY = re.compile(
    r"token|secret|password|credential|authorization|api[-_]?key|access[-_]?key",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b|\bBearer\s+\S+", re.IGNORECASE)
_PROVENANCE_KEYS = frozenset(
    {
        "status",
        "source_repository",
        "source_revision",
        "source_checkpoint_sha256",
        "source_step",
        "training_seed",
        "training_run_id",
        "tracking_local_run_id",
        "tracking_run_id",
        "tracking_experiment_id",
        "prior",
        "data",
        "diagnostics",
    }
)
_REFERENCE_KEYS = frozenset(
    {"name", "sha256", "config_hash", "content_hash", "manifest_hash", "uri", "status", "recipe"}
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


def exact_keys(value: object, keys: set | frozenset, label: str) -> dict:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("artifact JSON contains duplicate keys")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    if path.stat().st_size > 8 << 20:
        raise ValueError("artifact metadata exceeds the 8 MiB format limit")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("artifact metadata is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("artifact metadata must be a JSON object")
    _validate_json(value)
    return value


def _validate_json(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact metadata keys must be strings")
            _validate_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("artifact metadata must contain only finite numbers")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("artifact metadata must contain only JSON values")


def validate_profile(profile: str | None) -> str | None:
    if profile is not None and (
        not isinstance(profile, str) or PROFILE_PATTERN.fullmatch(profile) is None
    ):
        raise ValueError("profile must be a simple 1-64 character alphanumeric slug")
    return profile


def _check_private_metadata(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_KEY.search(key) or key.lower() in {
                "observations",
                "training_data",
                "raw_data",
                "user_data",
                "environment",
            }:
                raise ValueError("provenance must not contain credentials or observation data")
            _check_private_metadata(item)
    elif isinstance(value, list):
        for item in value:
            _check_private_metadata(item)
    elif isinstance(value, str):
        if _SECRET_VALUE.search(value):
            raise ValueError("provenance must not contain credentials")
        if "://" in value:
            url = urlsplit(value)
            if (
                url.username
                or url.password
                or any(
                    _SECRET_KEY.search(key)
                    or key.lower() in {"auth", "sig", "signature"}
                    or key.lower().startswith(("x-amz-", "x-goog-"))
                    for key, _ in (*parse_qsl(url.query), *parse_qsl(url.fragment))
                )
            ):
                raise ValueError("provenance URLs must not contain credentials")


def provenance_metadata(value: Mapping | None) -> dict:
    """Accept only explicit metadata, never arbitrary run payloads or data arrays."""
    if value is None:
        return {"status": "not_recorded"}
    if not isinstance(value, Mapping) or value.keys() - _PROVENANCE_KEYS:
        raise ValueError("provenance contains unsupported fields; pass only artifact metadata")
    result = dict(value)
    result.setdefault("status", "recorded" if result else "not_recorded")
    _validate_json(result)
    _check_private_metadata(result)
    if result["status"] not in ("recorded", "not_recorded"):
        raise ValueError("provenance status must be recorded or not_recorded")
    for key in _PROVENANCE_KEYS - {"source_step", "training_seed", "prior", "data", "diagnostics"}:
        if key in result and (not isinstance(result[key], str) or not result[key].strip()):
            raise ValueError(f"provenance {key} must be a nonempty string")
    if "source_checkpoint_sha256" in result and not SHA256_PATTERN.fullmatch(
        result["source_checkpoint_sha256"]
    ):
        raise ValueError("source_checkpoint_sha256 must be a full lowercase SHA-256")
    for key in ("source_step", "training_seed"):
        if key in result and type(result[key]) is not int:
            raise ValueError(f"provenance {key} must be an integer")
    if "source_step" in result and result["source_step"] < 0:
        raise ValueError("provenance source_step must be nonnegative")
    if "prior" in result:
        prior = result["prior"]
        if (
            not isinstance(prior, dict)
            or not prior
            or prior.keys()
            - {
                "name",
                "config",
                "config_hash",
                "reconstructed",
                "source_revision",
                "status",
            }
        ):
            raise ValueError("prior must contain only prior identity and configuration metadata")
        for key in prior.keys() - {"config", "reconstructed"}:
            if not isinstance(prior[key], str) or not prior[key].strip():
                raise ValueError("prior identity metadata must be nonempty strings")
        if "config" in prior and not isinstance(prior["config"], dict):
            raise ValueError("prior config must be a JSON object, not observation arrays")
        if "config_hash" in prior and not SHA256_PATTERN.fullmatch(prior["config_hash"]):
            raise ValueError("prior config_hash must be a full lowercase SHA-256")
        if "reconstructed" in prior and type(prior["reconstructed"]) is not bool:
            raise ValueError("prior reconstructed flag must be boolean")
    for key in ("data", "diagnostics"):
        if key in result:
            if not isinstance(result[key], list):
                raise ValueError(f"provenance {key} must be a list of artifact references")
            for reference in result[key]:
                if (
                    not isinstance(reference, dict)
                    or not reference
                    or reference.keys() - _REFERENCE_KEYS
                ):
                    raise ValueError(f"provenance {key} contains an invalid artifact reference")
                if any(not isinstance(item, str) or not item for item in reference.values()):
                    raise ValueError("artifact reference values must be nonempty strings")
                for field in ("sha256", "config_hash", "content_hash", "manifest_hash"):
                    if field in reference and not SHA256_PATTERN.fullmatch(reference[field]):
                        raise ValueError("artifact reference hashes must be full lowercase SHA-256")
    if len(json_bytes(result)) > 1 << 20:
        raise ValueError("provenance exceeds the 1 MiB metadata limit")
    return result


def semantic_schema(config: ModelConfig) -> dict:
    """Ordered factories describe variable entity geometry without imposing a channel count."""
    feature_columns = {
        "response_moments": [
            *[f"partial_log_sales_correlation_decay_{decay:g}" for decay in RESPONSE_MOMENT_DECAYS],
            "tanh_standardized_log_sales_slope",
        ],
        "adstock_moments": [
            f"{metric}_decay_{decay:g}"
            for decay in ADSTOCK_MOMENT_DECAYS
            for metric in (
                "signed_log1p_mean_over_sales_scale",
                "tanh_partial_slope",
                "clipped_level",
            )
        ],
        "control_moments": [f"tanh_abs_t_stat_decay_{decay:g}" for decay in ADSTOCK_MOMENT_DECAYS],
        "control_location_scale": ["raw_mean", "normalization_scale"],
        "control_effect_moments": [
            f"{metric}_decay_{decay:g}"
            for decay in ADSTOCK_MOMENT_DECAYS
            for metric in ("tanh_standardized_slope", "clipped_level")
        ],
    }
    return {
        "version": 1,
        "geometry": {"channels": "input", "controls": "input", "demand": config.n_demand},
        "representation": {
            "cells": ["channels", "controls", "outcome"],
            "base_columns": [
                "value",
                "spend_mean",
                "spend_std",
                "is_query",
                "is_revenue",
                "week",
                "is_control",
            ],
            "feature_blocks": [
                {"name": name, "width": width, "columns": feature_columns[name]}
                for name, width in MOMENT_FEATURE_BLOCKS
            ],
            "readouts": ["weekly", "channels", "controls"],
            "attention": "causal_over_context_then_query",
            "query_outcome": "unobserved_zero_cell",
            "feature_source": "observed_context_only",
            "response_decays": list(RESPONSE_MOMENT_DECAYS),
            "response_max_lag": RESPONSE_MOMENT_L_MAX,
            "adstock_decays": list(ADSTOCK_MOMENT_DECAYS),
            "adstock_max_lag": ADSTOCK_MOMENT_L_MAX,
            "level_limit": ADSTOCK_LEVEL_LIMIT,
            "control_t_divisor": CONTROL_MOMENT_T_DIVISOR,
        },
        "normalization": {
            "spend": "context_zscore_log1p",
            "spend_std_floor": 1e-6,
            "spend_mean_affine": [9.5, 1.5],
            "spend_std_affine": [0.55, 0.25],
            "outcome": "context_centered_log_without_std_division",
            "controls": "context_zscore_raw",
            "control_std_threshold": 1e-3,
            "control_small_std_replacement": 1.0,
            "control_z_clip": [-10.0, 10.0],
            "week": "zero_based_index_divided_by_total_window_length",
            "target": "signed_natural_components_divided_by_context_p95_abs_outcome",
            "sales_scale_quantile": 0.95,
            "sales_scale_floor": 1e-6,
        },
        "targets": {
            "ordered_blocks": [
                "baseline_intrinsic",
                "contribution[channel]",
                "control_contribution[control]",
                *DECOMPOSITION_FIXED_TARGETS,
            ],
            "fixed_columns": list(DECOMPOSITION_FIXED_TARGETS),
            "latent_reduction": "sum",
            "noise_included": False,
            "support_gates": [
                {"component": gate.column, "edge_type": gate.family, "reduction": gate.reduction}
                for gate in ZERO_ATOM_GATES
            ],
        },
        "graph": {
            "families": list(STRUCT_EDGE_KEYS),
            "block_shapes": ["K", "J*K", "J*M", "J", "M", "M*K", "K*(K-1)", "M*(M-1)"],
            "within_block": "row_major_without_square_diagonals",
            "node_bank": ["channels", "controls", "demand", "baseline", "sales"],
            "generator_families": ["cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"],
            "db_zb_interpretation": "packed_dy_zy_outcome_edges_not_intrinsic_intercept_effects",
            "training_transform": "2*g-1",
            "sample_rule": "mean_over_window_strictly_greater_than_zero",
            "activity": "both_endpoints_active",
            "full_graph_mode_ties": "lexicographically_first",
        },
        "trajectory": {
            "feature_version": _TRAJECTORY_FEATURE_VERSION,
            "rank": _TRAJECTORY_RANK,
            "constant_loading_scale": _TRAJECTORY_CONSTANT_LOADING_SCALE,
            "context_query": "separately_sampled_marginals_not_paired_full_horizon_draws",
        },
    }


def inference_metadata(config: ModelConfig) -> dict:
    return {
        "head": "conditional",
        "mode": "model_average",
        "graph_threshold": 0.0,
        "flow_steps": config.flow_steps_sample,
        "ode_method": config.flow_ode_method,
        "trajectory_correlation": config.flow_trajectory_correlation,
        "component_units": "context_sales_scale",
        "natural_units": "outcome",
    }


def producer_metadata() -> dict:
    return {name: version(name) for name in ("pymc-mfm", "numpy", "torch", "safetensors")}
