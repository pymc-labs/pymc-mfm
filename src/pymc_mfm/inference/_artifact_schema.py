"""Versioned, non-executable metadata for the one canonical architecture."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from packaging.version import InvalidVersion, Version

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
    FLOW_STEPS_SAMPLE,
    FLOW_TRAJECTORY_CORRELATION,
    ZERO_ATOM_GATES,
)
from pymc_mfm.model.graph import STRUCT_EDGE_KEYS, EdgeSlotLayout

from ._digest import json_bytes as _json_bytes

FORMAT_VERSION = 1
SEMANTIC_SCHEMA_VERSION = 2
ARCHITECTURE = "mfm-graph-conditional-v1"
BUNDLE_FILES = ("config.json", "model.safetensors", "manifest.json")
#: Optional sidecar next to a bundle: synthetic inputs and their expected outputs.
RECORDED_OUTPUTS_FILE = "recorded_outputs.json"
COMPATIBILITY = {
    "python": ">=3.13",
    "pymc-mfm": ">=0.0.1.dev0",
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
        "training_evaluation",
    }
)
#: Frozen shape of recorded evaluation metric keys, which omit the leg. Only the
#: shape is checked on load; membership in the live evaluator key set is checked
#: where training code is available.
TRAINING_EVALUATION_KEY_PATTERN = re.compile(
    r"^(joint|conditional)/(query|context)/[a-z0-9_]+$"
    r"|^structure_head/[a-z]{2}/[a-z_]+$"
    r"|^joint_router/(edge/[a-z]{2}/[a-z_]+|graph/[a-z0-9_]+)$"
    r"|^counts/[a-z_]+$"
)
_TRAINING_EVALUATION_KEYS = frozenset(
    {
        "status",
        "source",
        "step",
        "leg",
        "head",
        "eval_seed",
        "n_worlds",
        "n_samples",
        "flow_steps",
        "worlds",
        "metrics",
        "aggregation",
    }
)
#: ``slot`` averages each world's slots, then worlds; ``slot_pooled`` pools every
#: world's slots, the convention of historically recorded evaluations.
_AGGREGATIONS = frozenset({"world", "slot", "slot_pooled", "pooled_edges"})
_RECORDED_OUTPUTS_KEYS = frozenset(
    {
        "format_version",
        "profile",
        "source_checkpoint_sha256",
        "state_sha256",
        "world",
        "observations",
        "graph",
        "n_samples",
        "cases",
        "producer",
    }
)
_RECORDED_OBSERVATION_KEYS = frozenset(
    {"spend", "controls", "context_outcome", "channel_mask", "control_mask", "latent_mask"}
)
#: One expected ``decompose`` output per mode and window.
_RECORDED_CASES = frozenset(
    f"{mode}/{window}"
    for mode in ("model_average", "map", "fixed")
    for window in ("query", "context")
)
_RECORDED_PRODUCER_VERSIONS = ("python", "pymc-mfm", "pymc-generator", "torch", "numpy")
#: Recorded seeds are int64 integers.
_INT64 = range(-(1 << 63), 1 << 63)
#: ``platform.system()`` and ``platform.machine()`` values of the platforms torch
#: supports; anything else, such as a host, user or node name, is rejected. Tuples,
#: so that membership of an unhashable JSON value is False rather than a TypeError.
_PLATFORM_SYSTEMS = ("Linux", "Darwin", "Windows")
_PLATFORM_MACHINES = ("x86_64", "AMD64", "aarch64", "arm64", "ARM64")
#: The only local version labels a producer may record, those of torch builds, as
#: packaging normalizes them: the CPU, CUDA, ROCm and XPU wheels, optionally with the
#: suffix of a CPU cxx11-ABI or AMD ROCm wheel (``cpu.cxx11.abi``,
#: ``rocm7.0.0.git56b79e13``, ``rocm7.0.0.lw.git2e48b21f``), and the commit of a
#: source or NVIDIA NGC build (``git5228986``, ``5228986c39.nv25.5``).
_TORCH_BUILD = re.compile(
    r"(cpu|cu\d+|rocm\d+(\.\d+)*|xpu)(\.cxx11\.abi|(\.lw)?\.git[0-9a-f]{7,40})?"
    r"|(git)?[0-9a-f]{7,40}(\.nv\d+\.\d+)?"
)
_REFERENCE_KEYS = frozenset(
    {"name", "sha256", "config_hash", "content_hash", "manifest_hash", "uri", "status", "recipe"}
)


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
    return parse_json(path.read_bytes())


def parse_json(data: bytes) -> dict:
    """Strict artifact JSON: one object, unique keys and finite numbers only."""
    if len(data) > 8 << 20:
        raise ValueError("artifact metadata exceeds the 8 MiB format limit")
    try:
        value = json.loads(data, object_pairs_hook=_pairs)
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
    for key in _PROVENANCE_KEYS - {
        "source_step",
        "training_seed",
        "prior",
        "data",
        "diagnostics",
        "training_evaluation",
    }:
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
    if "training_evaluation" in result:
        _validate_training_evaluation(result["training_evaluation"])
    if len(_json_bytes(result)) > 1 << 20:
        raise ValueError("provenance exceeds the 1 MiB metadata limit")
    return result


def _validate_training_evaluation(value: object) -> None:
    """Check the recorded evaluation's structure; values are never recomputed here.

    ``provenance_metadata`` has already rejected non-finite numbers.
    """
    record = exact_keys(value, _TRAINING_EVALUATION_KEYS, "provenance training_evaluation")
    if record["status"] != "recorded":
        raise ValueError("training_evaluation status must be recorded")
    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError("training_evaluation source must be a nonempty string")
    if record["leg"] not in ("holdout", "seen"):
        raise ValueError("training_evaluation leg must be holdout or seen")
    if record["head"] not in ("joint", "conditional"):
        raise ValueError("training_evaluation head must be joint or conditional")
    if type(record["eval_seed"]) is not int:
        raise ValueError("training_evaluation eval_seed must be an integer")
    for key, minimum in (("step", 0), ("n_worlds", 1), ("n_samples", 1), ("flow_steps", 1)):
        if type(record[key]) is not int or record[key] < minimum:
            raise ValueError(f"training_evaluation {key} must be an integer >= {minimum}")
    worlds = record["worlds"]
    if not isinstance(worlds, list) or len(worlds) != record["n_worlds"]:
        raise ValueError("training_evaluation worlds must list exactly n_worlds worlds")
    for world in worlds:
        exact_keys(world, {"shard_seed", "row"}, "training_evaluation world")
        if any(type(world[key]) is not int or world[key] < 0 for key in world):
            raise ValueError("training_evaluation worlds need nonnegative integer seeds and rows")
    metrics, aggregation = record["metrics"], record["aggregation"]
    if (
        not isinstance(metrics, dict)
        or not metrics
        or not isinstance(aggregation, dict)
        or metrics.keys() != aggregation.keys()
    ):
        raise ValueError("training_evaluation metrics and aggregation need equal nonempty keys")
    for key, metric in metrics.items():
        if TRAINING_EVALUATION_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"training_evaluation metric {key!r} is not an evaluator key")
        if metric is not None and type(metric) is not float:
            raise ValueError(f"training_evaluation metric {key!r} must be a float or null")
        if not isinstance(aggregation[key], str) or aggregation[key] not in _AGGREGATIONS:
            raise ValueError(f"training_evaluation aggregation of {key!r} is unknown")


def recorded_outputs_metadata(
    value: object,
    *,
    profile: str | None,
    provenance: Mapping,
    state_sha256: str,
    model_config: ModelConfig,
) -> dict:
    """Bind a recorded-output sidecar to the one exported artifact it describes.

    The sidecar holds one row of a ``SHARD_PLANS`` shard: its raw observations, a
    supplied graph and the expected ``decompose`` outputs per mode and window,
    recorded with the weights whose state digest is ``state_sha256``. It names the
    bundle profile and the original checkpoint, which export never rehashes. Every
    array must have the shape its observations imply, the observations need the
    bundle's control and latent widths (``model_config``) and an active channel and
    control, numbers must be finite float64 values and seeds int64 integers, and
    ``producer`` holds only release versions (see ``_release_version``), the
    platform system and machine, and the torch thread count, so credentials, paths,
    host or user names and other free text are rejected.
    """
    _validate_json(value)
    if any(_SECRET_VALUE.search(text) for text in _strings(value)):
        raise ValueError("recorded outputs must not contain credentials")
    record = exact_keys(value, _RECORDED_OUTPUTS_KEYS, "recorded outputs")
    if type(record["format_version"]) is not int or record["format_version"] != 1:
        raise ValueError("unsupported recorded outputs version")
    if record["profile"] != profile:
        raise ValueError("recorded outputs describe another profile")
    source = provenance.get("source_checkpoint_sha256")
    if source is None:
        raise ValueError("recorded outputs require a recorded source_checkpoint_sha256")
    if record["source_checkpoint_sha256"] != source:
        raise ValueError("recorded outputs describe another source checkpoint")
    if record["state_sha256"] != state_sha256:
        raise ValueError("recorded outputs were recorded with other weights")
    _recorded_world(record["world"])
    observations = exact_keys(
        record["observations"], _RECORDED_OBSERVATION_KEYS, "recorded observations"
    )
    steps, channels = _array(observations["spend"], 2, "spend")
    control_steps, controls = _array(observations["controls"], 2, "controls")
    (context,) = _array(observations["context_outcome"], 1, "context_outcome")
    (demand,) = _array(observations["latent_mask"], 1, "latent_mask", binary=True)
    # Every window's cases need nonempty arrays, so the context is a proper prefix.
    if (
        control_steps != steps
        or _array(observations["channel_mask"], 1, "channel_mask", binary=True) != (channels,)
        or _array(observations["control_mask"], 1, "control_mask", binary=True) != (controls,)
    ):
        raise ValueError("recorded observations disagree in length or entity counts")
    if controls != model_config.n_controls or demand != model_config.n_demand:
        raise ValueError("recorded observations need the artifact's control and latent widths")
    if 1 not in observations["channel_mask"] or 1 not in observations["control_mask"]:
        raise ValueError("recorded observations need an active channel and an active control")
    slots = EdgeSlotLayout(channels, controls, demand).n_slots
    if _array(record["graph"], 1, "graph", binary=True) != (slots,):
        raise ValueError("recorded graph needs one 0/1 entry per edge slot")
    n_samples = record["n_samples"]
    if type(n_samples) is not int:
        raise ValueError("recorded n_samples must be an integer")
    columns = 1 + channels + controls + len(DECOMPOSITION_FIXED_TARGETS)
    for name, case in exact_keys(record["cases"], _RECORDED_CASES, "recorded cases").items():
        exact_keys(case, {"seed", "sales_scale", "components_mean", "graphs"}, "recorded case")
        rows = context if name.endswith("/context") else steps - context
        if (
            not (type(case["seed"]) is int and case["seed"] in _INT64)
            or not (_float64(case["sales_scale"]) and case["sales_scale"] > 0)
            or _array(case["components_mean"], 2, "components_mean") != (rows, columns)
            or _array(case["graphs"], 2, "graphs", binary=True) != (n_samples, slots)
        ):
            raise ValueError(f"recorded case {name} does not match the recorded world")
    _recorded_producer(record["producer"])
    return record


def recorded_outputs_bytes(record: Mapping) -> bytes:
    """The canonical serialization of a validated recorded-output sidecar."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def _strings(value: object):
    """Every string of a JSON value, object keys included."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _float64(value: object) -> bool:
    """Whether a JSON number converts to a finite float64; huge integers do not."""
    try:
        return type(value) in (int, float) and math.isfinite(float(value))
    except OverflowError:
        return False


def _array(value: object, ndim: int, label: str, *, binary: bool = False) -> tuple[int, ...]:
    """Shape of a nonempty rectangular array of finite float64 numbers, or of 0/1 integers."""
    shape = []
    item = value
    for _ in range(ndim):
        if not isinstance(item, list) or not item:
            raise ValueError(f"recorded {label} must be a nonempty {ndim}-dimensional array")
        shape.append(len(item))
        item = item[0]
    items = [value]
    for size in shape:
        if any(not isinstance(row, list) or len(row) != size for row in items):
            raise ValueError(f"recorded {label} must be a rectangular array")
        items = [entry for row in items for entry in row]
    if not all(
        type(entry) is int and entry in (0, 1) if binary else _float64(entry) for entry in items
    ):
        kind = "0/1 integers" if binary else "finite float64 numbers"
        raise ValueError(f"recorded {label} must hold {kind}")
    return tuple(shape)


def _recorded_world(value: object) -> None:
    from pymc_mfm.data.recipes import SHARD_PLANS

    world = exact_keys(value, {"plan", "shard", "shard_seed", "row"}, "recorded world")
    plan = SHARD_PLANS.get(world["plan"]) if isinstance(world["plan"], str) else None
    # ``plan.seed`` raises ValueError unless the shard is an index of the plan.
    if (
        plan is None
        or any(type(world[key]) is not int for key in ("shard", "shard_seed", "row"))
        or world["shard_seed"] != plan.seed(world["shard"])
        or not 0 <= world["row"] < plan.n_cells * plan.draws_per_cell
    ):
        raise ValueError("recorded world must be a row of a SHARD_PLANS shard")


def _release_version(text: object, *, build: bool) -> bool:
    """Whether ``text`` is a release version: canonical PEP 440 text, no local label.

    Pre-, post- and development releases (``2.8.0a0``, ``0.0.1.dev0``) qualify as
    packaging parses them. Only a torch ``build`` may carry a local label, and only
    a ``_TORCH_BUILD`` label; ``normalized_version`` gives the canonical text.
    """
    try:
        parsed = Version(text) if isinstance(text, str) else None
    except InvalidVersion:
        return False
    if parsed is None or str(parsed) != text:
        return False
    return parsed.local is None or (build and _TORCH_BUILD.fullmatch(parsed.local) is not None)


def normalized_version(text: str) -> str:
    """``text`` as packaging normalizes it, e.g. ``nv25.05`` as ``nv25.5``.

    Text that is no PEP 440 version is returned unchanged, for the release-version
    rule to refuse.
    """
    try:
        return str(Version(text))
    except InvalidVersion:
        return text


def python_version() -> str:
    """The running Python as a PEP 440 version, from ``sys.version_info``.

    ``platform.python_version()`` is no PEP 440 version for a CPython built from an
    untagged commit (``3.13.9+``).
    """
    info = sys.version_info
    level = {"alpha": "a", "beta": "b", "candidate": "rc"}.get(info.releaselevel)
    release = f"{info.major}.{info.minor}.{info.micro}"
    return release if level is None else f"{release}{level}{info.serial}"


def _recorded_producer(value: object) -> None:
    producer = exact_keys(
        value,
        {*_RECORDED_PRODUCER_VERSIONS, "system", "machine", "torch_threads"},
        "recorded producer",
    )
    for name in _RECORDED_PRODUCER_VERSIONS:
        if not _release_version(producer[name], build=name == "torch"):
            raise ValueError(f"recorded producer {name} must be a release version")
    if (
        producer["system"] not in _PLATFORM_SYSTEMS
        or producer["machine"] not in _PLATFORM_MACHINES
        or not (type(producer["torch_threads"]) is int and producer["torch_threads"] >= 1)
    ):
        raise ValueError("recorded producer must name a supported platform and thread count")


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
                "signed_log1p_canonical_mean_over_canonical_sales_scale",
                "tanh_partial_slope",
                "clipped_level",
            )
        ],
        "control_moments": [f"tanh_abs_t_stat_decay_{decay:g}" for decay in ADSTOCK_MOMENT_DECAYS],
        "control_location_scale": ["canonical_mean", "canonical_normalization_scale"],
        "control_effect_moments": [
            f"{metric}_decay_{decay:g}"
            for decay in ADSTOCK_MOMENT_DECAYS
            for metric in ("tanh_standardized_slope", "clipped_level")
        ],
    }
    return {
        "version": SEMANTIC_SCHEMA_VERSION,
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
            "feature_source": "canonical_observed_context_only",
            "response_decays": list(RESPONSE_MOMENT_DECAYS),
            "response_max_lag": RESPONSE_MOMENT_L_MAX,
            "adstock_decays": list(ADSTOCK_MOMENT_DECAYS),
            "adstock_max_lag": ADSTOCK_MOMENT_L_MAX,
            "level_limit": ADSTOCK_LEVEL_LIMIT,
            "control_t_divisor": CONTROL_MOMENT_T_DIVISOR,
        },
        "canonicalization": {
            "fit": "context_only",
            "apply": "unchanged_scales_over_context_and_query",
            "precision": "float64_before_log1p_regression_zscore_clipping_and_float32",
            "reductions": "max_magnitude_factored_mean_and_rms",
            "spend": "divide_each_channel_by_context_mean",
            "controls": "divide_each_control_by_context_rms_without_centering",
            "outcome": "divide_positive_outcome_by_context_p95_absolute_outcome",
            "zero_context": "all_zero_column_stays_zero_nonzero_query_rejected",
            "constant_nonzero_controls": "valid_sign_and_origin_preserved",
            "raw_unit_features": False,
        },
        "normalization": {
            "spend": "context_zscore_log1p_canonical_spend",
            "spend_std_floor": 1e-6,
            "spend_mean": "context_mean_log1p_canonical_spend",
            "spend_std": "context_std_log1p_canonical_spend",
            "outcome": "context_centered_log_canonical_outcome_without_std_division",
            "controls": "context_zscore_canonical_controls",
            "control_std_threshold": 1e-3,
            "control_small_std_replacement": 1.0,
            "control_z_clip": [-10.0, 10.0],
            "week": "zero_based_index_divided_by_total_window_length",
            "target": "signed_natural_components_divided_by_original_context_p95_abs_outcome",
            "sales_scale_quantile": 0.95,
            "sales_scale": {
                "source": "original_positive_context_outcome",
                "floor": None,
                "dtype": "float64",
                "device": "cpu",
                "network_input": False,
                "use": "target_normalization_and_original_unit_output_restoration",
            },
            "model_tensors": "float32_at_batch_boundary_then_model_dtype_on_model_device",
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


def validate_semantic_schema(
    value: object, config: ModelConfig | None = None, *, label: str
) -> None:
    """Check the representation version, then its full contract when geometry is known."""
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != SEMANTIC_SCHEMA_VERSION
        or (config is not None and _json_bytes(value) != _json_bytes(semantic_schema(config)))
    ):
        raise ValueError(
            f"{label} has an incompatible input representation; expected semantic schema "
            f"version {SEMANTIC_SCHEMA_VERSION} with context-only unit-invariant scaling. "
            "Retrain with the current preprocessing; dimensional A/B weights cannot be "
            "converted by relabeling their schema."
        )


def inference_metadata() -> dict:
    return {
        "head": "conditional",
        "mode": "model_average",
        "graph_threshold": 0.0,
        "flow_steps": FLOW_STEPS_SAMPLE,
        "ode_method": "euler",
        "trajectory_correlation": FLOW_TRAJECTORY_CORRELATION,
        "component_units": "context_sales_scale",
        "natural_units": "outcome",
    }
