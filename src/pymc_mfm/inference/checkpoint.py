"""Strict local safetensors bundles; transport and training resume are separate."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import torch
from packaging.specifiers import SpecifierSet
from safetensors import safe_open
from safetensors.torch import save_file

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM

from ._artifact_schema import (
    ARCHITECTURE,
    BUNDLE_FILES,
    COMPATIBILITY,
    FORMAT_VERSION,
    SHA256_PATTERN,
    digest_bytes,
    exact_keys,
    file_record,
    inference_metadata,
    json_bytes,
    producer_metadata,
    provenance_metadata,
    read_json,
    semantic_schema,
    validate_profile,
)

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_CONFIG_KEYS = frozenset(
    {
        "format_version",
        "architecture",
        "model_config",
        "schemas",
        "inference",
        "compatibility",
        "producer",
        "provenance",
        "profile",
        "training_resume",
    }
)
_MANIFEST_KEYS = frozenset(
    {"format_version", "architecture", "profile", "files", "tensors", "bundle_sha256"}
)


def _canonical_model(config: ModelConfig) -> MFM:
    # No random initialization, allocation, or RNG advancement on restoration.
    with torch.device("meta"):
        return MFM(config)


def _validate_state(
    reference: MFM, state: Mapping[str, torch.Tensor], *, specs: list | None = None
) -> list[dict]:
    expected = reference.state_dict()
    if state.keys() != expected.keys():
        raise ValueError(
            "artifact tensor keys differ from the complete architecture: "
            f"missing={sorted(expected.keys() - state.keys())}, "
            f"unexpected={sorted(state.keys() - expected.keys())}"
        )
    parameters = dict(reference.named_parameters())
    catalog = []
    for name, template in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            raise ValueError(f"artifact tensor {name} has no materialized tensor data")
        if tensor.layout != torch.strided or tensor.shape != template.shape:
            raise ValueError(f"artifact tensor {name} has an incompatible layout or shape")
        if template.is_floating_point():
            if tensor.dtype not in _FLOAT_DTYPES or not torch.isfinite(tensor).all().item():
                raise ValueError(f"artifact tensor {name} requires finite supported floating data")
        elif tensor.dtype != template.dtype:
            raise ValueError(f"artifact tensor {name} has an incompatible dtype")
        entry = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "kind": "parameter" if name in parameters else "buffer",
        }
        if name in parameters:
            entry["requires_grad"] = parameters[name].requires_grad
        catalog.append(entry)
    if specs is not None:
        if not isinstance(specs, list) or len(specs) != len(catalog):
            raise ValueError("artifact tensor catalog is missing or incomplete")
        for actual, declared in zip(catalog, specs, strict=True):
            if isinstance(declared, dict) and "requires_grad" in actual:
                if type(declared.get("requires_grad")) is not bool:
                    raise ValueError("artifact parameter requires_grad must be boolean")
                actual["requires_grad"] = declared["requires_grad"]
            if json_bytes(actual) != json_bytes(declared):
                raise ValueError(f"artifact tensor catalog disagrees with tensor {actual['name']}")
    return catalog


def _validate_runtime(model: MFM, reference: MFM) -> None:
    """Fail rather than silently serialize a modified, unconstructable module tree."""
    actual_modules = dict(model.named_modules())
    expected_modules = dict(reference.named_modules())
    if actual_modules.keys() != expected_modules.keys() or any(
        type(actual_modules[name]) is not type(module) for name, module in expected_modules.items()
    ):
        raise ValueError("save_pretrained requires the complete canonical module tree")
    if dict(model.named_buffers()).keys() != dict(reference.named_buffers()).keys():
        raise ValueError("save_pretrained cannot discard unrecognized buffers")
    for module_name in ("encoder", "head", "conditional_head"):
        actual = actual_modules[module_name]
        expected = expected_modules[module_name]
        for attribute in (
            "config",
            "ode_method",
            "n_steps_sample",
            "mc_samples",
            "n_controls",
            "n_demand",
            "trajectory_correlation",
            "n_freq",
            "graph_router",
        ):
            if hasattr(expected, attribute) and getattr(actual, attribute) != getattr(
                expected, attribute
            ):
                raise ValueError(f"{module_name}.{attribute} disagrees with model.config")
    actual_parameters = dict(model.named_parameters(remove_duplicate=False))
    expected_parameters = dict(reference.named_parameters(remove_duplicate=False))
    if actual_parameters.keys() != expected_parameters.keys():
        raise ValueError("save_pretrained requires all canonical parameters")
    if len({id(value) for value in actual_parameters.values()}) != len(actual_parameters):
        raise ValueError("save_pretrained cannot change canonical parameter sharing")


def save_pretrained(
    model: MFM,
    path: str | Path,
    *,
    provenance: Mapping | None = None,
    profile: str | None = None,
) -> Path:
    """Export one complete, immutable, offline model bundle into a new directory.

    The three files are ``config.json``, ``model.safetensors``, and
    ``manifest.json``. All parameters and persistent buffers are retained,
    including auxiliary training modules, numerical dtypes and trainability.
    Optimizer/RNG state is not included: this is not a training-resume archive.

    ``profile`` labels an artifact, not an architecture. For a collection use
    ``root / "profiles" / profile`` as ``path`` for each explicitly named export.
    A nonempty destination is never overwritten (including Hub snapshots).
    Only whitelisted, caller-approved metadata is accepted as provenance.
    Stop concurrent model updates before exporting.
    """
    if type(model) is not MFM:
        raise ValueError("save_pretrained accepts only the canonical MFM architecture")
    config = ModelConfig.from_dict(model.config.to_dict())
    reference = _canonical_model(config)
    _validate_runtime(model, reference)
    state = model.state_dict()
    catalog = _validate_state(reference, state)
    parameters = dict(model.named_parameters())
    for entry in catalog:
        if entry["kind"] == "parameter":
            entry["requires_grad"] = parameters[entry["name"]].requires_grad
    # Exercise custom semantic-buffer checks without copying into initialized weights.
    reference.load_state_dict(state, strict=True, assign=True)
    previous = getattr(model, "artifact_metadata", {})
    if provenance is None:
        provenance = previous.get("provenance")
    if profile is None:
        profile = previous.get("profile")
    profile = validate_profile(profile)
    metadata = {
        "format_version": FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "model_config": config.to_dict(),
        "schemas": semantic_schema(config),
        "inference": inference_metadata(config),
        "compatibility": dict(COMPATIBILITY),
        "producer": producer_metadata(),
        "provenance": provenance_metadata(provenance),
        "profile": profile,
        "training_resume": False,
    }
    path = Path(path).expanduser()
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError("save_pretrained requires a new or empty export directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        config_bytes = json_bytes(metadata)
        (staging / "config.json").write_bytes(config_bytes)
        tensors = {name: value.detach().cpu().contiguous() for name, value in state.items()}
        save_file(
            tensors,
            str(staging / "model.safetensors"),
            metadata={
                "format": "pt",
                "architecture": ARCHITECTURE,
                "config_sha256": digest_bytes(config_bytes),
            },
        )
        manifest = {
            "format_version": FORMAT_VERSION,
            "architecture": ARCHITECTURE,
            "profile": profile,
            "files": {name: file_record(staging / name) for name in BUNDLE_FILES[:2]},
            "tensors": catalog,
        }
        manifest["bundle_sha256"] = digest_bytes(json_bytes(manifest))
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        os.replace(staging, path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return path


def _read_manifest(path: Path) -> dict:
    manifest = exact_keys(read_json(path), _MANIFEST_KEYS, "artifact manifest")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported artifact manifest version")
    if manifest["architecture"] != ARCHITECTURE:
        raise ValueError("unsupported artifact architecture")
    validate_profile(manifest["profile"])
    identity = manifest["bundle_sha256"]
    if not isinstance(identity, str) or not SHA256_PATTERN.fullmatch(identity):
        raise ValueError("artifact bundle_sha256 must be a full lowercase SHA-256")
    if identity != digest_bytes(
        json_bytes({key: value for key, value in manifest.items() if key != "bundle_sha256"})
    ):
        raise ValueError("artifact manifest integrity check failed")
    files = exact_keys(manifest["files"], set(BUNDLE_FILES[:2]), "artifact files")
    for record in files.values():
        exact_keys(record, {"sha256", "size"}, "artifact file record")
        if (
            not isinstance(record["sha256"], str)
            or not SHA256_PATTERN.fullmatch(record["sha256"])
            or type(record["size"]) is not int
            or record["size"] < 1
        ):
            raise ValueError("invalid artifact file identity")
    return manifest


def _load_local(path: str | Path, *, device: str | torch.device = "cpu") -> MFM:
    """The sole reconstruction path, also used after Hub snapshot retrieval."""
    path = Path(path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError("expected a safe MFM bundle directory, not a checkpoint file")
    if any(not (path / name).is_file() for name in BUNDLE_FILES):
        raise FileNotFoundError(f"model bundle requires all of {BUNDLE_FILES}")
    manifest = _read_manifest(path / "manifest.json")
    for name, record in manifest["files"].items():
        if file_record(path / name) != record:
            raise ValueError(f"artifact integrity check failed for {name}")
    metadata = exact_keys(read_json(path / "config.json"), _CONFIG_KEYS, "artifact configuration")
    if (
        type(metadata["format_version"]) is not int
        or metadata["format_version"] != FORMAT_VERSION
        or metadata["architecture"] != ARCHITECTURE
    ):
        raise ValueError("unsupported artifact configuration or architecture")
    config = ModelConfig.from_dict(metadata["model_config"])
    for key, expected in (
        ("schemas", semantic_schema(config)),
        ("inference", inference_metadata(config)),
        ("compatibility", COMPATIBILITY),
    ):
        if json_bytes(metadata[key]) != json_bytes(expected):
            raise ValueError(f"artifact {key} do not match the canonical contract")
    if metadata["training_resume"] is not False:
        raise ValueError("model bundles do not contain training-resume state")
    if validate_profile(metadata["profile"]) != manifest["profile"]:
        raise ValueError("artifact profile metadata disagree")
    if json_bytes(provenance_metadata(metadata["provenance"])) != json_bytes(
        metadata["provenance"]
    ):
        raise ValueError("artifact provenance is incomplete")
    producer = exact_keys(
        metadata["producer"], {"pymc-mfm", "numpy", "torch", "safetensors"}, "artifact producer"
    )
    for name, requirement in COMPATIBILITY.items():
        installed = platform.python_version() if name == "python" else version(name)
        if not SpecifierSet(requirement).contains(installed, prereleases=True):
            raise ValueError(
                f"installed {name} does not satisfy artifact compatibility {requirement}"
            )
        if name != "python" and (
            not isinstance(producer[name], str)
            or not SpecifierSet(requirement).contains(producer[name], prereleases=True)
        ):
            raise ValueError(
                f"artifact producer {name} does not satisfy its compatibility contract"
            )
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as stream:
        if stream.metadata() != {
            "format": "pt",
            "architecture": ARCHITECTURE,
            "config_sha256": manifest["files"]["config.json"]["sha256"],
        }:
            raise ValueError("safetensors metadata disagree with the configuration identity")
        state = {name: stream.get_tensor(name) for name in stream.keys()}
    model = _canonical_model(config)
    catalog = _validate_state(model, state, specs=manifest["tensors"])
    model.load_state_dict(state, strict=True, assign=True)
    parameters = dict(model.named_parameters())
    for entry in catalog:
        if entry["kind"] == "parameter":
            parameters[entry["name"]].requires_grad_(entry["requires_grad"])
    model.artifact_metadata = {**metadata, "manifest": manifest}
    exported = manifest["files"]["model.safetensors"]["sha256"]
    model.provenance = {
        **metadata["provenance"],
        "profile": metadata["profile"],
        "checkpoint_sha256": metadata["provenance"].get("source_checkpoint_sha256", exported),
        "exported_checkpoint_sha256": exported,
        "bundle_sha256": manifest["bundle_sha256"],
    }
    return model.to(device).eval()
