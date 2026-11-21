"""Strict local safetensors bundles; transport and training resume are separate."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import torch
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from safetensors import safe_open
from safetensors.torch import save_file

from pymc_mktfm.model.config import ModelConfig
from pymc_mktfm.model.model import MFM

from ._artifact_schema import (
    ARCHITECTURE,
    BUNDLE_FILES,
    COMPATIBILITY,
    FORMAT_VERSION,
    SHA256_PATTERN,
    _release_version,
    exact_keys,
    file_record,
    inference_metadata,
    normalized_version,
    provenance_metadata,
    python_version,
    read_json,
    semantic_schema,
    validate_profile,
    validate_semantic_schema,
)
from ._digest import digest_bytes, json_bytes

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
_PRODUCER_KEYS = frozenset({"pymc-mktfm", "numpy", "torch", "safetensors"})
_COMPATIBILITY_KEYS = _PRODUCER_KEYS | {"python"}
#: Recorded compatibility ranges hold only these bounds; loading enforces the lower ones.
_RANGE_OPERATORS = frozenset({">=", ">", "<", "<="})
_LOWER_BOUNDS = frozenset({">=", ">"})


def _canonical_model(config: ModelConfig) -> MFM:
    # Built on the CPU without advancing PyTorch's generator; restoration replaces its tensors
    # (assign=True). On the meta device, initialization ran PyTorch's Python reference
    # kernels, whose first call imports torch._dynamo (and with it sympy, triton and
    # torch.distributed): 0.4 s on an Apple M3 Max and 2-3 s on Linux hosts, against 8 ms to
    # initialize River-1's 2.4 M parameters.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
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


def _state_sha256(state: Mapping[str, torch.Tensor], catalog: list[dict]) -> str:
    """Digest of the weights: catalog-ordered name, dtype, shape and tensor bytes.

    Unlike a safetensors file hash, it does not depend on how the file was written.
    """
    digest = hashlib.sha256()
    for entry in catalog:
        tensor = state[entry["name"]].detach().cpu().contiguous()
        digest.update(json_bytes([entry["name"], str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


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
        for attribute in ("config", "n_controls", "n_demand", "graph_router"):
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


def _check_environment() -> dict[str, str]:
    """Installed package versions, refused outside this release's ``COMPATIBILITY``.

    The installed release decides what it supports; a later release can relax it.
    """
    installed = {
        name: python_version() if name == "python" else version(name)
        for name in COMPATIBILITY.keys() | _COMPATIBILITY_KEYS
    }
    for name, supported in COMPATIBILITY.items():
        if not SpecifierSet(supported).contains(installed[name], prereleases=True):
            raise ValueError(
                f"installed {name} {installed[name]} is outside this release's "
                f"supported range {supported}"
            )
    return installed


def _check_compatibility(
    compatibility: object, producer: object, installed: Mapping[str, str]
) -> None:
    """Refuse an artifact whose recorded environment the ``installed`` one cannot serve.

    ``installed`` has passed ``_check_environment``: this release's ``COMPATIBILITY``,
    which a later release can relax. Installed versions must also satisfy every
    lower bound the artifact records. The recorded upper bounds state the producing
    release's tested range and are not enforced, so no published artifact becomes
    permanently unloadable. The recorded ranges use only ``>=``, ``>``, ``<`` and
    ``<=`` bounds, and the producer versions are release versions inside them
    (``_artifact_schema._release_version``: torch alone may carry a build label).
    """
    producer = exact_keys(producer, _PRODUCER_KEYS, "artifact producer")
    compatibility = exact_keys(compatibility, _COMPATIBILITY_KEYS, "artifact compatibility")
    for name, requirement in compatibility.items():
        try:
            recorded = SpecifierSet(requirement) if isinstance(requirement, str) else None
        except InvalidSpecifier:
            recorded = None
        if not recorded or any(bound.operator not in _RANGE_OPERATORS for bound in recorded):
            raise ValueError(
                f"artifact compatibility {name} must be a nonempty range of >=, >, <, <= bounds"
            )
        if any(
            bound.operator in _LOWER_BOUNDS
            and not bound.contains(installed[name], prereleases=True)
            for bound in recorded
        ):
            raise ValueError(
                f"installed {name} {installed[name]} is below the artifact's minimum {requirement}"
            )
        if name in producer and not (
            _release_version(producer[name], build=name == "torch")
            and recorded.contains(producer[name], prereleases=True)
        ):
            raise ValueError(
                f"artifact producer {name} {producer[name]!r} must be a release version "
                "within its compatibility"
            )


def _export_environment() -> tuple[dict[str, str], dict[str, str]]:
    """The compatibility and producer versions that an export here records.

    The installed versions must satisfy this release's ``COMPATIBILITY`` and, as
    packaging normalizes them, be release versions (``_release_version``: torch alone
    may carry a build label). Loading the export here then passes
    ``_check_compatibility``: it records ``COMPATIBILITY`` and producer versions
    inside it. ``save_pretrained`` runs it, and ``train`` runs it before training, so
    its terminal export is never refused for the environment.
    """
    installed = _check_environment()
    producer = {name: normalized_version(installed[name]) for name in sorted(_PRODUCER_KEYS)}
    for name, number in producer.items():
        if not _release_version(number, build=name == "torch"):
            label = "a build label" if name == "torch" else "a local label"
            raise ValueError(
                f"installed {name} {installed[name]} carries {label} this release cannot record"
            )
    return dict(COMPATIBILITY), producer


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
    Annotations on a loaded model must preserve its existing source/run identity.
    Loading keeps the weights' digest in ``model.artifact_metadata`` (and shows it
    as ``model.provenance["state_sha256"]``); once the weights differ from it,
    e.g. after a dtype cast or an edit, the model is another artifact. It then
    inherits neither the loaded provenance nor the loaded profile, so
    ``provenance`` must be passed explicitly (``{}`` records none) and only an
    explicit ``profile`` labels it. The installed environment must satisfy this
    release's ``COMPATIBILITY`` and its package versions must be release versions,
    torch's with at most a build label (see ``_export_environment``), so that the
    export loads where it was written. Stop concurrent model updates before
    exporting.
    """
    if type(model) is not MFM:
        raise ValueError("save_pretrained accepts only the canonical MKTFM architecture")
    compatibility, producer = _export_environment()
    config = ModelConfig.from_dict(model.config.to_dict())
    previous = getattr(model, "artifact_metadata", {})
    if previous:
        validate_semantic_schema(previous.get("schemas"), config, label="loaded artifact")
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
    loaded_state = previous.get("state_sha256")
    if loaded_state is not None and _state_sha256(state, catalog) != loaded_state:
        # The loaded source, run and evaluation records and the profile label
        # describe the loaded weights, not these.
        if provenance is None and previous["provenance"].get("status") == "recorded":
            raise ValueError(
                "the weights changed since loading, so the loaded provenance no longer "
                "describes them; pass provenance explicitly ({} records none)"
            )
        previous = {}
    elif provenance is not None:
        previous_provenance = previous.get("provenance", {})
        supplied = provenance_metadata(provenance)
        for key in provenance:
            if (
                key != "status"
                and key in previous_provenance
                and supplied[key] != previous_provenance[key]
            ):
                raise ValueError(f"provenance {key!r} conflicts with the loaded artifact")
        if previous:
            # Without a recorded source, the loaded checkpoint is the exported file.
            source = previous_provenance.get(
                "source_checkpoint_sha256",
                previous["manifest"]["files"]["model.safetensors"]["sha256"],
            )
            if supplied.get("source_checkpoint_sha256", source) != source:
                raise ValueError("source_checkpoint_sha256 conflicts with the loaded checkpoint")
        provenance = {**previous_provenance, **supplied} if provenance else previous_provenance
    else:
        provenance = previous.get("provenance") or None
    if profile is None:
        profile = previous.get("profile")
    elif previous.get("profile") is not None and profile != previous["profile"]:
        raise ValueError("profile conflicts with the loaded artifact")
    profile = validate_profile(profile)
    metadata = {
        "format_version": FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "model_config": config.to_dict(),
        "schemas": semantic_schema(config),
        "inference": inference_metadata(),
        "compatibility": compatibility,
        "producer": producer,
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
        raise FileNotFoundError("expected a safe MKTFM bundle directory, not a checkpoint file")
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
    validate_semantic_schema(metadata["schemas"], label="artifact schemas")
    config = ModelConfig.from_dict(metadata["model_config"])
    # Geometry alone cannot distinguish dimensional weights from canonical ones.
    validate_semantic_schema(metadata["schemas"], config, label="artifact schemas")
    if json_bytes(metadata["inference"]) != json_bytes(inference_metadata()):
        raise ValueError("artifact inference does not match the canonical contract")
    if metadata["training_resume"] is not False:
        raise ValueError("model bundles do not contain training-resume state")
    if validate_profile(metadata["profile"]) != manifest["profile"]:
        raise ValueError("artifact profile metadata disagree")
    if json_bytes(provenance_metadata(metadata["provenance"])) != json_bytes(
        metadata["provenance"]
    ):
        raise ValueError("artifact provenance is incomplete")
    _check_compatibility(metadata["compatibility"], metadata["producer"], _check_environment())
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as stream:
        if stream.metadata() != {
            "format": "pt",
            "architecture": ARCHITECTURE,
            "config_sha256": manifest["files"]["config.json"]["sha256"],
        }:
            raise ValueError("safetensors metadata disagree with the configuration identity")
        # assign=True keeps these tensors as the weights. get_tensor views the file
        # buffer at its byte offsets, not at the allocator's 64-byte alignment, and
        # x86 BLAS kernels round unaligned operands differently: a bundle would not
        # reproduce the saved model's outputs bit for bit. Own fresh allocations.
        state = {name: stream.get_tensor(name).clone() for name in stream.keys()}
    model = _canonical_model(config)
    catalog = _validate_state(model, state, specs=manifest["tensors"])
    model.load_state_dict(state, strict=True, assign=True)
    parameters = dict(model.named_parameters())
    for entry in catalog:
        if entry["kind"] == "parameter":
            parameters[entry["name"]].requires_grad_(entry["requires_grad"])
    state_sha256 = _state_sha256(state, catalog)
    # Beside the provenance an export inherits, where editing model.provenance
    # cannot hide that the weights have changed since loading.
    model.artifact_metadata = {**metadata, "manifest": manifest, "state_sha256": state_sha256}
    exported = manifest["files"]["model.safetensors"]["sha256"]
    model.provenance = {
        **metadata["provenance"],
        "profile": metadata["profile"],
        "checkpoint_sha256": metadata["provenance"].get("source_checkpoint_sha256", exported),
        "exported_checkpoint_sha256": exported,
        "bundle_sha256": manifest["bundle_sha256"],
        "state_sha256": state_sha256,
    }
    return model.to(device).eval()
