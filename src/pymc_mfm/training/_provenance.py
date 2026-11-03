"""Content identities for the code and arrays actually consumed by a run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pymc_mfm.training.checkpoint import file_sha256


def environment_record(device: str) -> dict[str, Any]:
    packages = {}
    for name in (
        "pymc-mfm",
        "torch",
        "numpy",
        "mlflow",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pymc-generator",
        "safetensors",
        "huggingface-hub",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "device": device,
        "cuda_runtime": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.device(device))
        hardware.update(
            accelerator=properties.name,
            accelerator_memory_bytes=properties.total_memory,
            accelerator_capability=[properties.major, properties.minor],
        )
    return {
        "pymc_mfm": packages["pymc-mfm"],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": packages,
        "hardware": hardware,
    }


def code_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    files = {
        path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(root.rglob("*.py"))
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    record: dict[str, Any] = {
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_files": files,
        "revision": None,
        "modified_from_revision": None,
        "revision_status": "not_available",
    }
    # Only a source checkout owns its Git revision; an installed environment may
    # be nested under an unrelated application's repository.
    repository = root.parent.parent if root.parent.name == "src" else None
    if repository is not None and (repository / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            changed = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            record.update(
                revision=revision, modified_from_revision=bool(changed), revision_status="observed"
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return record


def corpus_identity(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Hash actual inputs without retaining arrays or asserting missing identity."""
    arrays = {}
    other = {}
    unavailable = {}
    for key, value in sorted(corpus.items()):
        try:
            if isinstance(value, (np.ndarray, torch.Tensor)):
                array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                if array.dtype.hasobject:
                    raise TypeError("object array has no canonical byte identity")
                contiguous = np.ascontiguousarray(array)
                digest = hashlib.sha256()
                digest.update(str(array.dtype).encode())
                digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
                digest.update(memoryview(contiguous).cast("B"))
                arrays[str(key)] = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "sha256": digest.hexdigest(),
                }
            else:
                if isinstance(value, np.generic):
                    value = value.item()
                # Do not use repr/default=str: NumPy representations truncate
                # large arrays and could falsely assert identical inputs.
                encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
                other[str(key)] = hashlib.sha256(encoded).hexdigest()
        except (TypeError, ValueError, RuntimeError) as exc:
            unavailable[str(key)] = {"status": "unavailable", "error_type": type(exc).__name__}
    encoded = json.dumps(
        {"arrays": arrays, "metadata": other}, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "content_sha256": None if unavailable else hashlib.sha256(encoded).hexdigest(),
        "arrays": arrays,
        "metadata_sha256": other,
        "unavailable": unavailable,
        "identity_scope": "actual in-memory inputs",
    }


def verified_prior_sha256(record: Mapping[str, Any]) -> str | None:
    """Hash of a corpus record's generator prior when its sidecar is bound to the corpus file.

    ``record`` is a run's corpus record, e.g. :func:`pymc_mfm.data.recipes.provenance`.
    The hash exists only when the record claims ``data_origin == "synthetic"``,
    its ``prior_sidecar`` is a dict with a dict ``prior``, and its ``sha256`` is a
    full lowercase SHA-256 equal to the sidecar's ``corpus_sha256``; otherwise
    the result is ``None``. It is the SHA-256 of the compact key-sorted JSON of
    ``{"prior": sidecar["prior"], "generator_version":
    sidecar.get("pymc_generator_version")}``, the value :func:`export_lineage`
    records under ``prior.config.corpus_prior_hashes``.
    """
    sidecar = record.get("prior_sidecar")
    source_digest = record.get("sha256")
    if not (
        record.get("data_origin") == "synthetic"
        and isinstance(sidecar, dict)
        and isinstance(sidecar.get("prior"), dict)
        and isinstance(source_digest, str)
        and re.fullmatch(r"[a-f0-9]{64}", source_digest)
        and sidecar.get("corpus_sha256") == source_digest
    ):
        return None
    prior_record = {
        "prior": sidecar["prior"],
        "generator_version": sidecar.get("pymc_generator_version"),
    }
    return hashlib.sha256(
        json.dumps(prior_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def export_lineage(run: Mapping[str, Any], checkpoint_sha256: str) -> dict[str, Any]:
    """Project safe, explicit data/prior/report identities into a model bundle."""
    data = []
    priors = {}
    for index, corpus in enumerate(run["corpus"]):
        actual = corpus.get("actual_inputs", {}).get("content_sha256")
        if actual:
            data.append(
                {
                    "name": f"corpus_{index}_actual_inputs",
                    "content_hash": actual,
                    "status": "recorded",
                    "recipe": "typed-array-and-metadata-content-sha256",
                }
            )
        else:
            data.append({"name": f"corpus_{index}_actual_inputs", "status": "identity_unavailable"})
        source_digest = corpus.get("sha256")
        valid_source_digest = isinstance(source_digest, str) and re.fullmatch(
            r"[a-f0-9]{64}", source_digest
        )
        if valid_source_digest:
            data.append(
                {
                    "name": f"corpus_{index}_source_file",
                    "sha256": corpus["sha256"],
                    "status": "caller_recorded_source_file",
                }
            )
        digest = verified_prior_sha256(corpus)
        if digest is not None:
            priors[f"corpus_{index}"] = digest
    prior = {"status": "not_recorded"}
    if priors:
        prior = {
            "name": "verified_generation_priors",
            "config": {"corpus_prior_hashes": priors},
            "config_hash": hashlib.sha256(
                json.dumps(priors, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "status": "recorded_hash_references"
            if len(priors) == len(run["corpus"])
            else "partial_hash_references",
        }
    diagnostics = [
        {key: record[key] for key in ("name", "sha256", "status", "recipe") if key in record}
        for record in run.get("diagnostics", [])
        if record["checkpoint_sha256"] == checkpoint_sha256
    ]
    return {
        "data": data,
        "prior": prior,
        "diagnostics": diagnostics
        or [{"name": "terminal_checkpoint_diagnostics", "status": "not_generated"}],
    }


def corpus_key(record: Mapping[str, Any]) -> str | None:
    """The identity that matches a corpus record across runs, or ``None`` when it has none.

    This is the content SHA-256 of the in-memory arrays when every array has one. Generator
    corpora hold object arrays, which have no canonical byte identity, so their
    ``content_sha256`` is ``None``; the ``.npz`` file SHA-256 (``sha256``) identifies them.
    """
    content = (record.get("actual_inputs") or {}).get("content_sha256")
    if content is not None:
        return content
    file_sha256 = record.get("sha256")
    return None if file_sha256 is None else f"file:{file_sha256}"
