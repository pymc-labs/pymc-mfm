"""Optional Hub transport over the same strict local MFM bundle contract."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from pymc_mfm.model.model import MFM

from ._artifact_schema import BUNDLE_FILES, json_bytes, validate_profile
from .checkpoint import _load_local, _read_manifest, save_pretrained

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_LICENSE = re.compile(r"[a-z0-9][a-z0-9.+-]{0,79}\Z")


class HubArtifactError(RuntimeError):
    """A credential-free transport error with a machine-readable failure code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError("Hub operations require the pymc-mfm[hub] extra") from None
    return huggingface_hub


def _repo_id(value: str) -> str:
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(value)
    except (ValueError, TypeError):
        raise ValueError(
            "expected a valid Hub model repository ID, not a URL or credential"
        ) from None
    return value


def _transport(call, **kwargs):
    """Never persist or surface backend exception text, which can include auth headers."""
    try:
        return call(**kwargs)
    except Exception as exc:
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        if isinstance(exc, LocalEntryNotFoundError):
            code, message = (
                "cache_miss",
                "The requested artifact is not available in the local Hub cache.",
            )
        elif isinstance(exc, RevisionNotFoundError):
            code, message = (
                "revision_not_found",
                "The requested Hub revision does not exist or is inaccessible.",
            )
        elif isinstance(exc, (GatedRepoError, RepositoryNotFoundError)) or getattr(
            getattr(exc, "response", None), "status_code", None
        ) in (401, 403):
            code, message = (
                "access_denied_or_missing",
                (
                    "The Hub model is missing, private, or gated; check normal HF "
                    "authentication and access."
                ),
            )
        else:
            code, message = (
                "transport_error",
                (
                    "The Hub artifact operation failed; check connectivity, access, "
                    "and the requested revision."
                ),
            )
        raise HubArtifactError(code, message) from None


def _snapshot(
    repo_id: str, *, revision, patterns, cache_dir, local_files_only, token
) -> tuple[Path, str]:
    hub = _hub()
    path = Path(
        _transport(
            hub.snapshot_download,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            allow_patterns=patterns,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    )
    # Official snapshot_download returns <cache>/snapshots/<full commit>, including offline.
    commit = path.name
    if path.parent.name != "snapshots" or not _COMMIT.fullmatch(commit):
        raise HubArtifactError(
            "invalid_snapshot", "Hub did not return an immutable cache snapshot."
        )
    if isinstance(revision, str) and _COMMIT.fullmatch(revision) and commit != revision:
        raise HubArtifactError(
            "revision_mismatch", "Hub returned a different commit than requested."
        )
    return path, commit


def _local_reference(value: str | Path) -> Path | None:
    path = Path(value).expanduser()
    if (
        isinstance(value, Path)
        or path.exists()
        or path.is_absolute()
        or str(value).startswith((".", "~"))
    ):
        return path
    return None


def _select_bundle(root: Path, profile: str | None) -> Path:
    if profile is not None and (root / "profiles" / profile / "config.json").is_file():
        return root / "profiles" / profile
    if (root / "config.json").is_file():
        return root
    if profile is not None:
        return root / "profiles" / profile
    raise FileNotFoundError("no root bundle; use list_profiles and select an explicit profile")


def from_pretrained(
    path_or_repo: str | Path,
    *,
    revision: str | None = None,
    profile: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    device: str | torch.device = "cpu",
) -> MFM:
    """Load a local directory or one profile from an immutable Hub snapshot.

    A direct bundle and a collection under ``profiles/<name>/`` both work with
    standard ``hf download`` output. Local directories never import the Hub SDK
    or perform authentication/network I/O. Remote branches/tags are resolved
    once; all bundle files then come from that exact commit. Use a full commit
    in release instructions. There is no built-in mutable release repository.

    Cache location, offline mode and tokens follow huggingface_hub conventions;
    ``token=False`` disables implicit authentication. Remote ``profile`` selects
    ``profiles/<name>/``; omit it for a root bundle. Only the selected
    configuration and its three bundle files are downloaded.
    """
    profile = validate_profile(profile)
    local = _local_reference(path_or_repo)
    commit = None
    if local is not None:
        if revision is not None:
            raise ValueError("revision applies to Hub references, not local directories")
        bundle = _select_bundle(local, profile)
    else:
        _hub()
        repo_id = _repo_id(path_or_repo)
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a nonempty branch, tag, or full commit")
        prefix = "" if profile is None else f"profiles/{profile}/"
        configs = [f"{prefix}config.json"]
        root, commit = _snapshot(
            repo_id,
            revision=revision,
            patterns=configs,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        bundle = root / prefix
        patterns = [f"{prefix}{name}" for name in BUNDLE_FILES]
        pinned_root, pinned_commit = _snapshot(
            repo_id,
            revision=commit,
            patterns=patterns,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        if pinned_commit != commit or pinned_root != root:
            raise HubArtifactError(
                "revision_mismatch", "The artifact files do not share one immutable snapshot."
            )
    model = _load_local(bundle, device=device)
    if profile is not None and model.artifact_metadata["profile"] != profile:
        raise ValueError("requested profile disagrees with bundle metadata")
    if commit is not None:
        model.artifact_metadata["hub"] = {"repo_id": repo_id, "revision": commit}
        model.provenance.update({"hub_repo_id": repo_id, "hub_revision": commit})
    return model


def list_profiles(
    path_or_repo: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> dict[str, dict]:
    """Discover bundle identities without downloading weights, reports, or corpora.

    Keys are collection profile slugs; ``'.'`` denotes a direct root bundle
    (load it without a profile argument). Each entry records its profile label,
    relative directory, bundle digest and, for remote discovery, resolved commit.
    Reuse that commit in a subsequent from_pretrained call to pin discovery.
    """
    root = _local_reference(path_or_repo)
    commit = None
    if root is None:
        _hub()
        repo_id = _repo_id(path_or_repo)
        root, commit = _snapshot(
            repo_id,
            revision=revision,
            patterns=["manifest.json", "profiles/*/manifest.json"],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    elif revision is not None:
        raise ValueError("revision applies to Hub references, not local directories")
    paths = []
    if (root / "manifest.json").is_file():
        paths.append(root / "manifest.json")
    if (root / "profiles").is_dir():
        paths.extend(sorted((root / "profiles").glob("*/manifest.json")))
    profiles = {}
    for path in paths:
        manifest = _read_manifest(path)
        relative = path.parent.relative_to(root).as_posix()
        name = "." if relative == "." else validate_profile(path.parent.name)
        if name != "." and manifest["profile"] != name:
            raise ValueError("profile directory and manifest label disagree")
        profiles[name] = {
            "profile": manifest["profile"],
            "path": relative,
            "bundle_sha256": manifest["bundle_sha256"],
            "revision": commit,
        }
    if not profiles:
        raise FileNotFoundError("no MFM bundles were found in the selected directory or snapshot")
    return profiles


def _model_card(metadata: dict, *, repo_id: str, profile: str | None, license: str, hub) -> str:
    card_data = hub.ModelCardData(
        license=license,
        library_name="pymc-mfm",
        tags=["marketing-mix-modeling", "graph-conditional", "safetensors"],
    )
    provenance = metadata["provenance"]
    profile_argument = "" if profile is None else f", profile={profile!r}"
    text = f"""---
{card_data.to_yaml()}
---
# Marketing Foundation Model

A complete trainable graph-conditional MFM artifact, including the shared
encoder, joint graph router, auxiliary structure head, graph conditioner,
and conditional decomposition decoder. A/B and new checkpoints are artifact
profiles of this same architecture, not alternative model implementations.

## Install and load

```sh
pip install 'pymc-mfm[hub]'
```

```python
from pymc_mfm import from_pretrained, list_profiles

# Use the full immutable commit returned by publication or profile discovery.
profiles = list_profiles({repo_id!r})
revision = next(iter(profiles.values()))["revision"]
model = from_pretrained({repo_id!r}, revision=revision{profile_argument})
```

## Task, units, and domain

Conditional additive decomposition of a positive observed outcome from
nonnegative media spend and finite controls. All statistics use observed
context only. Components retain their signs; multiply scaled model draws by
the context p95 absolute-outcome ruler to obtain outcome units. Context and
query are separately sampled marginal windows, not paired full-horizon draws.
Inference supports graph averaging, an empirical full-graph mode, or a supplied
fixed graph. Masked nodes have no active incident edges. The ordered feature,
target, graph, numerical, and normalization contracts are in `config.json`.
The architecture is count-agnostic over media/control inputs; artifact-specific
training support is described only by the explicit provenance below.

## Provenance and evidence

This commit exports profile {profile!r}. Each bundle records original checkpoint
identity separately from exported safetensors and configuration hashes in its
manifest. Producing tracking IDs, prior configuration, data identities, and
saved diagnostic references are included only when explicitly supplied.
No empirical calibration, out-of-domain validity, or conditional performance
is implied by export. Missing diagnostic references mean evidence was not
provided, not that an evaluation passed. Joint-router metrics must not be
interpreted as conditional-decoder evidence.

```json
{json_bytes(provenance).decode().rstrip()}
```

## Serialization, privacy, and limitations

Only safetensors and strict JSON metadata are loaded; no remote code or
executable object payload is needed. Model bundles preserve trainable state
but contain no optimizer/RNG resume state, observations, or training corpora.
Discover profiles with `list_profiles`; standard `hf download` directories use
the identical local loader. This package does not provide Transformers
AutoModel registration, hosted inference, or inference widgets.

## License

The publishing maintainer explicitly supplied `{license}` as the approved
artifact license. This declaration does not infer rights from a code license.
"""
    return hub.ModelCard(text).content


def push_to_hub(
    model: MFM,
    repo_id: str,
    *,
    revision: str | None = None,
    private: bool = True,
    token: str | bool | None = None,
    provenance: Mapping | None = None,
    profile: str | None = None,
    license: str | None = None,
):
    """Explicitly publish one complete bundle and model card in a single commit.

    Supplying ``license`` attests that the maintainer approved that artifact
    license; no license is inferred or approved by this function. Missing
    approval blocks publication before any network operation. New repositories
    default private; existing repository visibility is never changed.
    A collection must share one explicit root license; conflicting or unknown
    existing artifact licensing is rejected rather than implicitly replaced.
    Only freshly generated allowlisted files are committed, never a caller's
    working directory. The returned official CommitInfo contains the immutable
    ``oid`` to use for reproducible retrieval. No import/training/load publishes.
    """
    if (
        not isinstance(license, str)
        or not _LICENSE.fullmatch(license)
        or license in {"unknown", "other", "unlicensed"}
    ):
        raise ValueError("publishing requires an explicitly approved artifact license identifier")
    profile = validate_profile(profile)
    hub = _hub()
    repo_id = _repo_id(repo_id)
    if type(private) is not bool:
        raise ValueError("private must be boolean")
    with tempfile.TemporaryDirectory(prefix="mfm-hub-") as temporary:
        bundle = save_pretrained(
            model, Path(temporary) / "bundle", provenance=provenance, profile=profile
        )
        # The exporter validates the complete model and metadata before any remote action.
        from ._artifact_schema import read_json

        metadata = read_json(bundle / "config.json")
        profile = metadata["profile"]
        prefix = "" if profile is None else f"profiles/{profile}/"
        card = _model_card(metadata, repo_id=repo_id, profile=profile, license=license, hub=hub)
        operations = [
            hub.CommitOperationAdd(path_in_repo=f"{prefix}{name}", path_or_fileobj=bundle / name)
            for name in BUNDLE_FILES
        ]
        operations.append(
            hub.CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8"))
        )
        if profile is not None:
            operations.append(
                hub.CommitOperationAdd(
                    path_in_repo=f"{prefix}README.md", path_or_fileobj=card.encode("utf-8")
                )
            )
        api = hub.HfApi(token=token)
        _transport(
            api.create_repo, repo_id=repo_id, repo_type="model", private=private, exist_ok=True
        )
        info = _transport(api.model_info, repo_id=repo_id, revision=revision)
        card_data = getattr(info, "card_data", None)
        existing_license = (
            card_data.get("license")
            if isinstance(card_data, Mapping)
            else getattr(card_data, "license", None)
        )
        existing_artifacts = any(
            sibling.rfilename == "manifest.json"
            or re.fullmatch(r"profiles/[^/]+/manifest\.json", sibling.rfilename)
            for sibling in (getattr(info, "siblings", None) or ())
        )
        if existing_license is not None and existing_license != license:
            raise ValueError("publication cannot replace the existing repository license")
        if existing_artifacts and existing_license is None:
            raise ValueError(
                "existing artifacts need explicit repository license metadata before publication"
            )
        return _transport(
            api.create_commit,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            parent_commit=info.sha,
            operations=operations,
            commit_message="Upload complete MFM model artifact",
        )
