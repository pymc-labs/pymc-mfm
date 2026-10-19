"""Optional Hub transport over the same strict local MFM bundle contract."""

from __future__ import annotations

import json
import math
import re
import secrets
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from pymc_mfm.model.model import MFM

from ._artifact_schema import (
    BUNDLE_FILES,
    RECORDED_OUTPUTS_FILE,
    json_bytes,
    parse_json,
    recorded_outputs_bytes,
    recorded_outputs_metadata,
    validate_profile,
)
from .checkpoint import _load_local, _read_manifest, save_pretrained

if TYPE_CHECKING:
    from pymc_mfm.data.recipes import ShardPlan

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_LICENSE = re.compile(r"[a-z0-9][a-z0-9.+-]{0,79}\Z")
_PROFILE_MANIFEST = re.compile(r"profiles/([^/]+)/manifest\.json")
_CARD_TAGS = ("marketing-mix-modeling", "graph-conditional", "safetensors")
#: Profile-recipe fields shown in a card's training-corpus table.
_CORPUS_FIELDS = (
    "n_time_steps",
    "query_frac",
    "p_long_horizon",
    "val_cell_frac",
    "n_treatments",
    "n_treatments_active_range",
    "n_covariates",
    "n_covariates_active_range",
    "n_latent",
    "n_latent_active_range",
    "edge_budget",
    "l_max",
    "carryover_family_probs",
    "carryover_alpha_range",
    "saturation_family_probs",
    "confounding_strength_range",
    "rw_baseline_std_range",
    "rw_outcome_std_range",
)


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


def _cell(value: object) -> str:
    """An exact JSON rendering that cannot break a Markdown table row."""
    return json.dumps(value, sort_keys=True, separators=(", ", ": ")).replace("|", "\\|")


def _table(header: tuple[str, ...], rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _front_matter(license: str) -> str:
    tags = [f"- {tag}" for tag in _CARD_TAGS]
    return "\n".join(
        ["---", f"license: {json.dumps(license)}", "library_name: pymc-mfm", "tags:", *tags, "---"]
    )


def _load_snippet(repo_id: str | None, profile: str | None) -> str:
    argument = "" if profile is None else f", profile={profile!r}"
    if repo_id is None:
        return f"""```python
from pymc_mfm import from_pretrained

# A local copy or `hf download` directory of the published repository.
model = from_pretrained("<downloaded directory>"{argument})
```"""
    key = "." if profile is None else profile
    return f"""```sh
pip install 'pymc-mfm[hub]'
```

```python
from pymc_mfm import from_pretrained, list_profiles

# Use the full immutable commit returned by publication or profile discovery.
profiles = list_profiles({repo_id!r})
revision = profiles[{key!r}]["revision"]
model = from_pretrained({repo_id!r}, revision=revision{argument})
```"""


def _architecture_section(metadata: Mapping) -> str:
    parameters = sum(
        math.prod(entry["shape"])
        for entry in metadata["manifest"]["tensors"]
        if entry["kind"] == "parameter"
    )
    rows = [
        ("architecture", metadata["architecture"]),
        *(
            (f"model_config.{name}", value)
            for name, value in metadata["model_config"].items()
            if name != "format_version"
        ),
        ("parameters", parameters),
        *((f"inference.{name}", value) for name, value in metadata["inference"].items()),
    ]
    return _table(("field", "value"), [(name, _cell(value)) for name, value in rows])


def _training_plan(name: object) -> ShardPlan | None:
    """The ``SHARD_PLANS`` profile training plan that ``corpus_plan`` names, if any."""
    if name is None:
        return None
    from pymc_mfm.data.recipes import SHARD_PLANS

    names = sorted(key for key, plan in SHARD_PLANS.items() if plan.profile.name == key)
    if not isinstance(name, str) or name not in names:
        raise ValueError(f"corpus_plan must name a profile training plan: {', '.join(names)}")
    return SHARD_PLANS[name]


def _check_plan_worlds(plan: ShardPlan | None, evaluation: Mapping | None) -> None:
    """Refuse a training plan whose shards do not hold the recorded evaluation worlds."""
    if plan is None or evaluation is None:
        return
    seeds = {plan.seed(shard) for shard in range(plan.n_shards)}
    rows = plan.n_cells * plan.draws_per_cell
    if any(
        world["shard_seed"] not in seeds or world["row"] >= rows for world in evaluation["worlds"]
    ):
        raise ValueError(
            f"the recorded training_evaluation worlds are not rows of plan {plan.name!r}'s shards"
        )


def _corpus_section(plan: ShardPlan | None) -> str:
    if plan is None:
        return (
            "No shard plan was named when this card was rendered, so the training corpus\n"
            "is identified only by the provenance data references below."
        )
    from pymc_mfm.data.recipes import recipe_for

    payload = recipe_for(plan.profile).payload
    rows = [
        ("plan.name", plan.name),
        ("plan.n_shards", plan.n_shards),
        ("plan.first_seed", plan.first_seed),
        ("plan.last_seed", plan.seed(plan.n_shards - 1)),
        ("plan.seed_stride", plan.seed_stride),
        ("plan.n_cells", plan.n_cells),
        ("plan.draws_per_cell", plan.draws_per_cell),
        ("plan.worlds", plan.worlds),
        *((f"prior.{field}", payload[field]) for field in _CORPUS_FIELDS),
    ]
    return f"""The maintainer who rendered this card named MFM's shard plan
`{plan.name}` as the training corpus. Shard `i` is one generator call with seed
`plan.first_seed + i * plan.seed_stride`; regenerate it into directory `<dir>` with
`python -m pymc_mfm.data generate --profile {plan.name} --shard <i> --out <dir>`.
The generator marks a `prior.val_cell_frac` share of cells as held out
(`is_val`). The table lists the prior settings that bound the training domain;
the complete prior is `recipe_for(plan.profile).payload` with
`plan = SHARD_PLANS[{plan.name!r}]` from `pymc_mfm.data.recipes`.

{_table(("field", "value"), [(name, _cell(value)) for name, value in rows])}"""


def _evaluation_section(record: Mapping | None) -> str:
    if record is None:
        return "No training evaluation is recorded in this bundle's provenance."
    protocol = [
        (f"training_evaluation.{name}", _cell(record[name]))
        for name in (
            "source",
            "step",
            "leg",
            "head",
            "eval_seed",
            "n_worlds",
            "n_samples",
            "flow_steps",
            "worlds",
        )
    ]
    metrics = [
        (name, _cell(value), record["aggregation"][name])
        for name, value in sorted(record["metrics"].items())
    ]
    head = (
        ""
        if record["head"] == "conditional"
        else f"\nThe scored head is `{record['head']}`, so these values do not describe the\n"
        "conditional decoder that `decompose` samples.\n"
    )
    return f"""Values recorded by the producing training run, stored as the provenance key
`training_evaluation` in `config.json`. Loading checks their structure, never
their values; `null` marks a metric that was undefined. Aggregation `world`
averages worlds equally; `slot` averages each world's channel or control slots,
then the worlds equally; `slot_pooled` pools the channel or control slots of all
worlds, weighting each slot rather than each world equally; `pooled_edges`
scores the pooled edge slots of all worlds once.
{head}
{_table(("field", "value"), protocol)}

{_table(("metric", "value", "aggregation"), metrics)}"""


def _license_section(license: str) -> str:
    return f"""Artifact license: `{license}`, supplied explicitly by the maintainer who rendered
this card. It is not inferred from the code license."""


def _bundle_card(
    metadata: Mapping, *, license: str, repo_id: str | None, plan: ShardPlan | None
) -> str:
    profile = metadata["profile"]
    provenance = dict(metadata["provenance"])
    evaluation = provenance.pop("training_evaluation", None)
    _check_plan_worlds(plan, evaluation)
    title = "" if profile is None else f": profile {profile}"
    return f"""{_front_matter(license)}
# Marketing Foundation Model{title}

A complete trainable graph-conditional MFM artifact, including the shared
encoder, joint graph router, auxiliary structure head, graph conditioner,
and conditional decomposition decoder. Profiles are artifacts of this same
fixed architecture trained on different corpora, not alternative model
implementations.

## Architecture

{_architecture_section(metadata)}

## Install and load

{_load_snippet(repo_id, profile)}

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
training support is described only by the corpus and provenance below.

## Training corpus

{_corpus_section(plan)}

## Recorded training evaluation

{_evaluation_section(evaluation)}

## Provenance and evidence

Each bundle records original checkpoint identity separately from exported
safetensors and configuration hashes in its manifest. Producing tracking IDs,
prior configuration, data identities, and saved diagnostic references are
included only when explicitly supplied. No empirical calibration,
out-of-domain validity, or conditional performance is implied by export.
Missing diagnostic references mean evidence was not provided, not that an
evaluation passed. Joint-router metrics must not be interpreted as
conditional-decoder evidence.

```json
{json_bytes(provenance).decode().rstrip()}
```

## Serialization, privacy, and limitations

Only safetensors and strict JSON metadata are loaded; no remote code or
executable object payload is needed. Model bundles preserve trainable state
but contain no optimizer/RNG resume state, observations, or training corpora.
An optional `recorded_outputs.json` beside the bundle holds one synthetic
world's inputs and the expected `decompose` outputs used to check an install.
Discover profiles with `list_profiles`; standard `hf download` directories use
the identical local loader. This package does not provide Transformers
AutoModel registration, hosted inference, or inference widgets. The weights
were fitted only to synthetic worlds of the training corpus above; accuracy
for other horizons, entity or live-effect counts, response families,
confounding or noise levels is not established.

## License

{_license_section(license)}
"""


def _collection_card(profiles: list[str], *, license: str, repo_id: str | None) -> str:
    rows = [(name, f"[profiles/{name}/README.md](profiles/{name}/README.md)") for name in profiles]
    return f"""{_front_matter(license)}
# Marketing Foundation Model collection

Every profile below is an artifact of the same fixed graph-conditional MFM
architecture, trained on its own corpus. Each profile's model card describes its
architecture, training corpus, recorded training evaluation, provenance and
limitations.

## Profiles

{_table(("profile", "model card"), rows)}

## Install and load

{_load_snippet(repo_id, profiles[0])}

## License

{_license_section(license)}
"""


def write_model_cards(
    path: str | Path,
    *,
    license: str,
    repo_id: str | None = None,
    corpus_plan: str | None = None,
) -> list[Path]:
    """Render the model cards of a local bundle or collection, e.g. before publication.

    Each ``profiles/<name>/`` bundle receives its own ``README.md`` and the root
    ``README.md`` indexes every profile; a lone root bundle's card is the root
    ``README.md``. A root bundle beside ``profiles/`` is refused, since its card
    and the index would both be the root ``README.md``. ``license`` is explicit,
    nonempty text, which may state that no license has been selected yet;
    ``push_to_hub`` renders and publishes its own cards and accepts only approved
    identifiers. ``corpus_plan`` names the ``SHARD_PLANS`` training plan (``A`` or
    ``B``) that the one bundle in ``path`` was trained on, which its card then
    describes; a card never infers its corpus from the profile label, and a plan
    whose shards do not hold every world of the bundle's recorded
    ``training_evaluation`` is refused. To describe the corpora of a collection,
    pass each ``profiles/<name>`` directory with its own plan. Every described
    bundle is validated by the local loader before any card is written. Each card
    is written to a sibling temporary file renamed onto its target, so a symlinked
    ``README.md``, as in a Hub cache snapshot, is replaced rather than written
    through into a blob other revisions share.
    ``repo_id`` (validated with the ``hub`` extra) selects Hub loading
    instructions; no Hub request is made. Returns the written paths.
    """
    if not isinstance(license, str) or not license.strip():
        raise ValueError("model cards require an explicit license text")
    plan = _training_plan(corpus_plan)
    if repo_id is not None:
        repo_id = _repo_id(repo_id)
    root = Path(path).expanduser()
    entries = list_profiles(root)
    profiles = sorted(name for name in entries if name != ".")
    if profiles and "." in entries:
        raise ValueError("a directory cannot hold both a root bundle and profiles/<name>/ bundles")
    if plan is not None and len(entries) > 1:
        raise ValueError("corpus_plan describes one bundle; pass each profiles/<name> directory")

    def card(name: str) -> str:
        metadata = _load_local(root / entries[name]["path"]).artifact_metadata
        return _bundle_card(metadata, license=license, repo_id=repo_id, plan=plan)

    cards = {root / "profiles" / name / "README.md": card(name) for name in profiles}
    cards[root / "README.md"] = (
        _collection_card(profiles, license=license, repo_id=repo_id) if profiles else card(".")
    )
    token = secrets.token_hex(4)
    for target, text in cards.items():
        temporary = target.with_name(f".{target.name}.{token}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return list(cards)


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
    recorded_outputs: str | Path | None = None,
    corpus_plan: str | None = None,
):
    """Explicitly publish one complete bundle and its model cards in a single commit.

    Supplying ``license`` attests that the maintainer approved that artifact
    license; no license is inferred or approved by this function. Missing
    approval blocks publication before any network operation. New repositories
    default private; existing repository visibility is never changed.
    A collection must share one explicit root license; conflicting or unknown
    existing artifact licensing is rejected rather than implicitly replaced.
    Only freshly generated allowlisted files are committed, never a caller's
    working directory. The returned official CommitInfo contains the immutable
    ``oid`` to use for reproducible retrieval. No import/training/load publishes.

    A profile's card is ``profiles/<name>/README.md``. The root ``README.md``
    indexes every profile already published plus this one, so publishing a
    profile never rewrites another profile's card. A repository holds either
    ``profiles/<name>/`` bundles or one root bundle, whose card is the root
    ``README.md``; publishing a profile beside a root bundle, or a root bundle
    beside profiles, is refused before any commit. ``corpus_plan`` names the
    ``SHARD_PLANS`` training plan (``A`` or ``B``) the model was trained on, which
    the card then describes, provided its shards hold every world of the recorded
    ``training_evaluation``; without it the card identifies the corpus only by
    provenance, never by the profile label. ``recorded_outputs`` names the JSON
    sidecar of synthetic inputs and expected outputs for this artifact. It must
    name the same profile, source checkpoint and weights (``state_sha256``, as
    recorded by ``from_pretrained``), have the bundle's control and latent widths
    and pass the strict sidecar schema; its canonical serialization is uploaded as
    ``recorded_outputs.json`` beside the bundle files. Republishing a profile
    without a sidecar deletes its previous one in the same commit.
    """
    if (
        not isinstance(license, str)
        or not _LICENSE.fullmatch(license)
        or license in {"unknown", "other", "unlicensed"}
    ):
        raise ValueError("publishing requires an explicitly approved artifact license identifier")
    profile = validate_profile(profile)
    plan = _training_plan(corpus_plan)
    recorded = None
    if recorded_outputs is not None:
        recorded_outputs = Path(recorded_outputs).expanduser()
        if not recorded_outputs.is_file():
            raise FileNotFoundError("recorded_outputs must name an existing JSON file")
        recorded = parse_json(recorded_outputs.read_bytes())
    hub = _hub()
    repo_id = _repo_id(repo_id)
    if type(private) is not bool:
        raise ValueError("private must be boolean")
    with tempfile.TemporaryDirectory(prefix="mfm-hub-") as temporary:
        bundle = save_pretrained(
            model, Path(temporary) / "bundle", provenance=provenance, profile=profile
        )
        # The exported files must load exactly as they will be published.
        exported = _load_local(bundle)
        metadata = exported.artifact_metadata
        profile = metadata["profile"]
        prefix = "" if profile is None else f"profiles/{profile}/"
        sidecar = None
        if recorded is not None:
            sidecar = recorded_outputs_bytes(
                recorded_outputs_metadata(
                    recorded,
                    profile=profile,
                    provenance=metadata["provenance"],
                    state_sha256=metadata["state_sha256"],
                    model_config=exported.config,
                )
            )
        bundle_card = hub.ModelCard(
            _bundle_card(metadata, license=license, repo_id=repo_id, plan=plan)
        ).content
        operations = [
            hub.CommitOperationAdd(path_in_repo=f"{prefix}{name}", path_or_fileobj=bundle / name)
            for name in BUNDLE_FILES
        ]
        sidecar_path = f"{prefix}{RECORDED_OUTPUTS_FILE}"
        if sidecar is not None:
            operations.append(
                hub.CommitOperationAdd(path_in_repo=sidecar_path, path_or_fileobj=sidecar)
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
        files = {sibling.rfilename for sibling in (getattr(info, "siblings", None) or ())}
        published = {match.group(1) for match in map(_PROFILE_MANIFEST.fullmatch, files) if match}
        if existing_license is not None and existing_license != license:
            raise ValueError("publication cannot replace the existing repository license")
        if (published or "manifest.json" in files) and existing_license is None:
            raise ValueError(
                "existing artifacts need explicit repository license metadata before publication"
            )
        profiles = sorted(published | ({profile} if profile is not None else set()))
        if profiles and (profile is None or "manifest.json" in files):
            # The root bundle's card and the profile index would both be README.md.
            raise ValueError(
                "a repository cannot hold both a root bundle and profiles/<name>/ bundles"
            )
        if sidecar is None and sidecar_path in files:
            operations.append(hub.CommitOperationDelete(path_in_repo=sidecar_path))
        if profile is not None:
            operations.append(
                hub.CommitOperationAdd(
                    path_in_repo=f"{prefix}README.md", path_or_fileobj=bundle_card.encode("utf-8")
                )
            )
        root_card = (
            hub.ModelCard(_collection_card(profiles, license=license, repo_id=repo_id)).content
            if profiles
            else bundle_card
        )
        operations.append(
            hub.CommitOperationAdd(
                path_in_repo="README.md", path_or_fileobj=root_card.encode("utf-8")
            )
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
