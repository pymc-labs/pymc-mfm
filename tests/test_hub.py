"""Hub contracts use local artifacts and isolated transports, never real publication."""

import builtins
import fnmatch
import hashlib
import itertools
import json
import re
import shlex
import shutil
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub as hf
import huggingface_hub.errors as hf_errors
import pytest
import torch
from packaging.version import Version

from pymc_mfm.inference._artifact_schema import recorded_outputs_bytes
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import (
    HubArtifactError,
    from_pretrained,
    list_profiles,
    push_to_hub,
    write_model_cards,
)
from pymc_mfm.model.flow import DECOMPOSITION_FIXED_TARGETS
from pymc_mfm.model.graph import EdgeSlotLayout

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
REPO = "test-owner/mfm-artifacts"


def _forbid_hub_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] == "huggingface_hub":
            pytest.fail(f"unexpected Hub SDK import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _assert_same_draws(original, restored, batch):
    torch.manual_seed(19)
    expected, expected_graphs = original.sample_disentangled(batch, 2)
    torch.manual_seed(19)
    actual, actual_graphs = restored.sample_disentangled(batch, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_graphs, expected_graphs, rtol=0, atol=0)


def _card_facts(text):
    """Rows of every Markdown table in a card: first cell -> remaining cells."""
    return {
        cells[0]: cells[1:]
        for cells in (line[2:-2].split(" | ") for line in text.splitlines() if line[:2] == "| ")
    }


def _sidecar(state_sha256, *, n_controls=2, n_demand=2):
    """A recorded-output sidecar of profile A of checkpoint ``f * 64`` with those weights.

    Its world has 3 weeks, 2 of them context, one channel, ``n_controls`` controls
    and ``n_demand`` latent nodes (by default the widths of the ``model`` fixture);
    each case holds posterior-mean components and 4 graph draws.
    """
    slots = EdgeSlotLayout(1, n_controls, n_demand).n_slots
    columns = 2 + n_controls + len(DECOMPOSITION_FIXED_TARGETS)
    cases = {
        f"{mode}/{window}": {
            "seed": seed,
            "sales_scale": 2.5,
            "components_mean": [[0.5] * columns for _ in range(2 if window == "context" else 1)],
            "graphs": [[0] * slots for _ in range(4)],
        }
        for seed, (mode, window) in enumerate(
            itertools.product(("model_average", "map", "fixed"), ("query", "context"))
        )
    }
    return {
        "format_version": 1,
        "profile": "A",
        "source_checkpoint_sha256": "f" * 64,
        "state_sha256": state_sha256,
        "world": {"plan": "evaluation", "shard": 0, "shard_seed": 540000, "row": 0},
        "observations": {
            "spend": [[1.0], [2.0], [3.0]],
            "controls": [
                [0.25 * week + control for control in range(n_controls)] for week in (2, 1, 3)
            ],
            "context_outcome": [4.0, 5.0],
            "channel_mask": [1],
            "control_mask": [1] * n_controls,
            "latent_mask": [1] + [0] * (n_demand - 1),
        },
        "graph": [0] * slots,
        "n_samples": 4,
        "cases": cases,
        "producer": {
            "python": "3.13.9",
            "pymc-mfm": "0.0.1.dev0",
            "pymc-generator": "0.0.2",
            "torch": "2.8.0+cpu",
            "numpy": "2.4.6",
            "system": "Linux",
            "machine": "x86_64",
            "torch_threads": 2,
        },
    }


def _state_sha256(model):
    """The weights' digest, as loading the model's bundle records it."""
    with tempfile.TemporaryDirectory() as directory:
        return from_pretrained(save_pretrained(model, Path(directory) / "bundle")).provenance[
            "state_sha256"
        ]


def test_local_and_cli_profile_layout_do_not_require_hub(model, batch, tmp_path, monkeypatch):
    save_pretrained(model, tmp_path / "profiles" / "A", profile="A")
    save_pretrained(model, tmp_path / "profiles" / "B", profile="B")

    _forbid_hub_imports(monkeypatch)
    profiles = list_profiles(tmp_path)
    assert set(profiles) == {"A", "B"}
    _assert_same_draws(model, from_pretrained(tmp_path, profile="B"), batch)
    _assert_same_draws(model, from_pretrained(tmp_path / "profiles" / "A"), batch)
    with pytest.raises(FileNotFoundError):
        from_pretrained(tmp_path)
    with pytest.raises(ValueError):
        from_pretrained(tmp_path / "profiles" / "A", profile="B")


def test_mutable_revision_is_resolved_once_and_only_selected_files_arrive(
    model, batch, tmp_path, monkeypatch
):
    source = tmp_path / "source"
    save_pretrained(model, source)
    save_pretrained(model, source / "profiles" / "A", profile="A")
    save_pretrained(model, source / "profiles" / "B", profile="B")
    (source / "observations.csv").write_text("private observations")
    (source / "profiles" / "B" / "report.html").write_text("not requested")
    cache = tmp_path / "cache"
    calls = []
    token = "hf_notForLogsOrArtifacts"

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        requested = kwargs["revision"]
        # A moving branch would serve B on a second unpinned resolution.
        commit = COMMIT_A if requested == "release" and len(calls) == 1 else requested
        if requested == "release" and len(calls) > 1:
            commit = COMMIT_B
        destination = cache / "models--test-owner--mfm-artifacts" / "snapshots" / commit
        destination.mkdir(parents=True, exist_ok=True)
        for file in source.rglob("*"):
            if file.is_file() and any(
                fnmatch.fnmatchcase(file.relative_to(source).as_posix(), pattern)
                for pattern in kwargs["allow_patterns"]
            ):
                relative = file.relative_to(source)
                (destination / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, destination / relative)
        return str(destination)

    monkeypatch.setattr(hf, "snapshot_download", snapshot_download)
    loaded = from_pretrained(REPO, revision="release", profile="B", cache_dir=cache, token=token)
    _assert_same_draws(model, loaded, batch)
    assert [call["revision"] for call in calls] == ["release", COMMIT_A]
    assert all(call["token"] == token and call["cache_dir"] == cache for call in calls)
    assert loaded.provenance["hub_revision"] == COMMIT_A
    assert token not in json.dumps(loaded.artifact_metadata)
    snapshot = cache / "models--test-owner--mfm-artifacts" / "snapshots" / COMMIT_A
    assert {
        file.relative_to(snapshot).as_posix() for file in snapshot.rglob("*") if file.is_file()
    } == {
        "profiles/B/config.json",
        "profiles/B/model.safetensors",
        "profiles/B/manifest.json",
    }
    before = {file: file.read_bytes() for file in snapshot.rglob("*") if file.is_file()}
    _assert_same_draws(loaded, from_pretrained(snapshot, profile="B"), batch)
    assert all(file.read_bytes() == data for file, data in before.items())


@pytest.mark.parametrize("profile", [None, "B"])
def test_real_sdk_offline_cache_and_cli_snapshot_are_identical(model, batch, tmp_path, profile):
    cache = tmp_path / "cache"
    repo = cache / "models--test-owner--mfm-artifacts"
    snapshot = repo / "snapshots" / COMMIT_A
    bundle = snapshot if profile is None else snapshot / "profiles" / profile
    save_pretrained(model, bundle, profile=profile)
    (repo / "refs").mkdir()
    (repo / "refs" / "release").write_text(COMMIT_A)
    sdk_model = from_pretrained(
        REPO,
        revision="release",
        profile=profile,
        cache_dir=cache,
        local_files_only=True,
        token=False,
    )
    local_model = from_pretrained(snapshot, profile=profile)
    _assert_same_draws(sdk_model, local_model, batch)
    assert sdk_model.provenance["bundle_sha256"] == local_model.provenance["bundle_sha256"]
    assert sdk_model.provenance["hub_revision"] == COMMIT_A
    with pytest.raises(HubArtifactError) as missing:
        from_pretrained(
            REPO, revision=COMMIT_B, cache_dir=cache, local_files_only=True, token=False
        )
    assert missing.value.code == "cache_miss"


def test_offline_partial_cache_never_loads_random_missing_weights(model, tmp_path):
    cache = tmp_path / "cache"
    snapshot = save_pretrained(
        model, cache / "models--test-owner--mfm-artifacts" / "snapshots" / COMMIT_A
    )
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        from_pretrained(
            REPO, revision=COMMIT_A, cache_dir=cache, local_files_only=True, token=False
        )


def test_hub_commit_mismatch_fails_instead_of_combining_snapshots(model, tmp_path, monkeypatch):
    first = save_pretrained(model, tmp_path / "snapshots" / COMMIT_A)
    second = save_pretrained(model, tmp_path / "snapshots" / COMMIT_B)
    paths = iter([str(first), str(second)])
    monkeypatch.setattr(hf, "snapshot_download", lambda **kwargs: next(paths))
    with pytest.raises(HubArtifactError) as mismatch:
        from_pretrained(REPO, revision="moving")
    assert mismatch.value.code == "revision_mismatch"


def test_discovery_only_fetches_manifests_and_pins_followup(model, tmp_path, monkeypatch):
    root = tmp_path / "snapshots" / COMMIT_A
    save_pretrained(model, root / "profiles" / "A", profile="A")
    save_pretrained(model, root / "profiles" / "B", profile="B")
    requests = []

    def snapshot_download(**kwargs):
        requests.append(kwargs)
        return str(root)

    monkeypatch.setattr(hf, "snapshot_download", snapshot_download)
    profiles = list_profiles(REPO, revision="release", token=False)
    assert set(profiles) == {"A", "B"}
    assert {entry["revision"] for entry in profiles.values()} == {COMMIT_A}
    assert requests[0]["allow_patterns"] == ["manifest.json", "profiles/*/manifest.json"]
    from_pretrained(REPO, profile="A", revision=profiles["A"]["revision"])
    assert all(request["revision"] == COMMIT_A for request in requests[1:])


@pytest.mark.parametrize(
    "error_type,code",
    [
        (hf_errors.GatedRepoError, "access_denied_or_missing"),
        (hf_errors.RepositoryNotFoundError, "access_denied_or_missing"),
        (hf_errors.RevisionNotFoundError, "revision_not_found"),
        (hf_errors.LocalEntryNotFoundError, "cache_miss"),
        (RuntimeError, "transport_error"),
    ],
)
def test_auth_revision_and_transport_errors_do_not_leak_credentials(
    tmp_path, monkeypatch, caplog, error_type, code
):
    token = "hf_neverExposeThisCredential"

    def failure(**kwargs):
        message = f"backend included Authorization: Bearer {token}"
        if error_type in (
            hf_errors.GatedRepoError,
            hf_errors.RepositoryNotFoundError,
            hf_errors.RevisionNotFoundError,
        ):
            url = "https://huggingface.co/api/models/test-owner/mfm-artifacts"
            status = 403 if error_type is hf_errors.GatedRepoError else 404
            if Version(hf.__version__).major >= 1:
                import httpx

                response = httpx.Response(status, request=httpx.Request("GET", url))
            else:
                import requests

                response = requests.Response()
                response.status_code = status
                response.request = requests.Request("GET", url).prepare()
            raise error_type(message, response=response)
        raise error_type(message)

    monkeypatch.setattr(hf, "snapshot_download", failure)
    with pytest.raises(HubArtifactError) as caught:
        from_pretrained(REPO, revision=COMMIT_A, cache_dir=tmp_path, token=token)
    assert caught.value.code == code
    assert token not in "".join(traceback.format_exception(caught.value))
    assert token not in caplog.text


def test_publication_is_explicit_license_gated_and_preserves_complete_state(
    model, batch, tmp_path, monkeypatch
):
    with monkeypatch.context() as patch:
        _forbid_hub_imports(patch)
        # A local card's free-text license is not an approved publication identifier.
        for license in (None, "not selected (publication pending)", "other"):
            with pytest.raises(ValueError):
                push_to_hub(model, REPO, license=license)
        # The evaluation plan is no training corpus.
        with pytest.raises(ValueError):
            push_to_hub(model, REPO, license="mit", corpus_plan="evaluation")
        # A description that could carry a credential is refused before any transport.
        with pytest.raises(ValueError, match="credentials"):
            push_to_hub(model, REPO, license="mit", description="Token hf_abcdefghijklmnop.")
    token = "hf_doNotSerializePublisherToken"
    published = tmp_path / "published"
    events = []
    model.user_observations = {"secret": "user data must never upload"}

    class LocalCommitApi:
        def __init__(self, *, token):
            assert token == "hf_doNotSerializePublisherToken"

        def create_repo(self, **kwargs):
            events.append(("create_repo", kwargs))

        def model_info(self, **kwargs):
            return SimpleNamespace(sha=COMMIT_A)

        def create_commit(self, **kwargs):
            events.append(("create_commit", kwargs))
            assert kwargs["parent_commit"] == COMMIT_A
            allowed = {
                "profiles/A/config.json",
                "profiles/A/manifest.json",
                "profiles/A/model.safetensors",
                "profiles/A/recorded_outputs.json",
                "profiles/A/README.md",
                "README.md",
            }
            assert {operation.path_in_repo for operation in kwargs["operations"]} == allowed
            for operation in kwargs["operations"]:
                source = operation.path_or_fileobj
                data = source if isinstance(source, bytes) else Path(source).read_bytes()
                assert token.encode() not in data
                assert b"user data must never upload" not in data
                target = published / operation.path_in_repo
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return SimpleNamespace(oid=COMMIT_B)

    monkeypatch.setattr(hf, "HfApi", LocalCommitApi)
    provenance = {
        "source_checkpoint_sha256": "f" * 64,
        "source_step": 17,
        "source_revision": "c" * 40,
        "prior": {"name": "local-training", "config_hash": "e" * 64},
        "data": [{"name": "synthetic", "sha256": "d" * 64}],
        "diagnostics": [],
        "tracking_run_id": "run-17",
    }
    payload = _sidecar(_state_sha256(model))
    sidecar = tmp_path / "recorded_outputs.json"
    sidecar.write_text(json.dumps(payload, indent=2))
    commit = push_to_hub(
        model,
        REPO,
        revision="release",
        token=token,
        profile="A",
        provenance=provenance,
        license="apache-2.0",
        recorded_outputs=sidecar,
        description="Profile A, trained on **local** worlds.",
    )
    assert commit.oid == COMMIT_B
    assert [name for name, _ in events] == ["create_repo", "create_commit"]
    assert events[0][1]["private"] is True
    # The canonical serialization of the validated sidecar is published, not the
    # caller's bytes.
    assert (published / "profiles" / "A" / "recorded_outputs.json").read_bytes() == (
        recorded_outputs_bytes(payload)
    )
    # The loader accepts a bundle directory that also holds its cards and sidecar.
    restored = from_pretrained(published, profile="A")
    _assert_same_draws(model, restored, batch)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for path in ("README.md", "profiles/A/README.md"):
        card = hf.ModelCard((published / path).read_text())
        assert card.data.license == "apache-2.0"
        assert card.data.library_name == "pymc-mfm"
        assert not getattr(card.data, "pipeline_tag", None)
    root = _card_facts(hf.ModelCard((published / "README.md").read_text()).text)
    assert "profiles/A/README.md" in root["A"][0]
    # Without corpus_plan, the card of a bundle labelled A names no shard plan.
    card = _card_facts(hf.ModelCard((published / "profiles/A/README.md").read_text()).text)
    assert not any(key.startswith("plan.") for key in card)
    assert restored.provenance["source_checkpoint_sha256"] == "f" * 64
    # The description belongs to its profile's card, not to the collection index.
    profile_card = (published / "profiles/A/README.md").read_text()
    assert "## Model description\n\nProfile A, trained on **local** worlds.\n" in profile_card
    assert "Model description" not in (published / "README.md").read_text()


def test_publish_rejects_private_metadata_before_remote_changes(model, monkeypatch):
    def no_remote(**kwargs):
        pytest.fail("invalid provenance must never create or update a repository")

    monkeypatch.setattr(hf, "HfApi", no_remote)
    with pytest.raises(ValueError):
        push_to_hub(model, REPO, provenance={"token": "secret"}, license="mit")


@pytest.mark.parametrize("existing_license", [None, "apache-2.0"])
def test_profile_publication_cannot_implicitly_relicense_existing_artifacts(
    model, monkeypatch, existing_license
):
    class ExistingRepositoryApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, **kwargs):
            pass

        def model_info(self, **kwargs):
            return SimpleNamespace(
                sha=COMMIT_A,
                card_data=hf.ModelCardData(license=existing_license),
                siblings=[SimpleNamespace(rfilename="profiles/A/manifest.json")],
            )

        def create_commit(self, **kwargs):
            pytest.fail("a profile cannot replace unknown or different collection licensing")

    monkeypatch.setattr(hf, "HfApi", ExistingRepositoryApi)
    with pytest.raises(ValueError):
        push_to_hub(model, REPO, profile="B", license="mit")


def _replace(*path, value):
    """A sidecar edit that sets the entry at ``path``."""

    def edit(sidecar, provenance):
        target = sidecar
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _without_source(sidecar, provenance):
    # The sidecar and the bundle agree, so only the required-source rule rejects it.
    sidecar["source_checkpoint_sha256"] = None
    del provenance["source_checkpoint_sha256"]


#: One edit per sidecar rule; each is rejected by that rule alone.
INVALID_SIDECARS = {
    "other_profile": _replace("profile", value="B"),
    "other_checkpoint": _replace("source_checkpoint_sha256", value="e" * 64),
    "no_source_checkpoint": _without_source,
    "unknown_version": _replace("format_version", value=2),
    "unknown_field": _replace("notes", value="free text"),
    "unplanned_world": _replace("world", "plan", value="custom"),
    "float_shard_seed": _replace("world", "shard_seed", value=540000.0),
    "other_shard_seed": _replace("world", "shard_seed", value=540001),
    "row_outside_shard": _replace("world", "row", value=128),
    "observation_field": _replace("observations", "note", value="example-host"),
    "scalar_array": _replace("observations", "spend", value=1.0),
    "ragged_array": lambda sidecar, _: sidecar["observations"]["spend"][1].append(1.0),
    "text_in_array": _replace("cases", "map/query", "components_mean", 0, 0, value="0.5"),
    "overflowing_number": _replace("observations", "spend", 0, 0, value=10**400),
    "non_binary_graph": _replace("graph", 0, value=2),
    "misaligned_controls": lambda sidecar, _: sidecar["observations"]["controls"].pop(),
    "channel_mask_width": lambda sidecar, _: sidecar["observations"]["channel_mask"].append(1),
    "control_mask_width": lambda sidecar, _: sidecar["observations"]["control_mask"].append(1),
    # Self-consistent worlds of other widths than the bundle's.
    "other_control_width": lambda sidecar, _: sidecar.update(
        _sidecar(sidecar["state_sha256"], n_controls=1)
    ),
    "other_latent_width": lambda sidecar, _: sidecar.update(
        _sidecar(sidecar["state_sha256"], n_demand=3)
    ),
    "no_active_channel": _replace("observations", "channel_mask", value=[0]),
    "no_active_control": _replace("observations", "control_mask", value=[0, 0]),
    "graph_width": lambda sidecar, _: sidecar["graph"].append(0),
    "float_draw_count": _replace("n_samples", value=4.0),
    "missing_case": lambda sidecar, _: sidecar["cases"].pop("fixed/query"),
    "case_field": _replace("cases", "map/query", "note", value="example-host"),
    "fractional_case_seed": _replace("cases", "map/query", "seed", value=0.5),
    "int64_overflowing_seed": _replace("cases", "map/query", "seed", value=2**63),
    "nonpositive_scale": _replace("cases", "map/query", "sales_scale", value=0.0),
    "overflowing_scale": _replace("cases", "map/query", "sales_scale", value=10**400),
    "case_rows": lambda sidecar, _: sidecar["cases"]["fixed/context"]["components_mean"].pop(),
    "draw_count": lambda sidecar, _: sidecar["cases"]["map/context"]["graphs"].pop(),
    "noncanonical_version": _replace("producer", "numpy", value="v2.4.6"),
    "local_version": _replace("producer", "numpy", value="2.4.6+exampleuser"),
    "torch_local_version": _replace("producer", "torch", value="2.8.0+example.host"),
    "numpy_build_label": _replace("producer", "numpy", value="2.4.6+cpu"),
    "host_as_system": _replace("producer", "system", value="example-host"),
    "host_as_machine": _replace("producer", "machine", value="example-host"),
    "zero_threads": _replace("producer", "torch_threads", value=0),
    "producer_field": _replace("producer", "hostname", value="example-host"),
}


@pytest.mark.parametrize("edit", INVALID_SIDECARS.values(), ids=INVALID_SIDECARS.keys())
def test_recorded_outputs_must_describe_the_published_artifact(model, tmp_path, monkeypatch, edit):
    def no_remote(**kwargs):
        pytest.fail("an unbound or malformed sidecar must never create or update a repository")

    provenance = {"source_checkpoint_sha256": "f" * 64}
    sidecar = _sidecar(_state_sha256(model))
    edit(sidecar, provenance)
    path = tmp_path / "recorded_outputs.json"
    path.write_text(json.dumps(sidecar))
    monkeypatch.setattr(hf, "HfApi", no_remote)
    with pytest.raises(ValueError):
        push_to_hub(
            model, REPO, profile="A", provenance=provenance, license="mit", recorded_outputs=path
        )


def test_recorded_outputs_must_be_an_existing_json_object(model, tmp_path, monkeypatch):
    def no_remote(**kwargs):
        pytest.fail("an unreadable sidecar must never create or update a repository")

    monkeypatch.setattr(hf, "HfApi", no_remote)
    arguments = {"profile": "A", "provenance": {"source_checkpoint_sha256": "f" * 64}}
    path = tmp_path / "recorded_outputs.json"
    with pytest.raises(FileNotFoundError):
        push_to_hub(model, REPO, license="mit", recorded_outputs=path, **arguments)
    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError):
        push_to_hub(model, REPO, license="mit", recorded_outputs=path, **arguments)


@pytest.mark.parametrize("change", ["bfloat16", "scaled_tensor"])
def test_recorded_outputs_of_other_weights_are_never_published(
    model, tmp_path, monkeypatch, change
):
    def no_remote(**kwargs):
        pytest.fail("changed weights must never be published with the loaded sidecar")

    source = {"source_checkpoint_sha256": "f" * 64}
    loaded = from_pretrained(save_pretrained(model, tmp_path / "A", provenance=source, profile="A"))
    path = tmp_path / "recorded_outputs.json"
    path.write_text(json.dumps(_sidecar(loaded.provenance["state_sha256"])))
    if change == "bfloat16":
        loaded = loaded.to(torch.bfloat16)
    else:
        with torch.no_grad():
            next(tensor for tensor in loaded.conditional_head.parameters() if tensor.any()).mul_(
                1.05
            )
    monkeypatch.setattr(hf, "HfApi", no_remote)
    # Restating the source checkpoint is allowed; the sidecar still names other weights.
    with pytest.raises(ValueError):
        push_to_hub(
            loaded, REPO, profile="A", provenance=source, license="mit", recorded_outputs=path
        )


def test_profile_publication_keeps_other_cards_and_indexes_every_profile(model, monkeypatch):
    existing = [
        "README.md",
        "profiles/A/README.md",
        "profiles/A/config.json",
        "profiles/A/manifest.json",
        "profiles/A/model.safetensors",
        "profiles/A/recorded_outputs.json",
        "profiles/B/config.json",
        "profiles/B/manifest.json",
        "profiles/B/model.safetensors",
        "profiles/B/recorded_outputs.json",
    ]
    committed = {}

    class CollectionApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, **kwargs):
            pass

        def model_info(self, **kwargs):
            return SimpleNamespace(
                sha=COMMIT_A,
                card_data=hf.ModelCardData(license="apache-2.0"),
                siblings=[SimpleNamespace(rfilename=name) for name in existing],
            )

        def create_commit(self, **kwargs):
            committed.update({item.path_in_repo: item for item in kwargs["operations"]})
            return SimpleNamespace(oid=COMMIT_B)

    monkeypatch.setattr(hf, "HfApi", CollectionApi)
    push_to_hub(model, REPO, profile="B", license="apache-2.0", corpus_plan="B")
    assert not any(path.startswith("profiles/A/") for path in committed)
    # Republishing B without a sidecar removes B's previous one in the same commit.
    assert isinstance(committed["profiles/B/recorded_outputs.json"], hf.CommitOperationDelete)
    root = hf.ModelCard(committed["README.md"].path_or_fileobj.decode())
    card = hf.ModelCard(committed["profiles/B/README.md"].path_or_fileobj.decode())
    assert root.data.license == card.data.license == "apache-2.0"
    facts = _card_facts(root.text)
    assert "profiles/A/README.md" in facts["A"][0]
    assert "profiles/B/README.md" in facts["B"][0]
    assert json.loads(_card_facts(card.text)["plan.name"][0]) == "B"


@pytest.mark.parametrize(
    "profile,existing",
    [
        ("B", ["README.md", "config.json", "manifest.json", "model.safetensors"]),
        (
            None,
            [
                "README.md",
                "profiles/A/README.md",
                "profiles/A/config.json",
                "profiles/A/manifest.json",
                "profiles/A/model.safetensors",
            ],
        ),
    ],
    ids=["profile_beside_root_bundle", "root_bundle_beside_profiles"],
)
def test_publication_never_mixes_a_root_bundle_with_profiles(model, monkeypatch, profile, existing):
    class ExistingRepositoryApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, **kwargs):
            pass

        def model_info(self, **kwargs):
            return SimpleNamespace(
                sha=COMMIT_A,
                card_data=hf.ModelCardData(license="mit"),
                siblings=[SimpleNamespace(rfilename=name) for name in existing],
            )

        def create_commit(self, **kwargs):
            pytest.fail("the root README.md cannot be both a bundle's card and the index")

    monkeypatch.setattr(hf, "HfApi", ExistingRepositoryApi)
    with pytest.raises(ValueError):
        push_to_hub(model, REPO, profile=profile, license="mit")


def test_local_cards_refuse_a_root_bundle_beside_profiles(model, tmp_path):
    root = save_pretrained(model, tmp_path / "collection")
    save_pretrained(model, root / "profiles" / "A", profile="A")
    with pytest.raises(ValueError):
        write_model_cards(root, license="mit")
    assert not list(root.rglob("README.md"))


def _evaluation(*worlds):
    """A recorded joint holdout evaluation of ``worlds``, each ``{shard_seed, row}``."""
    return {
        "status": "recorded",
        "source": "terminal in-training evaluation",
        "step": 12,
        "leg": "holdout",
        "head": "joint",
        "eval_seed": 616161,
        "n_worlds": len(worlds),
        "n_samples": 4,
        "flow_steps": 16,
        "worlds": list(worlds),
        "metrics": {
            "joint/query/decomposition_energy_score": 0.25,
            "structure_head/cy/auroc": None,
        },
        "aggregation": {
            "joint/query/decomposition_energy_score": "world",
            "structure_head/cy/auroc": "pooled_edges",
        },
    }


def test_local_cards_describe_each_profile_and_index_the_collection(model, tmp_path, monkeypatch):
    from pymc_mfm.data.recipes import SHARD_PLANS, recipe_for

    # Row 3 of the first shard of plan A.
    evaluation = _evaluation({"shard_seed": 530200, "row": 3})
    metrics = evaluation["metrics"]
    provenance = {"source_checkpoint_sha256": "a" * 64, "training_evaluation": evaluation}
    save_pretrained(model, tmp_path / "profiles" / "A", profile="A", provenance=provenance)
    save_pretrained(model, tmp_path / "profiles" / "B", profile="B")
    with pytest.raises(ValueError):
        write_model_cards(tmp_path, license=" ")
    license = "not selected (publication pending)"
    with monkeypatch.context() as patch:
        _forbid_hub_imports(patch)
        paths = write_model_cards(tmp_path, license=license)
    cards = {
        path.relative_to(tmp_path).as_posix(): hf.ModelCard(path.read_text()) for path in paths
    }
    assert cards.keys() == {"README.md", "profiles/A/README.md", "profiles/B/README.md"}
    assert {card.data.license for card in cards.values()} == {license}
    root = _card_facts(cards["README.md"].text)
    assert all(f"profiles/{name}/README.md" in root[name][0] for name in ("A", "B"))
    for name in ("A", "B"):
        facts = _card_facts(cards[f"profiles/{name}/README.md"].text)
        # A profile label alone never names the training corpus.
        assert not any(key.startswith("plan.") for key in facts)
        assert json.loads(facts["model_config.n_controls"][0]) == model.config.n_controls
    recorded = _card_facts(cards["profiles/A/README.md"].text)
    assert {key: (json.loads(recorded[key][0]), recorded[key][1]) for key in metrics} == {
        key: (number, evaluation["aggregation"][key]) for key, number in metrics.items()
    }
    # Profile B records no training evaluation, so its card has no metric rows.
    assert not any("/" in key for key in _card_facts(cards["profiles/B/README.md"].text))
    written = {path: path.read_bytes() for path in tmp_path.rglob("README.md")}
    # A plan describes exactly one bundle, and only a profile training plan qualifies.
    with pytest.raises(ValueError):
        write_model_cards(tmp_path, license=license, corpus_plan="A")
    for corpus_plan in ("evaluation", "C", ["A"]):
        with pytest.raises(ValueError):
            write_model_cards(tmp_path / "profiles" / "A", license=license, corpus_plan=corpus_plan)
    assert {path: path.read_bytes() for path in tmp_path.rglob("README.md")} == written
    for name in ("A", "B"):
        [path] = write_model_cards(tmp_path / "profiles" / name, license=license, corpus_plan=name)
        text = hf.ModelCard(path.read_text()).text
        facts = _card_facts(text)
        plan = SHARD_PLANS[name]
        payload = json.loads(json.dumps(dict(recipe_for(plan.profile).payload)))
        expected = {
            "plan.worlds": plan.worlds,
            "plan.first_seed": plan.first_seed,
            "plan.last_seed": plan.seed(plan.n_shards - 1),
            "prior.edge_budget": payload["edge_budget"],
            "prior.n_treatments_active_range": payload["n_treatments_active_range"],
            "model_config.n_controls": model.config.n_controls,
            "parameters": sum(tensor.numel() for tensor in model.parameters()),
        }
        assert {key: json.loads(facts[key][0]) for key in expected} == expected
        # The regeneration command is a complete CLI call for the plan's shards.
        [command] = re.findall(r"`(python -m pymc_mfm\.data [^`]*)`", text)
        tokens = shlex.split(command)
        assert tokens[:4] == ["python", "-m", "pymc_mfm.data", "generate"]
        options = dict(zip(tokens[4::2], tokens[5::2], strict=True))
        assert options.keys() == {"--profile", "--shard", "--out"}
        assert options["--profile"] == name
    from_pretrained(tmp_path, profile="A")


def test_a_card_description_precedes_the_generated_sections_of_one_bundle(model, tmp_path):
    save_pretrained(model, tmp_path / "profiles" / "A", profile="A")
    save_pretrained(model, tmp_path / "profiles" / "B", profile="B")
    bundle = tmp_path / "profiles" / "A"
    plain = hf.ModelCard(write_model_cards(bundle, license="mit")[0].read_text()).text
    assert "Model description" not in plain
    written = {path: path.read_bytes() for path in tmp_path.rglob("README.md")}
    # Headings, issue numbers, harmless queries and variable names are prose, not
    # a credential-bearing URL.
    description = (
        "River test model from pull request #39.\n\n### Training\n\n4 updates on "
        "[toy worlds](https://hf.co/x?lang=en); set `HF_TOKEN` to load it."
    )
    for invalid in (
        "",
        " \n",
        ["River"],
        "Fetched with https://hf.co/x?token=abc",
        "See [the data](https://user:secret@hf.co/x).",
        "Bearer abcdefgh",
    ):
        with pytest.raises(ValueError):
            write_model_cards(bundle, license="mit", description=invalid)
    # A description names one bundle, never every profile of a collection.
    with pytest.raises(ValueError):
        write_model_cards(tmp_path, license="mit", description=description)
    assert {path: path.read_bytes() for path in tmp_path.rglob("README.md")} == written
    [path] = write_model_cards(bundle, license="mit", description=f"\n{description}\n\n")
    text = hf.ModelCard(path.read_text()).text
    head, architecture = text.split("## Architecture", 1)
    assert head.endswith(f"## Model description\n\n{description}\n\n")
    # The generated sections are unchanged by the description.
    assert plain.split("## Architecture", 1)[1] == architecture
    assert _card_facts(text) == _card_facts(plain)


#: Plan B's shards are seeded 530400..530719 and hold rows 0..1249; each plan
#: contradicts the second recorded world by one rule.
CONTRADICTED_PLANS = {
    "seed_outside_plan": ({"shard_seed": 530400, "row": 1249}, "A"),
    "row_outside_shard": ({"shard_seed": 530400, "row": 1250}, "B"),
}


@pytest.mark.parametrize(
    "world,corpus_plan", CONTRADICTED_PLANS.values(), ids=CONTRADICTED_PLANS.keys()
)
def test_cards_never_name_a_plan_that_the_recorded_worlds_contradict(
    model, tmp_path, monkeypatch, world, corpus_plan
):
    def no_remote(**kwargs):
        pytest.fail("a contradicted training plan must never create or update a repository")

    provenance = {"training_evaluation": _evaluation({"shard_seed": 530400, "row": 0}, world)}
    bundle = save_pretrained(model, tmp_path / "B", profile="B", provenance=provenance)
    with pytest.raises(ValueError):
        write_model_cards(bundle, license="mit", corpus_plan=corpus_plan)
    assert not (bundle / "README.md").exists()
    monkeypatch.setattr(hf, "HfApi", no_remote)
    with pytest.raises(ValueError):
        push_to_hub(
            model, REPO, profile="B", provenance=provenance, license="mit", corpus_plan=corpus_plan
        )


def test_cards_accept_the_last_row_of_the_last_shard(model, tmp_path):
    from pymc_mfm.data.recipes import SHARD_PLANS

    plan = SHARD_PLANS["B"]
    first = {"shard_seed": plan.seed(0), "row": 0}
    last = {
        "shard_seed": plan.seed(plan.n_shards - 1),
        "row": plan.n_cells * plan.draws_per_cell - 1,
    }
    provenance = {"training_evaluation": _evaluation(first, last)}
    bundle = save_pretrained(model, tmp_path / "B", profile="B", provenance=provenance)
    assert write_model_cards(bundle, license="mit", corpus_plan="B") == [bundle / "README.md"]


def test_cards_note_calibration_scores_of_one_class_edge_families(model, tmp_path):
    evaluation = _evaluation({"shard_seed": 530200, "row": 3})
    recorded = {
        # As recorded for A: no positive dz edge, so ece is 0.0 from negatives alone.
        "joint_router/edge/dz/ece": 0.0,
        "joint_router/edge/dz/mcb": None,
        "joint_router/edge/dz/brier": 0.0,
        "joint_router/edge/dz/n": 81.0,
        "joint_router/edge/dz/n_pos": 0.0,
        "structure_head/zb/ece": 0.01,
        "structure_head/zb/n": 4.0,
        "structure_head/zb/n_pos": 4.0,
        "joint_router/edge/cy/ece": 0.03,
        "joint_router/edge/cy/n": 172.0,
        "joint_router/edge/cy/n_pos": 80.0,
    }
    evaluation["metrics"].update(recorded)
    evaluation["aggregation"].update(dict.fromkeys(recorded, "pooled_edges"))
    provenance = {"training_evaluation": evaluation}
    bundle = save_pretrained(model, tmp_path / "A", profile="A", provenance=provenance)
    [path] = write_model_cards(bundle, license="mit")
    facts = _card_facts(hf.ModelCard(path.read_text()).text)
    notes = {key: facts[key][2] for key in evaluation["metrics"] if facts[key][2]}
    assert notes == {
        "joint_router/edge/dz/ece": "no positive edges — calibration not evidenced",
        "joint_router/edge/dz/mcb": "no positive edges — calibration not evidenced",
        "structure_head/zb/ece": "no negative edges — calibration not evidenced",
    }
    assert json.loads(facts["joint_router/edge/dz/ece"][0]) == 0.0


def test_local_cards_replace_cached_symlinks_without_touching_shared_blobs(model, tmp_path):
    export = save_pretrained(model, tmp_path / "export", profile="A")
    (export / "README.md").write_text("card that another revision shares\n")
    # A Hub cache snapshot links each file to a content-addressed blob, which every
    # revision holding the same bytes shares.
    blobs = tmp_path / "cache" / "blobs"
    snapshot = tmp_path / "cache" / "snapshots" / COMMIT_A
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for file in export.iterdir():
        blob = blobs / hashlib.sha256(file.read_bytes()).hexdigest()
        blob.write_bytes(file.read_bytes())
        (snapshot / file.name).symlink_to(blob)
    shared = {blob: blob.read_bytes() for blob in blobs.iterdir()}
    [card] = write_model_cards(snapshot, license="mit")
    assert {blob: blob.read_bytes() for blob in blobs.iterdir()} == shared
    assert hf.ModelCard(card.read_text()).data.license == "mit"
    # Only the card's entry is replaced, and no temporary file is left behind.
    assert {path.name: path.is_symlink() for path in snapshot.iterdir()} == {
        "README.md": False,
        "config.json": True,
        "manifest.json": True,
        "model.safetensors": True,
    }
