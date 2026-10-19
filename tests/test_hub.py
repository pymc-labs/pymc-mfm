"""Hub contracts use local artifacts and isolated transports, never real publication."""

import fnmatch
import json
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import HubArtifactError, from_pretrained, list_profiles, push_to_hub

hf = pytest.importorskip("huggingface_hub")
hf_errors = pytest.importorskip("huggingface_hub.errors")

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
REPO = "test-owner/mfm-artifacts"


def _assert_same_draws(original, restored, batch):
    torch.manual_seed(19)
    expected, expected_graphs = original.sample_disentangled(batch, 2)
    torch.manual_seed(19)
    actual, actual_graphs = restored.sample_disentangled(batch, 2)
    # Deserialized tensors can take a different x86 kernel path than freshly
    # built ones (1 ULP); tensors are compared bit-for-bit elsewhere.
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_graphs, expected_graphs, rtol=0, atol=0)


def test_local_and_cli_profile_layout_do_not_require_hub(model, batch, tmp_path, monkeypatch):
    import pymc_mfm.inference.hub as transport

    save_pretrained(model, tmp_path / "profiles" / "A", profile="A")
    save_pretrained(model, tmp_path / "profiles" / "B", profile="B")

    def unavailable():
        pytest.fail("a local/CLI-downloaded directory must not require Hub transport")

    monkeypatch.setattr(transport, "_hub", unavailable)
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
    import pymc_mfm.inference.hub as transport

    def no_transport():
        pytest.fail("missing license approval must fail before any Hub operation")

    with monkeypatch.context() as patch:
        patch.setattr(transport, "_hub", no_transport)
        with pytest.raises(ValueError):
            push_to_hub(model, REPO)
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
    commit = push_to_hub(
        model,
        REPO,
        revision="release",
        token=token,
        profile="A",
        provenance=provenance,
        license="apache-2.0",
    )
    assert commit.oid == COMMIT_B
    assert [name for name, _ in events] == ["create_repo", "create_commit"]
    assert events[0][1]["private"] is True
    restored = from_pretrained(published, profile="A")
    _assert_same_draws(model, restored, batch)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    card = hf.ModelCard((published / "README.md").read_text())
    assert card.data.license == "apache-2.0"
    assert card.data.library_name == "pymc-mfm"
    assert not getattr(card.data, "pipeline_tag", None)
    assert restored.provenance["source_checkpoint_sha256"] == "f" * 64


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
