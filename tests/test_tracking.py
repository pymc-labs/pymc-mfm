"""Consumer contracts for durable optional tracking and numerical independence."""

import builtins
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mfm.model.model import MFM
from pymc_mfm.training import MlflowTrainingObserver, TrackingConfig, train
from pymc_mfm.training import trainer as trainer_module
from pymc_mfm.training.checkpoint import file_sha256
from pymc_mfm.training.stream import batches_for_rows


@pytest.fixture(autouse=True)
def preserve_python_numpy_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_installed_code_identity_ignores_an_enclosing_applications_git_history(
    tmp_path, monkeypatch
):
    from pymc_mfm.training import _provenance

    module = tmp_path / ".venv/lib/site-packages/pymc_mfm/training/_provenance.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Installed source content\n")
    monkeypatch.setattr(_provenance, "__file__", str(module))
    before = _provenance.code_record()
    subprocess.run(
        ["git", "-c", "init.templateDir=", "init", "-q", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=MFM test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "Unrelated app",
        ],
        check=True,
        capture_output=True,
    )
    assert _provenance.code_record() == before
    assert before["revision"] is None
    module.write_text("# Different installed source content\n")
    assert _provenance.code_record()["source_sha256"] != before["source_sha256"]


def events(path):
    return [
        json.loads(file.read_text())
        for file in sorted((path / "tracking" / "events").glob("*.json"))
    ]


def artifact_paths(client, run_id, prefix=""):
    paths = set()
    for artifact in client.list_artifacts(run_id, prefix):
        if artifact.is_dir:
            paths.update(artifact_paths(client, run_id, artifact.path))
        else:
            paths.add(artifact.path)
    return paths


def tiny_run(path, *, tracking=None, corpus=None, **kwargs):
    return train(
        [
            make_synthetic_corpus(n_cells=4, draws_per_cell=1, context_lengths=(6,), seed=8)
            if corpus is None
            else corpus
        ],
        out_dir=path,
        model_config=TINY,
        training=reduced(
            steps=2,
            eval_every=1,
            eval_worlds=1,
            eval_samples=2,
            milestone_steps=(1,),
            prefetch_batches=0,
        ),
        device="cpu",
        tracking=tracking,
        **kwargs,
    )


def assert_scalar_histories(client, run_id, records):
    expected = defaultdict(list)
    for record in records:
        for scalar in record["scalars"]:
            expected[scalar["key"]].append(
                (scalar["step"], record["timestamp_ms"], scalar["value"])
            )
    for key, history in expected.items():
        actual = client.get_metric_history(run_id, key)
        assert [(item.step, item.timestamp, item.value) for item in actual] == history
    assert set(client.get_run(run_id).data.metrics) == set(expected)


def test_actual_client_histories_reports_and_checkpoint_versions(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    out = tmp_path / "tracked"
    result = tiny_run(
        out,
        corpus_provenance=[
            {"data_origin": "synthetic", "profile_note": "test-generated, not A/B weights"}
        ],
    )
    summary = result.run["tracking"]
    assert summary["status"] == summary["lifecycle_status"] == "finished"
    assert summary["delivery_status"] == "healthy"
    assert summary["tracking_uri"] == f"sqlite:///{out / 'tracking' / 'mlflow.db'}"
    client = mlflow.MlflowClient(tracking_uri=summary["tracking_uri"])
    run_id = summary["run_id"]
    remote = client.get_run(run_id)
    assert remote.info.status == "FINISHED"
    assert remote.data.tags["mfm.local_run_id"] == result.run["run_id"]
    records = events(out)

    def numeric_fields(mapping, prefix=""):
        for key, value in mapping.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            name = f"{prefix}/{escaped}" if prefix else escaped
            if isinstance(value, dict):
                yield from numeric_fields(value, name)
            elif isinstance(value, (bool, int, float)) and math.isfinite(value):
                yield name, float(value)

    local_steps = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    delivered_steps = [event for event in records if event["kind"] == "step"]
    for original, event in zip(local_steps, delivered_steps, strict=True):
        assert event["payload"] == original
        assert {item["original"]: item["value"] for item in event["scalars"]} == {
            "train/" + name: value for name, value in numeric_fields(original)
        }
    local_evaluations = [
        json.loads(line) for line in (out / "evaluations.jsonl").read_text().splitlines()
    ]
    delivered_evaluations = [event for event in records if event["kind"] == "evaluation"]
    for original, event in zip(local_evaluations, delivered_evaluations, strict=True):
        assert event["payload"] == original
        scalars = {item["original"]: item["value"] for item in event["scalars"]}
        for name, value in original["flat"].items():
            if value is not None and math.isfinite(value):
                assert scalars["evaluation/" + name] == value
        for name, value in numeric_fields(original["selection"], "selection"):
            assert scalars["evaluation/" + name] == value
    assert_scalar_histories(client, run_id, records)
    configuration = next(event["payload"] for event in records if event["kind"] == "configuration")
    assert configuration["training_config"] == result.run["training_config"]
    assert configuration["model_config"] == TINY.to_dict()
    assert configuration["code"]["source_sha256"] == result.run["code"]["source_sha256"]
    assert (
        configuration["corpus"][0]["actual_inputs"]["content_sha256"]
        == result.run["corpus"][0]["actual_inputs"]["content_sha256"]
    )
    names = json.loads((out / "tracking" / "names.json").read_text())
    for key, metadata in names.items():
        if metadata["kind"] == "parameter":
            assert key in remote.data.params
    paths = artifact_paths(client, run_id)
    for event_path in (out / "tracking" / "events").glob("*.json"):
        downloaded = Path(client.download_artifacts(run_id, "events/" + event_path.name))
        assert json.loads(downloaded.read_text()) == json.loads(event_path.read_text())
    checkpoints = [event["payload"] for event in records if event["kind"] == "checkpoint"]
    last = [record for record in checkpoints if record["kind"] == "last"]
    assert [record["step"] for record in last] == [1, 2]
    assert last[0]["artifact"]["path"] != last[1]["artifact"]["path"]
    for record in checkpoints:
        assert file_sha256(out / "tracking" / record["artifact"]["path"]) == record["sha256"]
        assert record["artifact"]["path"] in paths
    reports = [event["payload"] for event in records if event["kind"] == "diagnostic"]
    assert {report["step"] for report in reports} == {1, 2}
    # Per-world report tables must not multiply run-level chart series or
    # re-log embedded historical records at the current evaluation step.
    assert not any(
        metadata["original"].startswith("diagnostic/")
        for key, metadata in names.items()
        if key in remote.data.metrics
    )
    assert all(not event["scalars"] for event in records if event["kind"] == "diagnostic")
    assert not [
        event for event in records if event["kind"] in ("diagnostic_failure", "artifact_policy")
    ]
    for report in reports:
        manifest = report["manifest"]
        assert manifest["metric_representation"] == "per_world_table"
        assert manifest["provenance"]["checkpoint_sha256"] == next(
            record["sha256"] for record in last if record["step"] == report["step"]
        )
        assert manifest["provenance"]["training_local_run_id"] == result.run["run_id"]
        prefix = report["artifact_path"]
        assert prefix + "/manifest.json" in paths
        for artifact in manifest["artifacts"]:
            remote_path = prefix + "/" + artifact["path"]
            assert remote_path in paths
            downloaded = Path(client.download_artifacts(run_id, remote_path))
            assert file_sha256(downloaded) == artifact["sha256"]
    assert any(path.endswith("/report.html") for path in paths)
    assert any(path.endswith("/model.safetensors") for path in paths)


def test_tracking_on_off_preserves_weights_rng_consumption_and_predictions(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    corpus = make_synthetic_corpus(n_cells=4, draws_per_cell=1, context_lengths=(6,), seed=9)
    results = []
    states = []
    for enabled in (False, True):
        random.seed(417)
        np.random.seed(718)
        result = tiny_run(
            tmp_path / str(enabled),
            tracking=TrackingConfig(enabled=enabled),
            corpus=corpus,
            corpus_provenance=[{"data_origin": "synthetic"}],
        )
        results.append(result)
        states.append((random.getstate(), np.random.get_state(), torch.random.get_rng_state()))
    for name, parameter in results[0].model.state_dict().items():
        torch.testing.assert_close(parameter, results[1].model.state_dict()[name], rtol=0, atol=0)
    assert results[0].run["consumed"] == results[1].run["consumed"]
    assert results[0].run["corpus"] == results[1].run["corpus"]
    assert states[0][0] == states[1][0]
    assert states[0][1][0] == states[1][1][0]
    np.testing.assert_array_equal(states[0][1][1], states[1][1][1])
    assert states[0][1][2:] == states[1][1][2:]
    torch.testing.assert_close(states[0][2], states[1][2], rtol=0, atol=0)
    for kind in ("step", "evaluation"):
        a = [event for event in events(results[0].run_dir) if event["kind"] == kind]
        b = [event for event in events(results[1].run_dir) if event["kind"] == kind]
        for left, right in zip(a, b, strict=True):
            if kind == "step":
                for key in (
                    "loss",
                    "flow_loss",
                    "context_flow_loss",
                    "graph_conditional_flow_loss",
                    "graph_conditional_context_flow_loss",
                    "struct_loss",
                    "lr",
                    "batch_rows",
                ):
                    assert left["payload"][key] == right["payload"][key]
            else:
                assert left["payload"]["flat"] == right["payload"]["flat"]
                assert left["payload"]["selection"] == right["payload"]["selection"]
                for leg in ("holdout", "seen"):
                    assert (
                        left["payload"]["legs"][leg]["rows"]
                        == right["payload"]["legs"][leg]["rows"]
                    )
    row = int(np.flatnonzero(corpus["is_val"])[0])
    batch = batches_for_rows([corpus], [(0, row)], device="cpu")[0][0]
    predictions = []
    for result in results:
        torch.manual_seed(161)
        predictions.append(result.model.sample_disentangled(batch, 2))
    for a, b in zip(predictions[0], predictions[1], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert results[0].run["tracking"]["status"] == "disabled"
    assert results[0].run["tracking"]["run_id"] is None


def test_unrelated_active_run_is_neither_attached_nor_terminated(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'shared.db'}"
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "caller", artifact_location=(tmp_path / "caller-artifacts").as_uri()
    )
    try:
        with mlflow.start_run(experiment_id=experiment) as caller:
            result = tiny_run(tmp_path / "training", tracking=TrackingConfig(diagnostics=False))
            assert mlflow.active_run().info.run_id == caller.info.run_id
            assert result.run["tracking"]["run_id"] != caller.info.run_id
            assert client.get_run(caller.info.run_id).info.status == "RUNNING"
            assert client.get_run(caller.info.run_id).data.metrics == {}
            assert result.run["tracking"]["tracking_uri"] == uri
    finally:
        mlflow.set_tracking_uri(old_uri)


@pytest.mark.parametrize("mode", ["disabled", "missing_dependency", "backend_unavailable"])
def test_absent_tracking_never_prevents_numerical_training_or_hides_status(
    tmp_path, monkeypatch, mode
):
    original_import = builtins.__import__

    def no_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            if mode == "disabled":
                raise AssertionError("disabled tracking imported MLflow")
            raise ImportError("missing optional dependency")
        return original_import(name, *args, **kwargs)

    if mode in ("disabled", "missing_dependency"):
        monkeypatch.setattr(builtins, "__import__", no_mlflow)
    else:
        mlflow = pytest.importorskip("mlflow")

        def unavailable(*args, **kwargs):
            raise RuntimeError("password=DO_NOT_SAVE_BACKEND_SECRET")

        monkeypatch.setattr(mlflow, "MlflowClient", unavailable)
    config = TrackingConfig(
        enabled=mode != "disabled",
        diagnostics=False,
        tracking_uri="https://user:DO_NOT_SAVE_URI_SECRET@example.invalid/?token=DO_NOT_SAVE_QUERY_SECRET",
    )
    if mode == "disabled":
        result = tiny_run(tmp_path / mode, tracking=config)
    else:
        with pytest.warns(RuntimeWarning):
            result = tiny_run(tmp_path / mode, tracking=config)
    assert result.status == "finished"
    status = json.loads((result.run_dir / "tracking" / "state.json").read_text())
    assert status["lifecycle_status"] == "finished"
    assert status["status"] == ("disabled" if mode == "disabled" else "degraded")
    assert status["run_id"] is None
    assert [event["step"] for event in events(result.run_dir) if event["kind"] == "step"] == [1, 2]
    for path in (result.run_dir / "tracking").rglob("*.json*"):
        assert "DO_NOT_SAVE" not in path.read_text()
    if mode != "disabled":
        failures = [
            json.loads(line)
            for line in (result.run_dir / "tracking" / "delivery_failures.jsonl")
            .read_text()
            .splitlines()
        ]
        assert any(failure["operation"] == "start_run" for failure in failures)
        assert any(
            failure["artifact"] and "checkpoint" in failure["artifact"] for failure in failures
        )


def test_backend_artifact_failure_keeps_local_versions_and_marks_delivery_failed(
    tmp_path, monkeypatch
):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    def fail_delivery(*args, **kwargs):
        raise OSError("DO_NOT_SAVE_FAILURE_SECRET")

    monkeypatch.setattr(mlflow.MlflowClient, "log_artifact", fail_delivery)
    with pytest.warns(RuntimeWarning):
        result = tiny_run(tmp_path / "run", tracking=TrackingConfig(diagnostics=False))
    assert result.status == "finished"
    assert result.run["tracking"]["status"] == "degraded"
    client = mlflow.MlflowClient(tracking_uri=result.run["tracking"]["tracking_uri"])
    remote = client.get_run(result.run["tracking"]["run_id"])
    assert remote.info.status == "FAILED"
    assert remote.data.tags["mfm.lifecycle_status"] == "finished"
    assert remote.data.tags["mfm.delivery_status"] == "degraded"
    assert_scalar_histories(client, remote.info.run_id, events(result.run_dir))
    for event in events(result.run_dir):
        if event["kind"] == "checkpoint":
            artifact = event["payload"]["artifact"]
            assert file_sha256(result.run_dir / "tracking" / artifact["path"]) == artifact["sha256"]
    assert (
        "DO_NOT_SAVE" not in (result.run_dir / "tracking" / "delivery_failures.jsonl").read_text()
    )


@pytest.mark.parametrize(
    ("failure", "status", "remote_status"),
    [(RuntimeError, "failed", "FAILED"), (KeyboardInterrupt, "interrupted", "KILLED")],
)
def test_numerical_failure_and_interrupt_finalize_owned_run(
    tmp_path, monkeypatch, failure, status, remote_status
):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    original = MFM.loss_terms
    calls = 0

    def fail_second_update(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure("stopped update")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MFM, "loss_terms", fail_second_update)
    out = tmp_path / status
    with pytest.raises(failure):
        tiny_run(out, tracking=TrackingConfig(diagnostics=False))
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == run["tracking"]["lifecycle_status"] == status
    assert run["final_step"] == run["consumed"]["updates"] == 1
    client = mlflow.MlflowClient(tracking_uri=run["tracking"]["tracking_uri"])
    assert client.get_run(run["tracking"]["run_id"]).info.status == remote_status
    assert [event["step"] for event in events(out) if event["kind"] == "step"] == [1]
    assert not (out / "model").exists()


def test_model_initialization_failure_has_early_run_identity(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    def fail_model(*args, **kwargs):
        raise RuntimeError("model construction failed")

    monkeypatch.setattr(trainer_module, "MFM", fail_model)
    out = tmp_path / "startup"
    with pytest.raises(RuntimeError):
        tiny_run(out, tracking=TrackingConfig(diagnostics=False))
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["consumed"]["updates"] == 0
    client = mlflow.MlflowClient(tracking_uri=run["tracking"]["tracking_uri"])
    assert client.get_run(run["tracking"]["run_id"]).info.status == "FAILED"
    records = events(out)
    assert records[0]["payload"]["status"] == "running"
    assert records[-1]["payload"]["status"] == "failed"
    assert not any(event["kind"] == "step" for event in records)


def test_original_names_nonfinite_and_structured_values_are_lossless(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    observer = MlflowTrainingObserver(tmp_path, TrackingConfig(diagnostics=False))
    observer.on_run_start({"run_id": "independent-event-contract", "status": "running"})
    long_value = list(range(2000))
    observer.on_run_config(
        {
            "run_id": "independent-event-contract",
            "full_values": long_value,
            "api_token": "DO_NOT_SAVE",
        }
    )
    record = {
        "step": 7,
        "loss[a]": 1.25,
        "loss_a_": 2.5,
        "x" * 400: 9.0,
        "x" * 399 + "y": 3.0,
        "ranks": [[1, 3], [0, 4]],
        "bad": float("nan"),
        "absent": None,
    }
    record.update(
        api_token=847293.125, apiKey="DO_NOT_SAVE_CAMEL", **{"X-Api-Key": "DO_NOT_SAVE_HEADER"}
    )
    observer.on_step(record)
    observer.on_run_end(
        {"run_id": "independent-event-contract", "status": "finished", "final_step": 7}
    )
    client = mlflow.MlflowClient(tracking_uri=observer.tracking_uri)
    records = events(tmp_path)
    assert_scalar_histories(client, observer.run_id, records)
    step = next(event for event in records if event["kind"] == "step")
    assert step["payload"]["ranks"] == record["ranks"]
    assert step["payload"]["bad"] == {"status": "nonfinite", "value": "nan"}
    assert step["payload"]["absent"] is None
    assert len({scalar["key"] for scalar in step["scalars"]}) == 5
    assert not any(
        scalar["original"].endswith(("/bad", "/absent", "/ranks")) for scalar in step["scalars"]
    )
    config = next(event["payload"] for event in records if event["kind"] == "configuration")
    assert config["full_values"] == long_value
    assert config["api_token"] == "[redacted]"
    assert step["payload"]["api_token"] == "[redacted]"
    assert step["payload"]["apiKey"] == step["payload"]["X-Api-Key"] == "[redacted]"
    assert not any(item["value"] == 847293.125 for item in step["scalars"])
    assert all(len(key) <= 250 for key in client.get_run(observer.run_id).data.metrics)


def test_private_report_uploads_only_safe_receipt_and_keeps_complete_local_evidence(
    tmp_path, monkeypatch
):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    report = tmp_path / "report"
    (report / "tables" / "nested").mkdir(parents=True)
    table = report / "tables" / "nested" / "ranks.json"
    table.write_text(json.dumps({"ranks": [0, 2, 1], "reference_status": "not_available"}))
    private = report / "observations.npz"
    np.savez(private, observed=np.array([12.0, 13.0]))
    stray = report / "business.csv"
    stray.write_text("do not upload unrelated files")
    observer = MlflowTrainingObserver(tmp_path / "run")
    observer.on_run_start({"run_id": "allowlist", "status": "running"})
    manifest = {
        "provenance": {
            "source_step": 4,
            "checkpoint_sha256": "a" * 64,
            "input_support": {"query": {"mean": 879031.25}},
        },
        "dimensions": {
            "window": "query",
            "head": "conditional",
            "mode": "model_average",
            "units": "outcome",
            "data_split": "heldout",
        },
        "metrics": {"coverage": None, "closure": 0.125},
        "statuses": {"coverage": {"status": "unavailable", "reason": "truth_missing"}},
        "artifacts": [
            {"path": "tables/nested/ranks.json", "sha256": file_sha256(table), "kind": "table"},
            {
                "path": "observations.npz",
                "sha256": file_sha256(private),
                "kind": "array",
                "sensitive": True,
            },
        ],
    }
    observer.on_diagnostic(manifest, report)
    (observer.out_dir / "diagnostics.jsonl").write_text(json.dumps({"manifest": manifest}) + "\n")
    observer.on_run_end({"run_id": "allowlist", "status": "finished", "final_step": 4})
    client = mlflow.MlflowClient(tracking_uri=observer.tracking_uri)
    paths = artifact_paths(client, observer.run_id)
    assert not any(
        path.endswith(("observations.npz", "business.csv", "ranks.json", "diagnostics.jsonl"))
        for path in paths
    )
    assert client.get_run(observer.run_id).data.metrics == {}
    for path in paths:
        downloaded = Path(client.download_artifacts(observer.run_id, path))
        assert "879031.25" not in downloaded.read_text()
    local = next(
        event["payload"] for event in events(observer.out_dir) if event["kind"] == "diagnostic"
    )
    assert local["manifest"] == manifest
    policy = [
        event["payload"]
        for event in events(observer.out_dir)
        if event["kind"] == "diagnostic_receipt"
    ]
    assert policy[0]["status"] == "withheld_sensitive"
    assert set(policy[0]) == {"step", "checkpoint_sha256", "report_sha256", "status", "reason"}
    assert observer.summary["status"] == "finished"


def test_tampered_or_escaping_report_artifact_is_not_delivered(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    report = tmp_path / "report"
    report.mkdir()
    (report / "changed.json").write_text('{"changed": true}')
    (tmp_path / "outside.json").write_text('{"private": true}')
    observer = MlflowTrainingObserver(tmp_path / "run")
    observer.on_run_start({"run_id": "unsafe-report", "status": "running"})
    manifest = {
        "provenance": {"source_step": 1, "synthetic_training_evaluation": True},
        "metrics": {},
        "artifacts": [
            {"path": "changed.json", "sha256": "0" * 64},
            {"path": "../outside.json", "sha256": file_sha256(tmp_path / "outside.json")},
        ],
    }
    with pytest.warns(RuntimeWarning):
        observer.on_diagnostic(manifest, report)
    observer.on_run_end({"run_id": "unsafe-report", "status": "finished", "final_step": 1})
    client = mlflow.MlflowClient(tracking_uri=observer.tracking_uri)
    assert client.get_run(observer.run_id).info.status == "FAILED"
    assert not any(
        path.endswith(("changed.json", "outside.json"))
        for path in artifact_paths(client, observer.run_id)
    )
    failures = [
        json.loads(line)
        for line in (observer.root / "delivery_failures.jsonl").read_text().splitlines()
    ]
    assert {failure["operation"] for failure in failures} == {
        "diagnostic_artifact",
        "diagnostic_allowlist",
    }


def test_interrupt_during_terminal_artifact_upload_is_honestly_finalized(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    original = mlflow.MlflowClient.log_artifact
    interrupted = False

    def interrupt_final_delivery(self, run_id, local_path, artifact_path=None):
        nonlocal interrupted
        if not interrupted and artifact_path and "terminal_evidence" in artifact_path:
            interrupted = True
            raise KeyboardInterrupt()
        return original(self, run_id, local_path, artifact_path=artifact_path)

    monkeypatch.setattr(mlflow.MlflowClient, "log_artifact", interrupt_final_delivery)
    out = tmp_path / "interrupted-delivery"
    with pytest.raises(KeyboardInterrupt), pytest.warns(RuntimeWarning):
        tiny_run(out, tracking=TrackingConfig(diagnostics=False))
    run = json.loads((out / "run.json").read_text())
    assert run["final_step"] == 2
    assert run["status"] == run["tracking"]["lifecycle_status"] == "interrupted"
    client = mlflow.MlflowClient(tracking_uri=run["tracking"]["tracking_uri"])
    remote = client.get_run(run["tracking"]["run_id"])
    assert remote.info.status == "KILLED"
    assert remote.data.tags["mfm.lifecycle_status"] == "interrupted"
    assert events(out)[-1]["payload"]["status"] == "interrupted"


def test_first_step_nonfinite_trace_and_loss_are_retained(tmp_path, monkeypatch):
    from pymc_mfm.training import TrainingError

    original = MFM.loss_terms

    def poison_first_loss(self, *args, **kwargs):
        terms, trace = original(self, *args, **kwargs)
        terms["loss"] = terms["loss"] * float("nan")
        trace["loss_reconstruction"]["loss"] = terms["loss"]
        return terms, trace

    monkeypatch.setattr(MFM, "loss_terms", poison_first_loss)
    out = tmp_path / "nonfinite"
    with pytest.raises(TrainingError):
        tiny_run(out, tracking=TrackingConfig(enabled=False))
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed"
    trace = json.loads((out / "objective_trace.json").read_text())
    assert trace["loss_reconstruction"]["loss"]["status"] == "nonfinite"
    loss = json.loads((out / "metrics.jsonl").read_text())
    assert loss["loss"] is None and loss["nonfinite"]["loss"] == "nan"
    delivered = next(event for event in events(out) if event["kind"] == "step")
    assert delivered["payload"]["nonfinite"]["loss"] == "nan"
    assert not any(item["original"] == "train/loss" for item in delivered["scalars"])


def test_diagnostic_dimension_boundaries_and_per_metric_steps_do_not_collide(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    observer = MlflowTrainingObserver(tmp_path)
    observer.on_run_start({"run_id": "dimensions", "status": "running"})
    for units, world, step, value in (("USD/day", "x", 5, 0.25), ("USD", "day/x", 9, 0.75)):
        manifest = {
            "provenance": {"synthetic_training_evaluation": True, "world_id": world},
            "dimensions": {
                "units": units,
                "window": "query",
                "head": "conditional",
                "mode": "model_average",
            },
            "metrics": {"score": value},
            "metric_dimensions": {"score": {"source_step": step}},
            "artifacts": [],
        }
        observer.on_diagnostic(manifest, tmp_path)
    observer.on_run_end({"run_id": "dimensions", "status": "finished", "final_step": 9})
    client = mlflow.MlflowClient(tracking_uri=observer.tracking_uri)
    histories = [
        client.get_metric_history(observer.run_id, key)
        for key in client.get_run(observer.run_id).data.metrics
    ]
    assert len(histories) == 2
    assert sorted((history[0].step, history[0].value) for history in histories) == [
        (5, 0.25),
        (9, 0.75),
    ]


@pytest.mark.parametrize("secret_location", ["manifest", "record"])
def test_credential_bearing_report_withholds_html_and_figures_too(
    tmp_path, monkeypatch, secret_location
):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    report = tmp_path / "report"
    report.mkdir()
    html = report / "report.html"
    html.write_text("<pre>DO_NOT_SAVE_API_SECRET</pre>")
    evidence = report / "evidence.json"
    evidence.write_text(
        json.dumps(
            {"reference": {"label": "https://user:DO_NOT_SAVE_API_SECRET@example.invalid/fit"}}
            if secret_location == "record"
            else {}
        )
    )
    provenance = {"synthetic_training_evaluation": True}
    if secret_location == "manifest":
        provenance["apiKey"] = "DO_NOT_SAVE_API_SECRET"
    observer = MlflowTrainingObserver(tmp_path / "run")
    observer.on_run_start({"run_id": "credentials", "status": "running"})
    observer.on_diagnostic(
        {
            "provenance": provenance,
            "metrics": {"score": 0.25},
            "artifacts": [
                {"path": "report.html", "sha256": file_sha256(html), "sensitive": False},
                {"path": "evidence.json", "sha256": file_sha256(evidence), "sensitive": False},
            ],
        },
        report,
    )
    observer.on_run_end({"run_id": "credentials", "status": "finished"})
    client = mlflow.MlflowClient(tracking_uri=observer.tracking_uri)
    paths = artifact_paths(client, observer.run_id)
    assert not any(path.endswith("report.html") for path in paths)
    assert client.get_run(observer.run_id).data.metrics == {}
    for path in paths:
        assert (
            "DO_NOT_SAVE" not in Path(client.download_artifacts(observer.run_id, path)).read_text()
        )
    receipt = next(
        event["payload"]
        for event in events(observer.out_dir)
        if event["kind"] == "diagnostic_receipt"
    )
    assert receipt["status"] == "withheld_credentials"


@pytest.mark.parametrize("hook", ["on_step", "on_diagnostic"])
def test_observer_exception_credentials_never_enter_failure_artifacts(tmp_path, monkeypatch, hook):
    from pymc_mfm.training import TrainingObserver

    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    class FailingObserver(TrainingObserver):
        pass

    def fail(*args):
        raise RuntimeError("Authorization: Bearer DO_NOT_SAVE_CALLER_SECRET")

    setattr(FailingObserver, hook, fail)

    out = tmp_path / "caller-failure"
    with pytest.raises(RuntimeError):
        tiny_run(
            out,
            tracking=TrackingConfig(diagnostics=hook == "on_diagnostic"),
            observers=[FailingObserver()],
        )
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and run["error_type"] == "RuntimeError"
    assert "DO_NOT_SAVE" not in json.dumps(run)
    if hook == "on_diagnostic":
        diagnostic = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
        assert "manifest" in diagnostic
    client = mlflow.MlflowClient(tracking_uri=run["tracking"]["tracking_uri"])
    for path in artifact_paths(client, run["tracking"]["run_id"]):
        if path.endswith((".json", ".jsonl")):
            assert (
                "DO_NOT_SAVE"
                not in Path(client.download_artifacts(run["tracking"]["run_id"], path)).read_text()
            )
