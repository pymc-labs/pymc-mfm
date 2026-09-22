"""Read complete tracked histories and saved reports without resampling.

Install ``pymc-mfm[training]``. Canonical training enables MLflow by default:

    export MLFLOW_TRACKING_URI=sqlite:///outputs/mlflow.db
    python -m pymc_mfm.training --corpus generated.npz --out-dir outputs/seed31
    python examples/tracking.py outputs/seed31 --metric train/loss --download-reports downloads

These commands select outputs/mlflow.db. Readback uses the run's saved backend
unless ``--tracking-uri`` overrides it. To compare several runs in one UI, select
the same output-local SQLite URI for all training runs:

    export MLFLOW_TRACKING_URI=sqlite:////absolute/worktree/outputs/mlflow.db
    python -m pymc_mfm.training --corpus generated.npz --out-dir outputs/seed32 --seed 32
    mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1

A configured server uses the same client path (no hosted server is required):

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python -m pymc_mfm.training --corpus generated.npz --out-dir outputs/server-run

Set normal MLflow credential environment variables separately; do not put
secrets in run names/provenance. ``--no-tracking`` is the explicit opt-out;
``--no-diagnostics`` retains all existing evaluator results without reports.
Generated corpora with verified synthetic provenance permit complete report
uploads. Unknown/business origin keeps sensitive report artifacts local.

No metric filter means every full scalar history is printed, using original
names from the name table rather than guessed sanitized keys. Structured
arrays, undefined reasons and figure inputs are available in versioned events
and report artifacts, not misleading scalar averages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def artifact_files(client, run_id, path=""):
    for entry in client.list_artifacts(run_id, path):
        if entry.is_dir:
            yield from artifact_files(client, run_id, entry.path)
        else:
            yield entry.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--tracking-uri", help="override the saved URI; authentication stays in environment"
    )
    parser.add_argument("--metric", help="one complete original name, e.g. train/loss")
    parser.add_argument("--download-reports", type=Path)
    args = parser.parse_args(argv)
    state = json.loads((args.run_dir / "tracking" / "state.json").read_text())
    if state["run_id"] is None:
        raise SystemExit(
            (
                "No MLflow run exists: "
                f"{state['status']}"
                "; inspect local tracking/events and delivery_failures.jsonl"
            )
        )
    from mlflow import MlflowClient

    uri = args.tracking_uri or state["tracking_uri"]
    client = MlflowClient(tracking_uri=uri)
    run_id = state["run_id"]
    run = client.get_run(run_id)
    print(
        json.dumps({"run_id": run_id, "status": run.info.status, "tags": run.data.tags}, indent=2)
    )
    names_path = client.download_artifacts(run_id, "tracking/names.json")
    names = json.loads(Path(names_path).read_text())
    for key, metadata in sorted(names.items(), key=lambda item: item[1]["original"]):
        if metadata["kind"] != "metric" or (args.metric and args.metric != metadata["original"]):
            continue
        history = client.get_metric_history(run_id, key)
        print(
            json.dumps(
                {
                    "metric": metadata["original"],
                    "source_path": metadata["path"],
                    "history": [
                        {"step": point.step, "timestamp_ms": point.timestamp, "value": point.value}
                        for point in history
                    ],
                }
            )
        )
    artifacts = list(artifact_files(client, run_id))
    print(json.dumps({"artifacts": artifacts}, indent=2))
    configuration = [
        path for path in artifacts if path.startswith("events/") and "_configuration_" in path
    ]
    for path in configuration:
        full = json.loads(Path(client.download_artifacts(run_id, path)).read_text())
        print(json.dumps({"full_configuration": full["payload"]}, indent=2))
    if args.download_reports:
        args.download_reports.mkdir(parents=True, exist_ok=True)
        for path in artifacts:
            if path.startswith("diagnostics/"):
                local = client.download_artifacts(run_id, path, dst_path=str(args.download_reports))
                if path.endswith(".html"):
                    print(f"Saved report: {Path(local).resolve().as_uri()}")
    comparisons = client.search_runs(
        [run.info.experiment_id], order_by=["attributes.start_time DESC"]
    )
    print(
        json.dumps(
            {
                "comparison_runs": [
                    {
                        "run_id": other.info.run_id,
                        "profile": other.data.tags.get("mfm.profile"),
                        "local_run_id": other.data.tags.get("mfm.local_run_id"),
                        "lifecycle": other.data.tags.get("mfm.lifecycle_status"),
                        "delivery": other.data.tags.get("mfm.delivery_status"),
                    }
                    for other in comparisons
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
