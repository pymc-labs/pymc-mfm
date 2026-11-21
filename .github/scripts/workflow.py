"""Generate, train, reload, infer and report using public APIs and a local backend.

Requires [generation,training]. This two-update integration exercise checks
unit-equivariant inference but produces neither usable pretrained weights nor
calibrated attribution. The full model uses a reduced world count, horizon and
training budget. No Hub writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mlflow import MlflowClient

from pymc_mktfm import from_pretrained
from pymc_mktfm.data.corpus import corpus_batch
from pymc_mktfm.data.profiles import PROFILE_A
from pymc_mktfm.data.recipes import (
    SHARD_PLANS,
    generate_corpus,
    load_corpora,
    profile_prior,
    provenance,
)
from pymc_mktfm.diagnostics import build_report, render_report
from pymc_mktfm.inference import DecompositionResult, decompose
from pymc_mktfm.training import TrackingConfig, TrainingConfig, load_checkpoint, train


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New local output directory")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prior = profile_prior(
        PROFILE_A,
        n_cells=6,
        draws_per_cell=2,
        seed=SHARD_PLANS["A"].seed(0),
        n_time_steps=16,
        l_max=4,
        carryover_burn_in=0,
        query_frac=0.5,
        p_long_horizon=0.0,
    )
    generate_corpus(prior, out / "corpus.npz")
    loaded = load_corpora([out / "corpus.npz"])[0]
    run_dir = out / "run"
    trained = train(
        [loaded.corpus],
        out_dir=run_dir,
        device="cpu",
        training=TrainingConfig(
            steps=2,
            batch_size=2,
            warmup_steps=1,
            eval_every=2,
            eval_worlds=1,
            eval_samples=8,
            log_every=1,
            milestone_steps=(),
        ),
        corpus_provenance=[provenance(loaded)],
        # Explicit local URI prevents an ambient remote URI from receiving this run.
        tracking=TrackingConfig(tracking_uri=f"sqlite:///{out / 'mlflow.db'}"),
    )
    assert trained.status == "finished" and trained.final_step == 2
    model = from_pretrained(run_dir / "model", device="cpu")
    checkpoint_model, _ = load_checkpoint(run_dir / "checkpoint_last.pt", device="cpu")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint_model.state_dict()[name], rtol=0, atol=0)
    checkpoint_hash = digest(run_dir / "checkpoint_last.pt")
    assert model.provenance["checkpoint_sha256"] == checkpoint_hash
    assert model.provenance["tracking_local_run_id"] == trained.run["run_id"]

    # Holdout truth is kept outside the inference input, only passed to diagnostics.
    corpus = loaded.corpus
    row = int(np.flatnonzero(corpus["is_val"] == 1)[0])
    context = int(corpus["support_mask"][row].sum())
    observations = {
        "spend": corpus["treatment_raw"][row],
        "controls": corpus["covariates"][row],
        "context_outcome": corpus["outcome_raw"][row, :context],
        "channel_mask": corpus["treatment_active_mask"][row],
        "control_mask": corpus["covariate_active_mask"][row],
        "latent_mask": corpus["latent_active_mask"][row],
    }
    (out / "observations.json").write_text(
        json.dumps({key: value.tolist() for key, value in observations.items()}, indent=2) + "\n"
    )
    torch.manual_seed(31)
    result = decompose(model, observations, n_samples=19, window="query")
    torch.manual_seed(31)
    checkpoint_result = decompose(checkpoint_model, observations, n_samples=19, window="query")
    np.testing.assert_allclose(
        result.components, checkpoint_result.components, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(result.graphs, checkpoint_result.graphs)
    result.save(out / "result")
    saved = DecompositionResult.load(out / "result")
    np.testing.assert_array_equal(saved.components, result.components)
    assert saved.provenance == result.provenance

    # Reuse the fitted context scales in both windows. All contributions are in
    # outcome units, so only the sales multiplier is undone on the returned draws.
    rescalings = {
        "case_10_units": (10.0, 10_000.0, 100.0),
        "tiny_units": (1e-8, 1e-8, 1e-8),
        "independent_columns": (
            1e6,
            np.logspace(-8, 8, observations["spend"].shape[1]),
            np.logspace(8, -8, observations["controls"].shape[1]),
        ),
    }
    unit_checks = []
    for window in ("context", "query"):
        torch.manual_seed(31)
        reference = decompose(model, observations, n_samples=19, window=window)
        for label, (sales_units, spend_units, control_units) in rescalings.items():
            transformed = {
                **observations,
                "spend": observations["spend"].astype(np.float64) * spend_units,
                "controls": observations["controls"].astype(np.float64) * control_units,
                "context_outcome": (
                    observations["context_outcome"].astype(np.float64) * sales_units
                ),
            }
            torch.manual_seed(31)
            changed = decompose(model, transformed, n_samples=19, window=window)
            restored = changed.components / sales_units
            np.testing.assert_allclose(restored, reference.components, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(changed.graphs, reference.graphs)
            unit_checks.append(
                {
                    "window": window,
                    "units": label,
                    "max_abs_error_original_units": float(
                        np.max(np.abs(restored - reference.components))
                    ),
                    "graphs_identical": True,
                }
            )
    observed_report = build_report(saved, out / "observed-report")
    replay = render_report(out / "observed-report", out / "observed-rerender")
    assert replay["provenance"] == observed_report["provenance"]
    assert replay["provenance"]["checkpoint_sha256"] == checkpoint_hash
    batch = corpus_batch(corpus, [row])
    truth = batch["targets_z"][0].numpy().astype(np.float64) * float(batch["sales_scale"][0])
    build_report(
        saved,
        out / "synthetic-report",
        truth=truth,
        graph_truth=corpus["g"][row],
        observed=corpus["outcome_raw"][row, context:],
    )
    for path in (out / "observed-report", out / "synthetic-report", out / "observed-rerender"):
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["panels"]["target_recovery"]["render_status"] == "ok"
        assert all(
            panel.get("render_status") == "ok"
            for panel in manifest["panels"].values()
            if panel["status"] not in ("unavailable", "not_applicable")
        )

    # Successful numerical training alone does not prove tracking delivery.
    tracking = trained.run["tracking"]
    assert tracking["delivery_status"] == "healthy" and tracking["failure_count"] == 0
    client = MlflowClient(tracking_uri=tracking["tracking_uri"])
    run_id = tracking["run_id"]
    assert client.get_run(run_id).info.status == "FINISHED"
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    names = json.loads((run_dir / "tracking/names.json").read_text())
    loss_key = next(
        key
        for key, metadata in names.items()
        if metadata["kind"] == "metric" and metadata["original"] == "train/loss"
    )
    history = sorted(
        (item.step, item.value) for item in client.get_metric_history(run_id, loss_key)
    )
    assert history == [(record["step"], record["loss"]) for record in records]
    assert saved.provenance["tracking_run_id"] == run_id
    events = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "tracking/events").glob("*.json"))
    ]
    checkpoint = next(event["payload"] for event in events if event["kind"] == "checkpoint")
    downloaded = Path(client.download_artifacts(run_id, checkpoint["artifact"]["path"]))
    assert digest(downloaded) == checkpoint["sha256"]
    report = next(event["payload"] for event in events if event["kind"] == "diagnostic")
    for artifact in report["manifest"]["artifacts"]:
        downloaded = Path(
            client.download_artifacts(run_id, report["artifact_path"] + "/" + artifact["path"])
        )
        assert digest(downloaded) == artifact["sha256"]
    summary = {
        "status": trained.status,
        "steps": trained.final_step,
        "budget": trained.run["training_config"]["budget"],
        "checkpoint_sha256": checkpoint_hash,
        "bundle_sha256": model.provenance["bundle_sha256"],
        "tracking_run_id": run_id,
        "tracking_delivery": tracking["delivery_status"],
        "loss_history": history,
        "query_shape": list(saved.components.shape),
        "unit_checks": unit_checks,
        "report": str(out / "synthetic-report/report.html"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
