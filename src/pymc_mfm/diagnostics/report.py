"""Versioned, hash-bound local reports and inference-free rerendering."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .core import PANEL_NAMES, compute_diagnostics, json_safe

if TYPE_CHECKING:
    from pymc_mfm.inference.result import DecompositionResult

FORMAT_VERSION = 1
_INPUT_FILES = (
    "result/result.json",
    "result/arrays.npz",
    "evidence.json",
    "evidence.npz",
    "diagnostics.json",
)


class DiagnosticError(RuntimeError):
    """Scoring/rendering failed; the attached manifest retains every failure."""

    def __init__(self, message, manifest_path):
        super().__init__(message)
        self.manifest_path = Path(manifest_path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _output(path):
    path = Path(path)
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"report output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact(root, relative, kind, *, sensitive, **metadata):
    path = root / relative
    return {
        "path": relative,
        "kind": kind,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "sensitive": sensitive,
        **metadata,
    }


def _safe_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact paths must be nonempty strings")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"artifact path escapes report directory: {relative}")
    joined = root / path
    if (
        joined.is_symlink()
        or root.resolve() not in joined.resolve().parents
        or not joined.is_file()
    ):
        raise ValueError(f"unsafe or missing artifact: {relative}")
    return joined


def _human_report(manifest, root):
    escape = html.escape
    conditional_sbc = manifest["kind"] == "conditional_calibration_report"
    sections = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>Conditional live-effect SBC report</title>"
        if conditional_sbc
        else "<title>Conditional decomposition diagnostic report</title>",
        (
            "<style>body{font:15px "
            "system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 "
            "1em;}"
        ),
        (
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{pa"
            "dding:.45em;text-align:left;border-bottom:1px solid #ddd}"
        ),
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f6f6;padding:1em}img{max-width:100%}",
        (
            ".failed{border-left:5px solid "
            "#a00;padding-left:1em}.unavailable{color:#555}h2{margin-top:2em}s"
            "ummary{cursor:pointer;font-weight:600;padding:.7em 0}</style>"
        ),
        "<h1>Conditional live-effect SBC</h1>"
        if conditional_sbc
        else "<h1>Conditional decomposition diagnostics</h1>",
        (
            "<p>Report format "
            f"{manifest['format_version']}"
            " · status: <strong>"
            f"{escape(manifest['status'])}"
            "</strong></p>"
        ),
        "<details><summary>Evidence identity, dimensions and configuration</summary><pre>"
        + escape(
            json.dumps(
                {"dimensions": manifest["dimensions"], "provenance": manifest["provenance"]},
                indent=2,
            )
        )
        + "</pre></details>",
        "<h2>Scientific qualifications</h2><ul>",
    ]
    sections.extend(f"<li>{escape(note)}</li>" for note in manifest["qualifications"])
    sections.append("</ul><h2>Panels</h2>")
    for name, panel in manifest["panels"].items():
        sections.append(
            f"<section class='{escape(panel['status'])}'><h3>{escape(name.replace('_', ' '))}</h3>"
        )
        sections.append(f"<p>{escape(panel['status'])}: {escape(panel.get('reason') or '')}</p>")
        if panel.get("path"):
            image = base64.b64encode(_safe_file(root, panel["path"]).read_bytes()).decode("ascii")
            sections.append(f"<img src='data:image/png;base64,{image}' alt='{escape(name)}'>")
        sections.append("</section>")
    sections.append(
        (
            "<details><summary>All scalar "
            "diagnostics</summary><table><tr><th>Name</th><th>Value</th><th>Un"
            "it</th><th>Status / qualification</th></tr>"
        )
    )
    for name, value in manifest["metrics"].items():
        status = manifest["statuses"].get(f"metric/{name}", {})
        rendered = "unavailable" if value is None else f"{value:.8g}"
        sections.append(
            f"<tr><td>{escape(name)}</td><td>{rendered}</td>"
            f"<td>{escape(manifest['metric_units'].get(name, ''))}</td>"
            f"<td>{escape(status.get('status', ''))} {escape(status.get('reason') or '')}</td></tr>"
        )
    sections.append(
        (
            "</table></details><details><summary>Full applicability and "
            "failure inventory</summary><pre>"
        )
        + escape(json.dumps(manifest["statuses"], indent=2))
        + "</pre></details>"
    )
    if conditional_sbc:
        sections.append(
            "<p>Full per-family and per-entity PIT evidence, skip reasons and "
            "integrity hashes are saved in <a href='calibration.json'>calibration.json</a> "
            "and <a href='manifest.json'>manifest.json</a>. "
            "Rendering uses saved values only; missing tests are not non-rejections.</p></html>"
        )
    else:
        sections.append(
            "<p>Raw evidence, diagnostic arrays, training/evaluation records "
            "and integrity hashes are listed in <a href='manifest.json'>manifest.json</a>. "
            "Rendering never runs inference or reference fitting.</p></html>"
        )
    return "\n".join(sections)


def _finish(summary, root, provenance, *, strict, source_manifest=None):
    sensitive = provenance.get("synthetic_training_evaluation") is not True
    metadata = summary["metadata"]
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "conditional_diagnostic_report",
        "status": "complete",
        "provenance": json_safe(provenance),
        "dimensions": {
            "window": metadata["window"],
            "head": provenance.get("head", "conditional"),
            "mode": metadata["mode"],
            "units": metadata["units"],
            "data_split": provenance.get("data_split"),
            "n_samples": metadata["n_samples"],
            "n_time": len(metadata["time"]),
            "n_components": len(metadata["component_names"]),
            "n_edges": len(metadata["edge_names"]),
        },
        "metric_representation": (
            "per_world_table"
            if provenance.get("evaluation_data_origin") == "training_corpus"
            else "scalars"
        ),
        "metrics": summary["metrics"],
        "metric_units": summary["metric_units"],
        "statuses": dict(summary["statuses"]),
        "panels": {name: dict(record) for name, record in summary["panels"].items()},
        "qualifications": summary["qualifications"],
        "artifacts": [],
        "evidence": {
            "result": "result",
            "optional_arrays": "evidence.npz",
            "records_and_reference": "evidence.json",
            "computed": "diagnostics.json",
        },
        "arrays": {
            name: {
                "path": "diagnostics.json",
                "pointer": f"/arrays/{name}",
                "status": summary["statuses"].get(f"array/{name}", {}),
            }
            for name in summary["arrays"]
        },
        "tracking": {
            "metric_destination": (
                "manifest.json#/metrics (per-world table; canonical histories use train/evaluation)"
                if provenance.get("evaluation_data_origin") == "training_corpus"
                else "diagnostic/<identity>/<full metric key>"
            ),
            "artifact_destination": (
                "diagnostics/<step>/<checkpoint>/<window>/<mode>/<report identity>"
            ),
            "status_destination": "manifest.json#/statuses",
            "source_step": provenance.get("source_step"),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        },
    }
    for name in (
        "checkpoint_sha256",
        "source_step",
        "data_sha256",
        "data_split",
        "inference_config",
        "profile",
    ):
        present = provenance.get(name) is not None
        manifest["statuses"][f"provenance/{name}"] = {
            "status": "recorded" if present else "unavailable",
            "reason": "caller-supplied evidence identity"
            if present
            else "not supplied; never inferred",
        }
    if source_manifest is not None:
        manifest["rendered_from"] = source_manifest
    for relative in _INPUT_FILES:
        manifest["artifacts"].append(_artifact(root, relative, "evidence", sensitive=sensitive))
    failures = []
    if summary.get("scoring_error"):
        failures.append(f"scoring: {summary['scoring_error']}")
    (root / "figures").mkdir(exist_ok=True)
    for name, panel in manifest["panels"].items():
        manifest["statuses"][f"panel/{name}"] = dict(panel)
        if panel["status"] in ("unavailable", "not_applicable", "failed"):
            continue
        figure = None
        try:
            from pymc_mfm.plotting.panels import plot_panel

            figure = plot_panel(name, summary)
            relative = f"figures/{name}.png"
            figure.savefig(root / relative, dpi=120, bbox_inches="tight")
            panel["path"] = relative
            panel["render_status"] = "ok"
            manifest["artifacts"].append(
                _artifact(
                    root,
                    relative,
                    "figure",
                    sensitive=sensitive,
                    panel=name,
                    arrays="diagnostics.json#/arrays",
                )
            )
        except Exception as exc:
            panel.update(
                status="failed", render_status="failed", reason=f"{type(exc).__name__}: {exc}"
            )
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            if figure is not None:
                figure.clear()
        manifest["statuses"][f"panel/{name}"] = dict(panel)
    if failures:
        manifest["status"] = "failed"
    manifest["failures"] = failures
    (root / "report.html").write_text(_human_report(manifest, root), encoding="utf-8")
    manifest["artifacts"].append(_artifact(root, "report.html", "report", sensitive=sensitive))
    _write_json(root / "manifest.json", manifest)
    if strict and failures:
        raise DiagnosticError(
            "Diagnostic report failed: " + "; ".join(failures), root / "manifest.json"
        )
    return manifest


def _report_evidence(truth, graph_truth, observed, training_records, evaluation_records, reference):
    """Caller evidence normalized exactly as a report persists and scores it.

    Arrays are frozen copies: a caller reusing its array after scoring cannot
    change the evidence a later ``Diagnostics.report`` saves.
    """
    records = json_safe(
        {
            "training_records": list(training_records),
            "evaluation_records": list(evaluation_records),
            "reference": reference,
        }
    )
    optional_arrays = {}
    for name, value in (("truth", truth), ("graph_truth", graph_truth), ("observed", observed)):
        if value is not None:
            array = np.array(value, copy=True)
            array.setflags(write=False)
            optional_arrays[name] = array
    if any(
        value.dtype.kind not in "biuf" or not np.isfinite(value).all()
        for value in optional_arrays.values()
    ):
        raise ValueError("truth/graph_truth/observed must be finite real arrays")
    return records, optional_arrays


def _score(result, records, optional_arrays):
    return compute_diagnostics(
        result,
        truth=optional_arrays.get("truth"),
        graph_truth=optional_arrays.get("graph_truth"),
        observed=optional_arrays.get("observed"),
        training_records=records["training_records"],
        evaluation_records=records["evaluation_records"],
        reference=records["reference"],
    )


def _write_report(result, root, records, optional_arrays, summary, *, strict):
    """Persist evidence with an already computed summary; never rescores."""
    result.save(root / "result")
    _write_json(root / "evidence.json", records)
    np.savez_compressed(root / "evidence.npz", **optional_arrays)
    _write_json(root / "diagnostics.json", summary)
    return _finish(summary, root, result.provenance, strict=strict)


def build_report(
    result: DecompositionResult,
    out_dir,
    *,
    truth=None,
    graph_truth=None,
    observed=None,
    training_records=(),
    evaluation_records=(),
    reference=None,
    strict=True,
):
    """Compute and render a complete, locally persisted diagnostic report.

    ``out_dir`` must be new/empty so historical evidence is never overwritten.
    Missing truth/reference is an expected unavailable state. Actual scoring or
    rendering failures are persisted and raise :class:`DiagnosticError` in
    strict mode. Truth/observation arrays stay separate from inference results.
    ``result.diagnostics(...).report(out_dir)`` writes the same files from its
    already computed values.
    """
    records, optional_arrays = _report_evidence(
        truth, graph_truth, observed, training_records, evaluation_records, reference
    )
    root = _output(out_dir)
    try:
        summary = _score(result, records, optional_arrays)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        summary = {
            "format_version": FORMAT_VERSION,
            "metrics": {},
            "metric_units": {},
            "arrays": {},
            "statuses": {"scoring": {"status": "failed", "reason": reason}},
            "panels": {
                name: {"status": "failed", "reason": "scoring failed: " + reason}
                for name in PANEL_NAMES
            },
            "qualifications": [
                "Scoring failed; this report has no valid computed diagnostic values."
            ],
            "scoring_error": reason,
            "metadata": {
                "time": result.time.tolist(),
                "head": result.provenance.get("head", "conditional"),
                "window": result.window,
                "mode": result.mode,
                "units": result.units,
                "n_samples": result.n_samples,
                "component_names": list(result.component_names),
                "edge_names": list(result.edge_names),
            },
        }
    return _write_report(result, root, records, optional_arrays, summary, strict=strict)


def render_report(saved_path, out_dir, *, strict=True):
    """Rerender verified stored values without rescoring or loading any model.

    Accepts a report directory or its ``manifest.json``. All artifact hashes
    are verified before creating output. Diagnostic figures use saved
    diagnostics.json; SBC figures use saved calibration.json without inference.
    """
    path = Path(saved_path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") not in (
        "conditional_diagnostic_report",
        "calibration_report",
        "conditional_calibration_report",
    ):
        raise ValueError("unsupported diagnostic report format")
    descriptors = manifest.get("artifacts")
    if not isinstance(descriptors, list):
        raise ValueError("report artifact descriptors are missing")
    seen = set()
    for record in descriptors:
        relative = record.get("path")
        if relative in seen:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen.add(relative)
        artifact = _safe_file(source, relative)
        if _sha256(artifact) != record.get("sha256"):
            raise ValueError(f"diagnostic artifact checksum mismatch: {relative}")
    if manifest["kind"] in ("calibration_report", "conditional_calibration_report"):
        if "calibration.json" not in seen:
            raise ValueError("calibration report is missing its saved rank evidence")
        calibration = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
        if manifest.get("provenance") != calibration.get("provenance"):
            raise ValueError("calibration manifest provenance disagrees with saved evidence")
        expected_kind = (
            "independent_world_conditional_sbc"
            if manifest["kind"] == "conditional_calibration_report"
            else "independent_world_sbc"
        )
        if calibration.get("kind") != expected_kind:
            raise ValueError("calibration manifest disagrees with saved card kind")
        return write_calibration_report(calibration, out_dir, strict=strict)
    if not set(_INPUT_FILES) <= seen:
        raise ValueError("report is missing required raw/computed evidence artifacts")
    summary = json.loads((source / "diagnostics.json").read_text(encoding="utf-8"))
    if summary.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported computed diagnostics format")
    from pymc_mfm.inference.result import DecompositionResult

    saved_result = DecompositionResult.load(source / "result")
    if manifest.get("provenance") != saved_result.provenance:
        raise ValueError("report provenance disagrees with verified saved result identity")
    expected_metadata = {
        "time": saved_result.time.tolist(),
        "window": saved_result.window,
        "mode": saved_result.mode,
        "units": saved_result.units,
        "n_samples": saved_result.n_samples,
        "component_names": list(saved_result.component_names),
        "edge_names": list(saved_result.edge_names),
    }
    if any(summary["metadata"].get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("computed diagnostic geometry disagrees with saved result")
    expected_dimensions = {
        "window": saved_result.window,
        "head": saved_result.provenance.get("head", "conditional"),
        "mode": saved_result.mode,
        "units": saved_result.units,
        "data_split": saved_result.provenance.get("data_split"),
        "n_samples": saved_result.n_samples,
        "n_time": len(saved_result.time),
        "n_components": len(saved_result.component_names),
        "n_edges": len(saved_result.edge_names),
    }
    if manifest.get("dimensions") != expected_dimensions:
        raise ValueError("report dimensions disagree with saved result")
    if manifest.get("metrics") != summary["metrics"]:
        raise ValueError("report metrics disagree with stored diagnostic values")
    root = _output(out_dir)
    for relative in _INPUT_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_safe_file(source, relative), destination)
    return _finish(
        summary,
        root,
        saved_result.provenance,
        strict=strict,
        source_manifest={"sha256": _sha256(manifest_path), "format_version": FORMAT_VERSION},
    )


def write_calibration_report(report, out_dir, *, strict=True):
    """Persist and render an independent-world SBC report card without sampling.

    Accepts the complete output of :func:`pymc_mfm.diagnostics.sbc_report`
    or :func:`pymc_mfm.diagnostics.conditional_sbc_report`. Randomized ranks,
    PIT, IDs, seeds, null bands and qualifications are saved so
    :func:`render_report` can reproduce every figure without resimulation.
    """
    report = json_safe(report)
    if report.get("kind") == "independent_world_conditional_sbc":
        return _write_conditional_calibration_report(report, out_dir, strict=strict)
    if (
        report.get("format_version") != FORMAT_VERSION
        or report.get("kind") != "independent_world_sbc"
    ):
        raise ValueError("expected a versioned independent-world SBC report")
    if not report.get("quantities") or not report.get("world_ids"):
        raise ValueError("calibration report requires saved quantities and world IDs")
    provenance = report["provenance"]
    sensitive = provenance.get("synthetic_training_evaluation") is not True
    root = _output(out_dir)
    _write_json(root / "calibration.json", report)
    metrics = {}
    units = {}
    statuses = {}
    for name, quantity in report["quantities"].items():
        for key, value in quantity.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) or value is None:
                path = f"calibration/{name}/{key}"
                metrics[path] = value
                units[path] = "as_defined_in_rank_evidence"
                statuses[f"metric/{path}"] = {
                    "status": "independent_worlds_declared" if value is not None else "unavailable",
                    "reason": quantity["qualification"]
                    if value is not None
                    else "undefined under supplied rank geometry",
                }
                if key == "pot_c_pvalue":
                    statuses[f"metric/{path}"] = dict(quantity["pot_c_status"])
        for key, value in quantity["rank_coverage"].items():
            path = f"calibration/{name}/rank_coverage/{key}"
            metrics[path] = value
            units[path] = "fraction"
            statuses[f"metric/{path}"] = {
                "status": "independent_worlds_declared",
                "reason": "Rank-implied central coverage, distinct from sample-quantile coverage",
            }
    qualification = report["qualification"]
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "calibration_report",
        "status": "complete",
        "provenance": provenance,
        "dimensions": {
            "head": provenance.get("head", "unspecified"),
            "mode": provenance.get("mode", "unspecified"),
            "window": provenance.get("window", "unspecified"),
            "units": provenance.get("units", "rank"),
            "data_split": provenance.get("data_split"),
            "n_worlds": len(report["world_ids"]),
            "n_quantities": len(report["quantity_names"]),
        },
        "metrics": metrics,
        "metric_units": units,
        "statuses": statuses,
        "qualifications": [
            qualification,
            "Independent worlds are explicitly declared by the caller.",
            "Family-wise band control uses Bonferroni across possibly dependent quantities.",
        ],
        "panels": {},
        "failures": [],
        "artifacts": [
            _artifact(root, "calibration.json", "calibration_evidence", sensitive=sensitive)
        ],
        "arrays": {
            "ranks": {"path": "calibration.json", "pointer": "/ranks", "status": "recorded"},
            "quantities": {
                "path": "calibration.json",
                "pointer": "/quantities",
                "status": "recorded",
            },
        },
    }
    for kind in ("ranks", "ecdf", "coverage"):
        panel = {"status": "independent_worlds_declared", "reason": qualification}
        figure = None
        try:
            from pymc_mfm.plotting.panels import plot_sbc_report

            figure = plot_sbc_report(report, kind=kind)
            relative = f"{kind}.png"
            figure.savefig(root / relative, dpi=120, bbox_inches="tight")
            panel.update(path=relative, render_status="ok")
            manifest["artifacts"].append(
                _artifact(root, relative, "figure", sensitive=sensitive, panel=kind)
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            panel.update(status="failed", reason=reason, render_status="failed")
            manifest["failures"].append(f"{kind}: {reason}")
        finally:
            if figure is not None:
                figure.clear()
        manifest["panels"][kind] = panel
        manifest["statuses"][f"panel/{kind}"] = dict(panel)
    if manifest["failures"]:
        manifest["status"] = "failed"
    (root / "report.html").write_text(_human_report(manifest, root), encoding="utf-8")
    manifest["artifacts"].append(_artifact(root, "report.html", "report", sensitive=sensitive))
    _write_json(root / "manifest.json", manifest)
    if strict and manifest["failures"]:
        raise DiagnosticError("Calibration report rendering failed", root / "manifest.json")
    return manifest


def _conditional_calibration_inventory(report):
    """Flatten saved scalar tests and their eligibility into manifest entries."""
    metrics = {}
    units = {}
    statuses = {}
    qualifications = []
    for family in ("channel", "control"):
        group = report["families"].get(family) or {}
        entries = [(f"calibration/{family}", group)]
        entries.extend(
            (f"calibration/{family}/entities/{name}", entity)
            for name, entity in group.get("entities", {}).items()
        )
        for prefix, entry in entries:
            skips = entry.get("skip_reasons") or {}
            skip_reason = f"Saved skip reasons: {json.dumps(skips, sort_keys=True)}"
            statuses[f"{prefix}/skip_reasons"] = {
                "status": "recorded",
                "reason": skip_reason,
            }
            entry_status = entry.get("status", "unavailable")
            statuses[prefix] = {"status": entry_status, "reason": skip_reason}
            for mode in ("conditional", "model_average"):
                path = f"{prefix}/{mode}"
                record = entry.get(mode)
                if record is None:
                    missing_status = (
                        "not_applicable" if entry_status == "not_applicable" else "unavailable"
                    )
                    reason = (
                        f"Saved inference mode is not model_average; {mode} SBC is not "
                        f"applicable. {skip_reason}"
                        if missing_status == "not_applicable"
                        else f"No eligible {mode} rank record. {skip_reason}"
                    )
                    statuses[path] = {"status": missing_status, "reason": reason}
                    statuses[f"{path}/exact_full_draw_count"] = {
                        "status": missing_status,
                        "reason": reason,
                    }
                    statuses[f"{path}/rank_coverage"] = {
                        "status": missing_status,
                        "reason": reason,
                    }
                    for key in ("n_ranks", "ks_pvalue", "ecdf_band_reject"):
                        metric = f"{path}/{key}"
                        metrics[metric] = None
                        units[metric] = (
                            "count"
                            if key == "n_ranks"
                            else "fraction"
                            if key == "ks_pvalue"
                            else "indicator"
                        )
                        statuses[f"metric/{metric}"] = {
                            "status": missing_status,
                            "reason": reason,
                        }
                    qualifications.append(f"{path}: {reason}")
                    continue
                record_status = record.get("status", "recorded")
                reason = record.get("qualification") or report["qualification"]
                if isinstance(record_status, dict):
                    reason = record_status.get("reason") or reason
                    record_status = record_status.get("status", "recorded")
                statuses[path] = {"status": record_status, "reason": reason}
                for key, value in record.items():
                    if not (value is None or isinstance(value, (int, float))):
                        continue
                    metric = f"{path}/{key}"
                    metrics[metric] = int(value) if isinstance(value, bool) else value
                    units[metric] = (
                        "fraction"
                        if key.endswith("pvalue")
                        else "indicator"
                        if key == "ecdf_band_reject"
                        else "count"
                        if key in ("n_ranks", "exact_full_draw_count", "n_on")
                        else "as_defined_in_rank_evidence"
                    )
                    statuses[f"metric/{metric}"] = {
                        "status": record_status if value is not None else "unavailable",
                        "reason": reason
                        if value is not None
                        else (
                            f"No saved {key} test outcome; unavailable, not a non-rejection. "
                            f"{reason}"
                            if key in ("ks_pvalue", "ecdf_band_reject")
                            else f"Undefined for the saved rank geometry. {reason}"
                        ),
                    }
                    if value is None and key in ("ks_pvalue", "ecdf_band_reject"):
                        qualifications.append(f"{metric}: no saved test outcome. {reason}")
                coverage = record.get("rank_coverage") or {}
                statuses[f"{path}/rank_coverage"] = {
                    "status": record_status if coverage else "unavailable",
                    "reason": reason if coverage else "No saved rank-implied coverage.",
                }
                for key, value in coverage.items():
                    metric = f"{path}/rank_coverage/{key}"
                    metrics[metric] = value
                    units[metric] = "fraction"
                    statuses[f"metric/{metric}"] = {
                        "status": record_status if value is not None else "unavailable",
                        "reason": (
                            "Saved PIT-implied central coverage over eligible independent worlds. "
                            + reason
                            if value is not None
                            else "No saved rank-implied coverage; unavailable."
                        ),
                    }
                exact = record.get("exact_full_draw_count") or {}
                exact_path = f"{path}/exact_full_draw_count"
                statuses[exact_path] = {
                    "status": "recorded" if exact else "unavailable",
                    "reason": (
                        "Saved discrete-rank tests for full-draw-count subsets."
                        if exact
                        else "No subset has enough exact full-draw ranks for a discrete-rank test."
                    ),
                }
                for count, discrete in exact.items():
                    count_path = f"{exact_path}/{count}"
                    discrete_reason = discrete["qualification"]
                    statuses[count_path] = {
                        "status": discrete["status"],
                        "reason": discrete_reason,
                    }
                    for key in (
                        "n_ranks",
                        "n_post_samples",
                        "ks_pvalue",
                        "chi2_pvalue",
                        "ecdf_band_reject",
                        "pot_c_pvalue",
                    ):
                        value = discrete.get(key)
                        metric = f"{count_path}/{key}"
                        metrics[metric] = value
                        units[metric] = (
                            "count"
                            if key in ("n_ranks", "n_post_samples")
                            else "fraction"
                            if key.endswith("pvalue")
                            else "indicator"
                        )
                        test_status = (
                            discrete["pot_c_status"]
                            if key == "pot_c_pvalue"
                            else {"status": discrete["status"], "reason": discrete_reason}
                        )
                        statuses[f"metric/{metric}"] = (
                            dict(test_status)
                            if value is not None or key == "pot_c_pvalue"
                            else {
                                "status": "unavailable",
                                "reason": f"No saved {key} test outcome. {test_status['reason']}",
                            }
                        )
                        if value is None and key in (
                            "ks_pvalue",
                            "chi2_pvalue",
                            "ecdf_band_reject",
                            "pot_c_pvalue",
                        ):
                            qualifications.append(
                                f"{metric}: no saved test outcome. {test_status['reason']}"
                            )
    family_wise = report["family_wise"]
    statuses["calibration/family_wise"] = {
        "status": "recorded",
        "reason": "Saved multiple-testing decisions in calibration.json#/family_wise: "
        + json.dumps(family_wise, sort_keys=True)
        + ". No decisions are recalculated during rendering.",
    }
    return metrics, units, statuses, qualifications


def _write_conditional_calibration_report(report, out_dir, *, strict):
    if (
        report.get("format_version") != FORMAT_VERSION
        or not report.get("world_ids")
        or not report.get("cell_ids")
        or not isinstance(report.get("families"), dict)
    ):
        raise ValueError("expected a versioned independent-world conditional SBC report")
    provenance = report["provenance"]
    sensitive = provenance.get("synthetic_training_evaluation") is not True
    root = _output(out_dir)
    _write_json(root / "calibration.json", report)
    metrics, units, statuses, missing = _conditional_calibration_inventory(report)
    qualification = report["qualification"]
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "conditional_calibration_report",
        "status": "complete",
        "provenance": provenance,
        "dimensions": {
            "head": provenance.get("head", "unspecified"),
            "mode": provenance.get("mode", "unspecified"),
            "window": provenance.get("window", "unspecified"),
            "units": provenance.get("units", "rank"),
            "data_split": provenance.get("data_split"),
            "n_worlds": len(report["world_ids"]),
            "n_cells": len(report["cell_ids"]),
            "n_entities": sum(
                len(group.get("entities", {})) for group in report["families"].values()
            ),
        },
        "metrics": metrics,
        "metric_units": units,
        "statuses": statuses,
        "qualifications": [
            qualification,
            "Each selected generator cell contributes one independent world; cell and world IDs "
            "are saved with the card, not inferred by rendering.",
            "Conditional PIT uses only edge-on draws for live truth edges; graph-averaged PIT "
            "uses saved full-draw ranks for eligible effects, including dead truth edges.",
            "Family-wise rejecting quantities are saved Bonferroni-adjusted ECDF-band "
            "decisions; saved KS p-values are additional diagnostics.",
            "Missing eligible ranks or null tests are unavailable; map/fixed inference is "
            "not applicable. Neither state is a non-rejection.",
            *missing,
        ],
        "panels": {},
        "failures": [],
        "artifacts": [
            _artifact(root, "calibration.json", "calibration_evidence", sensitive=sensitive)
        ],
        "arrays": {
            "families": {
                "path": "calibration.json",
                "pointer": "/families",
                "status": "recorded",
            },
            "family_wise": {
                "path": "calibration.json",
                "pointer": "/family_wise",
                "status": "recorded",
            },
        },
    }
    has_rank_record = any(
        entry.get(mode) is not None
        for group in report["families"].values()
        for entry in (group, *group.get("entities", {}).values())
        for mode in ("conditional", "model_average")
    )
    all_not_applicable = all(
        group.get("status") == "not_applicable" for group in report["families"].values()
    )
    for kind in ("ranks", "ecdf", "coverage"):
        panel = {
            "status": (
                "independent_cells_declared"
                if has_rank_record
                else "not_applicable"
                if all_not_applicable
                else "unavailable"
            ),
            "reason": qualification
            if has_rank_record
            else (
                "Saved inference modes are not model_average; conditional SBC is not applicable."
                if all_not_applicable
                else "No eligible ranks in any family or entity; figure marks unavailable views."
            ),
        }
        figure = None
        try:
            from pymc_mfm.plotting.panels import plot_sbc_report

            figure = plot_sbc_report(report, kind=kind)
            relative = f"{kind}.png"
            figure.savefig(root / relative, dpi=120, bbox_inches="tight")
            panel.update(path=relative, render_status="ok")
            manifest["artifacts"].append(
                _artifact(root, relative, "figure", sensitive=sensitive, panel=kind)
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            panel.update(status="failed", reason=reason, render_status="failed")
            manifest["failures"].append(f"{kind}: {reason}")
        finally:
            if figure is not None:
                figure.clear()
        manifest["panels"][kind] = panel
        manifest["statuses"][f"panel/{kind}"] = dict(panel)
    if manifest["failures"]:
        manifest["status"] = "failed"
    (root / "report.html").write_text(_human_report(manifest, root), encoding="utf-8")
    manifest["artifacts"].append(_artifact(root, "report.html", "report", sensitive=sensitive))
    _write_json(root / "manifest.json", manifest)
    if strict and manifest["failures"]:
        raise DiagnosticError("Conditional SBC report rendering failed", root / "manifest.json")
    return manifest
