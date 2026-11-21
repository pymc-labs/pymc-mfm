"""Figure, graph-export and record accessors on results, diagnostics and training runs.

Plot methods delegate to the renderers behind reports. ``dag`` returns a native
GraphvizGraph; other figure methods return matplotlib ``Figure`` objects and
import matplotlib only when called. Nothing here samples a model, trains or fits
a reference. A method whose evidence is missing raises ``ValueError`` naming it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymc_mktfm.plotting import panels

from .core import record_series
from .coverage import evaluation_coverage_intervals
from .report import _output, _report_evidence, _score, _write_report

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from pymc_mktfm.inference.result import DecompositionResult
    from pymc_mktfm.plotting import GraphvizGraph
    from pymc_mktfm.plotting.inspector import GraphInspector


class Diagnostics:
    """The diagnostic catalog of one result, computed once from supplied evidence.

    ``summary`` is the complete catalog, as saved in ``diagnostics.json``;
    ``metrics``, ``statuses``, ``plot`` and ``report`` read it and never
    rescore. Obtain it from :meth:`DecompositionResult.diagnostics`.
    """

    __slots__ = ("result", "summary", "_records", "_arrays")

    def __init__(
        self,
        result: DecompositionResult,
        *,
        truth=None,
        graph_truth=None,
        observed=None,
        reference=None,
        training_records=(),
        evaluation_records=(),
        coverage_intervals=(),
    ):
        self.result = result
        self._records, self._arrays = _report_evidence(
            truth,
            graph_truth,
            observed,
            training_records,
            evaluation_records,
            reference,
            coverage_intervals,
        )
        self.summary = _score(result, self._records, self._arrays)

    @property
    def metrics(self) -> dict[str, float | None]:
        return self.summary["metrics"]

    @property
    def statuses(self) -> dict[str, dict[str, Any]]:
        return self.summary["statuses"]

    @property
    def plot(self) -> DiagnosticsPlots:
        return DiagnosticsPlots(self)

    def report(self, out_dir, *, strict=True) -> dict[str, Any]:
        """Write the files :func:`build_report` writes, from these computed values."""
        return _write_report(
            self.result, _output(out_dir), self._records, self._arrays, self.summary, strict=strict
        )


class _SummaryPlots:
    """Figures and graph exports from one computed diagnostic summary."""

    __slots__ = ()

    def _summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def _draw(self, panel, render=None, **options) -> Figure | GraphvizGraph:
        summary = self._summary()
        status = summary["panels"][panel]
        if status["status"] == "unavailable":
            raise ValueError(f"{panel} is unavailable: {status['reason']}")
        if render is None:
            return panels.plot_panel(panel, summary)
        return render(summary, **options)

    def components(self) -> Figure:
        """Family totals with 80% bands and the context/query boundary."""
        return self._draw("components")

    def channels(self) -> Figure:
        return self._draw("channel_contributions")

    def channel_totals(self, *, ax=None) -> Figure:
        return self._draw("channel_totals", panels.plot_channel_totals, ax=ax)

    def controls(self) -> Figure:
        return self._draw("controls")

    def indirect(self) -> Figure:
        return self._draw("indirect_sources")

    def reconstruction(self) -> Figure:
        """Latent total against the observations when they were supplied."""
        return self._draw("target_recovery")

    def graph(self) -> Figure:
        """Edge frequencies grouped by edge family."""
        return self._draw("structure_heatmap")

    def dag(self, *, graph=None) -> GraphvizGraph:
        """Native Graphviz Average or zero-based complete graph rank; requires ``dot``."""
        return self._draw("dag_recovery", panels.plot_dag, graph=graph)

    def dot(self, *, graph=None) -> str:
        """Prediction-only DOT: Average (``graph=None``, the default) or a ranked graph.

        Integer ranks are zero-based.
        """
        from pymc_mktfm.plotting.dot import graph_to_dot

        return graph_to_dot(self._summary(), graph=graph)

    def graph_inspector(self) -> GraphInspector:
        """Offline graph distribution selector for notebooks or HTML export."""
        from pymc_mktfm.plotting.inspector import plot_graph_inspector

        return plot_graph_inspector(self._summary())

    def graph_posterior(self, *, ax=None) -> Figure:
        return self._draw("dag_posterior", panels.plot_graph_posterior, ax=ax)

    def cardinality(self) -> Figure:
        return self._draw("graph_cardinality")

    def graph_statistics(self) -> Figure:
        return self._draw("graph_statistics")

    def cycles(self, *, ax=None) -> Figure:
        return self._draw("cycles", panels.plot_cycles, ax=ax)

    def zero_atoms(self, *, ax=None) -> Figure:
        return self._draw("zero_atoms", panels.plot_zero_atoms, ax=ax)

    def uncertainty(self, *, ax=None) -> Figure:
        return self._draw("graph_uncertainty", panels.plot_uncertainty, ax=ax)

    def support(self) -> Figure:
        return self._draw("support")

    def input_support(self) -> Figure:
        """Raw spend/control location and scale recorded at inference.

        ``decompose`` and the evaluator's result capture record them.
        """
        return self._draw("input_support")

    def set_posterior(self) -> Figure:
        """Cardinality-conditioned set approximation (needs ``set_cardinalities``)."""
        return self._draw("set_posterior")


class ResultPlots(_SummaryPlots):
    """Figures of one result's draws (``result.plot``); none needs truth."""

    __slots__ = ("_result",)

    def __init__(self, result: DecompositionResult):
        self._result = result

    def _summary(self) -> dict[str, Any]:
        return self._result.diagnostics().summary

    def reconstruction(self, observed=None) -> Figure:
        """Latent total; ``observed`` outcomes add the observation residual."""
        if observed is None:
            return super().reconstruction()
        return self._result.diagnostics(observed=observed).plot.reconstruction()


class DiagnosticsPlots(_SummaryPlots):
    """Figures of a :class:`Diagnostics` (``diag.plot``), including evidence panels."""

    __slots__ = ("_diagnostics",)

    #: Accessor kind: its catalog panel and the renderer's kind.
    _CALIBRATION = {
        "pit": ("calibration_pit", "ranks"),
        "ecdf": ("calibration_ecdf", "ecdf"),
        "coverage": ("coverage", "coverage"),
    }

    def __init__(self, diagnostics: Diagnostics):
        self._diagnostics = diagnostics

    def _summary(self) -> dict[str, Any]:
        return self._diagnostics.summary

    def recovery(self, *, ax=None) -> Figure:
        """Posterior means against component truth."""
        return self._draw("recovery_scatter", panels.plot_recovery_scatter, ax=ax)

    def attribution(self) -> Figure:
        """Live and dead errors of each channel and control (needs truth and graph truth)."""
        return self._draw("attribution")

    def window_totals(self) -> Figure:
        """Pointwise channel coverage and joint live-effect coverage of window means.

        Includes the latent-total fit; joint coverage uses a studentized 80%
        simultaneous band, distinct from all live pointwise intervals covering.
        """
        return self._draw("window_totals")

    def cumulative(self) -> Figure:
        """Running totals of each channel's contribution against truth."""
        return self._draw("cumulative")

    def temporal(self) -> Figure:
        """Lag-1 and residual autocorrelation per trajectory family (needs truth)."""
        return self._draw("temporal")

    def residuals(self, axis="true", *, ax=None) -> Figure:
        """Signed window-total residual per size bin: ``"true"`` or ``"predicted"`` size.

        Each channel and control effect's posterior mean minus truth in
        ``sales_scale`` units, binned by its true or posterior-mean size with
        this world's cuts (needs truth and the result's ``sales_scale``).
        """
        if axis not in panels.RESIDUAL_AXES:
            raise ValueError(f"residual axis must be one of {panels.RESIDUAL_AXES}, got {axis!r}")
        return self._draw(
            f"residual_{axis}_size", panels.plot_residual_by_magnitude, axis=axis, ax=ax
        )

    def strata(self, metric="coverage", by="true_size_bin", *, ax=None) -> Figure:
        """One stratified cell value of this world's effects, per stratum of ``by``.

        ``metric`` is ``"coverage"``, ``"crps"``, ``"signed_residual"`` or
        ``"abs_residual"``; ``by`` is one of
        :data:`pymc_mktfm.diagnostics.STRATA_AXES`.
        """
        return self._draw("strata", panels.plot_strata, metric=metric, by=by, ax=ax)

    def reliability(self, kind="corp", *, ax=None) -> Figure:
        """Router edge frequencies against graph truth: ``"corp"`` isotonic fit or ``"binned"``.

        The CORP curve is the bin-free isotonic fit with the pooled Brier
        decomposition; ``"binned"`` draws the equal-width bins and pooled ECE.
        """
        return self._draw("reliability", panels.plot_reliability, kind=kind, ax=ax)

    def calibration(self, kind="pit", *, ax=None) -> Figure:
        """One-world screening: ``"pit"`` ranks, ``"ecdf"`` or interval ``"coverage"``.

        Only the coverage figure is one axes, so only it draws into ``ax=``.
        """
        if kind not in self._CALIBRATION:
            raise ValueError(f"calibration kind must be one of {sorted(self._CALIBRATION)}")
        panel, drawn = self._CALIBRATION[kind]
        return self._draw(panel, panels.plot_calibration, kind=drawn, ax=ax)

    def posterior_sbc(self) -> Figure:
        return self._draw("posterior_sbc")

    def c2st(self) -> Figure:
        return self._draw("c2st")

    def reference(self) -> Figure:
        return self._draw("reference_comparison")

    def training(self) -> Figure:
        return self._draw("training")

    def evaluation(self) -> Figure:
        return self._draw("evaluation")


_RECORD_FILES = {"training": "metrics.jsonl", "evaluation": "evaluations.jsonl"}
#: Heads compared by ``run.plot.heads``: decomposition heads record per window,
#: the structure head and the joint router's edge marginals per edge family.
_HEADS = ("joint", "conditional", "structure_head", "joint_router/edge")
_EDGE_SCORE_SOURCES = ("structure_head", "joint_router/edge")
#: Characters of an unnamed run's ``run_id`` in figure headers: its UTC start
#: time and the first 8 of its 32 random hex digits (``run.json`` keeps all).
_RUN_ID_LABEL = 25


def _scope(window):
    """What the third part of a flat evaluation key selects."""
    if window in ("query", "context"):
        return f"{window} window"
    return "whole graph" if window == "graph" else f"{window} edges"


def _one_class(flat, family):
    """Whether an edge family's recorded counts hold one truth class only."""
    n, n_pos = flat.get(f"{family}/n"), flat.get(f"{family}/n_pos")
    return bool(n) and n_pos in (0, n)


def _decoding_error(exc):
    """Why bytes are not one JSON value: invalid UTF-8, malformed JSON or a decoder limit."""
    if isinstance(exc, UnicodeDecodeError):
        return f"invalid UTF-8: {exc.reason}"
    if isinstance(exc, json.JSONDecodeError):
        return exc.msg
    return str(exc)  # an integer past the digit limit, or nesting past the recursion limit


def _read_json(path, what):
    """The JSON value in ``path``; bytes that do not decode raise ``ValueError`` naming it."""
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: malformed {what} ({_decoding_error(exc)})") from exc


def _finite_number(value):
    """A JSON number within the float range; booleans are not numbers."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(value) <= sys.float_info.max
    )


def _corpus_support_problem(summary):
    """Why ``corpus_support.json`` content cannot be drawn, or ``None`` when it can."""
    if not isinstance(summary, dict):
        return "not a JSON object"
    worlds = summary.get("worlds")
    if not isinstance(worlds, list):
        return "worlds is not a list"
    for index, world in enumerate(worlds):
        if not isinstance(world, dict) or not all(
            _finite_number(world.get(key)) for key in ("mean", "std")
        ):
            return f"worlds[{index}] has no finite numeric mean and std"
    if not isinstance(summary.get("skipped", {}), dict):
        return "skipped is not a JSON object"
    total = summary.get("n_worlds", 0)
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        return "n_worlds is not a non-negative integer"
    if not isinstance(summary.get("qualification"), str):
        return "qualification is not a string"
    return None


def _read_records(path, partial):
    """JSON-lines records; only an unterminated last line may be an interrupted write.

    Each line decodes on its own, so bytes that are not UTF-8, malformed JSON
    and values past the decoder's limits are located (or, in that last line,
    tolerated) alike.
    """
    if not path.exists():
        return []
    data = path.read_bytes()
    terminated = data.endswith(b"\n")
    lines = data.split(b"\n")
    if terminated or not data:
        lines.pop()
    records = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line.decode("utf-8"))
        except (ValueError, RecursionError) as exc:
            error = _decoding_error(exc)
            if number == len(lines) and not terminated:
                partial.append({"file": path.name, "line": number, "error": error})
                break
            raise ValueError(f"{path}:{number}: malformed record ({error})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number}: record is not a JSON object")
        records.append(record)
    return records


class TrainingRun:
    """Recorded history of one :func:`pymc_mktfm.training.train` run directory.

    Reads ``run.json``, ``metrics.jsonl`` and ``evaluations.jsonl`` on load.
    ``plot.strata`` and ``derived_metrics`` read saved ``evaluation_tables``
    when called. It never loads a model and also follows a run still writing.
    ``training`` and ``evaluation`` are :func:`record_series` summaries; a
    missing records file gives empty series. ``partial_lines`` lists the
    tolerated unterminated last line of a file (a write in progress or cut
    short); a malformed line anywhere else raises with its ``file:line``.
    """

    __slots__ = ("run_dir", "run", "training", "evaluation", "partial_lines")

    def __init__(self, run_dir, run, training, evaluation, partial_lines=()):
        self.run_dir = Path(run_dir)
        self.run = run
        self.training = training
        self.evaluation = evaluation
        self.partial_lines = tuple(partial_lines)

    @classmethod
    def load(cls, run_dir: str | Path) -> TrainingRun:
        run_dir = Path(run_dir)
        path = run_dir / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; not a training run directory")
        run = _read_json(path, "run record")
        if not isinstance(run, dict):
            raise ValueError(f"{path}: run record is not a JSON object")
        partial = []
        summaries = {}
        for kind, name in _RECORD_FILES.items():
            records = _read_records(run_dir / name, partial)
            try:
                summaries[kind] = record_series(records, kind)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"{run_dir / name}: {exc}") from exc
        return cls(run_dir, run, summaries["training"], summaries["evaluation"], partial)

    def __repr__(self) -> str:
        return (
            f"TrainingRun({str(self.run_dir)!r}, status={self.run.get('status')!r}, "
            f"last_step={self.last_step})"
        )

    @property
    def last_step(self) -> int | None:
        """Latest recorded step; ``run.json`` states progress only when a run ends."""
        steps = [*(self.training["steps"] or ()), *(self.evaluation["steps"] or ())]
        return max(steps) if steps else None

    @property
    def plot(self) -> TrainingPlots:
        return TrainingPlots(self)

    def derived_metrics(
        self, *, seed: int = 0, bootstrap_samples: int = 2000
    ) -> list[dict[str, Any]]:
        """Report-only quantities per event, from recorded scores and coverage tables.

        ``holdout_minus_seen/<key>`` is the held-out minus the seen value of
        every key both legs record (positive: higher on held-out worlds).
        ``<leg>/<source>/mean_auroc`` and ``mean_ece`` average the edge
        families whose score is defined, for the structure head and the joint
        router's edge marginals; ``mean_ece`` leaves out families recorded with
        one truth class (``n_pos`` 0 or ``n``), whose ECE evidences no
        calibration. An undefined input gives ``None``.

        ``coverage_intervals[leg][key]`` holds the equal-world mean and its
        95% cell-clustered bootstrap interval, scored-world/cell/slot counts,
        eligibility and undefined reasons, seed and draw count. Slot weights
        affect only count metadata. Intervals use saved coverage evidence in
        ``evaluation_tables/step_<N:09d>.json``, never inference replay; absent
        tables or evidence are ``unavailable`` / ``not_recorded``.
        """
        derived = []
        for record in self.evaluation["records"]:
            flat = record.get("flat") or {}
            values: dict[str, Any] = {"step": record["step"]}
            for key, value in flat.items():
                leg, _, rest = key.partition("/")
                if leg == "holdout" and f"seen/{rest}" in flat:
                    seen = flat[f"seen/{rest}"]
                    values[f"holdout_minus_seen/{rest}"] = (
                        None if value is None or seen is None else value - seen
                    )
            for leg in dict.fromkeys(key.partition("/")[0] for key in flat):
                for source in _EDGE_SCORE_SOURCES:
                    prefix = f"{leg}/{source}/"
                    for metric in ("auroc", "ece"):
                        scores = {
                            key.rsplit("/", 1)[0]: value
                            for key, value in flat.items()
                            if key.startswith(prefix)
                            and key.endswith(f"/{metric}")
                            and key[len(prefix) :].count("/") == 1
                        }
                        if not scores:
                            continue
                        defined = [
                            value
                            for family, value in scores.items()
                            if value is not None
                            and not (metric == "ece" and _one_class(flat, family))
                        ]
                        values[f"{leg}/{source}/mean_{metric}"] = (
                            math.fsum(defined) / len(defined) if defined else None
                        )
            path = self.run_dir / "evaluation_tables" / f"step_{int(record['step']):09d}.json"
            tables = _read_json(path, "evaluation tables") if path.is_file() else None
            values["coverage_intervals"] = evaluation_coverage_intervals(
                record, tables, seed=seed, bootstrap_samples=bootstrap_samples
            )
            derived.append(values)
        return derived


class TrainingPlots:
    """Figures of a :class:`TrainingRun`'s records (``run.plot``).

    Every method takes ``ax=``: a figure of one axes is drawn into the caller's
    axes, and one that needs more raises ``ValueError`` stating how many.
    """

    __slots__ = ("_run",)

    def __init__(self, run: TrainingRun):
        self._run = run

    def _header(self, detail=None):
        run = self._run.run
        run_id = run.get("run_id")
        name = run.get("run_name") or (
            str(run_id)[:_RUN_ID_LABEL] if run_id else self._run.run_dir.name
        )
        line = (
            f"run {name} · status {run.get('status', 'unrecorded')} · "
            f"recorded through step {self._run.last_step}"
        )
        return line if detail is None else f"{line}\n{detail}"

    def _series(self, kind, names):
        records = getattr(self._run, kind)
        path = self._run.run_dir / _RECORD_FILES[kind]
        if not records["series"]:
            raise ValueError(f"no {kind} records in {path}")
        missing = [name for name in names if name not in records["series"]]
        if missing:
            raise ValueError(f"{path} records no series {missing}")
        return records

    def _training(self, names, ax) -> Figure:
        records = self._series("training", names)
        return panels.plot_records(records, metric_names=names, header=self._header(), ax=ax)

    def _evaluation(self, names, metric, window, heads, legs, ax) -> Figure:
        from pymc_mktfm.training.evaluation import metric_units

        records = self._series("evaluation", names)
        last = records["records"][-1]
        # Each leg record labels its heads; the conditional label states its sampling mode.
        labels = {
            head: label
            for leg in legs
            for head, label in ((last.get("legs") or {}).get(leg, {}).get("labels") or {}).items()
        }
        modes = " · ".join(
            f"{head}: {labels.get(head.split('/')[0], 'sampling mode unrecorded')}"
            for head in heads
        )
        try:
            units = metric_units(metric)
        except ValueError:  # a metric recorded by another release
            units = "unknown to this release"
        detail = (
            f"{metric} · units: {units} · {_scope(window)} · "
            f"{', '.join(legs)} {'leg' if len(legs) == 1 else 'legs'} · "
            f"{last.get('n_samples', 'unrecorded')} draws per world\n{modes}"
        )
        return panels.plot_records(
            records, kind="evaluation", metric_names=names, header=self._header(detail), ax=ax
        )

    def losses(self, *, ax=None) -> Figure:
        """Every recorded objective term, including reconstructed raw terms."""
        names = [name for name in self._run.training["series"] if "loss" in name]
        if self._run.training["series"] and not names:
            raise ValueError(f"{self._run.run_dir / 'metrics.jsonl'} records no objective terms")
        return self._training(names, ax)

    def learning_rate(self, *, ax=None) -> Figure:
        return self._training(["lr"], ax)

    def gradients(self, *, ax=None) -> Figure:
        """Pre-clipping gradient norm and whether clipping applied."""
        return self._training(["grad_norm", "grad_clipped"], ax)

    def evaluation(
        self, metric, head="conditional", window="query", legs=("holdout", "seen"), *, ax=None
    ) -> Figure:
        """One evaluation metric of one head and window across legs.

        The recorded key is ``{leg}/{head}/{window}/{metric}``; for the
        structure head pass the edge family as ``window`` (``head=
        "structure_head", window="cy"``), for router graph statistics
        ``head="joint_router", window="graph"``.
        """
        legs = (legs,) if isinstance(legs, str) else tuple(legs)
        names = [f"flat/{leg}/{head}/{window}/{metric}" for leg in legs]
        return self._evaluation(names, metric, window, (head,), legs, ax)

    def heads(self, metric, window="query", leg="holdout", *, ax=None) -> Figure:
        """One metric on one leg for every head that records it.

        The joint and conditional heads record decomposition metrics per window
        (``"query"`` or ``"context"``). Edge scores are recorded per edge family
        by the structure head and by the joint router's edge marginals (the
        joint head's graph probabilities): pass the family as ``window``, for
        example ``heads("auroc", window="cy")``.
        """
        names = {head: f"flat/{leg}/{head}/{window}/{metric}" for head in _HEADS}
        recorded = {
            head: name for head, name in names.items() if name in self._run.evaluation["series"]
        }
        shown = recorded or names
        return self._evaluation(list(shown.values()), metric, window, tuple(shown), (leg,), ax)

    def strata(
        self,
        metric="coverage",
        by="true_size_bin",
        *,
        head="conditional",
        window="query",
        leg="holdout",
        step=None,
        ax=None,
    ) -> Figure:
        """One stratified cell value per stratum, from a recorded evaluation's strata table.

        Reads ``evaluation_tables/step_<N:09d>.json`` of ``step`` (the last
        recorded evaluation by default) and draws its ``leg`` leg's
        ``strata[head][window]`` table; ``metric`` and ``by`` are those of
        :meth:`DiagnosticsPlots.strata`, for example ``strata("coverage",
        by="active_channels")`` or ``strata("signed_residual",
        by="true_size_bin")``. A step without that table raises ``ValueError``.
        """
        evaluation_path = self._run.run_dir / _RECORD_FILES["evaluation"]
        steps = self._run.evaluation["steps"] or []
        if not steps:
            raise ValueError(f"no evaluation records in {evaluation_path}")
        step = steps[-1] if step is None else step
        if step not in steps:
            raise ValueError(f"{evaluation_path} records no evaluation at step {step}")
        path = self._run.run_dir / "evaluation_tables" / f"step_{int(step):09d}.json"
        if not path.is_file():
            raise ValueError(f"{path} not found; this run recorded no tables at step {step}")
        tables = _read_json(path, "evaluation tables")
        try:
            table = tables["legs"][leg]["strata"][head][window]
        except (KeyError, TypeError):
            raise ValueError(
                f"{path} records no {head} {window} strata for the {leg} leg"
            ) from None
        record = self._run.evaluation["records"][steps.index(step)]
        label = ((record.get("legs") or {}).get(leg, {}).get("labels") or {}).get(
            head, "sampling mode unrecorded"
        )
        detail = (
            f"{leg} leg · step {step} · {window} window · "
            f"{record.get('n_samples', 'unrecorded')} draws per world\n{head}: {label}"
        )
        return panels.plot_strata_table(
            table, metric=metric, by=by, header=self._header(detail), ax=ax
        )

    def corpus(self, *, ax=None) -> Figure:
        """Training-corpus composition, as ``run.json`` recorded it when training started.

        The figure has six axes, so ``ax=`` raises ``ValueError``.
        """
        summary = self._run.run.get("corpus_summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{self._run.run_dir / 'run.json'} records no corpus_summary")
        if ax is not None:
            raise ValueError("the corpus figure needs 6 axes; ax= draws only single-axes figures")
        return panels.plot_corpus_summary(summary, header=self._header())

    def corpus_support(self, *, ax=None) -> Figure:
        """Context-outcome location and scale of the corpus worlds, from ``corpus_support.json``.

        Training writes the file when it starts, from at most
        ``CORPUS_SUPPORT_MAX_WORLDS`` evenly spaced worlds of both splits. The
        figure has three axes, so ``ax=`` raises ``ValueError``; so does a file
        that does not decode or lacks a finite mean and std for a world.
        """
        path = self._run.run_dir / "corpus_support.json"
        if not path.is_file():
            raise ValueError(f"{path} not found; this run recorded no corpus support")
        if ax is not None:
            raise ValueError(
                "the corpus-support figure needs 3 axes; ax= draws only single-axes figures"
            )
        summary = _read_json(path, "corpus support")
        problem = _corpus_support_problem(summary)
        if problem is not None:
            raise ValueError(f"{path}: {problem}")
        from pymc_mktfm.plotting.comparisons import plot_corpus_support

        return plot_corpus_support(summary)
