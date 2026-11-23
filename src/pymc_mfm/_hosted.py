"""Hosted River execution and conversion to the common result tree."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date as calendar_date
from numbers import Number
from typing import Any, cast

import numpy as np
import pandas as pd
from xarray import DataArray, Dataset, DataTree

from .client import Analysis, Client, Gone, curve
from .client._dates import check_plan_follows_history, check_weekly_dates
from .client._http import _SETTING_PADDING, _service_options, _setting
from .client._wire import dataset_body, plan_body, require_columns
from .client.plots import _warning_list

_BASELINE = "baseline"
_BAND = ["mean", "p05", "p95"]
_DRAWS_SCHEMA = "river-draws/1"
_POLL_SECONDS = 2.0
_UNITS = "outcome"
# River row statuses of a plan that has no result. Only these leave a gap in a curve.
_FAILED = frozenset({"failed", "cancelled"})
_SCALARS = {
    "baseline": "baseline",
    "latent": "latent",
    "indirect_cc": "indirect:cc",
    "indirect_zc": "indirect:zc",
    "indirect_dc": "indirect:dc",
}
# Flat files use the raw column name, but these names can describe something else.
_RESERVED = {
    "outcome",
    "graph",
    "channel_contribution",
    "control_contribution",
    *_SCALARS,
    *_SCALARS.values(),
}


@dataclass(frozen=True, eq=False)
class _Submitted:
    """Private copies of what the latest successful inference sent to River.

    Sweeps read these, never the public tree, which callers can edit. ``tree`` is the
    result the model holds now; only a sweep of that tree may use these inputs.
    """

    tree: DataTree
    context: pd.DataFrame  # the uploaded history columns, in date order
    query: pd.DataFrame  # the plan columns, in date order
    query_dates: np.ndarray  # the caller's query timestamps, as datetime64[ns]


class HostedEngine:
    def __init__(self, *, roles: dict[str, Any], base_url: str | None, token: str | None) -> None:
        self._roles = roles.copy()
        self._base_url, self._token = _service_options(base_url, token, key_parameter="token")
        if _is_hub_token(self._token):
            raise ValueError(
                "The River API key (token= or MFM_API_KEY) is a Hugging Face token, but the "
                "hosted engine sends it to River. Use your River API key from the Connect page."
            )
        self._latest: _Submitted | None = None

    def _client(self) -> Client:
        return Client(base_url=self._base_url, api_key=self._token)

    @contextmanager
    def _analysis(
        self,
        context: pd.DataFrame,
        plan: pd.DataFrame,
        *,
        n_samples: int,
        random_seed: int | None,
        explain_history: bool,
        curves: tuple[dict[str, Any], ...] = (),
    ) -> Iterator[tuple[tuple[str, ...], Analysis]]:
        """Upload the history, run one analysis to the end and yield the dataset warnings and it.

        Both request bodies are built and checked first, so a bad value sends nothing. Once
        River accepts the analysis, any failure, here or in the caller's block, names it.
        """
        roles = self._roles
        columns = {
            "date": roles["date_column"],
            "outcome": roles["target_column"],
            "spend": roles["channel_columns"],
            "controls": roles["control_columns"],
        }
        history = dataset_body(context, **columns, name=None)
        # A reused dataset bypasses the client's upload-time cadence check.
        check_weekly_dates(history["dates"], "context")
        wire_plan = plan_body(
            plan,
            channels=columns["spend"],
            controls=columns["controls"],
            date_column=columns["date"],
        )
        check_plan_follows_history(wire_plan, calendar_date.fromisoformat(history["dates"][-1]))
        _check_values(history, wire_plan, columns["outcome"])
        with self._client() as client:
            try:
                dataset = client.datasets.create(context, **columns, reuse=True)
            except Gone:
                # The content key can still refer to an expired dataset that is no longer listed.
                dataset = client.datasets.create(context, **columns, fresh=True)
            analysis = cast(
                Analysis,
                client.analyze(
                    dataset,
                    plan=wire_plan,
                    explain_history=explain_history,
                    fresh=True,
                    draws=n_samples,
                    seed=random_seed,
                    curves=curves,
                ),
            )
            with _resumable(analysis.id):
                _wait(analysis)
                yield dataset.warnings, analysis

    def inference(
        self,
        context: pd.DataFrame,
        query: pd.DataFrame,
        *,
        n_samples: int,
        random_seed: int | None,
        revision: int,
    ) -> DataTree:
        roles = self._roles
        date = roles["date_column"]
        target = roles["target_column"]
        channels = roles["channel_columns"]
        controls = roles["control_columns"]
        inputs = [date, *channels, *controls]
        context, context_dates, context_days = _ordered(
            context, date, [*inputs, target], "context", least=2
        )
        query, query_dates, query_days = _ordered(query, date, inputs, "query", least=1)
        history = context.loc[:, [*inputs, target]].copy()
        plan = query.loc[:, inputs].copy()  # Query outcomes are for plotting only.

        with self._analysis(
            history, plan, n_samples=n_samples, random_seed=random_seed, explain_history=True
        ) as (dataset_warnings, analysis):
            baseline = _baseline(analysis.table, analysis.id)
            draws = _baseline_draws(analysis.draws([_BASELINE]))
            notes = _distinct(
                [
                    *dataset_warnings,
                    *(note for value in baseline["warnings"] for note in _warning_list(value)),
                ]
            )
            seed = {} if random_seed is None else {"random_seed": random_seed}
            provenance = json.dumps(
                {
                    "engine": "hosted",
                    "analysis_id": analysis.id,
                    "draws_schema": _DRAWS_SCHEMA,
                    **seed,
                    **({"warnings": notes} if notes else {}),
                }
            )
            root = Dataset(
                attrs={
                    "schema_version": 1,
                    "inference_revision": revision,
                    "date_column": date,
                    "target_column": target,
                    "channel_columns": json.dumps(channels),
                    "control_columns": json.dumps(controls),
                    "marginal_windows": "context and query are separate marginals, not joint paths",
                    **seed,
                }
            )
            groups = {"/": root}
            for window, frame, dates, days, observed in (
                (
                    "context",
                    history,
                    context_dates,
                    context_days,
                    history[target].to_numpy(dtype=np.float64, copy=True),
                ),
                ("query", plan, query_dates, query_days, _query_outcome(query, target)),
            ):
                attrs = {
                    "window": window,
                    "provenance": provenance,
                    "units": _UNITS,
                    "n_samples": n_samples,
                    "context_length": len(history),
                }
                posterior, labels, families = _posterior(
                    draws, window, dates, days, channels, controls, n_samples
                )
                path = f"/{window}"
                groups[path] = Dataset(attrs=attrs)
                posterior.attrs = attrs
                groups[f"{path}/posterior"] = _freeze(posterior)
                groups[f"{path}/constant_data"] = _freeze(
                    Dataset(
                        {
                            "component_family": ("component", np.asarray(families, dtype=str)),
                            "media": (("date", "channel"), _matrix(frame, channels)),
                            "controls": (("date", "control"), _matrix(frame, controls)),
                        },
                        coords={
                            "date": dates,
                            "channel": np.asarray(channels, dtype=str),
                            "control": np.asarray(controls, dtype=str),
                            "component": np.asarray(labels, dtype=str),
                        },
                    )
                )
                if observed is not None:
                    groups[f"{path}/observed_data"] = _freeze(
                        Dataset({"outcome": ("date", observed)}, coords={"date": dates})
                    )
            tree = DataTree.from_dict(groups)
            _warn(notes)
        # Commit only after the warnings: one raised as an error discards this tree.
        self._latest = _Submitted(tree, history, plan, query_dates)
        return tree

    def response_curves(
        self,
        tree: DataTree,
        *,
        levels: tuple[float, ...],
        channels: tuple[str, ...],
        weeks: int,
        n_samples: int,
        random_seed: int | None,
    ) -> DataTree:
        latest = self._latest
        if latest is None or latest.tree is not tree:
            raise RuntimeError(
                "Run model.inference() again: this tree is not the model's latest hosted result."
            )
        plan = latest.query.iloc[:weeks]
        with np.errstate(over="ignore", invalid="ignore"):
            total_media = _matrix(plan, channels).sum(axis=0)
            media = total_media[:, None] * np.asarray(levels)
        if not (np.isfinite(media).all() and (media >= 0).all()):
            raise ValueError("Hosted response curve media must be finite and nonnegative")

        with self._analysis(
            latest.context,
            plan,
            n_samples=n_samples,
            random_seed=random_seed,
            explain_history=False,
            curves=tuple(curve(channel, levels) for channel in channels),
        ) as (_, analysis):
            table = analysis.table
            baseline = _baseline(table, analysis.id)
            stats, notes = _curve_summary(table, baseline, channels, levels, analysis.id)
            dims = ("channel", "level")
            coords = {
                "channel": np.asarray(channels, dtype=str),
                "level": np.asarray(levels),
                "date": latest.query_dates[:weeks],
            }
            path = "/response_curves"
            groups = {
                node.path: node.to_dataset(inherit=False)
                for node in tree.subtree
                if node.path != path and not node.path.startswith(f"{path}/")
            }
            groups[path] = Dataset(
                attrs={
                    "weeks": weeks,
                    "n_samples": n_samples,
                    "units": _UNITS,
                    "source_inference_revision": tree.attrs["inference_revision"],
                    "source_provenance": tree["/query"].attrs["provenance"],
                    "provenance": json.dumps({"engine": "hosted", "analysis_id": analysis.id}),
                    **({} if random_seed is None else {"random_seed": random_seed}),
                }
            )
            groups[f"{path}/summary"] = _freeze(
                Dataset(
                    {
                        "mean": (dims, stats[..., 0]),
                        "low": (dims, stats[..., 1]),
                        "high": (dims, stats[..., 2]),
                        "media": (dims, media),
                        "total_media": ("channel", total_media),
                        "warning": (dims, notes),
                    },
                    coords=coords,
                    attrs={"interval": 0.9},
                )
            )
            updated = DataTree.from_dict(groups, name=tree.name)
            _warn(_distinct(note for point in notes.flat for note in point.split("\n")))
        # After the warnings, as in inference: the model now holds this tree, so the next
        # sweep starts from it.
        self._latest = replace(latest, tree=updated)
        return updated


def _is_hub_token(key: str) -> bool:
    """Whether ``key`` is a Hugging Face credential: its ``hf_`` form or a Hub variable's value."""
    value = key.strip(_SETTING_PADDING)
    if value.startswith("hf_"):
        return True
    hub_tokens = (_setting(None, name) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    return any(value == hub for hub in hub_tokens if hub)


@contextmanager
def _resumable(analysis_id: str) -> Iterator[None]:
    """Name an accepted analysis on any later failure, so it is inspected, not run again."""
    try:
        yield
    except BaseException as error:
        quoted = json.dumps(analysis_id)
        error.add_note(
            f"River accepted analysis {quoted}. Calling the model again starts a new analysis. "
            "Inspect this one with "
            f"pymc_mfm.client.Client(base_url=..., api_key=...).get({quoted})."
        )
        raise


def _distinct(messages: Iterable[str]) -> list[str]:
    """River warnings in first-seen order, each on one line: stored text joins them with "\\n"."""
    return list(dict.fromkeys(line for message in messages if (line := " ".join(message.split()))))


def _warn(notes: list[str]) -> None:
    for note in notes:
        # Above this helper: the engine method, the Model method, then the caller's line.
        warnings.warn(note, UserWarning, stacklevel=4)


def _check_values(history: dict[str, Any], plan: dict[str, Any], target: str) -> None:
    """Refuse what the local engine and River refuse, before any request.

    Spend must be nonnegative, outcomes positive, and a channel or control that is zero
    in every history week must stay zero in the plan (River's ``history_all_zero``).
    """
    for window, body in (("context", history), ("query", plan)):
        negative = [name for name, values in body["spend"].items() if any(v < 0 for v in values)]
        if negative:
            raise ValueError(
                f"{window} channel spend must be nonnegative; negative values in {negative}"
            )
    if any(value <= 0 for value in history["outcome"]):
        raise ValueError(
            f"context outcomes must be positive; {target!r} has zero or negative values"
        )
    for role in ("spend", "controls"):
        silent = [
            name
            for name, values in history[role].items()
            if not any(values) and any(plan[role][name])
        ]
        if silent:
            raise ValueError(
                f"{silent} are zero in every context week but not in the query; "
                "a series with no history cannot be planned"
            )


def _wait(analysis: Analysis) -> None:
    """Poll until River ends the analysis, with no time limit: the model cannot resume one.

    A failed analysis stops at once, since its baseline failed and no window has a result;
    a deleted one raises ``Gone``.
    """
    while True:
        state = analysis.status()
        if state.status == "failed":
            raise RuntimeError(f"Hosted analysis {analysis.id!r} has no successful baseline run")
        if state.status == "deleted":
            raise Gone(410, "deleted", f"Analysis {analysis.id} was deleted")
        if state.finished:
            return
        time.sleep(_POLL_SECONDS)


def _ordered(
    frame: pd.DataFrame, date: str, required: list[str], window: str, *, least: int
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Rows in date order, their own timestamps and the calendar days River receives.

    Dates follow the local engine's rules: no numeric labels and no time zones.
    """
    if not isinstance(frame, pd.DataFrame) or len(frame) < least:
        raise ValueError(f"{window} must be a pandas DataFrame with at least {least} rows")
    if not frame.columns.is_unique:
        raise ValueError(f"{window} column names must be unique")
    require_columns(frame, required)
    values = frame[date]
    if pd.api.types.is_numeric_dtype(values.dtype) or any(
        isinstance(value, (Number, np.bool_)) for value in values
    ):
        raise ValueError(f"{window} column {date!r} must contain dates, not numeric row labels")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
        if dates.tz is not None:
            raise ValueError("dates must be timezone-naive; convert them before inference")
        dates = dates.as_unit("ns")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid dates in {window} column {date!r}: {error}") from None
    if dates.hasnans:
        raise ValueError(f"{window} column {date!r} has missing dates")
    order = np.argsort(dates.asi8, kind="stable")
    dates = dates[order]
    return frame.iloc[order], dates.to_numpy(), dates.strftime("%Y-%m-%d").tolist()


def _matrix(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(names)].to_numpy(dtype=np.float64, copy=True, na_value=np.nan)


def _query_outcome(frame: pd.DataFrame, target: str) -> np.ndarray | None:
    if target not in frame:
        return None
    series = frame[target]
    if pd.api.types.is_datetime64_any_dtype(series.dtype) or pd.api.types.is_timedelta64_dtype(
        series.dtype
    ):
        return None
    try:
        series = cast("pd.Series", pd.to_numeric(series, errors="coerce"))
    except (TypeError, ValueError, OverflowError):
        series = series.map(_outcome_number)
    if pd.api.types.is_complex_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        return None
    values = series.to_numpy(dtype=np.float64, copy=True, na_value=np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return None
    values[~valid] = np.nan
    return values


def _outcome_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _freeze(dataset: Dataset) -> Dataset:
    for variable in dataset.variables.values():
        cast("np.ndarray", variable.data).setflags(write=False)
    return dataset


def _baseline(table: pd.DataFrame, analysis_id: str) -> pd.DataFrame:
    rows = cast("pd.DataFrame", table.loc[table["scenario_id"] == _BASELINE])
    status = cast("pd.Series", rows["status"])
    if rows.empty or not status.eq("ok").all():
        raise RuntimeError(f"Hosted analysis {analysis_id!r} has no successful baseline run")
    return rows


def _coordinate(draws: Dataset, dim: str, path: str) -> pd.Index:
    if dim not in draws.coords or draws.coords[dim].dims != (dim,):
        raise ValueError(f"{path}: hosted draws are missing the {dim!r} coordinate")
    index = draws.get_index(dim)
    if not index.is_unique or index.hasnans:
        raise ValueError(
            f"{path}: hosted {dim!r} coordinate must contain unique, nonmissing labels"
        )
    return index


def _baseline_draws(draws: Dataset) -> Dataset:
    path = "Hosted baseline draws"
    if draws.attrs.get("schema") != _DRAWS_SCHEMA:
        raise ValueError(f"{path}: expected schema {_DRAWS_SCHEMA!r}")
    scenarios = _coordinate(draws, "scenario", path)
    if tuple(scenarios) != (_BASELINE,) or draws.attrs.get("scenario_id", _BASELINE) != _BASELINE:
        raise ValueError(f"{path}: expected exactly the {_BASELINE!r} scenario")
    return draws.isel(scenario=0, drop=True)


def _contribution(draws: Dataset, name: str, dims: tuple[str, ...], path: str) -> DataArray:
    if name not in draws.data_vars:
        raise ValueError(f"{path}: hosted draws are missing {name!r}")
    variable = draws[name]
    if len(variable.dims) != len(dims) or set(variable.dims) != set(dims):
        raise ValueError(f"{path}: hosted {name!r} must have dimensions {dims}")
    if variable.dtype.kind not in "fiu":
        raise ValueError(f"{path}: hosted {name!r} must be numeric with dimensions {dims}")
    if not np.isfinite(variable.to_numpy()).all():
        raise ValueError(f"{path}: hosted {name!r} has nonfinite draws")
    return variable.transpose(*dims)


def _role_draws(
    draws: Dataset,
    *,
    prefix: str,
    date_dim: str,
    family: str,
    names: tuple[str, ...],
    path: str,
) -> np.ndarray:
    dims = ("draw", date_dim)
    grouped = f"{prefix}{family}_contribution"
    if grouped in draws.data_vars:
        variable = _contribution(draws, grouped, (*dims, family), path)
        labels = _coordinate(draws, family, path)
        if set(labels) != set(names):
            raise ValueError(f"{path}: hosted {family} labels {tuple(labels)} differ from {names}")
        if tuple(labels) != names:
            variable = variable.sel({family: list(names)})
        return variable.to_numpy().astype(np.float64, copy=False)
    # Early river-draws/1 files used one variable per column, as in the client fixture.
    if not _RESERVED.isdisjoint(names):
        raise ValueError(f"{path}: hosted draws are missing grouped variable {grouped!r}")
    columns = [_contribution(draws, f"{prefix}{name}", dims, path).to_numpy() for name in names]
    return np.stack(columns, axis=-1, dtype=np.float64)


def _posterior(
    draws: Dataset,
    window: str,
    dates: np.ndarray,
    days: list[str],
    channels: tuple[str, ...],
    controls: tuple[str, ...],
    n_samples: int,
) -> tuple[Dataset, list[str], list[str]]:
    """The window's draws, labelled with the caller's own ``dates``.

    River received only the calendar ``days``, so its labels are checked against those.
    """
    path = f"/{window}/posterior"
    prefix = "history_" if window == "context" else ""
    wire_date = f"{prefix}date"
    index = _coordinate(draws, wire_date, path)
    # Compare every label, not just the axis length.
    if index.dtype.kind not in "MOUS" or not np.array_equal(
        pd.to_datetime(index, errors="coerce").to_numpy(),
        np.asarray(days, dtype="datetime64[ns]"),
    ):
        raise ValueError(f"{path}: hosted {wire_date!r} labels differ from the {window} inputs")
    draw_index = _coordinate(draws, "draw", path)
    if len(draw_index) != n_samples:
        raise ValueError(f"{path}: expected {n_samples} draws, got {len(draw_index)}")

    dims = ("chain", "draw", "date")
    coords: dict[str, Any] = {"chain": [0], "draw": draw_index.to_numpy(), "date": dates}
    variables = {}
    labels, families = [], []
    if f"{prefix}baseline" not in draws.data_vars:
        raise ValueError(f"{path}: hosted draws are missing {prefix + 'baseline'!r}")
    for name, label in _SCALARS.items():
        wire_name = f"{prefix}{name}"
        if wire_name not in draws.data_vars:
            continue
        values = _contribution(draws, wire_name, ("draw", wire_date), path).to_numpy()
        variables[name] = (dims, values.astype(np.float64, copy=False)[None])
        labels.append(label)
        families.append(label.split(":")[0])

    # Match the local slot ordering: baseline, channels, controls, latent and indirect.
    slot = 1
    for family, names in (("channel", channels), ("control", controls)):
        if not names:
            continue
        values = _role_draws(
            draws, prefix=prefix, date_dim=wire_date, family=family, names=names, path=path
        )
        variables[f"{family}_contribution"] = ((*dims, family), values[None])
        coords[family] = np.asarray(names, dtype=str)
        labels[slot:slot] = [f"{family}:{name}" for name in names]
        families[slot:slot] = [family] * len(names)
        slot += len(names)

    # If the wire file also carries expected-outcome draws, check the decomposition.
    wire_outcome = f"{prefix}outcome"
    if wire_outcome in draws.data_vars:
        outcome = _contribution(draws, wire_outcome, ("draw", wire_date), path).to_numpy()
        total = variables["baseline"][1][0].copy()
        for name, (_, values) in variables.items():
            if name != "baseline":
                total += values[0].sum(axis=-1) if values.ndim == 4 else values[0]
        if not np.allclose(total, outcome, rtol=1e-5, atol=1e-5):
            raise ValueError(f"{path}: hosted contributions do not sum to its outcome draws")

    graph = draws.get(f"{prefix}graph")
    if (
        graph is not None
        and len(graph.dims) == 2
        and set(graph.dims) == {"draw", "edge"}
        and "edge" in graph.coords
    ):
        edges = graph.coords["edge"]
        if edges.dims == ("edge",) and edges.size:
            names = edges.to_numpy()
            values = graph.transpose("draw", "edge").to_numpy()
            if (
                all(isinstance(name, str) and name and "\x00" not in name for name in names)
                and len(set(names)) == len(names)
                and values.dtype.kind in "bfiu"
                and np.isin(values, (0, 1)).all()
            ):
                variables["graph"] = (("chain", "draw", "edge"), values.astype(np.int8)[None])
                coords["edge"] = names
    return Dataset(variables, coords=coords), labels, families


def _curve_summary(
    table: pd.DataFrame,
    baseline: pd.DataFrame,
    channels: tuple[str, ...],
    levels: tuple[float, ...],
    analysis_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """The reported mean, 5th and 95th percentiles and warnings at each channel and level.

    Only a point whose plan River reports failed is a gap: NaN statistics and no warning.
    Every other point needs one successful row with a finite band, so a missing plan or
    row raises ``ValueError`` instead of looking like a gap. A channel with no successful
    level other than 1 has no curve, so that raises ``RuntimeError``.
    """
    stats = np.full((len(channels), len(levels), len(_BAND)), np.nan)
    notes = np.full((len(channels), len(levels)), "", dtype=object)
    # The plan already contains only the swept weeks, so whole-query bands cover that window.
    curves = table.loc[table["kind"].eq("curve") & table["base_scenario_id"].eq(_BASELINE)]
    for row, channel in enumerate(channels):
        for column, level in enumerate(levels):
            point = _curve_point(curves, baseline, channel, level)
            if point is None:
                continue
            band = point[_BAND].to_numpy(dtype=np.float64, na_value=np.nan)
            if not (np.isfinite(band).all() and band[1] <= band[2]):
                raise ValueError(
                    f"Hosted response curve statistics for {channel!r} at level {level:g} "
                    "do not have a finite 90% band"
                )
            stats[row, column] = band
            notes[row, column] = "\n".join(_distinct(_warning_list(point["warnings"])))
    columns = [index for index, level in enumerate(levels) if level != 1]
    failed = [
        channel
        for channel, means in zip(channels, stats[:, columns, 0], strict=True)
        if np.isnan(means).all()
    ]
    if failed:
        raise RuntimeError(
            f"Hosted analysis {analysis_id!r} has no successful response curve level "
            f"other than 1 for channels {failed}"
        )
    return stats, notes.astype(str)


def _curve_point(
    curves: pd.DataFrame, baseline: pd.DataFrame, channel: str, level: float
) -> pd.Series | None:
    """The one reported row of ``channel`` at ``level``, or ``None`` if its plan failed."""
    if level == 1:
        # River does not run level 1 again: it is the baseline plan.
        rows = baseline.loc[baseline["component"].eq(channel)]
        if len(rows) != 1:
            raise ValueError(
                f"Hosted response curves are missing baseline statistics for {channel!r}"
            )
        return rows.iloc[0]
    plan = curves.loc[curves["varied_channel"].eq(channel) & curves["level"].eq(level)]
    if plan.empty:
        raise ValueError(
            f"Hosted response curves are missing the {channel!r} plan at level {level:g}"
        )
    status = set(plan["status"])
    if status <= _FAILED:
        return None
    rows = plan.loc[plan["component"].eq(channel)]
    if len(rows) > 1:
        raise ValueError("Hosted response curves returned duplicate channel/level points")
    if status != {"ok"} or rows.empty:
        raise ValueError(
            f"Hosted response curves are missing a successful {channel!r} row at level {level:g}"
        )
    return rows.iloc[0]
