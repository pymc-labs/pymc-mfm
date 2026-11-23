"""Hosted River execution and conversion to the common result tree."""

from __future__ import annotations

import json
from datetime import date as calendar_date
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from .client import Analysis, Client, Gone, curve
from .client._dates import check_plan_follows_history, check_weekly_dates
from .client._http import _service_options
from .client._wire import iso_dates, require_columns
from .client.plots import curve_points

if TYPE_CHECKING:
    from xarray import DataArray, Dataset, DataTree

_BASELINE = "baseline"
_DRAWS_SCHEMA = "river-draws/1"
_UNITS = "outcome"
_WAIT_TIMEOUT = 30 * 60
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


class HostedEngine:
    def __init__(self, *, roles: dict[str, Any], base_url: str | None, token: str | None) -> None:
        self._roles = roles.copy()
        self._base_url, self._token = _service_options(base_url, token)

    def _client(self) -> Client:
        return Client(base_url=self._base_url, api_key=self._token)

    def _analyze(
        self,
        client: Client,
        context: pd.DataFrame,
        plan: pd.DataFrame,
        *,
        n_samples: int,
        random_seed: int | None,
        explain_history: bool,
        curves: tuple[dict[str, Any], ...] = (),
    ) -> Analysis:
        roles = self._roles
        columns = {
            "date": roles["date_column"],
            "outcome": roles["target_column"],
            "spend": roles["channel_columns"],
            "controls": roles["control_columns"],
        }
        try:
            dataset = client.datasets.create(context, **columns, reuse=True)
        except Gone:
            # The content key can still refer to an expired dataset that is no longer listed.
            dataset = client.datasets.create(context, **columns, fresh=True)
        analysis = cast(
            Analysis,
            client.analyze(
                dataset,
                plan=plan,
                explain_history=explain_history,
                fresh=True,
                draws=n_samples,
                seed=random_seed,
                curves=curves,
            ),
        )
        return analysis.wait(timeout=_WAIT_TIMEOUT, announce=False)

    def inference(
        self,
        context: pd.DataFrame,
        query: pd.DataFrame,
        *,
        n_samples: int,
        random_seed: int | None,
        revision: int,
    ) -> DataTree:
        xr = _require_xarray()
        roles = self._roles
        date = roles["date_column"]
        target = roles["target_column"]
        channels = roles["channel_columns"]
        controls = roles["control_columns"]
        inputs = [date, *channels, *controls]
        context, context_days = _ordered(context, date, [*inputs, target], "context", least=2)
        # A reused dataset bypasses the client's upload-time cadence check.
        check_weekly_dates(context_days, "context")
        query, query_days = _ordered(query, date, inputs, "query", least=1)
        check_plan_follows_history(
            {"dates": query_days}, calendar_date.fromisoformat(context_days[-1])
        )
        context_dates = pd.to_datetime(context_days).to_numpy(dtype="datetime64[ns]")
        query_dates = pd.to_datetime(query_days).to_numpy(dtype="datetime64[ns]")

        with self._client() as client:
            analysis = self._analyze(
                client,
                context.loc[:, [*inputs, target]],
                query.loc[:, inputs],  # Query outcomes are for plotting only.
                n_samples=n_samples,
                random_seed=random_seed,
                explain_history=True,
            )
            _baseline(analysis.table, analysis.id)
            draws = _baseline_draws(analysis.draws([_BASELINE]))

        seed = {} if random_seed is None else {"random_seed": random_seed}
        provenance = json.dumps(
            {"engine": "hosted", "analysis_id": analysis.id, "draws_schema": _DRAWS_SCHEMA, **seed}
        )
        root = xr.Dataset(
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
        for window, frame, dates, observed in (
            (
                "context",
                context,
                context_dates,
                context[target].to_numpy(dtype=np.float64, copy=True),
            ),
            ("query", query, query_dates, _query_outcome(query, target)),
        ):
            attrs = {
                "window": window,
                "provenance": provenance,
                "units": _UNITS,
                "n_samples": n_samples,
                "context_length": len(context),
            }
            posterior, labels, families = _posterior(
                draws, window, dates, channels, controls, n_samples
            )
            path = f"/{window}"
            groups[path] = xr.Dataset(attrs=attrs)
            posterior.attrs = attrs
            groups[f"{path}/posterior"] = _freeze(posterior)
            groups[f"{path}/constant_data"] = _freeze(
                xr.Dataset(
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
                    xr.Dataset({"outcome": ("date", observed)}, coords={"date": dates})
                )
        return xr.DataTree.from_dict(groups)

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
        xr = _require_xarray()
        context = self._frame(tree, "context")
        plan = self._frame(tree, "query").iloc[:weeks]
        with np.errstate(over="ignore", invalid="ignore"):
            total_media = _matrix(plan, channels).sum(axis=0)
            media = total_media[:, None] * np.asarray(levels)
        if not (np.isfinite(media).all() and (media >= 0).all()):
            raise ValueError("Hosted response curve media must be finite and nonnegative")

        with self._client() as client:
            analysis = self._analyze(
                client,
                context,
                plan,
                n_samples=n_samples,
                random_seed=random_seed,
                explain_history=False,
                curves=tuple(curve(channel, levels) for channel in channels),
            )
            table = analysis.table
            baseline = _baseline(table, analysis.id)
        stats = _curve_stats(table, baseline, channels, levels)
        dims = ("channel", "level")
        coords = {"channel": np.asarray(channels, dtype=str), "level": np.asarray(levels)}
        path = "/response_curves"
        groups = {
            node.path: node.to_dataset(inherit=False)
            for node in tree.subtree
            if node.path != path and not node.path.startswith(f"{path}/")
        }
        groups[path] = xr.Dataset(
            attrs={
                "weeks": weeks,
                "n_samples": n_samples,
                "units": _UNITS,
                "source_inference_revision": tree.attrs["inference_revision"],
                "source_provenance": tree["/query"].attrs["provenance"],
                **({} if random_seed is None else {"random_seed": random_seed}),
            }
        )
        groups[f"{path}/summary"] = _freeze(
            xr.Dataset(
                {
                    "mean": (dims, stats[..., 0]),
                    "low": (dims, stats[..., 1]),
                    "high": (dims, stats[..., 2]),
                    "media": (dims, media),
                    "total_media": ("channel", total_media),
                },
                coords=coords,
                attrs={"interval": 0.9},
            )
        )
        return xr.DataTree.from_dict(groups, name=tree.name)

    def _frame(self, tree: DataTree, window: str) -> pd.DataFrame:
        """Reconstruct only the uploaded columns from a completed inference."""
        roles = self._roles
        window_node = tree.children[window]
        constants = window_node.children["constant_data"].to_dataset(inherit=False)
        columns = {roles["date_column"]: constants["date"].to_numpy()}
        for variable, dim in (("media", "channel"), ("controls", "control")):
            values = constants[variable]
            for name in roles[f"{dim}_columns"]:
                columns[name] = values.sel({dim: name}).to_numpy()
        if window == "context":
            observed = window_node.children["observed_data"].to_dataset(inherit=False)
            columns[roles["target_column"]] = observed["outcome"].to_numpy()
        return pd.DataFrame(columns)


def _require_xarray():
    try:
        import h5netcdf  # noqa: F401
        import xarray
    except ImportError as error:
        raise ImportError(
            "Hosted model results need xarray and h5netcdf; install pymc-mfm[draws]."
        ) from error
    return xarray


def _ordered(
    frame: pd.DataFrame, date: str, required: list[str], window: str, *, least: int
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(frame, pd.DataFrame) or len(frame) < least:
        raise ValueError(f"{window} must be a pandas DataFrame with at least {least} rows")
    if not frame.columns.is_unique:
        raise ValueError(f"{window} column names must be unique")
    require_columns(frame, required)
    if pd.api.types.is_numeric_dtype(frame[date].dtype):
        raise ValueError(f"{window} column {date!r} must contain dates, not numeric row labels")
    days = iso_dates(frame[date], name=date)
    order = sorted(range(len(days)), key=days.__getitem__)
    return frame.iloc[order], [days[index] for index in order]


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
    channels: tuple[str, ...],
    controls: tuple[str, ...],
    n_samples: int,
) -> tuple[Dataset, list[str], list[str]]:
    import xarray as xr

    path = f"/{window}/posterior"
    prefix = "history_" if window == "context" else ""
    wire_date = f"{prefix}date"
    index = _coordinate(draws, wire_date, path)
    # Compare the full labels, not just axis lengths or calendar days.
    if index.dtype.kind not in "MOUS" or not np.array_equal(
        pd.to_datetime(index, errors="coerce").to_numpy(), dates
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
    return xr.Dataset(variables, coords=coords), labels, families


def _curve_stats(
    table: pd.DataFrame,
    baseline: pd.DataFrame,
    channels: tuple[str, ...],
    levels: tuple[float, ...],
) -> np.ndarray:
    columns = ["mean", "p05", "p95"]
    # The plan already contains only the swept weeks, so whole-query bands cover that window.
    successful = cast("pd.DataFrame", table.loc[table["status"].eq("ok")])
    points = curve_points(successful)
    points = points[(points["base_scenario_id"] == _BASELINE) & (points["level"] != 1)]
    points = points.set_index(["varied_channel", "level"])
    if not points.index.is_unique:
        raise ValueError("Hosted response curves returned duplicate channel/level points")
    index = pd.MultiIndex.from_product([channels, levels])
    stats = points.reindex(index)[columns].to_numpy(dtype=np.float64, copy=True)
    stats = stats.reshape(len(channels), len(levels), len(columns))
    if 1.0 in levels:
        level_one = levels.index(1.0)
        for row, channel in enumerate(channels):
            base = baseline.loc[baseline["component"] == channel, columns]
            band = base.to_numpy(dtype=np.float64)
            if len(base) != 1 or not np.isfinite(band).all():
                raise ValueError(
                    f"Hosted response curves are missing baseline statistics for {channel!r}"
                )
            stats[row, level_one] = band[0]
    missing = np.isnan(stats).all(axis=-1)
    valid = np.isfinite(stats).all(axis=-1) & (stats[..., 1] <= stats[..., 2])
    invalid = np.argwhere(~(missing | valid))
    if invalid.size:
        row, col = invalid[0]
        raise ValueError(
            f"Hosted response curve statistics for {channels[row]!r} at level "
            f"{levels[col]} do not have a finite 90% band"
        )
    if not valid.any():
        raise RuntimeError("Hosted response curves have no successful requested points")
    return stats
