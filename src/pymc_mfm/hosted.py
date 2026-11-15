"""A model that runs on the hosted service, with the ``add_context`` / ``inference`` API of ``MFM``.

Get one with ``pymc_mfm.from_pretrained("pymc-labs/mfm", profile="river-1", api_key=...)``.
It needs no PyTorch: it talks to the service through :mod:`pymc_mfm.client` and returns the
same :class:`~pymc_mfm.inference.result.DecompositionResult` as a local model. Reading draws
needs the ``draws`` extra (``pip install "pymc-mfm[client]"``).

Differences from a local ``MFM``, all stated again in the method docstrings:

- ``inference`` needs ``n_samples >= 100``. The local default is 32, so a call without
  ``n_samples`` raises ``ValueError`` before any request.
- The run uses the service's default seed (0). There is no ``seed`` argument.
- Dates must be weekly.
- The draws file is float32 and holds no graphs. The result keeps float32 (it is not
  widened to float64), records ``provenance["draws_dtype"]``, and has one all-padded
  placeholder edge. Graph plots and graph methods raise :class:`HostedUnsupported`.
- Methods that need the model in this process raise :class:`HostedUnsupported`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymc_mfm.client import Client, ClientError
from pymc_mfm.client._dates import check_weekly_dates
from pymc_mfm.client._wire import iso_dates

if TYPE_CHECKING:
    from typing import Self

    from pymc_mfm.client import Dataset
    from pymc_mfm.inference.result import DecompositionResult

#: The Hugging Face repository and profile the hosted service runs.
SERVED_REPO = "pymc-labs/mfm"
SERVED_PROFILE = "river-1"
#: The fewest draws the service accepts.
MIN_HOSTED_DRAWS = 100
#: ``MFM.inference`` default (``pymc_mfm.model.DEFAULT_N_SAMPLES``); a test keeps them equal.
LOCAL_DEFAULT_N_SAMPLES = 32
DRAWS_SCHEMA = "river-draws/1"

#: Public ``MFM`` methods that need the model in this process.
_REFUSED_METHODS = (
    "to",
    "parameters",
    "compile_sampling",
    "decompose",
    "build_idata",
    "input_usage",
)
#: Plot and export methods of a result that read graphs, which the hosted file lacks.
GRAPH_METHODS = (
    "graph",
    "dag",
    "dot",
    "graph_inspector",
    "graph_posterior",
    "cardinality",
    "graph_statistics",
    "cycles",
    "zero_atoms",
    "uncertainty",
    "support",
    "set_posterior",
)
_SCALARS = (
    ("baseline", "baseline"),
    ("latent", "latent"),
    ("indirect_cc", "indirect:cc"),
    ("indirect_zc", "indirect:zc"),
    ("indirect_dc", "indirect:dc"),
)


class HostedUnsupported(ClientError):
    """A method that needs the model in this process, or graphs, on a hosted model or result.

    The message names the method.
    """

    def __init__(self, method: str, reason: str | None = None) -> None:
        self.method = method
        super().__init__(
            f"{method} is not available on the hosted model"
            + (f": {reason}" if reason else ".")
            + " Load the model locally (without api_key) to use it."
        )


def _refuse(method: str):
    def refused(self, *args: Any, **kwargs: Any):
        raise HostedUnsupported(f"{method}()", "it needs the model weights in this process")

    refused.__name__ = method
    refused.__doc__ = "Not available on the hosted model: raises ``HostedUnsupported``."
    return refused


class HostedMFM:
    """The hosted model: ``add_context`` then ``inference``, as on a local ``MFM``.

    Build it with ``pymc_mfm.from_pretrained(..., api_key=...)``. ``client`` is a
    :class:`pymc_mfm.client.Client`.
    """

    def __init__(self, client: Client, *, repo: str = SERVED_REPO, profile: str = SERVED_PROFILE):
        self._client = client
        self.repo = repo
        self.profile = profile
        self._context: _Context | None = None

    @classmethod
    def connect(
        cls,
        path_or_repo: Any,
        *,
        api_key: str,
        base_url: str | None = None,
        revision: str | None = None,
        profile: str | None = None,
        cache_dir: Any = None,
        local_files_only: bool = False,
        token: Any = None,
        device: Any = "cpu",
        transport: Any = None,
    ) -> HostedMFM:
        """Check the arguments, then make the client. Nothing is sent."""
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a nonempty string")
        if str(path_or_repo) != SERVED_REPO:
            raise ValueError(
                f"The hosted service runs {SERVED_REPO!r}, not {str(path_or_repo)!r}. "
                "Load a local folder or another repository without api_key."
            )
        if profile is not None and profile != SERVED_PROFILE:
            raise ValueError(
                f"The hosted service runs profile {SERVED_PROFILE!r}, not {profile!r}."
            )
        if revision is not None:
            raise ValueError(
                "revision cannot be checked on the hosted service: it does not report its "
                "model revision before a run. Omit revision; each result records "
                'provenance["model_revision"].'
            )
        if cache_dir is not None or local_files_only or token not in (None, False):
            raise ValueError(
                "cache_dir, local_files_only and token apply to Hugging Face loading, "
                "not to the hosted service"
            )
        if device != "cpu":
            raise ValueError("device applies to local models, not to the hosted service")
        client = Client(base_url, api_key, transport=transport)
        return cls(client, repo=SERVED_REPO, profile=SERVED_PROFILE)

    def __repr__(self) -> str:
        return f"HostedMFM(repo={self.repo!r}, profile={self.profile!r})"

    def add_context(
        self,
        data: pd.DataFrame,
        *,
        date_column: str,
        target_column: str,
        channel_columns: Sequence[str],
        control_columns: Sequence[str],
    ) -> Self:
        """Check the history, store it on the service and remember its column roles.

        The column, date, sign and finiteness checks are those of ``MFM.add_context``,
        and run before any request. Dates must also be weekly (one row for each week,
        in order after sorting); a local model accepts other regular cadences. The
        history is uploaded with ``client.datasets.create(..., reuse=True)``, so an
        identical history already stored is reused. A failed call keeps the last
        successful context.
        """
        from pymc_mfm.inference.dataframe import (
            _column_names,
            _numeric_columns,
            _ordered_frame,
        )

        date_column, target_column = _column_names(
            (date_column, target_column), "date_column and target_column"
        )
        channels = _column_names(channel_columns, "channel_columns")
        controls = _column_names(control_columns, "control_columns")
        required = (date_column, target_column, *channels, *controls)
        if len(set(required)) != len(required):
            raise ValueError("date, target, channel and control column roles must be distinct")
        frame, dates = _ordered_frame(data, required, date_column)
        if len(dates) < 2:
            raise ValueError("add_context needs at least two dated context rows")
        spend = _numeric_columns(frame, channels)
        _numeric_columns(frame, controls)
        outcome = _numeric_columns(frame, (target_column,))[:, 0]
        if np.any(spend < 0) or np.any(outcome <= 0):
            raise ValueError("channel spend must be nonnegative and context outcomes positive")
        check_weekly_dates(iso_dates(dates), "history")
        upload = frame.loc[:, list(required)].copy()
        upload[date_column] = dates
        dataset = self._client.datasets.create(
            upload,
            date=date_column,
            outcome=target_column,
            spend=list(channels),
            controls=list(controls),
            reuse=True,
        )
        self._context = _Context(date_column, channels, controls, dataset)
        return self

    def inference(
        self,
        data: pd.DataFrame,
        *,
        n_samples: int = LOCAL_DEFAULT_N_SAMPLES,
        mode: str = "model_average",
        graph=None,
        router_samples: int | None = None,
        units: str = "outcome",
        provenance: Mapping[str, object] | None = None,
    ) -> DecompositionResult:
        """Run the hosted model on the weeks after the context; return the draws.

        ``data`` holds the future weeks: the configured date, channel and control columns,
        weekly, starting the week after the context. The result has the variables,
        coordinates and dims of a local result (see ``DecompositionResult``), except:

        - ``n_samples`` must be at least 100, the service minimum. The default is the
          local default, 32, so a call without ``n_samples`` raises ``ValueError`` before
          any request. Use for example ``n_samples=1000``.
        - The seed is the service default, 0. There is no ``seed`` argument.
        - The draws file is float32, so ``components`` is float32 and
          ``provenance["draws_dtype"]`` says so. Do not expect local float64 precision.
        - The file has no graphs. ``graphs`` is one all-zero placeholder edge that
          ``edge_mask`` marks as padding. ``result.plot.dag()`` and the other graph
          plots raise :class:`HostedUnsupported`.
        - Only the default ``mode="model_average"`` exists; ``mode``, ``graph`` and
          ``router_samples`` raise :class:`HostedUnsupported` when set.

        ``units`` is a label; ``provenance`` entries are added to the result's provenance
        and must not replace an existing key. Raises client errors (``ValidationError``,
        ``Gone``, ...) as ``pymc_mfm.client`` does, and ``ClientError`` if the analysis
        fails.
        """
        from pymc_mfm.inference.dataframe import _numeric_columns, _ordered_frame

        context = self._context
        if context is None:
            raise RuntimeError("call add_context(history, ...) before inference(future_data)")
        if mode != "model_average":
            raise HostedUnsupported("inference(mode=...)", "only model_average is hosted")
        if graph is not None:
            raise HostedUnsupported("inference(graph=...)", "graphs are not hosted")
        if router_samples is not None:
            raise HostedUnsupported("inference(router_samples=...)", "the router is not hosted")
        if not isinstance(n_samples, int | np.integer) or isinstance(n_samples, bool):
            raise ValueError("n_samples must be an integer")
        if n_samples < MIN_HOSTED_DRAWS:
            raise ValueError(
                f"The hosted service needs n_samples >= {MIN_HOSTED_DRAWS}, got {n_samples}. "
                f"The local default is {LOCAL_DEFAULT_N_SAMPLES}, so pass n_samples explicitly, "
                "for example n_samples=1000."
            )
        if not isinstance(units, str) or not units:
            raise ValueError("units must be a nonempty label")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        required = (context.date_column, *context.channels, *context.controls)
        frame, dates = _ordered_frame(data, required, context.date_column)
        _numeric_columns(frame, context.channels)
        _numeric_columns(frame, context.controls)
        plan = frame.loc[:, list(required)].copy()
        plan[context.date_column] = dates
        analysis = self._client.analyze(context.dataset, plan, draws=int(n_samples))
        analysis.wait()
        return _result_from_analysis(
            analysis,
            context,
            units=units,
            requested=int(n_samples),
            annotations=dict(provenance) if provenance else {},
        )


for _name in _REFUSED_METHODS:
    setattr(HostedMFM, _name, _refuse(_name))
del _name


class _Context:
    """The remembered column roles and stored dataset."""

    __slots__ = ("date_column", "channels", "controls", "dataset")

    def __init__(self, date_column, channels, controls, dataset: Dataset) -> None:
        self.date_column = date_column
        self.channels = tuple(channels)
        self.controls = tuple(controls)
        self.dataset = dataset


def _result_from_analysis(
    analysis, context: _Context, *, units: str, requested: int, annotations: dict
) -> DecompositionResult:
    if analysis.status().status == "failed":
        raise ClientError(
            f"Analysis {analysis.id} failed and has no draws. Run again with a new history or plan."
        )
    table = analysis.table
    baseline = table.loc[table["kind"] == "baseline", "scenario_id"]
    if baseline.empty:
        raise ClientError(f"Analysis {analysis.id} has no baseline plan.")
    scenario_id = str(baseline.iloc[0])
    draws = analysis.draws([scenario_id]).sel(scenario=scenario_id)
    return result_from_draws(
        draws,
        channels=context.channels,
        controls=context.controls,
        context_length=context.dataset.weeks,
        units=units,
        provenance={
            "backend": "hosted",
            "analysis_id": analysis.id,
            "dataset_id": context.dataset.dataset_id,
            "draws_requested": requested,
            **annotations,
        },
    )


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def result_from_draws(
    draws,
    *,
    channels: Sequence[str],
    controls: Sequence[str],
    context_length: int,
    units: str = "outcome",
    provenance: Mapping[str, Any] | None = None,
) -> DecompositionResult:
    """A ``DecompositionResult`` from one plan's draws file (``river-draws/1``, one scenario).

    Component order is the local one: ``baseline``, ``channel:*``, ``control:*``, ``latent``,
    ``indirect:*``, with channels and controls in the order given. The file has no graphs, so
    ``graphs`` is one all-zero placeholder edge, masked as padding, and ``provenance["graphs"]``
    is ``"unavailable"``. Arrays keep the file's dtype (float32).
    """
    from pymc_mfm.inference.result import DecompositionResult

    attrs = dict(draws.attrs)
    schema = attrs.get("schema")
    if schema != DRAWS_SCHEMA:
        raise ClientError(f"Unsupported draws file schema {schema!r}; expected {DRAWS_SCHEMA!r}.")
    columns: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []

    def take(variable: str, entity: str | None = None, member: str | None = None):
        array = draws[variable]
        if entity is not None:
            array = array.sel({entity: member})
        return np.asarray(array.transpose("draw", "date").values)

    if "baseline" in draws:
        columns.append(take("baseline"))
        names.append("baseline")
        families.append("baseline")
    for family, wanted in (("channel", tuple(channels)), ("control", tuple(controls))):
        variable = f"{family}_contribution"
        present = (
            tuple(str(item) for item in draws[family].values) if family in draws.coords else ()
        )
        if variable not in draws or set(present) != set(wanted):
            raise ClientError(
                f"The draws file has {family}s {list(present)}, but the context has {list(wanted)}."
            )
        for member in wanted:
            columns.append(take(variable, family, member))
            names.append(f"{family}:{member}")
            families.append(family)
    for variable, name in _SCALARS[1:]:
        if variable in draws:
            columns.append(take(variable))
            names.append(name)
            families.append(name.partition(":")[0])
    components = np.stack(columns, axis=2)
    dtype = str(components.dtype)
    n_draws = components.shape[0]
    info = {
        "draws_schema": schema,
        "draws_dtype": dtype,
        "graphs": "unavailable",
        "graph_note": "The hosted draws file has no graphs; graphs is a masked placeholder.",
        "inference_config": {"mode": "model_average", "window": "query", "n_samples": n_draws},
        **{
            key: _plain(attrs[key])
            for key in ("seed", "model_revision", "library_revision")
            if key in attrs
        },
        **(provenance or {}),
    }
    return DecompositionResult(
        components=components,
        graphs=np.zeros((n_draws, 1), dtype=np.int8),
        component_names=tuple(names),
        component_families=tuple(families),
        component_mask=np.ones(len(names), dtype=bool),
        edge_names=("graphs:unavailable",),
        edge_mask=np.zeros(1, dtype=bool),
        channel_names=tuple(channels),
        control_names=tuple(controls),
        time=np.asarray(draws["date"].values).astype("datetime64[ns]"),
        window="query",
        mode="model_average",
        units=units,
        provenance=info,
        context_length=int(context_length),
    )


@cache
def hosted_plots_class() -> type:
    """``ResultPlots`` with the graph methods replaced by ``HostedUnsupported`` refusals."""
    from pymc_mfm.diagnostics.accessors import ResultPlots

    namespace = {name: _graph_refusal(name) for name in GRAPH_METHODS if hasattr(ResultPlots, name)}
    return type("HostedResultPlots", (ResultPlots,), {"__slots__": (), **namespace})


def _graph_refusal(name: str):
    def refused(self, *args: Any, **kwargs: Any):
        raise HostedUnsupported(f"result.plot.{name}()", "the hosted draws file has no graphs")

    refused.__name__ = name
    return refused
