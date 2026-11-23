"""The Python model workflow, independent of where the model runs."""

from __future__ import annotations

from numbers import Integral
from secrets import randbelow
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pandas import DataFrame
    from xarray import DataTree

    from ._hosted import HostedEngine
    from .model import MFM
    from .plotting.charts import Plots

_RIVER_REPO = "pymc-labs/mfm"
_RIVER_REVISION = "970e6095f14824d7e25ecace7e20a47bebde35b8"
_SEED_LIMIT = 2**31


class _Roles(TypedDict):
    date_column: str
    target_column: str
    channel_columns: tuple[str, ...]
    control_columns: tuple[str, ...]


class NotSupported(RuntimeError):
    """The selected model engine does not support this operation or option."""


class ModelBusy(RuntimeError):
    """Another operation is already running on this model."""


class Result:
    """One completed run. Plots read its DataTree without running the model again."""

    __slots__ = ("_idata",)

    def __init__(self, idata: DataTree) -> None:
        self._idata = idata

    @property
    def idata(self) -> DataTree:
        """The original context and query marginals, plus curves when requested."""
        return self._idata

    @property
    def plot(self) -> Plots:
        """The library's plot suite for this run; it never calls the engine."""
        from .plotting.charts import Plots

        return Plots(self._idata)


class _LocalEngine:
    def __init__(self, network: MFM, roles: _Roles) -> None:
        self.network = network
        self._roles = roles

    def inference(self, context, query, *, n_samples, random_seed, revision) -> DataTree:
        tree = self.network.inference(
            context, query, **self._roles, n_samples=n_samples, random_seed=random_seed
        )
        assert tree is not None  # return_idata=True
        return tree

    def response_curves(self, tree, *, levels, channels, weeks, n_samples, random_seed) -> DataTree:
        if self.network.idata is not tree:
            raise RuntimeError("Run model.inference() again after using the research network.")
        self.network.response_curves(
            levels=levels,
            channels=channels,
            weeks=weeks,
            n_samples=n_samples,
            random_seed=random_seed,
        )
        updated = self.network.idata
        assert updated is not None  # the underlying model has completed inference
        return updated


class Model:
    """A model opened with :func:`pymc_mfm.load`, regardless of how it runs.

    Models come from :func:`pymc_mfm.load`; do not construct one directly.
    """

    def __init__(self, engine: _LocalEngine | HostedEngine, roles: _Roles) -> None:
        self._engine = engine
        self._roles = roles
        self._result: Result | None = None
        self._revision = 0
        self._lock = Lock()

    @property
    def result(self) -> Result | None:
        """The latest completed result, or ``None`` before inference."""
        return self._result

    @property
    def idata(self) -> DataTree | None:
        """The latest result's tree, or ``None`` before inference."""
        return None if self._result is None else self._result.idata

    @property
    def plot(self) -> Plots:
        """The library's charts for the latest result."""
        from .plotting.charts import Plots

        return Plots._live(lambda: self.idata)

    @property
    def network(self) -> MFM:
        """The underlying research model for local execution only."""
        if not isinstance(self._engine, _LocalEngine):
            raise NotSupported("The hosted engine does not expose the model network or weights.")
        return self._engine.network

    def _seed(self, seed: int | None) -> int | None:
        if seed is not None:
            return _bounded_int(seed, "random_seed", 0, _SEED_LIMIT - 1)
        # The service defaults to seed zero, even on a fresh analysis. Give a new
        # seed to each unseeded hosted run, as the local random stream would.
        return None if isinstance(self._engine, _LocalEngine) else randbelow(_SEED_LIMIT)

    def _sample_count(self, n_samples: int) -> int:
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        count = int(n_samples)
        if not isinstance(self._engine, _LocalEngine) and not 100 <= count <= 2000:
            raise NotSupported("Hosted River supports 100–2,000 draws per window.")
        return count

    def inference(
        self,
        context: DataFrame,
        query: DataFrame,
        *,
        n_samples: int = 1_000,
        random_seed: int | None = None,
    ) -> Result:
        """Decompose history and query into a single result with separate marginals.

        ``context`` holds observed outcomes. ``query`` is the following weeks, with
        media and controls and optional outcomes for evaluation only. Hosted users
        send history, query inputs and column roles to River, not query outcomes.
        Hosted River supports 100–2,000 draws per window.
        """
        if not self._lock.acquire(blocking=False):
            raise ModelBusy("Another inference or response curve run is already in progress.")
        try:
            n_samples = self._sample_count(n_samples)
            tree = self._engine.inference(
                context,
                query,
                n_samples=n_samples,
                random_seed=self._seed(random_seed),
                revision=self._revision + 1,
            )
            self._result = Result(tree)
            self._revision += 1
            return self._result
        finally:
            self._lock.release()

    def response_curves(
        self,
        *,
        levels: Sequence[float] | None = None,
        channels: Sequence[str] | None = None,
        weeks: int | None = None,
        n_samples: int | None = None,
        random_seed: int | None = None,
    ) -> Result:
        """Sweep media spend over the first query weeks of a completed inference.

        Return a new result; previous results are unchanged, and a failed run keeps the
        model's latest result. Hosted results store a reported 90% curve band, not the
        draws behind that band.
        """
        from .inference.curves import _levels, select_channels

        if not self._lock.acquire(blocking=False):
            raise ModelBusy("Another inference or response curve run is already in progress.")
        try:
            tree = self.idata
            if tree is None:
                raise RuntimeError("Call model.inference() before model.response_curves().")
            posterior = tree["/query/posterior"]
            available_weeks = posterior.sizes["date"]
            multipliers = tuple(_levels(levels).tolist())
            updated = self._engine.response_curves(
                tree,
                levels=multipliers,
                channels=select_channels(channels, self._roles["channel_columns"]),
                weeks=available_weeks
                if weeks is None
                else _bounded_int(weeks, "weeks", 1, available_weeks),
                n_samples=self._sample_count(
                    posterior.sizes["draw"] if n_samples is None else n_samples
                ),
                random_seed=self._seed(random_seed),
            )
            self._result = Result(updated)
            return self._result
        finally:
            self._lock.release()


def _bounded_int(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    return int(value)


def _names(names: Sequence[str], role: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise ValueError(f"{role} must be a sequence of column names, not a string")
    values = tuple(names)
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError(f"{role} must name at least one nonempty column")
    return tuple(str(name) for name in values)


def load(
    source: str | Path,
    *,
    date_column: str,
    target_column: str,
    channel_columns: Sequence[str],
    control_columns: Sequence[str],
    engine: Literal["local", "hosted"] = "local",
    token: str | bool | None = None,
    base_url: str | None = None,
    profile: str | None = None,
    revision: str | None = None,
    device: str = "cpu",
) -> Model:
    """Open a model to use with the same ``inference`` and plot calls on either engine.

    The local engine accepts a bundle directory or a Hugging Face repository.
    ``"river-1"`` selects the pinned River-1 Hub bundle; its weights need Hub access.
    ``token`` on the local engine is a Hugging Face credential (or ``False`` to disable
    implicit Hub auth). The hosted engine runs ``"river-1"`` on River without fetching
    weights. Its credential and service URL are ``token`` and ``base_url``, defaulting
    to ``MFM_API_KEY`` and ``MFM_API_URL``. These credentials are not interchangeable.
    """
    channels = _names(channel_columns, "channel_columns")
    controls = _names(control_columns, "control_columns")
    columns = (date_column, target_column, *channels, *controls)
    if any(not isinstance(name, str) or not name for name in columns) or len(set(columns)) != len(
        columns
    ):
        raise ValueError(
            "Date, target, channel, and control column names must be distinct and nonempty"
        )
    # Built-in str on both engines, as local inference stores: NumPy text such as
    # numpy.str_ cannot be an HDF5 attribute of the result tree.
    roles: _Roles = {
        "date_column": str(date_column),
        "target_column": str(target_column),
        "channel_columns": channels,
        "control_columns": controls,
    }
    if engine == "hosted":
        from ._hosted import HostedEngine

        if source != "river-1":
            raise NotSupported("The hosted engine currently serves 'river-1'.")
        if profile is not None or revision is not None or device != "cpu":
            raise NotSupported(
                "profile, revision and device are only available on the local engine."
            )
        if token is not None and not isinstance(token, str):
            raise TypeError("The hosted engine expects a River API key as a string token.")
        if isinstance(token, str) and _is_hub_token(token):
            raise ValueError(
                "token is a Hugging Face token, but the hosted engine sends token to River. "
                "Pass your River API key as token=, or set MFM_API_KEY and omit token."
            )
        return Model(HostedEngine(roles=dict(roles), base_url=base_url, token=token), roles)
    if engine != "local":
        raise ValueError("engine must be 'local' or 'hosted'")
    if base_url is not None:
        raise ValueError("base_url is only for the hosted engine")
    if isinstance(token, str):
        from .client._http import _SETTING_PADDING, _setting

        # The local engine sends token to the Hugging Face Hub as a bearer credential.
        river_key = _setting(None, "MFM_API_KEY")
        if river_key and token.strip(_SETTING_PADDING) == river_key:
            raise ValueError(
                "token is the River API key in MFM_API_KEY, but the local engine sends token "
                "to the Hugging Face Hub. Use engine='hosted' to run on River, or pass a "
                "Hugging Face token."
            )

    from . import from_pretrained

    if source == "river-1":
        source = _RIVER_REPO
        profile = "river-1" if profile is None else profile
        revision = _RIVER_REVISION if revision is None else revision
    network = from_pretrained(
        source, profile=profile, revision=revision, device=device, token=token
    )
    return Model(_LocalEngine(network, roles), roles)


def _is_hub_token(token: str) -> bool:
    """Whether ``token`` is a Hugging Face credential: its ``hf_`` form or the Hub variables."""
    from .client._http import _SETTING_PADDING, _setting

    value = token.strip(_SETTING_PADDING)
    if value.startswith("hf_"):
        return True
    hub_tokens = (_setting(None, name) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    return any(value == hub for hub in hub_tokens if hub)
