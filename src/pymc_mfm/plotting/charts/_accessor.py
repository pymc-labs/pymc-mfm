"""``Plots`` and ``model.plot``: sampling-free charts read from a model tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import contributions as _contributions
from . import curves as _curves
from . import fit as _fit
from . import inputs as _inputs
from . import media as _media
from . import shares as _shares
from . import totals as _totals
from ._data import IdataReader

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from xarray import DataTree


class Plots:
    """Charts of a model tree: ``Plots(idata)``, or the live ``model.plot``.

    ``Plots(idata)`` draws the tree it was given (``model.idata``, or one saved and opened
    again) and never changes. ``model.plot`` reads ``model.idata`` at each chart call, so
    it follows the model's last successful paired ``inference(history, future, ...)``.

    Each chart computes its numbers once, then draws them with Matplotlib (the default)
    or Plotly: pass ``backend="plotly"`` or call :func:`pymc_mfm.plotting.set_backend`. Matplotlib
    charts return a ``matplotlib.figure.Figure``, which notebooks display without
    pyplot; plotly charts return an interactive ``plotly.graph_objects.Figure``.
    They never sample; ``predict`` does not change them.

    Intervals are central intervals of the model's draws: pointwise over weeks or levels,
    one per series for shares and window totals. Draws are the expected outcome
    without observation noise, and contributions are conditional on the inputs, not
    causal lift. Context and query draws are separate marginals.
    """

    __slots__ = ("_get",)

    def __init__(self, idata: DataTree):
        self._get: Callable[[], DataTree | None] = lambda: idata

    @classmethod
    def _live(cls, get: Callable[[], DataTree | None]) -> Plots:
        """Charts that read the tree ``get()`` returns at each chart call."""
        plots = cls.__new__(cls)
        plots._get = get
        return plots

    @property
    def _source(self) -> IdataReader:
        return IdataReader(self._get())

    def inputs(self, *, backend: str | None = None, title: str | None = None) -> Any:
        """The outcome, each media covariate and each control covariate by week.

        Needs a successful ``inference(history, future, ...)``. Forecast weeks are
        shaded, with the query's observed outcome where its data had the target column.
        """
        return _inputs.inputs(self._source, backend=backend, title=title)

    def fit(
        self,
        *,
        window: str | None = None,
        interval: float = 0.9,
        backend: str | None = None,
        title: str | None = None,
    ) -> Any:
        """Expected against observed outcome by week, with goodness-of-fit scores.

        Draws the mean and central ``interval`` of the expected-outcome draws over the
        observed outcome, each window as its own segment, with the forecast weeks
        shaded when both windows are drawn. Below, the scores of each window with at
        least two observed weeks (cards in matplotlib, a table in plotly): MAPE (over
        weeks whose outcome is not zero) and R² (left out for a constant outcome) of
        the mean line, each ± its standard deviation over draws, and the empirical
        coverage of the interval (the share of observed weeks inside it). The intervals
        exclude observation noise, so coverage can fall below ``interval``.

        Needs complete ``inference``. Forecast weeks show and score the outcome only
        when the query had the target column. ``window`` is ``"context"`` or ``"query"``
        for one window, or ``None`` for both; neither window samples.
        ``interval`` is a fraction between 0 and 1.
        """
        return _fit.fit(
            self._source, window=window, interval=interval, backend=backend, title=title
        )

    def shares(
        self,
        *,
        share: str = "mean",
        window: str = "context",
        view: str = "bars",
        interval: float = 0.9,
        backend: str | None = None,
        title: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Each channel's and control's share of the expected outcome.

        The expected outcome of a draw and week is the sum of every component, so all
        components' shares add to one; the chart shows the channels and controls. With
        ``share="mean"`` a draw's share is the average over weeks of contribution /
        expected outcome; with ``share="total"``, the window's total contribution over
        its total expected outcome. Weeks (``"mean"``) or draws (``"total"``) whose
        expected outcome is not positive are left out, as is a draw with no week left;
        the caption says how many.

        ``view="bars"`` draws each mean share with its central ``interval`` over draws:
        channels first, then controls, each largest first. ``view="dist"`` draws the
        distribution of each series' per-draw shares as a Gaussian kernel density
        (Scott's bandwidth), all scaled to the same peak height.

        Needs complete ``inference``. ``window`` is ``"context"`` (the history weeks)
        or ``"query"`` (the forecast weeks). Neither samples. ``interval`` is a
        fraction between 0 and 1; ``ax`` draws into existing matplotlib axes.
        """
        return _shares.shares(
            self._source,
            share=share,
            window=window,
            view=view,
            interval=interval,
            backend=backend,
            title=title,
            ax=ax,
        )

    def contribution_totals(
        self,
        *,
        window: str = "context",
        components: Sequence[str] | None = None,
        interval: float = 0.9,
        bins: int = 40,
        truth: pd.DataFrame | None = None,
        backend: str | None = None,
        title: str | None = None,
    ) -> Any:
        """Distribution over draws of each component's total contribution over a window.

        One panel per component: a histogram of its per-draw totals (each draw's
        weekly contributions summed over the window), with their mean (solid line),
        central ``interval`` (shaded) and, with ``truth``, the true total (dashed).

        Needs complete ``inference``. ``window`` is ``"context"`` (the history weeks)
        or ``"query"`` (the forecast weeks). Neither samples.
        ``components`` are display names of any family (``"baseline"``, a channel or
        control, ``"latent"``, ``"indirect:cc"``...); every channel and control by
        default. ``bins`` is the number of histogram bins per panel (or any ``bins``
        :func:`numpy.histogram` accepts). ``truth`` holds known contributions, from a
        simulation say: a DataFrame indexed by date with one column per display name.
        A drawn component's column needs a value in every week of the window, whose
        sum is its true total. ``interval`` is a fraction between 0 and 1.
        """
        return _totals.contribution_totals(
            self._source,
            window=window,
            components=components,
            interval=interval,
            bins=bins,
            truth=truth,
            backend=backend,
            title=title,
        )

    def contribution_vs_media(
        self,
        *,
        window: str = "context",
        channels: Sequence[str] | None = None,
        interval: float = 0.9,
        truth: pd.DataFrame | None = None,
        backend: str | None = None,
        title: str | None = None,
    ) -> Any:
        """Each media covariate's weekly contribution against its weekly value, a panel each.

        Each point is a week: the media covariate's raw value and the mean of its
        contribution draws, with a light bar over their central ``interval``;
        ``truth`` weeks are black crosses at the same value. A week's contribution also
        depends on earlier weeks of the media covariate (carryover), so the points need
        not lie on one curve. The panels share the contribution axis.

        Needs complete ``inference``. ``window`` is ``"context"`` (the history weeks)
        or ``"query"`` (the forecast weeks). Neither samples.
        ``channels`` are media covariate (channel) display names, all by default.
        ``truth`` holds known contributions: a DataFrame indexed by date with one column
        per display name; its media covariate columns are drawn, and weeks it lacks get
        no cross (a drawn column needs a value in at least one week of the window).
        ``interval`` is a fraction between 0 and 1.
        """
        return _media.contribution_vs_media(
            self._source,
            window=window,
            channels=channels,
            interval=interval,
            truth=truth,
            backend=backend,
            title=title,
        )

    def response_curves(
        self,
        *,
        channels: Sequence[str] | None = None,
        interval: float = 0.9,
        truth: pd.DataFrame | None = None,
        backend: str | None = None,
        title: str | None = None,
    ) -> Any:
        """Response curves: each media covariate's contribution by its level.

        One panel per media covariate, sharing the y axis: the mean and central
        ``interval`` of its contribution, summed over the weeks it was varied, against
        its level as a percent of the observed level, with a dashed line (and a dot on
        the mean) at the observed level. Draws the latest ``model.response_curves(...)``,
        which needs complete ``inference``; a new successful ``inference`` drops them,
        while ``predict`` leaves them unchanged.
        ``channels`` picks panels (default: every channel of the curves). ``truth``,
        a DataFrame indexed by level with one column per channel, adds known
        contributions (from a simulation, say) as black dots. ``interval`` is a
        fraction between 0 and 1.
        """
        return _curves.response_curves(
            self._source,
            channels=channels,
            interval=interval,
            truth=truth,
            backend=backend,
            title=title,
        )

    def contributions_over_time(
        self,
        *,
        window: str | None = None,
        components: Sequence[str] | None = None,
        stacked: bool = True,
        interval: float = 0.9,
        truth: pd.DataFrame | None = None,
        backend: str | None = None,
        title: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Weekly contribution of each component to the expected outcome.

        Components are drawn from the bottom of the stack up: baseline, latent,
        indirect, controls, then channels on top. ``stacked=True`` stacks each
        component's weekly mean over draws, positive means up from zero and negative
        means down, with the observed outcome as a line where it was given. Percentile
        intervals do not add up, so the stack shows none and ``interval`` is unused.
        ``stacked=False`` draws one panel per component, three to a row, sharing the
        dates: its weekly mean and central ``interval``, with ``truth`` dashed.

        Needs complete ``inference``. ``window`` is ``"context"`` or ``"query"`` for one
        window, or ``None`` for both, the forecast weeks shaded; neither window samples.
        ``components`` are display names (``"baseline"``, a channel or
        control, ``"latent"``, ``"indirect:cc"``...), all active components by
        default. ``truth`` (``stacked=False`` only) holds known contributions, from a
        simulation say: a DataFrame indexed by date with one column per display name;
        weeks it lacks are gaps. ``interval`` is a fraction between 0 and 1; ``ax``
        draws the stacked chart into existing matplotlib axes.
        """
        return _contributions.contributions_over_time(
            self._source,
            window=window,
            components=components,
            stacked=stacked,
            interval=interval,
            truth=truth,
            backend=backend,
            title=title,
            ax=ax,
        )
