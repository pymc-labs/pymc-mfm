"""``model.plot``: the charts of one model's latest ``add_context`` and ``inference``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import contributions as _contributions
from . import curves as _curves
from . import fit as _fit
from . import inputs as _inputs
from . import shares as _shares
from . import spend as _spend
from . import totals as _totals
from ._data import ModelSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from pymc_mfm.model.model import MFM


class ModelPlots:
    """Charts of a model's latest ``add_context`` and ``inference`` (``model.plot``).

    Each chart computes its numbers once, then draws them with matplotlib (the
    default; ``pymc-mfm[plot]``) or plotly (``pymc-mfm[interactive]``): pass
    ``backend="plotly"`` or call :func:`pymc_mfm.plotting.set_backend`. Matplotlib
    charts return a ``matplotlib.figure.Figure``, which notebooks display without
    pyplot; plotly charts return an interactive ``plotly.graph_objects.Figure``.
    Charts read the model when called, so they follow its latest ``inference``.

    Intervals are pointwise central intervals of the model's draws. Draws are the
    expected outcome without observation noise, and contributions are conditional
    on the inputs, not causal lift. Context and query draws are separate marginals.
    """

    __slots__ = ("_source",)

    def __init__(self, model: MFM):
        self._source = ModelSource(model)

    def inputs(self, *, backend: str | None = None, title: str | None = None) -> Any:
        """The outcome, each channel's spend and each control by week.

        Needs ``add_context``. After ``inference`` it adds the forecast weeks, shaded,
        with the outcome given to ``inference`` when its data had the target column.
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
        """How well the expected outcome follows the observed outcome, week by week.

        Draws the mean and central ``interval`` of the expected-outcome draws over the
        observed outcome, each window as its own segment, with the forecast weeks
        shaded when both windows are drawn. Below, the scores of each window with at
        least two observed weeks (cards in matplotlib, a table in plotly): MAPE (over
        weeks whose outcome is not zero) and R² (left out for a constant outcome) of
        the mean line, each ± its standard deviation over draws, and the share of
        observed weeks inside the interval. The intervals exclude observation noise,
        so fewer observed weeks than ``interval`` may fall inside them.

        Needs ``inference``. Forecast weeks show and score the outcome only when the
        data given to ``inference`` had the target column. ``window`` is ``"context"``
        or ``"query"`` for one window, or ``None`` for both; ``"query"`` never samples
        the context. ``interval`` is a fraction between 0 and 1.
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
        channels first, then controls, each largest first. ``view="spread"`` draws the
        per-draw shares of each as a Gaussian kernel density (Scott's bandwidth), all
        scaled to the same height.

        Needs ``inference``. ``window`` is ``"context"`` (the history weeks) or
        ``"query"`` (the forecast weeks; never samples the context). ``interval`` is a
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
        """How much each component adds over a window: the spread of its total over draws.

        One panel per component: a histogram of its per-draw totals (each draw's
        weekly contributions summed over the window), with their mean (solid line),
        central ``interval`` (shaded) and, with ``truth``, the true total (dashed).

        Needs ``inference``. ``window`` is ``"context"`` (the history weeks) or
        ``"query"`` (the forecast weeks, which never samples the context).
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

    def contribution_vs_spend(
        self,
        *,
        window: str = "context",
        channels: Sequence[str] | None = None,
        interval: float = 0.9,
        truth: pd.DataFrame | None = None,
        backend: str | None = None,
        title: str | None = None,
    ) -> Any:
        """Each channel's weekly contribution against its weekly spend, a panel each.

        Each point is a week: the channel's raw spend and the mean of its
        contribution draws, with a light bar over their central ``interval``;
        ``truth`` weeks are black crosses at the same spend. A week's contribution
        also depends on earlier spend (carryover), so the points need not lie on one
        curve. The panels share the contribution axis.

        Needs ``inference``. ``window`` is ``"context"`` (the history weeks) or
        ``"query"`` (the forecast weeks, which never samples the context).
        ``channels`` are channel display names, all by default. ``truth`` holds known
        contributions: a DataFrame indexed by date with one column per display name;
        its channel columns are drawn, and weeks it lacks get no cross (a drawn
        channel's column needs a value in at least one week of the window).
        ``interval`` is a fraction between 0 and 1.
        """
        return _spend.contribution_vs_spend(
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
        """Where each channel's extra spend stops adding to the outcome.

        One panel per channel, sharing the y axis: the mean and central ``interval``
        of the channel's contribution, summed over the weeks it was varied, against
        its spend as a percent of the actual spend, with a dashed line (and a dot on
        the mean) at the actual spend. Draws the latest ``model.response_curves(...)``,
        which needs ``inference``; a new ``add_context`` or ``inference`` drops them.
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
        """What each component contributed to the outcome, week by week.

        Components are drawn from the bottom of the stack up: baseline, latent,
        indirect, controls, then channels on top. ``stacked=True`` stacks each
        component's weekly mean over draws, positive means up from zero and negative
        means down, with the observed outcome as a line where it was given. Percentile
        intervals do not add up, so the stack shows none and ``interval`` is unused.
        ``stacked=False`` draws one panel per component, three to a row, sharing the
        dates: its weekly mean and central ``interval``, with ``truth`` dashed.

        Needs ``inference``. ``window`` is ``"context"`` or ``"query"`` for one window,
        or ``None`` for both, the forecast weeks shaded; ``"query"`` never samples the
        context. ``components`` are display names (``"baseline"``, a channel or
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
