"""``model.plot``: the charts of one model's latest ``add_context`` and ``inference``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import fit as _fit
from . import inputs as _inputs
from . import shares as _shares
from ._data import ModelSource

if TYPE_CHECKING:
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
