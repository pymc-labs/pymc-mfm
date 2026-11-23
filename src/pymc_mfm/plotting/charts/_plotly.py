"""Plotly pieces shared by the charts; only plotly renderers import this module."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._style import BAND_ALPHA, FORECAST, FORECAST_ALPHA, panel_grid, rgba

TEMPLATE = "plotly_white"
#: Characters per caption line: plotly never wraps titles, so captions wrap here.
CAPTION_WIDTH = 120


def figure(rows: int = 1, cols: int = 1, **options) -> go.Figure:
    """An empty figure of ``rows`` x ``cols`` subplots."""
    return make_subplots(rows=rows, cols=cols, **options)


def subplot_specs(count: int, columns: int = 2) -> list[list[dict | None]]:
    """``specs`` of :func:`~._style.panel_grid`: no axes in a cell past the last panel."""
    rows, columns = panel_grid(count, columns)
    return [
        [{} if row * columns + column < count else None for column in range(columns)]
        for row in range(rows)
    ]


def text(value: str) -> str:
    """``value`` safe inside plotly's HTML-like labels."""
    return escape(value, quote=False)


def finish(
    fig: go.Figure,
    *,
    title: str,
    caption: str | Sequence[str] | None = None,
    height: int,
    notes: Sequence[str] = (),
) -> go.Figure:
    """Title, optional caption (a string or lines, each wrapped), size and the shared look.

    Title, caption and a one-row legend sit in the top margin, in that order, and the
    margin keeps room below the legend for subplot titles: nothing overlaps. ``height``
    is the height of the plotting area plus the bottom margin. ``notes`` (each wrapped)
    go under the x-axis titles in a bottom margin grown to hold them, so the plotting
    area keeps its height.
    """
    parts = [] if caption is None else [caption] if isinstance(caption, str) else list(caption)
    lines = [wrapped for part in parts for wrapped in textwrap.wrap(part, CAPTION_WIDTH)]
    heading = "<br>".join([text(title), *(f"<sup>{text(line)}</sup>" for line in lines)])
    legend_top = 14 + 30 + 18 * len(lines)  # title padding, title line, caption lines
    top = legend_top + 28 + 28  # the legend row, then room for subplot titles
    footnote = [wrapped for note in notes for wrapped in textwrap.wrap(note, CAPTION_WIDTH)]
    extra = 18 * len(footnote) + 10 if footnote else 0
    total = height + top + extra
    fig.update_layout(
        template=TEMPLATE,
        title={
            "text": heading,
            "x": 0.01,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "pad": {"t": 14},
        },
        height=total,
        legend={
            "orientation": "h",
            "yref": "container",
            "y": 1 - legend_top / total,
            "yanchor": "top",
            "x": 1,
            "xanchor": "right",
        },
        margin={"t": top, "l": 60, "r": 20, "b": 50 + extra},
    )
    if footnote:
        fig.add_annotation(
            text="<br>".join(text(line) for line in footnote),
            xref="paper",
            yref="paper",
            x=0,
            y=-50 / (height - 50),  # below the 50 px that hold the x-axis titles
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font={"size": 11},
        )
    return fig


def _half_step(dates: np.ndarray) -> pd.Timedelta:
    if len(dates) < 2:
        return pd.Timedelta(days=0.5)
    steps = np.diff(dates.astype("datetime64[ns]").astype(np.int64))
    return pd.Timedelta(int(np.median(steps)) // 2, unit="ns")


def forecast_span(fig: go.Figure, dates: np.ndarray, first: int, **where) -> None:
    """Shade the weeks from ``dates[first]`` on: the weeks the model did not see."""
    half = _half_step(dates)
    fig.add_vrect(
        x0=pd.Timestamp(dates[first]) - half,
        x1=pd.Timestamp(dates[-1]) + half,
        fillcolor=FORECAST,
        opacity=FORECAST_ALPHA,
        line_width=0,
        layer="below",
        **where,
    )


def band(
    fig: go.Figure,
    x,
    low: np.ndarray,
    high: np.ndarray,
    *,
    color: str,
    name: str,
    legendgroup: str,
    showlegend: bool = True,
    **where,
) -> None:
    """A filled band from ``low`` to ``high``; hovering it shows nothing of its own."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=high,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=legendgroup,
        ),
        **where,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=low,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=rgba(color, BAND_ALPHA),
            hoverinfo="skip",
            name=name,
            showlegend=showlegend,
            legendgroup=legendgroup,
        ),
        **where,
    )
