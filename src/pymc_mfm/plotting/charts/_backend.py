"""The drawing backend of ``model.plot``; each backend is imported only when it draws."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_BACKENDS = ("matplotlib", "plotly")
_DEFAULT = {"backend": "matplotlib"}


def _checked(backend: object) -> str:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    return backend


def set_backend(backend: str) -> None:
    """Draw every chart called without ``backend=`` with ``backend``.

    ``"matplotlib"`` (the default) returns a ``matplotlib.figure.Figure``; ``"plotly"``
    returns an interactive ``plotly.graph_objects.Figure``.
    """
    _DEFAULT["backend"] = _checked(backend)


def resolve(backend: str | None) -> str:
    """``backend``, or the default set by :func:`set_backend` when it is ``None``."""
    return _DEFAULT["backend"] if backend is None else _checked(backend)


def check(backend: str | None, ax: Any = None) -> str:
    """The drawing backend, checked before a chart reads the completed inference.

    Refuses ``ax=`` (matplotlib axes) with plotly.
    """
    name = resolve(backend)
    if name == "plotly" and ax is not None:
        raise ValueError("ax= draws into matplotlib axes; it does not apply to backend='plotly'")
    return name


def draw(
    prepared: Any,
    *,
    backend: str | None,
    matplotlib: Callable[..., Any],
    plotly: Callable[[Any], Any],
    ax: Any = None,
) -> Any:
    """Draw ``prepared`` numbers with the chosen backend's renderer.

    ``ax`` (matplotlib axes) goes to the matplotlib renderer of a single-axes chart.
    Charts call :func:`check` before they prepare, so a bad backend fails first.
    """
    name = check(backend, ax)
    if name == "plotly":
        return plotly(prepared)
    return matplotlib(prepared) if ax is None else matplotlib(prepared, ax=ax)
