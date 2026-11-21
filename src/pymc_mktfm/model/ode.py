"""Differentiable fixed-grid Euler integration used by flow sampling."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import Tensor


def integrate_flow(
    x: Tensor,
    drift: Callable[[Tensor, float], Tensor],
    grid: Iterable[tuple[float, float, float]],
    *,
    project: Callable[[Tensor, float], Tensor] | None = None,
) -> Tensor:
    """Transport ``x`` with explicit Euler steps over caller-provided ``(t, dt, t_next)``.

    The caller initializes and constrains ``x`` before entry; this function
    does not project the initial state. Projections apply at ``t_next`` after
    each completed step. ``drift`` and ``project`` must be out of place: the
    solver preserves autograd, draws no randomness and does not mutate its
    input. Projection alone does not turn an unconditional drift into an exact
    conditional flow.
    """
    for t, dt, t_next in grid:
        x = x + dt * drift(x, t)
        if project is not None:
            x = project(x, t_next)
    return x
