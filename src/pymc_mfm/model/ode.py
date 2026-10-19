"""Differentiable fixed-grid integration shared by flow training and sampling."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import Tensor


def _validate_ode_method(method: str) -> str:
    if method not in ("euler", "heun"):
        raise ValueError(f"unknown ode_method: {method!r}; expected 'euler' or 'heun'")
    return method


def integrate_flow(
    x: Tensor,
    drift: Callable[[Tensor, float], Tensor],
    grid: Iterable[tuple[float, float, float]],
    *,
    method: str = "euler",
    project: Callable[[Tensor, float], Tensor] | None = None,
) -> Tensor:
    """Transport ``x`` over caller-provided ``(t, dt, t_next)`` steps.

    The caller initializes and constrains ``x`` before entry; this function
    does not project the initial state. Projections apply at ``t_next`` after
    each completed step and, for Heun, before evaluating the predictor drift.
    ``drift`` and ``project`` must be out of place: the solver preserves
    autograd, draws no randomness and does not mutate its input. Projection
    alone does not turn an unconditional drift into an exact conditional flow.
    """
    _validate_ode_method(method)
    for t, dt, t_next in grid:
        velocity = drift(x, t)
        if method == "euler":
            x = x + dt * velocity
        else:
            predictor = x + dt * velocity
            if project is not None:
                predictor = project(predictor, t_next)
            velocity_next = drift(predictor, t_next)
            x = x + 0.5 * dt * (velocity + velocity_next)
        if project is not None:
            x = project(x, t_next)
    return x
