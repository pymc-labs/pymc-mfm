"""Opt-in ``torch.compile`` of the live-coordinate flow velocity used by sampling.

Only the per-step velocity is compiled. Base noise is drawn before transport, outside the
compiled code, so seeded calls keep their draw order. Dynamo restores the CPU and current
CUDA random state after it compiles, so a call that compiles draws the same noise as one
that does not. This module never reads or writes the random state.
"""

from __future__ import annotations

import contextlib
import threading
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .flow import _LiveVelocity

#: Accepted :func:`torch.compile` modes. ``"default"`` was the fastest overall on an
#: NVIDIA L4; ``"reduce-overhead"`` adds CUDA graphs.
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")


@dataclass(frozen=True)
class SamplingStatus:
    """How a model's sampler runs.

    ``compiled`` says compilation is switched on, not that every later input is already
    compiled: a new input shape can still compile on first use. ``compiles`` counts the
    compilations of the velocity (the first one and every recompile) since it was switched
    on, or ``None`` when this PyTorch version does not report them. ``fallback`` holds the
    error that returned the sampler to eager execution, if any.
    """

    compiled: bool
    mode: str | None = None
    compiles: int | None = None
    fallback: str | None = None


def _live_velocity(live: _LiveVelocity, x: Tensor, time: Tensor, offset: Tensor) -> Tensor:
    return live._velocity(x, time, offset)


class _CompileHooks:
    """Counts Dynamo compilations of the compiled step running on this thread.

    Dynamo runs its callbacks on the compiling thread, so a per-thread stack of running
    steps says which model a compilation belongs to.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self.available = False
        try:
            from torch._dynamo.callback import callback_handler
        except ImportError:  # pragma: no cover - older PyTorch
            return
        try:
            from torch._dynamo.callback import CallbackTrigger
        except ImportError:  # pragma: no cover - callbacks without triggers
            dynamo = None
        else:
            dynamo = CallbackTrigger.DYNAMO
        self._handler = callback_handler

        def on_start(args=None) -> None:
            trigger = getattr(args, "callback_trigger", dynamo)
            active = getattr(self._local, "active", None)
            if trigger == dynamo and active:
                active[-1].compiles += 1

        self._on_start = on_start
        self.available = True

    @contextlib.contextmanager
    def attribute(self, step: CompiledVelocity) -> Iterator[None]:
        # ``torch.compiler.reset()`` clears every callback, so register again when needed.
        if self.available and self._on_start not in self._handler.start_callbacks:
            self._handler.register_start_callback(self._on_start)
        active = self._local.__dict__.setdefault("active", [])
        active.append(step)
        try:
            yield
        finally:
            active.pop()


_HOOKS: _CompileHooks | None = None


def _hooks() -> _CompileHooks:
    global _HOOKS
    if _HOOKS is None:
        _HOOKS = _CompileHooks()
    return _HOOKS


class CompiledVelocity:
    """A compiled live velocity with its own compile count and eager fallback."""

    def __init__(self, mode: str = "default", *, dynamic: bool | None = True):
        if mode not in COMPILE_MODES:
            raise ValueError(f"mode must be one of {COMPILE_MODES}, got {mode!r}")
        self.mode = mode
        self.dynamic = dynamic
        self.compiles = 0
        self.fallback: str | None = None
        self._hooks = _hooks()
        self._compiled: Callable[..., Tensor] | None = None

    def __getstate__(self) -> dict[str, object]:
        # Compiled functions do not copy or pickle; a copy compiles again on first use.
        return {"mode": self.mode, "dynamic": self.dynamic}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__init__(state["mode"], dynamic=state["dynamic"])

    def _function(self) -> Callable[..., Tensor]:
        if self._compiled is None:
            self._compiled = torch.compile(_live_velocity, mode=self.mode, dynamic=self.dynamic)
        return self._compiled

    def status(self) -> SamplingStatus:
        return SamplingStatus(
            compiled=True,
            mode=self.mode,
            compiles=self.compiles if self._hooks.available else None,
            fallback=self.fallback,
        )

    def __call__(self, live: _LiveVelocity, x: Tensor, time: Tensor, offset: Tensor) -> Tensor:
        if self.fallback is not None:
            return live._velocity(x, time, offset)
        try:
            function = self._function()
            with self._hooks.attribute(self):
                if x.device.type == "cuda" and self.mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                velocity = function(live, x, time, offset)
        except Exception as error:  # noqa: BLE001 - any compiler failure falls back to eager
            self.fallback = f"{type(error).__name__}: {error}"
            warnings.warn(
                f"compiled sampling failed and this model now samples eagerly: {self.fallback}",
                RuntimeWarning,
                stacklevel=2,
            )
            return live._velocity(x, time, offset)
        # A CUDA-graph output is overwritten by the next replay. The Euler step consumes it
        # into a new tensor before that, so no copy is needed.
        return velocity
