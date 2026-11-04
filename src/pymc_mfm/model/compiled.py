"""Opt-in ``torch.compile`` of the live-coordinate flow velocity used by sampling.

Only the per-step velocity is compiled. Base noise is drawn before transport, outside the
compiled code, so seeded calls keep their draw order. Compilation (for example autotuning
on CUDA) may consume the global random state; each compiled call restores it, so a call
that compiles draws the same noise as one that does not.
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


class _CompileCounter:
    """Attributes Dynamo compilations to the compiled step running on this thread.

    Dynamo runs its compile callbacks on the thread that compiles, so a per-thread stack
    of running steps says which model a compilation belongs to.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self.available = False
        try:
            from torch._dynamo.callback import CallbackTrigger, callback_handler
        except ImportError:  # pragma: no cover - older PyTorch
            return
        self._dynamo = CallbackTrigger.DYNAMO
        self._handler = callback_handler

        def on_start(args=None) -> None:
            trigger = getattr(args, "callback_trigger", self._dynamo)
            active = getattr(self._local, "active", None)
            if trigger == self._dynamo and active:
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


_COUNTER: _CompileCounter | None = None


def _counter() -> _CompileCounter:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = _CompileCounter()
    return _COUNTER


@contextlib.contextmanager
def preserved_rng(device: torch.device) -> Iterator[None]:
    """Restore the CPU and ``device`` generators on exit."""
    cpu = torch.get_rng_state()
    accelerator = None
    if device.type == "cuda":
        accelerator = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if accelerator is not None:
            torch.cuda.set_rng_state(accelerator, device)


class CompiledVelocity:
    """A compiled live velocity with its own compile count and eager fallback."""

    def __init__(self, mode: str = "default", *, dynamic: bool | None = True):
        if mode not in COMPILE_MODES:
            raise ValueError(f"mode must be one of {COMPILE_MODES}, got {mode!r}")
        self.mode = mode
        self.dynamic = dynamic
        self.compiles = 0
        self.fallback: str | None = None
        self._counter = _counter()
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
            compiles=self.compiles if self._counter.available else None,
            fallback=self.fallback,
        )

    def __call__(self, live: _LiveVelocity, x: Tensor, time: Tensor, offset: Tensor) -> Tensor:
        if self.fallback is not None:
            return live._velocity(x, time, offset)
        try:
            function = self._function()
            with preserved_rng(x.device), self._counter.attribute(self):
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
