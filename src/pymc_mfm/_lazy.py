"""Import helper for names that need the ``local`` extra."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LOCAL_DEPENDENCIES = ("torch", "safetensors")


class _NeedsLocal(ImportError):
    """The install hint; an outer lazy name replaces the inner name with its own."""


def _hint(public_name: str) -> str:
    return (
        f"{public_name} runs the model in this process and needs PyTorch. "
        'Install it with: pip install "pymc-mfm[local]". '
        "To call a hosted service without PyTorch, use pymc_mfm.client."
    )


def import_local(module_name: str, attribute: str | None, public_name: str) -> Any:
    """Import ``module_name`` (or one of its attributes) for ``public_name``.

    A missing PyTorch or safetensors becomes an ``ImportError`` with the install hint.
    Any other missing module is a real bug and keeps its own error.
    """
    try:
        module = import_module(module_name)
        return module if attribute is None else getattr(module, attribute)
    except _NeedsLocal as error:
        # ``pymc_mfm.decompose`` reaches ``pymc_mfm.inference.decompose``; name the outer one.
        raise _NeedsLocal(_hint(public_name)) from error.__cause__
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] not in _LOCAL_DEPENDENCIES:
            raise
        raise _NeedsLocal(_hint(public_name)) from error
