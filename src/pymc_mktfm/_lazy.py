"""Load the names of a package that needs PyTorch on first use (PEP 562)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

_LOCAL_DEPENDENCIES = ("torch", "safetensors")


def lazy_exports(
    package: str, exports: dict[str, str], namespace: dict[str, Any]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return ``__getattr__`` and ``__dir__`` for a package.

    ``exports`` maps each public name to the module that defines it. A missing PyTorch
    dependency becomes an install hint; any other missing module keeps its own error.
    """

    def __getattr__(name: str) -> Any:
        module_name = exports.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        try:
            value = getattr(import_module(module_name), name)
        except ModuleNotFoundError as error:
            if (error.name or "").partition(".")[0] not in _LOCAL_DEPENDENCIES:
                raise
            raise ImportError(
                f"{package}.{name} runs the model in this process and needs PyTorch. "
                'Install it with: pip install "pymc-mktfm[local]". '
                "To call a hosted service without PyTorch, use pymc_mktfm.client."
            ) from error
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *exports})

    return __getattr__, __dir__
