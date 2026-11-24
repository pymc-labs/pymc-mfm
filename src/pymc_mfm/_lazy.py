"""Load public package names on first use (PEP 562)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


def lazy_exports(
    package: str, exports: dict[str, str], namespace: dict[str, Any]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return ``__getattr__`` and ``__dir__`` for a package.

    ``exports`` maps each public name to the module that defines it.
    """

    def __getattr__(name: str) -> Any:
        module_name = exports.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(module_name), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *exports})

    return __getattr__, __dir__
