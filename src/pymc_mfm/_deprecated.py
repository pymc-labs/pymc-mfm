"""Warning shims for names that moved: a module ``__getattr__`` built from a map."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def deprecated_names(
    module: str, names: Mapping[str, tuple[str, Callable[[], Any]]], namespace: dict[str, Any]
) -> Callable[[str], Any]:
    """A module ``__getattr__`` that warns once for each old name in ``names``.

    ``names`` maps an old name to the text that says what to use now and a function that
    loads the object to return. ``namespace`` is the module's ``globals()``: the loaded
    object is stored there, so a name warns once per process. Any other name raises
    ``AttributeError``.
    """

    def __getattr__(name: str) -> Any:
        if name not in names:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        replacement, load = names[name]
        warnings.warn(
            f"{module}.{name} is deprecated: {replacement}", DeprecationWarning, stacklevel=2
        )
        namespace[name] = value = load()
        return value

    return __getattr__
