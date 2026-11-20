"""Warning shims for names that moved: a module ``__getattr__`` built from a map."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def deprecated_names(
    module: str,
    names: Mapping[str, tuple[str, Callable[[], Any]]],
    namespace: dict[str, Any],
    *,
    stacklevel: int = 2,
) -> Callable[[str], Any]:
    """A module ``__getattr__`` that warns when resolving an old name in ``names``.

    ``names`` maps an old name to the text that says what to use now and a function that
    loads the object to return. ``namespace`` is the module's ``globals()``: the loaded
    object is cached there after a successful load. Any other name raises ``AttributeError``.

    Warning contract:

    - A name warns on first access in a process. After a successful load it sits in
      ``namespace``, so later accesses are silent, even if warning filters change.
      Overlapping first accesses can each warn before the object is cached.
    - Python's default filters hide ``DeprecationWarning`` outside ``__main__``. A first
      access that a filter hid is not reported again after the filter changes.
    - A load that raises (for example, PyTorch is missing) is not cached. The next access
      warns again and raises the same error, including the ``[local]`` install hint.

    Pass ``stacklevel=3`` when a module ``__getattr__`` calls the
    returned function, so the warning names the caller's line.
    """

    def __getattr__(name: str) -> Any:
        if name not in names:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        replacement, load = names[name]
        warnings.warn(
            f"{module}.{name} is deprecated: {replacement}",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        namespace[name] = value = load()
        return value

    return __getattr__
