"""The old import name of PyMC-MKTFM.

``import pymc_mfm`` and ``from pymc_mfm.x.y import z`` keep working. Every ``pymc_mfm`` name
resolves to the same module object as its ``pymc_mktfm`` name, so ``isinstance`` checks,
registries and module state are shared and nothing is imported twice.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

import pymc_mktfm

_OLD = __name__
_NEW = "pymc_mktfm"


def _real_name(name: str) -> str:
    return _NEW + name[len(_OLD) :]


def _real_loader(name: str):
    spec = importlib.util.find_spec(_real_name(name))
    if spec is None or spec.loader is None:
        raise ImportError(f"no loader for {name}", name=name)
    return spec.loader


class _AliasLoader(importlib.abc.InspectLoader):
    """Hands the import system the existing ``pymc_mktfm`` module under the old name.

    The code, source and package questions go to the real module's own loader, so
    ``python -m pymc_mfm.x`` (which runpy answers with ``get_code``) runs the real file.
    """

    def __init__(self) -> None:
        self._specs: dict[str, object] = {}

    def create_module(self, spec):
        module = importlib.import_module(_real_name(spec.name))
        # The import system overwrites ``__spec__`` on the module it receives. Keep the
        # real one so the module still reports its own name.
        self._specs[module.__name__] = module.__spec__
        return module

    def exec_module(self, module) -> None:
        module.__spec__ = self._specs.pop(module.__name__)

    def get_code(self, fullname):
        return _real_loader(fullname).get_code(_real_name(fullname))

    def get_source(self, fullname):
        return _real_loader(fullname).get_source(_real_name(fullname))

    def is_package(self, fullname):
        return _real_loader(fullname).is_package(_real_name(fullname))


class _AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._loader = _AliasLoader()

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real = _real_name(fullname)
        try:
            # This imports the parent packages but never the module itself, so a
            # ``__main__`` module is not run by looking for it.
            real_spec = importlib.util.find_spec(real)
        except ModuleNotFoundError as error:
            if error.name == real or (error.name and real.startswith(error.name + ".")):
                return None
            raise
        if real_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            self._loader,
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# ``import pymc_mfm`` returns the new package itself.
sys.modules[_OLD] = pymc_mktfm
