"""Sampling-free charts of a model's last complete ``inference``: ``Plots`` and ``model.plot``.

Matplotlib (``pymc-mfm[plot]``) and plotly (``pymc-mfm[interactive]``) draw the same
numbers; each library is imported only when a chart is drawn with it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pymc_mfm._deprecated import deprecated_names

from ._accessor import Plots
from ._backend import _BACKENDS, set_backend

if TYPE_CHECKING:
    from pymc_mfm.model.model import MFM

__all__ = ["Plots", "set_backend"]


def ModelPlots(model: MFM) -> Plots:  # noqa: N802
    """Deprecated: use ``Plots(idata)`` or ``model.plot``."""
    warnings.warn(
        "ModelPlots is deprecated: use pymc_mfm.Plots(idata) or model.plot",
        DeprecationWarning,
        stacklevel=2,
    )
    return model.plot


__getattr__ = deprecated_names(
    __name__,
    {
        "BACKENDS": (
            'there is no replacement; the backends are "matplotlib" and "plotly"',
            lambda: _BACKENDS,
        ),
    },
    globals(),
)
