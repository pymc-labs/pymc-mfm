"""Charts of a model's latest ``add_context`` and ``inference``: ``model.plot``.

Matplotlib (``pymc-mfm[plot]``) and plotly (``pymc-mfm[interactive]``) draw the same
numbers; each library is imported only when a chart is drawn with it.
"""

from ._accessor import ModelPlots
from ._backend import BACKENDS, set_backend

__all__ = ["BACKENDS", "ModelPlots", "set_backend"]
