"""A client for the River API.

This package needs no PyTorch: ``pip install pymc-mfm`` is enough. Reading draws files
needs the ``draws`` extra and plots need the ``plot`` extra.

.. code-block:: python

    from pymc_mfm.client import Client

    client = Client("https://api.example.com", api_key="...")  # or MFM_API_URL, MFM_API_KEY
    dataset = client.datasets.create(
        history, date="week", outcome="sales", spend=["tv", "search"], controls=["price"]
    )
    analysis = client.explore(dataset, plan).wait()
    analysis.table
    analysis.plot_curves()
"""

from ._client import Client
from ._hash import dataset_content_hash, frame_content_hash
from .analysis import Analysis, AnalysisStatus, Preview
from .builders import curve, grid, move, scale, scenario, set_controls, shift
from .datasets import Dataset
from .errors import (
    ApiConnectionError,
    ApiError,
    ClientError,
    Conflict,
    Forbidden,
    Gone,
    NotFound,
    NotReady,
    RateLimited,
    ServerError,
    Unauthorized,
    ValidationError,
)

__all__ = [
    "Analysis",
    "AnalysisStatus",
    "ApiConnectionError",
    "ApiError",
    "Client",
    "ClientError",
    "Conflict",
    "Dataset",
    "Forbidden",
    "Gone",
    "NotFound",
    "NotReady",
    "Preview",
    "RateLimited",
    "ServerError",
    "Unauthorized",
    "ValidationError",
    "curve",
    "dataset_content_hash",
    "frame_content_hash",
    "grid",
    "move",
    "scale",
    "scenario",
    "set_controls",
    "shift",
]
