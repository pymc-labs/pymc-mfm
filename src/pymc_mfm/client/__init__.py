"""A client for the River API.

This package needs no PyTorch: ``pip install pymc-mfm`` is enough. Reading draws files
needs the ``draws`` extra and plots need the ``plot`` extra.

.. code-block:: python

    from pymc_mfm.client import Client, amount, scenario

    client = Client()  # reads MFM_API_URL and MFM_API_KEY
    dataset = client.datasets.create(
        history, date="week", outcome="sales", spend=["tv", "search"], controls=["price"]
    )
    analysis = client.explore(dataset, plan).wait()
    analysis.table
    analysis.plot_curves()
    again = analysis.again(
        [scenario("tv_to_search", {"tv": 0.8, "search": amount(30000)})]
    ).wait()
"""

from ._client import Client
from ._hash import dataset_content_hash, frame_content_hash
from .analysis import Analysis, AnalysisStatus, Preview
from .builders import amount, curve, grid, scenario
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
    PayloadTooLarge,
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
    "PayloadTooLarge",
    "Preview",
    "RateLimited",
    "ServerError",
    "Unauthorized",
    "ValidationError",
    "amount",
    "curve",
    "dataset_content_hash",
    "frame_content_hash",
    "grid",
    "scenario",
]
