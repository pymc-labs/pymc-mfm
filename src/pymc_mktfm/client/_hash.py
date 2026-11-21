"""The content hash of a dataset (``mfm-dataset-v1``).

The service computes the same hash for every stored dataset. Compare hashes to find out
whether a history is already stored. The name, the column order and the row order do not
change the hash; every value, date and column role does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ._wire import as_names, dataset_body

HASH_PREFIX = "mfm-dataset-v1:"


def _canonical_date(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return dt.datetime.fromisoformat(str(value)).date().isoformat()


def _canonical_floats(values: Sequence[Any]) -> list[float]:
    numbers = [float(value) for value in values]
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("Dataset values must be finite numbers")
    return [number + 0.0 for number in numbers]  # -0.0 + 0.0 is 0.0


def dataset_content_hash(body: Mapping[str, Any]) -> str:
    """Hash the dataset body fields ``dates``, ``outcome``, ``spend`` and ``controls``.

    ``name`` is not part of the hash. Rows are sorted by date, so the same history in a
    different row order has the same hash. A series under ``spend`` and the same series
    under ``controls`` have different hashes. Raises ``ValueError`` for NaN or infinity.
    """
    dates = [_canonical_date(value) for value in body["dates"]]
    order = sorted(range(len(dates)), key=dates.__getitem__)

    def sorted_floats(values: Sequence[Any]) -> list[float]:
        series = _canonical_floats(values)
        if len(series) != len(dates):
            raise ValueError("Every series needs one value for each date")
        return [series[i] for i in order]

    canonical = {
        "v": 1,
        "dates": [dates[i] for i in order],
        "outcome": sorted_floats(body["outcome"]),
        "spend": {name: sorted_floats(values) for name, values in body.get("spend", {}).items()},
        "controls": {
            name: sorted_floats(values) for name, values in body.get("controls", {}).items()
        },
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode()
    return HASH_PREFIX + hashlib.sha256(encoded).hexdigest()


def frame_content_hash(
    frame: pd.DataFrame,
    *,
    date: str,
    outcome: str,
    spend: Sequence[str] | str,
    controls: Sequence[str] | str = (),
) -> str:
    """The content hash of the history that ``client.datasets.create`` would upload."""
    body = dataset_body(
        frame,
        date=date,
        outcome=outcome,
        spend=as_names(spend),
        controls=as_names(controls),
        name=None,
    )
    return dataset_content_hash(body)
