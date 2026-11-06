import json

import numpy as np
import pandas as pd
import pytest

from .client_fixtures import dataset_record


def test_create_sends_the_wire_body(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    dataset = client.datasets.create(
        history,
        date="week",
        outcome="sales",
        spend=["tv", "search"],
        controls=["price"],
        name="brand-a",
    )
    assert json.loads(service.requests[0].content) == {
        "name": "brand-a",
        "dates": ["2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22"],
        "outcome": [1000.0, 1100.0, 1050.0, 1200.0],
        "spend": {"tv": [100.0, 0.0, 120.0, 110.0], "search": [50.0, 55.0, 60.0, 45.0]},
        "controls": {"price": [9.9, 9.9, 10.5, 10.5]},
    }
    assert dataset.dataset_id == "ds_123"
    assert dataset.name == "brand-a" and dataset.date_column == "week"


def test_create_omits_name_when_none_and_accepts_a_single_column_name(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    client.datasets.create(history, date="week", outcome="sales", spend="tv")
    body = json.loads(service.requests[0].content)
    assert "name" not in body
    assert list(body["spend"]) == ["tv"] and body["controls"] == {}


def test_dates_may_be_strings_or_timestamps(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    history["week"] = history["week"].dt.strftime("%d %b %Y")
    client.datasets.create(history, date="week", outcome="sales", spend="tv")
    assert json.loads(service.requests[0].content)["dates"][0] == "2024-07-01"


def test_record_is_parsed(client, service):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    dataset = client.datasets.get("ds_123")
    assert dataset.channels == ("tv", "search") and dataset.controls == ("price",)
    assert (dataset.weeks, str(dataset.start), str(dataset.end)) == (4, "2024-07-01", "2024-07-22")
    assert dataset.expires_at.year == 2030 and dataset.expires_at.tzinfo is not None
    assert dataset.warnings == ("history is shorter than 75 weeks",)
    ranges = dataset.history_ranges
    assert list(ranges.index) == ["tv", "search", "price"]
    assert {"p05", "p95", "max", "zero_share", "all_zero"} <= set(ranges.columns)


def test_delete(client, service):
    service.on("DELETE", "/v1/datasets/ds_123", (202, {"deletion": "in_progress"}))
    assert client.datasets.delete("ds_123") is None
    assert service.requests[0].method == "DELETE"


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_values_are_refused_before_sending(client, service, history, bad):
    history.loc[1, "sales"] = bad
    with pytest.raises(ValueError, match="sales"):
        client.datasets.create(history, date="week", outcome="sales", spend="tv")
    assert service.requests == []


def test_missing_columns_are_named(client, service, history):
    with pytest.raises(ValueError, match="nope"):
        client.datasets.create(history, date="week", outcome="sales", spend=["tv", "nope"])
    assert service.requests == []


def test_missing_dates_are_refused(client, service, history):
    history["week"] = pd.Series([pd.Timestamp("2024-07-01"), pd.NaT, pd.NaT, pd.NaT])
    with pytest.raises(ValueError, match="week"):
        client.datasets.create(history, date="week", outcome="sales", spend="tv")
