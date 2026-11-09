import json

import numpy as np
import pandas as pd
import pytest

from pymc_mfm.client import frame_content_hash

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
        "outcome_name": "sales",
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


def history_hash(history, **overrides):
    columns = {"date": "week", "outcome": "sales", "spend": ["tv", "search"], "controls": ["price"]}
    return frame_content_hash(history, **{**columns, **overrides})


def create_reusing(client, history, **overrides):
    return client.datasets.create(
        history,
        date="week",
        outcome="sales",
        spend=["tv", "search"],
        controls=["price"],
        name="brand-a",
        reuse=True,
        **overrides,
    )


def test_list_returns_datasets_with_lifecycle_fields(client, service, history):
    stored = dataset_record(dataset_id="ds_9", content_hash=history_hash(history), name="brand-a")
    service.on("GET", "/v1/datasets", (200, {"datasets": [stored]}))
    (dataset,) = client.datasets.list()
    assert dataset.dataset_id == "ds_9" and dataset.name == "brand-a"
    assert dataset.content_hash == history_hash(history)
    assert dataset.analyses == 2
    assert dataset.created_at.year == 2029 and dataset.last_used_at.day == 26


def test_records_from_an_older_server_have_no_lifecycle_fields(client, service):
    service.on("GET", "/v1/datasets", (200, {"datasets": [dataset_record()]}))
    (dataset,) = client.datasets.list()
    assert dataset.content_hash is None and dataset.created_at is None
    assert dataset.last_used_at is None and dataset.analyses is None


def test_reuse_returns_the_dataset_with_the_same_content_hash(client, service, history):
    stored = dataset_record(dataset_id="ds_9", content_hash=history_hash(history))
    service.on("GET", "/v1/datasets", (200, {"datasets": [stored]}))
    dataset = create_reusing(client, history)
    assert dataset.dataset_id == "ds_9" and dataset.date_column == "week"
    assert not service.sent("POST", "/v1/datasets")


def test_reuse_ignores_a_dataset_with_the_same_name_and_weeks_but_another_value(
    client, service, history
):
    changed = history.assign(sales=history["sales"] + [0.0, 0.0, 0.0, 1.0])
    other = dataset_record(dataset_id="ds_old", content_hash=history_hash(changed), name="brand-a")
    assert other["summary"]["weeks"] == len(history)
    service.on("GET", "/v1/datasets", (200, {"datasets": [other]}))
    service.on("POST", "/v1/datasets", (201, dataset_record(dataset_id="ds_new")))
    assert create_reusing(client, history).dataset_id == "ds_new"
    assert len(service.sent("POST", "/v1/datasets")) == 1


def test_reuse_uploads_when_the_server_sends_no_hashes(client, service, history):
    service.on("GET", "/v1/datasets", (200, {"datasets": [dataset_record(dataset_id="ds_old")]}))
    service.on("POST", "/v1/datasets", (201, dataset_record(dataset_id="ds_new")))
    assert create_reusing(client, history).dataset_id == "ds_new"


def test_create_without_reuse_never_lists(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    client.datasets.create(history, date="week", outcome="sales", spend="tv")
    assert not service.sent("GET", "/v1/datasets")


def test_find_returns_none_without_a_match(client, service, history):
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    found = client.datasets.find(
        history, date="week", outcome="sales", spend=["tv", "search"], controls=["price"]
    )
    assert found is None


def test_create_sends_outcome_name_by_default_and_optional_currency(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    client.datasets.create(history, date="week", outcome="sales", spend="tv", currency="USD")
    body = json.loads(service.requests[0].content)
    assert body["outcome_name"] == "sales" and body["currency"] == "USD"


def test_create_omits_currency_and_a_too_long_outcome_name(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    long_name = "s" * 61
    client.datasets.create(
        history.rename(columns={"sales": long_name}), date="week", outcome=long_name, spend="tv"
    )
    body = json.loads(service.requests[0].content)
    assert "currency" not in body and "outcome_name" not in body


@pytest.mark.parametrize("bad", ["usd", "US", "USDX", "U1D", "ÉUR"])
def test_a_bad_currency_is_refused_before_any_request(client, service, history, bad):
    with pytest.raises(ValueError, match="currency"):
        client.datasets.create(
            history, date="week", outcome="sales", spend="tv", currency=bad, reuse=True
        )
    assert service.requests == []


def test_summary_labels_are_read_and_default_to_none(client, service):
    record = dataset_record()
    service.on("GET", "/v1/datasets/ds_123", (200, record))
    dataset = client.datasets.get("ds_123")
    assert dataset.outcome_name is None and dataset.currency is None
    record["summary"].update(outcome="sales", currency="EUR")
    dataset = client.datasets.get("ds_123")
    assert (dataset.outcome_name, dataset.currency) == ("sales", "EUR")


def test_labels_do_not_change_reuse_matching(client, service, history):
    stored = dataset_record(dataset_id="ds_9", content_hash=history_hash(history))
    stored["summary"].update(outcome="revenue", currency="EUR")  # other labels, same values
    service.on("GET", "/v1/datasets", (200, {"datasets": [stored]}))
    dataset = create_reusing(client, history, currency="USD")
    assert dataset.dataset_id == "ds_9" and not service.sent("POST", "/v1/datasets")


def bad_histories(history):
    """The test history with one rule broken in each copy."""
    return {
        "missing": history.drop(index=2).reset_index(drop=True),
        "duplicate": history.iloc[[0, 1, 1, 2]].reset_index(drop=True),
        "reversed": history.iloc[[0, 2, 1, 3]].reset_index(drop=True),
    }


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "Row 3 of your history has the date 2024-07-22, but row 2 has 2024-07-08."),
        ("duplicate", "Row 3 of your history repeats the date 2024-07-08, which is also in row 2."),
        (
            "reversed",
            "Row 2 of your history has the date 2024-07-15, but row 1 has 2024-07-01. "
            "The next required date is 2024-07-08, which is in row 3.",
        ),
    ],
)
@pytest.mark.parametrize("reuse", [False, True])
def test_a_bad_history_is_refused_before_any_upload(client, service, history, case, message, reuse):
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    frame = bad_histories(history)[case]
    frame.index = [40, 10, 30, 20][: len(frame)]  # row positions, not index labels
    with pytest.raises(ValueError, match=message):
        client.datasets.create(
            frame, date="week", outcome="sales", spend=["tv", "search"], reuse=reuse
        )
    assert not service.sent("POST", "/v1/datasets")


def test_a_shuffled_history_is_reused_without_a_check_or_upload(client, service, history):
    stored = dataset_record(dataset_id="ds_9", content_hash=history_hash(history))
    service.on("GET", "/v1/datasets", (200, {"datasets": [stored]}))
    shuffled = history.iloc[[2, 0, 3, 1]]
    assert create_reusing(client, shuffled).dataset_id == "ds_9"
    assert not service.sent("POST", "/v1/datasets")


def test_a_shuffled_history_without_a_match_is_refused_locally(client, service, history):
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    with pytest.raises(ValueError, match="Row 2 of your history has the date 2024-07-01"):
        create_reusing(client, history.iloc[[2, 0, 3, 1]])
    assert not service.sent("POST", "/v1/datasets")


def test_the_corrected_history_uploads_and_hashes_the_same(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    shuffled = history.iloc[[2, 0, 3, 1]]
    assert history_hash(shuffled) == history_hash(history)
    client.datasets.create(history, date="week", outcome="sales", spend=["tv", "search"])
    assert len(service.sent("POST", "/v1/datasets")) == 1
