import httpx
import pytest

from pymc_mfm.client import (
    ApiConnectionError,
    Client,
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
from pymc_mfm.client._http import MAX_RETRY_WAIT, parse_retry_after

from .client_fixtures import BASE_URL, dataset_record, error_body


def create(client, history):
    return client.datasets.create(
        history, date="week", outcome="sales", spend=["tv", "search"], controls=["price"]
    )


def test_requests_carry_the_bearer_key(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    create(client, history)
    request = service.requests[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert str(request.url) == f"{BASE_URL}/v1/datasets"


def test_url_and_key_come_from_the_environment(monkeypatch, service):
    monkeypatch.setenv("MFM_API_URL", "https://api.example.com/")
    monkeypatch.setenv("MFM_API_KEY", "env-key")
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    with Client(transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    assert str(service.requests[0].url) == "https://api.example.com/v1/datasets/ds_123"
    assert service.requests[0].headers["Authorization"] == "Bearer env-key"


def test_missing_url_or_key_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv("MFM_API_URL", raising=False)
    monkeypatch.delenv("MFM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MFM_API_URL"):
        Client(api_key="k")
    with pytest.raises(ValueError, match="MFM_API_KEY"):
        Client(BASE_URL)


def test_post_sends_a_content_derived_idempotency_key(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    create(client, history)
    create(client, history)
    client.datasets.create(history, date="week", outcome="sales", spend="tv")
    same, again, other = (r.headers["Idempotency-Key"] for r in service.requests)
    assert same == again != other
    assert same.startswith("mfm1-")


def test_get_sends_no_idempotency_key(client, service):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    client.datasets.get("ds_123")
    assert "Idempotency-Key" not in service.requests[0].headers


def test_retry_reuses_the_idempotency_key_and_body(client, service, sleeps, history):
    service.on(
        "POST", "/v1/datasets", (503, error_body("unavailable", "later")), (201, dataset_record())
    )
    create(client, history)
    first, second = service.requests
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert first.content == second.content
    assert len(sleeps) == 1


def test_a_caller_key_is_used_as_given(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    client.datasets.create(
        history, date="week", outcome="sales", spend="tv", idempotency_key="mine-1"
    )
    assert service.requests[0].headers["Idempotency-Key"] == "mine-1"


def test_a_410_for_the_content_key_is_retried_once_with_a_random_key(client, service, history):
    service.on(
        "POST", "/v1/datasets", (410, error_body("expired", "deleted")), (201, dataset_record())
    )
    create(client, history)
    first, second = service.requests
    assert first.headers["Idempotency-Key"].startswith("mfm1-")
    assert not second.headers["Idempotency-Key"].startswith("mfm1-")
    assert second.headers["Idempotency-Key"] != first.headers["Idempotency-Key"]
    assert first.content == second.content


def test_a_second_410_is_raised(client, service, history):
    service.on("POST", "/v1/datasets", (410, error_body("expired", "deleted")))
    with pytest.raises(Gone):
        create(client, history)
    assert len(service.requests) == 2


def test_a_410_with_a_caller_key_is_raised_without_a_retry(client, service, history):
    service.on("POST", "/v1/datasets", (410, error_body("expired", "deleted")))
    with pytest.raises(Gone):
        client.datasets.create(
            history, date="week", outcome="sales", spend="tv", idempotency_key="mine-1"
        )
    assert len(service.requests) == 1


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_retryable_statuses_are_retried(client, service, sleeps, status):
    service.on(
        "GET", "/v1/datasets/ds_123", (status, error_body("busy", "wait")), (200, dataset_record())
    )
    assert client.datasets.get("ds_123").dataset_id == "ds_123"
    assert len(service.requests) == 2 and len(sleeps) == 1


def test_connection_errors_are_retried(client, service, sleeps):
    service.on("GET", "/v1/datasets/ds_123", httpx.ConnectError("refused"), (200, dataset_record()))
    client.datasets.get("ds_123")
    assert len(service.requests) == 2


def test_connection_errors_end_in_a_typed_error(client, service, sleeps):
    service.on("GET", "/v1/datasets/ds_123", httpx.ConnectError("refused"))
    with pytest.raises(ApiConnectionError, match="refused"):
        client.datasets.get("ds_123")
    assert len(service.requests) == 4  # one try and three retries
    assert len(sleeps) == 3


def test_retry_after_seconds_sets_the_wait(client, service, sleeps):
    service.on(
        "GET",
        "/v1/datasets/ds_123",
        (429, error_body("rate_limited", "slow down"), {"Retry-After": "7"}),
        (200, dataset_record()),
    )
    client.datasets.get("ds_123")
    assert sleeps == [7.0]


def test_a_huge_retry_after_is_capped(client, service, sleeps):
    service.on(
        "GET",
        "/v1/datasets/ds_123",
        (503, error_body("unavailable", "x"), {"Retry-After": "86400"}),
        (200, dataset_record()),
    )
    client.datasets.get("ds_123")
    assert sleeps == [MAX_RETRY_WAIT]


def test_max_retry_after_is_a_client_parameter(service, sleeps):
    client = Client(
        BASE_URL, "test-key", max_retry_after=600, transport=httpx.MockTransport(service)
    )
    client._http._sleep = sleeps.append
    service.on(
        "GET",
        "/v1/datasets/ds_123",
        (503, error_body("unavailable", "x"), {"Retry-After": "300"}),
        (503, error_body("unavailable", "x"), {"Retry-After": "86400"}),
        (200, dataset_record()),
    )
    client.datasets.get("ds_123")
    assert sleeps == [300.0, 600.0]


def test_backoff_doubles_without_retry_after(client, service, sleeps):
    service.on("GET", "/v1/datasets/ds_123", (503, error_body("unavailable", "x")))
    with pytest.raises(ServerError):
        client.datasets.get("ds_123")
    assert sleeps == [0.5, 1.0, 2.0]


def test_retry_after_parses_seconds_and_dates():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    date = "Wed, 21 Oct 2026 07:28:10 GMT"
    assert parse_retry_after(date, now=1792567680.0) == pytest.approx(10.0)
    assert parse_retry_after(date, now=1792567700.0) == 0.0


def test_rate_limit_error_keeps_retry_after_after_the_last_retry(service, sleeps):
    service.on(
        "GET",
        "/v1/datasets/ds_123",
        (429, error_body("rate_limited", "slow down"), {"Retry-After": "2"}),
    )
    client = Client(BASE_URL, "k", max_retries=1, transport=httpx.MockTransport(service))
    client._http._sleep = sleeps.append
    with pytest.raises(RateLimited) as caught:
        client.datasets.get("ds_123")
    assert caught.value.retry_after == 2.0
    assert len(service.requests) == 2


@pytest.mark.parametrize(
    ("status", "code", "error_class"),
    [
        (401, "unauthorized", Unauthorized),
        (403, "forbidden", Forbidden),
        (404, "not_found", NotFound),
        (409, "idempotency_conflict", Conflict),
        (409, "not_ready", NotReady),
        (410, "expired", Gone),
        (413, "payload_too_large", PayloadTooLarge),
        (422, "invalid", ValidationError),
        (400, "bad_request", ValidationError),
        (500, "internal", ServerError),
    ],
)
def test_errors_map_to_typed_exceptions(client, service, status, code, error_class):
    service.on("GET", "/v1/datasets/ds_123", (status, error_body(code, "It broke.", "spend.tv")))
    with pytest.raises(error_class) as caught:
        client.datasets.get("ds_123")
    error = caught.value
    assert (error.status, error.code, error.message, error.field) == (
        status,
        code,
        "It broke.",
        "spend.tv",
    )
    assert "spend.tv" in str(error) and code in str(error)


def test_not_ready_is_a_conflict(client, service):
    service.on("GET", "/v1/datasets/ds_123", (409, error_body("not_ready", "wait")))
    with pytest.raises(Conflict):
        client.datasets.get("ds_123")


def test_an_error_without_a_json_body_still_maps(client, service):
    service.on("GET", "/v1/datasets/ds_123", (404, b"<html>nope</html>"))
    with pytest.raises(NotFound) as caught:
        client.datasets.get("ds_123")
    assert caught.value.code == "http_404"


def test_non_retryable_errors_are_not_retried(client, service, sleeps):
    service.on("GET", "/v1/datasets/ds_123", (422, error_body("invalid", "bad")))
    with pytest.raises(ValidationError):
        client.datasets.get("ds_123")
    assert len(service.requests) == 1 and sleeps == []
