import re

import httpx
import pytest

from pymc_mfm.client import (
    ApiConnectionError,
    ApiError,
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
    with pytest.raises(ValueError) as caught:
        Client(api_key="k")
    assert str(caught.value) == (
        "Pass base_url or set the MFM_API_URL environment variable. The Connect page of "
        "the River API console shows the service URL."
    )
    with pytest.raises(ValueError) as caught:
        Client(BASE_URL)
    assert str(caught.value) == (
        "Pass api_key or set the MFM_API_KEY environment variable. Create or replace "
        "your API key on the Connect page of the River API console."
    )


def refused_url_message(scheme: str, host: str) -> str:
    return (
        "The service URL must start with https:// and name a host, so the API key is never "
        f"sent in clear text; got scheme {scheme!r} and host {host!r}. Use the https:// "
        "URL exactly as the Connect page of the River API console shows it."
    )


def test_a_plain_http_service_url_is_refused_before_any_request(service):
    with pytest.raises(ValueError) as caught:
        Client("http://api.example.com", "test-key", transport=httpx.MockTransport(service))
    assert str(caught.value) == (
        "The service URL must start with https:// and name a host, so the API key is never "
        "sent in clear text; got scheme 'http' and host 'api.example.com'. Use the https:// "
        "URL exactly as the Connect page of the River API console shows it."
    )
    assert service.requests == []


@pytest.mark.parametrize(
    ("url", "scheme", "host"),
    [
        ("HTTP://API.example.com/v1", "http", "api.example.com"),
        ("http://192.0.2.10:8000", "http", "192.0.2.10"),
        ("http://localhost.example.com", "http", "localhost.example.com"),
        ("http://user:secret@api.example.com", "http", "api.example.com"),
        ("ftp://api.example.com", "ftp", "api.example.com"),
        ("api.example.com", "", ""),
        ("https:///v1", "https", ""),
        ("https://:443", "https", ""),
    ],
)
def test_a_service_url_that_is_not_https_or_has_no_host_is_refused(service, url, scheme, host):
    with pytest.raises(ValueError) as caught:
        Client(url, "test-key", transport=httpx.MockTransport(service))
    assert str(caught.value) == refused_url_message(scheme, host)
    assert service.requests == []


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        (
            "https://user:secret@api.example.com",
            "must not contain a user name or password; pass the key in api_key or MFM_API_KEY",
        ),
        (
            "https://user@api.example.com",
            "must not contain a user name or password; pass the key in api_key or MFM_API_KEY",
        ),
        ("http://localhost:abc", "has a port that is not a number from 0 to 65535"),
        ("https://api.example.com:99999", "has a port that is not a number from 0 to 65535"),
        ("https://api.example.com/v1?x=1", "must not contain a query or a fragment"),
        ("https://api.example.com/v1#top", "must not contain a query or a fragment"),
    ],
)
def test_a_service_url_with_credentials_a_bad_port_or_a_query_is_refused(service, url, problem):
    with pytest.raises(ValueError) as caught:
        Client(url, "test-key", transport=httpx.MockTransport(service))
    assert str(caught.value) == (
        f"The service URL {problem}. Use the https:// URL exactly as the Connect page of the "
        "River API console shows it."
    )
    assert "secret" not in str(caught.value)
    assert service.requests == []


def redirecting_client(location="https://elsewhere.example/v1/x?token=abc", status=307):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "api.example.com":
            return httpx.Response(status, headers={"Location": location, "X-Secret": "hdr"})
        return httpx.Response(200, content=b"data")

    return Client(BASE_URL, "test-key", transport=httpx.MockTransport(handler)), seen


REDIRECT_MESSAGE = (
    "HTTP {status} redirect: The service redirected the request to host 'elsewhere.example'. "
    "The client does not follow redirects. Check that MFM_API_URL is the https:// URL exactly "
    "as the Connect page of the River API console shows it."
)


@pytest.mark.parametrize("status", [301, 302, 307, 308])
@pytest.mark.parametrize("call", ["request", "download_authenticated", "download"])
def test_a_redirect_raises_one_clear_error_and_is_not_followed(status, call):
    client, seen = redirecting_client(status=status)
    http = client._http
    args = {
        "request": ("GET", "/v1/a"),
        "download_authenticated": ("/v1/a",),
        "download": (f"{BASE_URL}/v1/a",),
    }[call]
    with pytest.raises(ApiError) as caught:
        getattr(http, call)(*args)
    assert str(caught.value) == REDIRECT_MESSAGE.format(status=status)
    assert [r.url.host for r in seen] == ["api.example.com"]


def test_a_redirect_without_a_location_host_is_still_one_clear_error():
    client, _ = redirecting_client(location="/other")
    with pytest.raises(ApiError) as caught:
        client._http.request("GET", "/v1/a")
    assert str(caught.value) == (
        "HTTP 307 redirect: The service redirected the request to a location without a host. "
        "The client does not follow redirects. Check that MFM_API_URL is the https:// URL "
        "exactly as the Connect page of the River API console shows it."
    )


@pytest.mark.parametrize(
    ("url", "scheme", "host"),
    [
        ("http://storage.example.com/x?sig=1", "http", "storage.example.com"),
        ("storage.example.com/x", "", ""),
    ],
)
def test_a_download_url_that_is_not_https_is_refused_before_any_request(service, url, scheme, host):
    client = Client(BASE_URL, "test-key", transport=httpx.MockTransport(service))
    with pytest.raises(ValueError) as caught:
        client._http.download(url)
    assert str(caught.value) == (
        "The download URL must start with https:// and name a host; got scheme "
        f"{scheme!r} and host {host!r}."
    )
    assert service.requests == []


def test_the_key_goes_only_to_the_service_host_and_signed_urls_get_no_key():
    # Signed storage URLs (draws files) come from the service result and are fetched
    # without credentials; only the API paths of the configured host carry the key.
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"data")

    client = Client(BASE_URL, "test-key", transport=httpx.MockTransport(handler))
    client._http.download("https://storage.example.com/draws.nc?sig=1")
    client._http.download_authenticated("/v1/analyses/an_1/export")
    assert [(r.url.host, r.headers.get("Authorization")) for r in seen] == [
        ("storage.example.com", None),
        ("api.example.com", "Bearer test-key"),
    ]


def test_a_plain_http_url_in_the_environment_is_refused(monkeypatch, service):
    monkeypatch.setenv("MFM_API_URL", "http://api.example.com")
    monkeypatch.setenv("MFM_API_KEY", "env-key")
    with pytest.raises(ValueError) as caught:
        Client(transport=httpx.MockTransport(service))
    assert str(caught.value) == refused_url_message("http", "api.example.com")
    assert service.requests == []


@pytest.mark.parametrize(
    ("url", "sent_to"),
    [
        (BASE_URL, f"{BASE_URL}/v1/datasets/ds_123"),
        ("HTTPS://api.example.com", f"{BASE_URL}/v1/datasets/ds_123"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/datasets/ds_123"),
        ("http://localhost", "http://localhost/v1/datasets/ds_123"),
        ("http://[::1]:8000/", "http://[::1]:8000/v1/datasets/ds_123"),
    ],
)
def test_https_and_loopback_http_service_urls_are_accepted(service, url, sent_to):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    with Client(url, "test-key", transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    assert [str(r.url) for r in service.requests] == [sent_to]
    assert service.requests[0].headers["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    "key",
    ["mktfm_abc\n", "mktfm_abc\r\n", "  mktfm_abc  ", "\tmktfm_abc \n", "\ufeffmktfm_abc\n"],
)
def test_surrounding_whitespace_and_bom_are_stripped_from_the_key(key, monkeypatch, service):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()), (200, dataset_record()))
    with Client(BASE_URL, key, transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    monkeypatch.setenv("MFM_API_KEY", key)
    with Client(BASE_URL, transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    assert [r.headers["Authorization"] for r in service.requests] == ["Bearer mktfm_abc"] * 2


def test_surrounding_whitespace_is_stripped_from_the_url(monkeypatch, service):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()), (200, dataset_record()))
    with Client(BASE_URL + "/\n", "k", transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    monkeypatch.setenv("MFM_API_URL", " " + BASE_URL + "\r\n")
    with Client(api_key="k", transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    assert [str(r.url) for r in service.requests] == [f"{BASE_URL}/v1/datasets/ds_123"] * 2


HINT = (
    "Check the key in MFM_API_KEY (or api_key). Create or replace your API key on the "
    "Connect page of the River API console."
)
QUOTES = f"The API key starts or ends with a quote mark. Remove the quote marks. {HINT}"
INNER = f"The API key has a space or line break inside it. Copy the whole key on one line. {HINT}"
ODD = (
    "The API key has a character that a key never holds (a control character or one "
    f"that is not plain ASCII). Copy the key again, as plain text. {HINT}"
)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ('"mktfm_secretvalue"', QUOTES),
        ("'mktfm_secretvalue'\n", QUOTES),
        ("mktfm_secret value", INNER),
        ("mktfm_secret\nvalue", INNER),
        ("mktfm_secr\u00e9tvalue", ODD),
        ("mktfm_secret\x07value", ODD),
        ("mktfm_secret\ufeffvalue", ODD),
    ],
)
def test_a_malformed_key_is_refused_without_showing_it(key, message, monkeypatch, service):
    for build in (
        lambda: Client(BASE_URL, key, transport=httpx.MockTransport(service)),
        lambda: (
            monkeypatch.setenv("MFM_API_KEY", key),
            Client(BASE_URL, transport=httpx.MockTransport(service)),
        ),
    ):
        with pytest.raises(ValueError) as caught:
            build()
        assert str(caught.value) == message
        assert "secret" not in repr(caught.value)
    assert service.requests == []


def test_a_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.delenv("MFM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Pass api_key or set the MFM_API_KEY"):
        Client(BASE_URL, " \n")


SECRET = "mktfm_TOPSECRETVALUE123"


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError(f"failed with Authorization: Bearer {SECRET} sent"),
        httpx.ReadTimeout(f"Authorization header {SECRET}"),
    ],
)
def test_a_connection_error_never_shows_the_key(failure, service, sleeps):
    service.on("GET", "/v1/datasets/ds_123", failure)
    with Client(BASE_URL, SECRET, transport=httpx.MockTransport(service)) as client:
        client._http._sleep = sleeps.append
        with pytest.raises(ApiConnectionError) as caught:
            client.datasets.get("ds_123")
        assert SECRET not in str(caught.value) + repr(caught.value)
        assert SECRET not in repr(client) + repr(client._http)
    assert str(caught.value).startswith(f"Could not reach {BASE_URL}/v1/datasets/ds_123: ")


@pytest.mark.parametrize(
    ("message", "shown"),
    [
        ("Bad key.", "Bad key."),
        ("bad key", "bad key."),  # a message without a full stop still reads as two sentences
    ],
)
def test_a_rejected_key_says_where_the_key_comes_from(client, service, message, shown):
    service.on("GET", "/v1/datasets/ds_123", (401, error_body("unauthorized", message)))
    with pytest.raises(Unauthorized) as caught:
        client.datasets.get("ds_123")
    assert str(caught.value) == (
        f"HTTP 401 unauthorized: {shown} Check the key in MFM_API_KEY (or api_key). "
        "Create or replace your API key on the Connect page of the River API console."
    )
    assert caught.value.message == message


def test_post_sends_a_content_derived_idempotency_key(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    create(client, history)
    create(client, history)
    client.datasets.create(history, date="week", outcome="sales", spend="tv")
    same, again, other = (r.headers["Idempotency-Key"] for r in service.requests)
    assert same == again != other
    assert same.startswith("mfm1-")


def test_fresh_sends_a_new_key_on_each_call_and_one_key_across_retries(
    client, service, sleeps, history
):
    service.on(
        "POST",
        "/v1/datasets",
        (503, error_body("unavailable", "later")),
        (201, dataset_record()),
    )
    kwargs = {"date": "week", "outcome": "sales", "spend": "tv"}
    client.datasets.create(history, fresh=True, **kwargs)
    client.datasets.create(history, fresh=True, **kwargs)
    client.datasets.create(history, **kwargs)
    first, retry, second, derived = (r.headers["Idempotency-Key"] for r in service.requests)
    assert first == retry  # a retry is not new work
    assert len({first, second, derived}) == 3
    assert first.startswith("mfm1-fresh-") and derived.startswith("mfm1-")
    assert re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", first)  # the service's key pattern


def test_fresh_and_a_caller_key_together_are_refused(client, service, history):
    with pytest.raises(ValueError, match="not both"):
        client.datasets.create(
            history, date="week", outcome="sales", spend="tv", fresh=True, idempotency_key="k"
        )
    assert service.requests == []


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


def test_a_410_for_the_content_key_is_raised_and_names_the_way_out(client, service, history):
    service.on(
        "POST", "/v1/datasets", (410, error_body("gone", "deleted")), (201, dataset_record())
    )
    with pytest.raises(Gone, match="Pass fresh=True") as caught:
        create(client, history)
    assert (caught.value.status, caught.value.code) == (410, "gone")
    assert len(service.requests) == 1  # no retry with a key of the client's choice


def test_a_410_with_a_caller_key_is_raised_without_a_retry(client, service, history):
    service.on("POST", "/v1/datasets", (410, error_body("expired", "deleted")))
    with pytest.raises(Gone, match="deleted$"):
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
