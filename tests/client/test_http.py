import json
import re
from traceback import format_exception

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

from .client_fixtures import BASE_URL, dataset_record, error_body, status_record
from .local_service import API_KEY, serve


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


@pytest.mark.parametrize(
    ("base_url", "api_key", "expected_url", "expected_key"),
    [
        (None, None, BASE_URL, "env-key"),
        (" \ufeff\r\n", " \ufeff\r\n", BASE_URL, "env-key"),
        (" \ufeffhttps://arg.example.com/\n", None, "https://arg.example.com", "env-key"),
        (None, " \ufeffarg-key\n", BASE_URL, "arg-key"),
    ],
)
def test_url_and_key_are_resolved_independently(
    monkeypatch, service, base_url, api_key, expected_url, expected_key
):
    monkeypatch.setenv("MFM_API_URL", f" \ufeff{BASE_URL}/\r\n")
    monkeypatch.setenv("MFM_API_KEY", " \ufeffenv-key\n")
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    with Client(base_url, api_key, transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    assert str(service.requests[0].url) == f"{expected_url}/v1/datasets/ds_123"
    assert service.requests[0].headers["Authorization"] == f"Bearer {expected_key}"


@pytest.mark.parametrize("argument", [None, " \ufeff\r\n"])
def test_missing_url_or_key_names_the_environment_variable(monkeypatch, argument):
    monkeypatch.delenv("MFM_API_URL", raising=False)
    monkeypatch.delenv("MFM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MFM_API_URL"):
        Client(base_url=argument, api_key="k")
    with pytest.raises(
        ValueError, match="Pass api_key or set the MFM_API_KEY environment variable"
    ):
        Client(BASE_URL, argument)


def test_explicit_settings_are_trimmed_and_preserve_path_prefix(monkeypatch, service):
    monkeypatch.setenv("MFM_API_URL", "http://unused.example.com")
    monkeypatch.setenv("MFM_API_KEY", "unused key")
    service.on(
        "GET",
        "/team@org/blue/v1/datasets/ds_123",
        (200, json.dumps(dataset_record()).encode()),
    )
    with Client(
        " \ufeffHTTPS://api.example.com/team@org%2Fblue///\r\n",
        "\ufeffarg-key \n",
        transport=httpx.MockTransport(service),
    ) as client:
        client.datasets.get("ds_123")
    (request,) = service.requests
    assert str(request.url) == f"{BASE_URL}/team@org%2Fblue/v1/datasets/ds_123"
    assert request.headers["Authorization"] == "Bearer arg-key"


@pytest.mark.parametrize(
    ("url", "hint"),
    [
        ("HTTP://api.example.com", "https://"),
        ("http://localhost.example.com", "https://"),
        ("http://127.0.0.2:8000", "https://"),
        ("ftp://api.example.com", "https://"),
        ("api.example.com", "https://"),
        ("https://:443", "name a host"),
        ("https://user:raw-marker@api.example.com", "user name or password"),
        ("https://@api.example.com", "user name or password"),
        ("https://api.example.com:", "port"),
        ("https://api.example.com:abc", "port"),
        ("https://api.example.com:0", "port"),
        ("https://api.example.com:65536", "port"),
        ("https://[::1]443", "port"),
        ("https://[raw-marker", "well-formed"),
        ("https://api.example.com\\raw-marker", "well-formed"),
        ("https://api.example.com?", "query or fragment"),
        ("https://api.example.com#", "query or fragment"),
        ("https://api.example.com?token=raw-marker", "query or fragment"),
        ("https://api.example.com#raw-marker", "query or fragment"),
    ],
)
def test_invalid_service_url_is_refused_before_a_request(service, url, hint):
    with pytest.raises(ValueError, match=hint) as caught:
        Client(url, "test-key", transport=httpx.MockTransport(service))
    assert "raw-marker" not in "".join(format_exception(caught.value))
    assert "Connect page" in str(caught.value)
    assert service.requests == []


def test_plain_http_service_url_in_environment_is_refused(monkeypatch, service):
    monkeypatch.setenv("MFM_API_URL", " \ufeffhttp://api.example.com\n")
    with pytest.raises(ValueError, match="MFM_API_URL"):
        Client(api_key="test-key", transport=httpx.MockTransport(service))
    assert service.requests == []


@pytest.mark.parametrize("url", ["http://localhost", "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_plain_http_is_allowed_for_literal_loopback(service, url):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    with Client(url, "test-key", transport=httpx.MockTransport(service)) as client:
        client.datasets.get("ds_123")
    (request,) = service.requests
    assert str(request.url) == f"{url}/v1/datasets/ds_123"
    assert request.headers["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_http_bypasses_proxy_without_disabling_https_proxy(monkeypatch, host):
    with serve() as (proxy_url, _):
        # An HTTP proxy receives an absolute request target, which this origin rejects.
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("https_proxy", proxy_url)
        monkeypatch.setenv("no_proxy", "")
        url = proxy_url.replace("127.0.0.1", host, 1)
        with Client(url, API_KEY, max_retries=0) as client:
            assert client.list_analyses().empty
        https_url = url.replace("http://", "https://", 1)
        with (
            Client(https_url, API_KEY, max_retries=0) as client,
            pytest.raises(ApiConnectionError, match="ProxyError"),
        ):
            client.list_analyses()  # The stub does not implement CONNECT.


@pytest.mark.parametrize("from_env", [False, True])
@pytest.mark.parametrize(
    ("key", "hint"),
    [
        (' "raw-marker"\n', "quote mark"),
        ("raw-marker'", "quote mark"),
        ("raw-marker value", "whitespace inside"),
        ("raw-marker\nvalue", "whitespace inside"),
        ("raw-marker\x07", "control or non-ASCII"),
        ("raw-markeré", "control or non-ASCII"),
        ("raw-marker\ufeffvalue", "control or non-ASCII"),
    ],
)
def test_malformed_key_is_refused_without_echoing_it(monkeypatch, service, key, hint, from_env):
    monkeypatch.setenv("MFM_API_KEY", key if from_env else "valid-env-key")
    with pytest.raises(ValueError, match=hint) as caught:
        Client(BASE_URL, None if from_env else key, transport=httpx.MockTransport(service))
    assert "raw-marker" not in "".join(format_exception(caught.value))
    assert "Check api_key or MFM_API_KEY." in str(caught.value)
    assert "Connect page" in str(caught.value)
    assert service.requests == []


@pytest.mark.parametrize(
    ("status", "location", "host"),
    [
        (
            301,
            "https://user:raw-marker@redirect.example.com/other?token=raw-marker",
            "redirect.example.com",
        ),
        (302, "https://[raw-marker", None),
        (302, "https://redirect.example.com\\raw-marker", None),
        (302, "https://xn--/target", "xn--"),
        (302, "//redirect.example.com/target?sig=raw-marker", "redirect.example.com"),
        (304, None, None),
        (307, "/another-path", None),
    ],
)
def test_request_redirects_raise_without_following_or_retrying(
    client, service, sleeps, status, location, host
):
    headers = {} if location is None else {"Location": location}
    service.on("GET", "/v1/datasets/ds_123", (status, b"not JSON", headers))
    with pytest.raises(ApiError) as caught:
        client.datasets.get("ds_123")
    error = caught.value
    assert (error.status, error.code) == (status, "redirect")
    assert "MFM_API_URL" in error.message
    assert "raw-marker" not in "".join(format_exception(error))
    if host:
        assert host in error.message
    assert len(service.requests) == 1
    assert sleeps == []


def test_create_redirect_does_not_replay_the_post(client, service, history, sleeps):
    service.on("POST", "/v1/datasets", (307, b"", {"Location": "/other-datasets"}))
    with pytest.raises(ApiError) as caught:
        create(client, history)
    assert (caught.value.status, caught.value.code) == (307, "redirect")
    assert [(r.method, r.url.path) for r in service.requests] == [("POST", "/v1/datasets")]
    assert sleeps == []


@pytest.mark.parametrize("close_fails", [False, True])
def test_redirect_ignores_ambiguous_location_and_stays_terminal(sleeps, close_fails):
    seen = []

    class ResponseBody(httpx.ByteStream):
        def close(self) -> None:
            if close_fails:
                raise httpx.CloseError("raw-marker")

    def handle(request):
        seen.append(request)
        request.headers.pop("Authorization")
        return httpx.Response(
            302,
            headers=[("Location", "https://redirect.example.com"), ("Location", "raw-marker")],
            stream=ResponseBody(b""),
        )

    with Client(BASE_URL, "test-key", transport=httpx.MockTransport(handle)) as client:
        client._http._sleep = sleeps.append
        with pytest.raises(ApiError) as caught:
            client.datasets.get("ds_123")
    error = caught.value
    assert (error.status, error.code) == (302, "redirect")
    assert "MFM_API_URL" in error.message
    assert "raw-marker" not in "".join(format_exception(error))
    assert len(seen) == 1
    assert sleeps == []


def test_export_redirect_does_not_write_a_file(client, service, tmp_path):
    service.on("GET", "/v1/analyses/an_1", (200, status_record()))
    analysis = client.get("an_1")
    service.on("GET", "/v1/analyses/an_1/export", (308, b"", {"Location": "/export"}))
    path = tmp_path / "export.json"
    with pytest.raises(ApiError) as caught:
        analysis.export(path)
    assert (caught.value.status, caught.value.code) == (308, "redirect")
    assert [request.url.path for request in service.requests] == [
        "/v1/analyses/an_1",
        "/v1/analyses/an_1/export",
    ]
    assert service.requests[-1].headers["Authorization"] == "Bearer test-key"
    assert not path.exists()


def test_signed_download_redirect_identifies_the_storage_problem(client, service):
    service.on("GET", "/draws.nc", (302, b"", {"Location": "/another-object?sig=raw-marker"}))
    with pytest.raises(ApiError) as caught:
        client._http.download(f"{BASE_URL}/draws.nc?sig=initial")
    error = caught.value
    assert (error.status, error.code) == (302, "redirect")
    assert "direct signed download URL" in error.message
    assert "MFM_API_URL" not in error.message
    assert "raw-marker" not in error.message
    assert [request.url.path for request in service.requests] == ["/draws.nc"]


@pytest.mark.parametrize(
    "url",
    [
        "http://storage.example.com/draws.nc?sig=1",
        "storage.example.com/draws.nc",
        "https://user:pw@storage.example.com/draws.nc",
    ],
)
def test_unsafe_signed_download_url_is_refused_before_a_request(client, service, url):
    with pytest.raises(ValueError, match="signed download URL"):
        client._http.download(url)
    assert service.requests == []


@pytest.mark.parametrize("origin", [BASE_URL, "https://storage.example.com"])
def test_signed_download_preserves_query_and_sends_no_service_credentials(client, service, origin):
    query = "sig=abc%2fdef%20ghi&part=1&sig=%7e"
    url = f"{origin}/draws.nc?{query}"
    service.on(
        "GET",
        "/v1/datasets/ds_123",
        (200, dataset_record(), {"Set-Cookie": "session=synthetic; Domain=example.com; Secure"}),
    )
    service.on("GET", "/draws.nc", (200, b"draws"))
    client.datasets.get("ds_123")
    assert client._http.download(url) == b"draws"
    client.datasets.get("ds_123")
    _, download, authenticated = service.requests
    assert download.url.raw_path == f"/draws.nc?{query}".encode("ascii")
    assert "Authorization" not in download.headers
    assert "Cookie" not in download.headers
    assert authenticated.headers["Cookie"] == "session=synthetic"


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
    failure = httpx.ConnectError("transport failed with Authorization: Bearer test-key")
    service.on("GET", "/v1/datasets/ds_123", failure)
    with pytest.raises(ApiConnectionError, match="api.example.com.*ConnectError") as caught:
        client.datasets.get("ds_123")
    assert len(service.requests) == 4  # one try and three retries
    assert len(sleeps) == 3
    assert "test-key" not in "".join(format_exception(caught.value))


def test_signed_download_transport_error_does_not_expose_the_url(client, service):
    url = "https://storage.example.com/draws.nc?sig=raw-marker"
    service.on("GET", "/draws.nc", httpx.ReadTimeout(f"timeout for {url}"))
    with pytest.raises(ApiConnectionError, match="storage.example.com.*ReadTimeout") as caught:
        client._http.download(url)
    assert "raw-marker" not in "".join(format_exception(caught.value))


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


def test_a_rejected_key_says_where_to_get_one(client, service):
    service.on(
        "GET", "/v1/datasets/ds_123", (401, error_body("unauthorized", "Bad key", "authorization"))
    )
    with pytest.raises(Unauthorized) as caught:
        client.datasets.get("ds_123")
    error = caught.value
    assert (error.status, error.code, error.message, error.field) == (
        401,
        "unauthorized",
        "Bad key",
        "authorization",
    )
    assert str(error) == "HTTP 401 unauthorized: Bad key (field: authorization)"
    assert error.args == (str(error),)
    rendered = "".join(format_exception(error))
    assert "Check the River API key you supplied or set in MFM_API_KEY." in rendered
    assert "Create or replace it on the Connect page of the River API console." in rendered
    assert "test-key" not in rendered


@pytest.mark.parametrize("authorization", [None, "Bearer another-key"], ids=["absent", "replaced"])
def test_401_does_not_suggest_replacing_a_key_that_was_not_sent(authorization):
    def handle(request):
        if authorization is None:
            del request.headers["Authorization"]
        else:
            request.headers["Authorization"] = authorization
        return httpx.Response(401)

    with Client(BASE_URL, "test-key", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(Unauthorized) as caught:
            client.datasets.get("ds_123")
    assert "MFM_API_KEY" not in "".join(format_exception(caught.value))


def test_401_without_request_metadata_keeps_the_typed_error(client, monkeypatch):
    monkeypatch.setattr(httpx.Client, "request", lambda *args, **kwargs: httpx.Response(401))
    with pytest.raises(Unauthorized) as caught:
        client.datasets.get("ds_123")
    assert caught.value.status == 401
    assert "MFM_API_KEY" not in "".join(format_exception(caught.value))


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
    if status != 401:
        assert "MFM_API_KEY" not in "".join(format_exception(error))


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
