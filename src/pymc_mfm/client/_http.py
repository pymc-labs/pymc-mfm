"""HTTP transport: auth, idempotency keys, retries and error mapping."""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

import httpx

from pymc_mfm import __version__

from ._wire import canonical_json, to_json
from .errors import ApiConnectionError, ApiError, Gone, error_from_response

RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_RETRY_WAIT = 60.0  # seconds; default cap on a Retry-After wait
BACKOFF_BASE = 0.5  # seconds; doubles on each retry when there is no Retry-After
BACKOFF_CAP = 8.0
CONTENT_KEY_PREFIX = "mfm1-"
FRESH_KEY_PREFIX = "mfm1-fresh-"
_SETTING_PADDING = " \t\r\n\v\f\ufeff"
_LOOPBACK_HTTP = {
    b"localhost": "http://localhost",
    b"127.0.0.1": "http://127.0.0.1",
    b"::1": "http://[::1]",
}
_HOST_AND_PORT = re.compile(r"(?:\[[^\s\[\]\\]+\]|[^\s:\[\]\\]+)(?::[0-9]+)?")
_SIGNED_DOWNLOAD = "pymc_mfm.signed_download"
_URL_SHAPE_PROBLEM = "must be well-formed, with a host and an optional port from 1 to 65535"
_SERVICE_URL_HINT = (
    "Set base_url or MFM_API_URL to the direct https:// URL shown on the Connect page "
    "of the River API console."
)
_KEY_HINT = (
    "Check {parameter} or MFM_API_KEY. Create or replace your API key on the Connect page "
    "of the River API console."
)


def _setting(value: str | None, env: str) -> str:
    return (value or "").strip(_SETTING_PADDING) or os.environ.get(env, "").strip(_SETTING_PADDING)


def _url_problem(value: str, *, service: bool) -> str | None:
    try:
        url = httpx.URL(value)
    except (httpx.InvalidURL, ValueError):
        return _URL_SHAPE_PROBLEM
    if not url.is_absolute_url or not (
        url.scheme == "https" or (url.scheme == "http" and url.raw_host in _LOOPBACK_HTTP)
    ):
        return "must use https:// and name a host (plain HTTP is allowed only on loopback)"

    # Check the original authority: HTTPX can normalize away an empty userinfo or port.
    authority = re.split(r"[/?#]", value.partition("://")[2], maxsplit=1)[0]
    if "@" in authority:
        return "must not contain a user name or password"
    port = url.port
    if not _HOST_AND_PORT.fullmatch(authority) or (port is not None and not 1 <= port <= 65535):
        return _URL_SHAPE_PROBLEM
    if service and ("?" in value or "#" in value):
        return "must not contain a query or fragment"
    return None


def _check_url(value: str, *, service: bool) -> None:
    problem = _url_problem(value, service=service)
    if problem is None:
        return
    if service:
        raise ValueError(f"The service URL {problem}. {_SERVICE_URL_HINT}")
    raise ValueError(f"The signed download URL {problem}.")


def _check_api_key(key: str, parameter: str) -> None:
    if key[0] in "\"'" or key[-1] in "\"'":
        problem = "starts or ends with a quote mark. Remove the quote marks"
    elif any(char.isspace() for char in key):
        problem = "has whitespace inside it. Copy the whole key on one line"
    elif not key.isascii() or not key.isprintable():
        problem = "has a control or non-ASCII character. Copy the key again as plain text"
    else:
        return
    raise ValueError(f"The API key {problem}. {_KEY_HINT.format(parameter=parameter)}")


def _service_options(
    base_url: str | None, api_key: str | None, *, key_parameter: str = "api_key"
) -> tuple[str, str]:
    """Resolve and validate one service URL and its credential without opening a client.

    Messages name the credential as the caller's ``key_parameter``.
    """
    base_url = _setting(base_url, "MFM_API_URL")
    api_key = _setting(api_key, "MFM_API_KEY")
    if not base_url:
        raise ValueError("Pass base_url or set the MFM_API_URL environment variable.")
    _check_url(base_url, service=True)
    if not api_key:
        raise ValueError(f"Pass {key_parameter} or set the MFM_API_KEY environment variable.")
    _check_api_key(api_key, key_parameter)
    return base_url.rstrip("/"), api_key


def _omit_download_credentials(request: httpx.Request) -> None:
    # The HTTPX client shares a cookie jar across requests, even without a bearer header.
    if request.extensions.get(_SIGNED_DOWNLOAD):
        request.headers.pop("authorization", None)
        request.headers.pop("cookie", None)


def _location_host(response: httpx.Response) -> str | None:
    locations = response.headers.get_list("location")
    if len(locations) != 1:
        return None
    try:
        location = urlsplit(locations[0])
        authority = location.netloc.rsplit("@", 1)[-1]
        return location.hostname if _HOST_AND_PORT.fullmatch(authority) else None
    except ValueError:
        return None


def _reject_redirect(response: httpx.Response) -> None:
    # HTTPX may parse Location to build a next request even with redirects disabled.
    if not 300 <= response.status_code < 400:
        return
    host = _location_host(response)
    target = f" to host {host!r}" if host else ""
    if response.request.extensions.get(_SIGNED_DOWNLOAD):
        subject = "The signed download"
        hint = "The service must provide a direct signed download URL."
    else:
        subject, hint = "The service", _SERVICE_URL_HINT
    # HTTPX closes responses on hook errors. Close now so a transport failure
    # during cleanup cannot replace the terminal redirect and trigger a retry.
    try:
        response.close()
    except httpx.TransportError:
        pass
    raise ApiError(
        response.status_code,
        "redirect",
        f"{subject} returned a redirect{target}. The client does not follow redirects. {hint}",
    )


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Return the wait in seconds from a ``Retry-After`` header (seconds or HTTP date)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def content_idempotency_key(path: str, body: Any) -> str:
    """A key derived from the request path and body: an equal request gets an equal key, so
    the service returns the earlier resource instead of starting the work again."""
    digest = hashlib.sha256(path.encode() + b"\n" + canonical_json(body)).hexdigest()
    return CONTENT_KEY_PREFIX + digest


def fresh_idempotency_key() -> str:
    """A random key, so the service treats the request as new work."""
    return FRESH_KEY_PREFIX + uuid.uuid4().hex


class Http:
    """A small wrapper over ``httpx.Client`` for the API.

    The API key goes only to the configured service URL. Signed download requests
    do not carry the API key.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_retry_after: float = MAX_RETRY_WAIT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base_url, api_key = _service_options(base_url, api_key)
        if max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        if max_retry_after < 0:
            raise ValueError("max_retry_after must be 0 or more")
        self.max_retry_after = max_retry_after
        self.base_url = base_url
        self.max_retries = max_retries
        self._api_key = api_key
        self._sleep = time.sleep
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            event_hooks={"request": [_omit_download_credentials], "response": [_reject_redirect]},
            # The cleartext loopback exception must not traverse an environment proxy.
            mounts=dict.fromkeys(_LOOPBACK_HTTP.values()),
        )

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Call the API and return the decoded JSON body.

        The same ``idempotency_key`` goes out on every retry of this call.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": f"pymc-mfm-client/{__version__}",
        }
        content = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            content = to_json(body)
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        response = self._send(
            method, self.base_url + path, headers=headers, content=content, params=params
        )
        return self._decode(response)

    def create(
        self,
        path: str,
        body: Any,
        *,
        idempotency_key: str | None = None,
        fresh: bool = False,
    ) -> Any:
        """POST a create request and return the decoded JSON body.

        Without ``idempotency_key`` the key comes from the request content, so an equal
        request gets the earlier resource back. With ``fresh=True`` the key is random, so
        the request always starts new work; its retries still share that one key.

        If an earlier equal resource was deleted or expired, the service answers 410 and this
        raises ``Gone``: pass ``fresh=True`` to run the request again.
        The client never retries a 410 with a new key itself,
        because each new key starts new work.
        """
        if idempotency_key is not None and fresh:
            raise ValueError("Pass idempotency_key or fresh=True, not both")
        if fresh:
            idempotency_key = fresh_idempotency_key()
        if idempotency_key is not None:
            return self.request("POST", path, body=body, idempotency_key=idempotency_key)
        try:
            return self.request(
                "POST", path, body=body, idempotency_key=content_idempotency_key(path, body)
            )
        except Gone as error:
            raise Gone(
                error.status,
                error.code,
                f"{error.message} The Idempotency-Key comes from the request content, so this "
                "can mean that an earlier equal request was deleted or expired. Pass fresh=True, "
                "or a new idempotency_key, to run it again.",
                error.field,
                retry_after=error.retry_after,
            ) from error

    def download_authenticated(self, path: str) -> bytes:
        """Fetch the raw bytes of an API path, with the key. Raises ``ApiError`` on failure."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"pymc-mfm-client/{__version__}",
        }
        return self._send("GET", self.base_url + path, headers=headers).content

    def download(self, url: str) -> bytes:
        """Fetch a signed URL without the API key."""
        _check_url(url, service=False)
        response = self._send(
            "GET",
            url,
            headers={"User-Agent": f"pymc-mfm-client/{__version__}"},
            extensions={_SIGNED_DOWNLOAD: True},
        )
        return response.content

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last_attempt = attempt == self.max_retries
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as error:
                if last_attempt:
                    kind = type(error).__name__
                    break
                self._sleep(self._backoff(attempt, None))
                continue
            if response.status_code < 400:
                return response
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if response.status_code in RETRY_STATUSES and not last_attempt:
                self._sleep(self._backoff(attempt, retry_after))
                continue
            raise self._error(response, retry_after)
        else:
            raise AssertionError("unreachable")  # pragma: no cover

        # Raise outside the handler and omit the raw transport error: both its text
        # and its exception chain can contain a bearer key or a signed URL.
        endpoint = httpx.URL(url)
        raise ApiConnectionError(
            f"Could not reach {endpoint.scheme}://{endpoint.netloc.decode('ascii')}: {kind}"
        )

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_retry_after)
        return min(BACKOFF_BASE * 2**attempt, BACKOFF_CAP)

    @staticmethod
    def _error(response: httpx.Response, retry_after: float | None) -> ApiError:
        try:
            body = response.json()
        except ValueError:
            body = None
        return error_from_response(response.status_code, body, retry_after=retry_after)

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        try:
            return json.loads(response.content)
        except ValueError as error:
            raise ApiError(
                response.status_code,
                "invalid_response",
                "The service sent a body that is not JSON.",
            ) from error
