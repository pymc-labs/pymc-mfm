"""HTTP transport: auth, idempotency keys, retries and error mapping."""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import time
import uuid
from typing import Any
from urllib.parse import SplitResult, urlsplit

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
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_https_or_loopback(parts: SplitResult) -> bool:
    host = parts.hostname or ""
    return bool(host) and (
        parts.scheme == "https" or (parts.scheme == "http" and host in LOOPBACK_HOSTS)
    )


def check_service_url(base_url: str) -> None:
    """Raise ``ValueError`` unless ``base_url`` is a plain ``https://`` URL with a host.

    Plain ``http://`` is allowed only to a loopback host, for local tests. The message names
    the scheme and host only, never the rest of the URL.
    """
    parts = urlsplit(base_url)
    if not _is_https_or_loopback(parts):
        problem = (
            "must start with https:// and name a host, so the API key is never sent in clear "
            f"text; got scheme {parts.scheme!r} and host {parts.hostname or ''!r}"
        )
    elif parts.username is not None or parts.password is not None:
        problem = "must not contain a user name or password; pass the key in api_key or MFM_API_KEY"
    elif not _has_valid_port(parts):
        problem = "has a port that is not a number from 0 to 65535"
    elif parts.query or parts.fragment:
        problem = "must not contain a query or a fragment"
    else:
        return
    raise ValueError(
        f"The service URL {problem}. Use the https:// URL exactly as the Connect page of the "
        "River API console shows it."
    )


def _has_valid_port(parts: SplitResult) -> bool:
    try:
        parts.port
    except ValueError:
        return False
    return True


def check_download_url(url: str) -> None:
    """Raise ``ValueError`` unless a download ``url`` follows the same ``https://`` rule."""
    parts = urlsplit(url)
    if not _is_https_or_loopback(parts):
        raise ValueError(
            "The download URL must start with https:// and name a host; got scheme "
            f"{parts.scheme!r} and host {parts.hostname or ''!r}."
        )


def redirect_error(response: httpx.Response) -> ApiError:
    """The error for a 3xx answer. It names the status and the ``Location`` host only."""
    try:
        host = urlsplit(response.headers.get("Location", "")).hostname
    except ValueError:
        host = None
    target = f"host {host!r}" if host else "a location without a host"
    return ApiError(
        response.status_code,
        "redirect",
        f"The service redirected the request to {target}. The client does not follow "
        "redirects. Check that MFM_API_URL is the https:// URL exactly as the Connect page "
        "of the River API console shows it.",
    )


_SURROUNDING = " \t\r\n\x0b\x0c\ufeff"  # whitespace and a byte order mark, from a file or a paste
_KEY_HINT = (
    "Check the key in MFM_API_KEY (or api_key). Create or replace your API key on the "
    "Connect page of the River API console."
)


def clean_setting(value: str | None) -> str | None:
    """Strip surrounding whitespace and a byte order mark from a URL or key setting."""
    return None if value is None else value.strip(_SURROUNDING)


def check_api_key(key: str) -> None:
    """Raise ``ValueError`` if ``key`` cannot go in an HTTP header.

    The messages never contain the key, so a tester can paste them to support.
    """
    if key[0] in "\"'" or key[-1] in "\"'":
        problem = "The API key starts or ends with a quote mark. Remove the quote marks."
    elif any(char.isspace() for char in key):
        problem = "The API key has a space or line break inside it. Copy the whole key on one line."
    elif not all("!" <= char <= "~" for char in key):
        problem = (
            "The API key has a character that a key never holds (a control character or one "
            "that is not plain ASCII). Copy the key again, as plain text."
        )
    else:
        return
    raise ValueError(f"{problem} {_KEY_HINT}")


def _scrub(text: str, secret: str) -> str:
    """Hide the key in the text of a transport error."""
    return text.replace(secret, "***")


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

    The API key goes only to ``base_url``, which must be ``https://`` (plain ``http://`` only to
    ``localhost``, ``127.0.0.1`` or ``::1``, for local tests). Signed download URLs are fetched
    with :meth:`download`, which sends no credentials.
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
        base_url = clean_setting(base_url) or clean_setting(os.environ.get("MFM_API_URL"))
        api_key = clean_setting(api_key) or clean_setting(os.environ.get("MFM_API_KEY"))
        if not base_url:
            raise ValueError(
                "Pass base_url or set the MFM_API_URL environment variable. The Connect page of "
                "the River API console shows the service URL."
            )
        check_service_url(base_url)
        if not api_key:
            raise ValueError(
                "Pass api_key or set the MFM_API_KEY environment variable. Create or replace "
                "your API key on the Connect page of the River API console."
            )
        check_api_key(api_key)
        if max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        if max_retry_after < 0:
            raise ValueError("max_retry_after must be 0 or more")
        self.max_retry_after = max_retry_after
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._api_key = api_key
        self._sleep = time.sleep
        self._client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)

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
        """Fetch a signed URL without credentials. Raises ``ApiError`` on failure."""
        check_download_url(url)
        response = self._send("GET", url, headers={"User-Agent": f"pymc-mfm-client/{__version__}"})
        return response.content

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last_attempt = attempt == self.max_retries
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as error:
                if last_attempt:
                    text = _scrub(str(error), self._api_key)
                    raise ApiConnectionError(
                        f"Could not reach {url.split('?')[0]}: {text}"
                    ) from error
                self._sleep(self._backoff(attempt, None))
                continue
            if 300 <= response.status_code < 400:
                raise redirect_error(response)
            if response.status_code < 400:
                return response
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if response.status_code in RETRY_STATUSES and not last_attempt:
                self._sleep(self._backoff(attempt, retry_after))
                continue
            raise self._error(response, retry_after)
        raise AssertionError("unreachable")  # pragma: no cover

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
