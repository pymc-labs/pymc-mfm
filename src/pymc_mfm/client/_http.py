"""HTTP transport: auth, idempotency keys, retries and error mapping."""

from __future__ import annotations

import email.utils
import json
import os
import time
import uuid
from typing import Any

import httpx

from pymc_mfm import __version__

from ._wire import to_json
from .errors import ApiConnectionError, ApiError, error_from_response

RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_RETRY_WAIT = 60.0  # seconds; caps a large Retry-After
BACKOFF_BASE = 0.5  # seconds; doubles on each retry when there is no Retry-After
BACKOFF_CAP = 8.0


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


def new_idempotency_key() -> str:
    """One key per logical call; retries of that call reuse it."""
    return str(uuid.uuid4())


class Http:
    """A small wrapper over ``httpx.Client`` for the API.

    The API key goes only to ``base_url``. Signed download URLs are fetched with
    :meth:`download`, which sends no credentials.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base_url = base_url or os.environ.get("MFM_API_URL")
        api_key = api_key or os.environ.get("MFM_API_KEY")
        if not base_url:
            raise ValueError("Pass base_url or set the MFM_API_URL environment variable.")
        if not api_key:
            raise ValueError("Pass api_key or set the MFM_API_KEY environment variable.")
        if max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._api_key = api_key
        self._sleep = time.sleep
        self._client = httpx.Client(timeout=timeout, transport=transport)

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

    def download(self, url: str) -> bytes:
        """Fetch a signed URL without credentials. Raises ``ApiError`` on failure."""
        response = self._send("GET", url, headers={"User-Agent": f"pymc-mfm-client/{__version__}"})
        return response.content

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last_attempt = attempt == self.max_retries
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as error:
                if last_attempt:
                    raise ApiConnectionError(
                        f"Could not reach {url.split('?')[0]}: {error}"
                    ) from error
                self._sleep(self._backoff(attempt, None))
                continue
            if response.status_code < 400:
                return response
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if response.status_code in RETRY_STATUSES and not last_attempt:
                self._sleep(self._backoff(attempt, retry_after))
                continue
            raise self._error(response, retry_after)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _backoff(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, MAX_RETRY_WAIT)
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
