"""Typed errors for the River API client."""

from __future__ import annotations

from typing import Any


class ClientError(Exception):
    """Base class for every error this client raises on purpose."""


class ApiConnectionError(ClientError):
    """The service could not be reached after all retries."""


class ApiError(ClientError):
    """The service answered with an error.

    ``code``, ``message`` and ``field`` come from the error body
    ``{"error": {"code", "message", "field"?}}``. ``field`` names the request field
    at fault, when the service says so.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        field: str | None = None,
        *,
        retry_after: float | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.retry_after = retry_after
        super().__init__(str(self))

    def __str__(self) -> str:
        where = f" (field: {self.field})" if self.field else ""
        return f"HTTP {self.status} {self.code}: {self.message}{where}"


class ValidationError(ApiError):
    """The request is invalid (400, 422)."""


class Unauthorized(ApiError):
    """The API key is missing or wrong (401)."""


class Forbidden(ApiError):
    """The key may not do this (403)."""


class NotFound(ApiError):
    """The resource does not exist, or belongs to another user (404)."""


class Conflict(ApiError):
    """The request conflicts with stored state (409), for example a reused
    idempotency key with a different body."""


class NotReady(Conflict):
    """The analysis has no result yet (409 ``not_ready``). Call ``Analysis.wait()``."""


class Gone(ApiError):
    """The resource expired or was deleted (410)."""


class PayloadTooLarge(ApiError):
    """The request body is larger than the service accepts (413). Send fewer weeks,
    scenarios or curves in one call."""


class RateLimited(ApiError):
    """Too many requests (429). ``retry_after`` is in seconds, when given."""


class ServerError(ApiError):
    """The service failed (5xx)."""


_BY_STATUS: dict[int, type[ApiError]] = {
    400: ValidationError,
    401: Unauthorized,
    403: Forbidden,
    404: NotFound,
    409: Conflict,
    410: Gone,
    413: PayloadTooLarge,
    422: ValidationError,
    429: RateLimited,
}


def error_from_response(status: int, body: Any, *, retry_after: float | None = None) -> ApiError:
    """Build the typed error for an error response with a decoded JSON ``body``."""
    detail = body.get("error") if isinstance(body, dict) else None
    detail = detail if isinstance(detail, dict) else {}
    code = str(detail.get("code") or f"http_{status}")
    message = str(detail.get("message") or "The service returned an error.")
    field = detail.get("field")
    if status == 409 and code == "not_ready":
        error_class: type[ApiError] = NotReady
    elif status in _BY_STATUS:
        error_class = _BY_STATUS[status]
    else:
        error_class = ServerError if status >= 500 else ApiError
    return error_class(status, code, message, field, retry_after=retry_after)
