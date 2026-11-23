from datetime import UTC, datetime, timedelta
from traceback import format_exception

import httpx
import numpy as np
import pytest
import xarray as xr

from pymc_mfm.client import ApiConnectionError, Forbidden, Gone
from pymc_mfm.client.analysis import URL_RENEWAL_MARGIN

from .client_fixtures import FakeService, result_record, result_rows, status_record

N_DRAWS, N_WEEKS = 5, 2
FUTURE = "2030-01-08T00:00:00Z"
PAST = "2020-01-01T00:00:00Z"


def soon() -> str:
    """An expiry inside the renewal margin that has not passed yet.

    The service caps a renewed URL at the analysis expiry, so near that expiry every renewed
    URL looks like this.
    """
    moment = datetime.now(UTC) + timedelta(seconds=URL_RENEWAL_MARGIN / 2)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def draws_file(tmp_path, scenario_id: str, offset: float) -> bytes:
    """A tiny file in the shape of the service's draws files, made at test time."""
    dataset = xr.Dataset(
        {
            "outcome": (("draw", "date"), np.full((N_DRAWS, N_WEEKS), offset, dtype="float32")),
            "tv": (("draw", "date"), np.full((N_DRAWS, N_WEEKS), offset / 2, dtype="float32")),
        },
        coords={
            "draw": np.arange(N_DRAWS),
            "date": np.array(["2024-07-29", "2024-08-05"], "M8[ns]"),
        },
        attrs={"scenario_id": scenario_id, "schema": "river-draws/1"},
    )
    path = tmp_path / f"{scenario_id}.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    return path.read_bytes()


def file_url(name: str, token: str) -> str:
    return f"https://files.example.com/{name}.nc?sig={token}"


def result_with_urls(token: str, expires: str = FUTURE):
    rows = result_rows("baseline", "baseline", url=file_url("baseline", token), url_expires=expires)
    rows += result_rows(
        "half", "scenario", base="baseline", url=file_url("half", token), url_expires=expires
    )
    rows += result_rows("broken", "scenario", base="baseline", failed=True)
    return result_record(rows)


class Storage:
    """Serves draws files by URL, and rejects a signature it no longer accepts."""

    def __init__(self, tmp_path, valid_token: str):
        self.files = {
            name: draws_file(tmp_path, name, offset)
            for name, offset in (("baseline", 100.0), ("half", 50.0))
        }
        self.valid_token = valid_token
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        name = request.url.path.removeprefix("/").removesuffix(".nc")
        if request.url.params.get("sig") != self.valid_token:
            return httpx.Response(403, content=b"<Error>ExpiredToken</Error>")
        return httpx.Response(200, content=self.files[name])


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path, valid_token="fresh")


@pytest.fixture
def analysis(client, service: FakeService, storage):
    """A client whose transport sends API calls to the fake service and files to storage."""
    api = httpx.MockTransport(service)
    files = httpx.MockTransport(storage)
    client._http._client._transport = httpx.MockTransport(
        lambda request: (files if request.url.host == "files.example.com" else api).handle_request(
            request
        )
    )
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    return client


def fetch(client, service, *results):
    service.on("GET", "/v1/analyses/an_1/result", *[(200, result) for result in results])
    return client.get("an_1")


def test_draws_are_concatenated_along_scenario(analysis, service, storage):
    done = fetch(analysis, service, result_with_urls("fresh"))
    draws = done.draws()
    assert list(draws["scenario"].values) == ["baseline", "half"]  # the failed plan has no file
    assert draws["outcome"].dims == ("scenario", "draw", "date")
    assert draws["outcome"].sel(scenario="half").mean() == 50.0
    assert draws["tv"].sel(scenario="baseline").mean() == 50.0
    assert draws.sizes == {"scenario": 2, "draw": N_DRAWS, "date": N_WEEKS}


def test_draws_can_be_limited_to_some_scenarios(analysis, service, storage):
    done = fetch(analysis, service, result_with_urls("fresh"))
    draws = done.draws(["half"])
    assert list(draws["scenario"].values) == ["half"]
    assert [r.url.path for r in storage.requests] == ["/half.nc"]


def test_the_api_key_is_not_sent_to_the_file_host(analysis, service, storage):
    fetch(analysis, service, result_with_urls("fresh")).draws(["half"])
    assert "Authorization" not in storage.requests[0].headers


def test_unknown_or_failed_scenarios_are_refused(analysis, service):
    done = fetch(analysis, service, result_with_urls("fresh"))
    with pytest.raises(ValueError, match="broken"):
        done.draws(["half", "broken"])


def test_an_expired_url_is_renewed_by_fetching_the_result_again(analysis, service, storage):
    stale = result_with_urls("old", expires=PAST)
    done = fetch(analysis, service, stale, result_with_urls("fresh"))
    draws = done.draws(["half"])
    assert draws.sizes["scenario"] == 1
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2
    assert [r.url.params["sig"] for r in storage.requests] == ["fresh"]  # no wasted fetch


def test_a_rejected_url_is_renewed_once_and_retried(analysis, service, storage):
    # The URL looks valid but storage rejects it, as when a clock is off.
    done = fetch(analysis, service, result_with_urls("old"), result_with_urls("fresh"))
    done.draws(["half"])
    assert [r.url.params["sig"] for r in storage.requests] == ["old", "fresh"]
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2


def test_a_url_rejected_again_after_renewal_is_an_error(analysis, service, storage):
    done = fetch(analysis, service, result_with_urls("old"))
    with pytest.raises(Forbidden):
        done.draws(["half"])
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2


def test_a_url_renewed_inside_the_margin_is_used_without_renewing_again(analysis, service, storage):
    done = fetch(
        analysis,
        service,
        result_with_urls("old", expires=soon()),
        result_with_urls("fresh", expires=soon()),
    )
    draws = done.draws(["half"])
    assert draws["outcome"].sel(scenario="half").mean() == 50.0
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2
    assert [r.url.params["sig"] for r in storage.requests] == ["fresh"]


def test_a_rejected_url_renewed_inside_the_margin_is_retried_without_renewing_again(
    analysis, service, storage
):
    done = fetch(
        analysis, service, result_with_urls("old"), result_with_urls("fresh", expires=soon())
    )
    draws = done.draws(["half"])
    assert draws["outcome"].sel(scenario="half").mean() == 50.0
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2
    assert [r.url.params["sig"] for r in storage.requests] == ["old", "fresh"]


@pytest.mark.parametrize(
    ("first", "tried"),
    [(result_with_urls("old", expires=PAST), []), (result_with_urls("old"), ["old"])],
    ids=["expired", "rejected"],
)
def test_a_url_already_expired_after_renewal_is_gone(analysis, service, storage, first, tried):
    # Storage would still serve "fresh": an expired URL is refused, not downloaded.
    done = fetch(analysis, service, first, result_with_urls("fresh", expires=PAST))
    with pytest.raises(Gone, match="analysis an_1 expired"):
        done.draws(["half"])
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2
    assert [r.url.params["sig"] for r in storage.requests] == tried


def test_failed_draw_renewal_omits_both_signatures_from_the_traceback(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record()))
    done = fetch(
        client, service, result_with_urls("expired-marker"), result_with_urls("renewed-marker")
    )
    service.on(
        "GET",
        "/half.nc",
        (403, b'{"error":{"code":"forbidden","message":"expired signature=expired-marker"}}'),
        httpx.ReadTimeout("could not fetch signature=renewed-marker"),
    )
    with pytest.raises(ApiConnectionError) as caught:
        done.draws(["half"])
    trace = "".join(format_exception(caught.value))
    assert "expired-marker" not in trace
    assert "renewed-marker" not in trace
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 2
    assert [r.url.params["sig"] for r in service.sent("GET", "/half.nc")] == [
        "expired-marker",
        *(["renewed-marker"] * 4),
    ]


def test_draws_never_start_a_run(analysis, service):
    fetch(analysis, service, result_with_urls("fresh")).draws()
    assert service.sent("POST", "/v1/analyses") == []
    assert all(r.method == "GET" for r in service.requests)
