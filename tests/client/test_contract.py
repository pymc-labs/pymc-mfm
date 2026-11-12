"""Drift tests: the SDK, its test fakes, the committed contract and the public HTTP reference agree.

``contract.json`` lists the routes, error codes and wire fields the SDK relies on.
``skills/mfm/reference/http.md`` is the public reference for the same API. These tests read only
those two files and the SDK source, never a service. When the API changes, change both in the
same pull request; a failure here names what drifted. A response key that a fake returns but the
contract lacks fails here, and so does a contract key that ``http.md`` does not name.
"""

import dataclasses
import json
import re
from pathlib import Path

import httpx
import pytest

from pymc_mfm.client import _client, errors
from pymc_mfm.client import analysis as analysis_module
from pymc_mfm.client._http import FRESH_KEY_PREFIX, RETRY_STATUSES, content_idempotency_key
from pymc_mfm.client.errors import ApiError, NotReady, error_from_response

from . import client_fixtures as fakes
from . import local_service
from .contract_check import (
    CONTRACT,
    FIELDS,
    allowed,
    check_fake_record,
    endpoint_cell,
    http_reference_text,
    required,
    row_drift,
    unnamed_keys,
)

CLIENT_SRC = Path(errors.__file__).parent
# `request("GET", f"/v1/...")`, `create("/v1/...")` and `download_authenticated(f"/v1/...")`.
ROUTE_CALL = re.compile(
    r"""(?:request\(\s*"(?P<method>[A-Z]+)",\s*|(?P<create>create\(\s*)|download_authenticated\(\s*)
        f?"(?P<path>/v1/[^"]*)\"""",
    re.VERBOSE,
)


def sdk_routes() -> set[tuple[str, str]]:
    routes = set()
    for source in CLIENT_SRC.glob("*.py"):
        for call in ROUTE_CALL.finditer(source.read_text()):
            method = call["method"] or ("POST" if call["create"] else "GET")
            routes.add((method, re.sub(r"\{[^}]*\}", "{id}", call["path"])))
    return routes


def test_the_routes_the_sdk_calls_are_the_routes_the_contract_marks_for_the_sdk():
    declared = {(r["method"], r["path"]) for r in CONTRACT["routes"] if r["sdk"]}
    assert sdk_routes() == declared


def test_every_error_status_of_the_contract_maps_to_a_specific_class():
    for status in {e["status"] for e in CONTRACT["errors"]}:
        error = error_from_response(status, {"error": {"code": "x", "message": "m"}})
        assert isinstance(error, ApiError)
        if status < 500:
            assert type(error) is not ApiError, f"{status} has no typed error"


def test_not_ready_is_found_by_its_contract_code():
    codes = {(e["status"], e["code"]) for e in CONTRACT["errors"]}
    assert (409, "not_ready") in codes
    error = error_from_response(409, {"error": {"code": "not_ready", "message": "m"}})
    assert isinstance(error, NotReady)


def test_error_codes_named_in_the_sdk_exist_in_the_contract():
    contract_codes = {e["code"] for e in CONTRACT["errors"]}
    named = set()
    for source in CLIENT_SRC.glob("*.py"):
        named |= set(re.findall(r"code == \"([a-z_]+)\"", source.read_text()))
        named |= set(re.findall(r"`(steps_removed)`", source.read_text()))
    assert named <= contract_codes


def test_the_retry_statuses_are_the_contract_retry_statuses():
    assert set(CONTRACT["retry_statuses"]) == RETRY_STATUSES


def test_result_and_list_columns_follow_the_contract_in_order():
    assert list(analysis_module.TABLE_COLUMNS) == FIELDS["result_row"]["ordered"]
    assert list(_client.LIST_COLUMNS) == FIELDS["analysis_list_item"]["ordered"]


def test_the_status_the_sdk_reads_covers_the_required_fields_and_stays_in_the_contract():
    sdk_reads = {f.name for f in dataclasses.fields(analysis_module.AnalysisStatus)}
    assert required("analysis_status_response") <= sdk_reads <= allowed("analysis_status_response")
    assert {"runs_cancelled", "warning_details"} <= sdk_reads


def test_every_response_key_of_the_contract_is_named_in_the_http_reference():
    missing = unnamed_keys(http_reference_text(), FIELDS)
    assert not missing, f"http.md does not name these response keys: {sorted(missing)}"


def test_each_http_reference_row_names_exactly_the_contract_keys_of_its_route():
    assert row_drift(http_reference_text()) == {}


def test_a_key_removed_from_one_http_reference_row_fails():
    reference = http_reference_text()
    status_row = endpoint_cell("GET", "/v1/analyses/{id}")
    shortened = reference.replace(status_row, status_row.replace("`warning_details`", "nothing"))
    assert row_drift(shortened) == {"GET /v1/analyses/{id}": ({"warning_details"}, set())}


def test_a_key_added_to_one_http_reference_row_fails():
    reference = http_reference_text()
    list_row = endpoint_cell("GET", "/v1/analyses")
    longer = reference.replace(list_row, list_row + " Also `dummy_key`.")
    assert row_drift(longer) == {"GET /v1/analyses": (set(), {"dummy_key"})}


def test_every_sdk_route_has_a_row_in_the_http_reference():
    for route in CONTRACT["routes"]:
        if route["sdk"]:
            endpoint_cell(route["method"], route["path"])


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("dataset_response", fakes.dataset_record(content_hash="h", name="n")),
        ("dataset_response", fakes.dataset_record()),
        ("analysis_started_response", fakes.started()),
        ("analysis_preview_response", fakes.preview_record()),
        ("analysis_status_response", fakes.status_record(dataset_id="ds", seed=1, draws=2)),
        ("analysis_status_response", fakes.status_record(runs_cancelled=1, warning_details=[{}])),
    ],
)
def test_the_test_fakes_stay_inside_the_contract(name, record):
    check_fake_record(name, record)


def test_a_dummy_response_key_in_a_fake_fails_the_drift_check():
    for name, record in [
        ("analysis_started_response", fakes.started()),
        ("dataset_response", fakes.dataset_record()),
    ]:
        with pytest.raises(AssertionError, match="dummy_key"):
            check_fake_record(name, {**record, "dummy_key": 1})


def test_a_dummy_contract_key_fails_until_the_http_reference_names_it():
    fields = {**FIELDS, "analysis_started_response": {"required": ["dummy_key"]}}
    reference = http_reference_text()
    dummy = ("analysis_started_response", "dummy_key")
    assert dummy in unnamed_keys(reference, fields)
    assert dummy not in unnamed_keys(reference + "\n`dummy_key`", fields)


def test_the_result_fake_has_every_contract_column():
    row_keys = {key for row in fakes.result_rows("baseline", "baseline") for key in row}
    assert row_keys == set(FIELDS["result_row"]["ordered"])


def test_request_bodies_stay_inside_the_contract(client, service, history, plan):
    service.on("POST", "/v1/datasets", (201, fakes.dataset_record()))
    service.on("POST", "/v1/analyses", (202, fakes.started()))
    dataset = client.datasets.create(
        history,
        date="week",
        outcome="sales",
        spend=["tv", "search"],
        controls="price",
        name="n",
        currency="EUR",
    )
    every_option = {"decision_weeks": (1, 2), "draws": 100, "seed": 1, "explain_history": True}
    client.analyze(dataset, plan, name="n", **every_option)
    client.analyze(
        dataset,
        context=("2024-07-01", "2024-07-15"),
        query_end="2024-07-22",
        channels="tv",
        controls="price",
        **every_option,
    )
    service.on("POST", "/v1/analyses", (200, {"plans": [], "runs_total": 0, "estimate_seconds": 0}))
    client.analyze(dataset, plan, preview=True)
    (upload,) = service.sent("POST", "/v1/datasets")
    assert set(json.loads(upload.content)) <= set(FIELDS["dataset_request"]["optional"])
    for request in service.sent("POST", "/v1/analyses"):
        assert set(json.loads(request.content)) <= set(FIELDS["analysis_request"]["optional"])


def test_idempotency_keys_match_the_service_pattern():
    pattern = re.compile(CONTRACT["idempotency_key_pattern"])
    keys = [content_idempotency_key("/v1/analyses", {"a": 1}), FRESH_KEY_PREFIX + "0" * 32]
    assert all(pattern.fullmatch(key) for key in keys)


def test_every_json_response_of_the_fake_service_is_checked_against_the_contract(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, {**fakes.status_record(), "dummy_key": 1}))
    with pytest.raises(AssertionError, match="dummy_key"):
        client.get("an_1")
    service.on(
        "GET", "/v1/datasets", (200, {"datasets": [{**fakes.dataset_record(), "dummy_key": 1}]})
    )
    with pytest.raises(AssertionError, match="dummy_key"):
        client.datasets.list()


def test_a_dummy_key_in_a_local_service_list_response_fails(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with pytest.raises(AssertionError, match="dummy_key"):
        with local_service.serve() as (base_url, state):
            monkeypatch.setattr(state, "listing", lambda: {"analyses": [{"dummy_key": 1}]})
            httpx.get(f"{base_url}/v1/analyses", headers={"Authorization": "Bearer test-key"})
