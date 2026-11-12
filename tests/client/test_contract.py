"""Drift tests: the SDK, its test fakes and the committed contract fixture agree.

``contract.json`` lists the routes, error codes and wire fields the SDK relies on. These tests
read only that file and the SDK source, never a service. When the API changes, change
``contract.json`` in the same pull request; a failure here names what drifted.
"""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from pymc_mfm.client import _client, errors
from pymc_mfm.client import analysis as analysis_module
from pymc_mfm.client._http import FRESH_KEY_PREFIX, RETRY_STATUSES, content_idempotency_key
from pymc_mfm.client.errors import ApiError, NotReady, error_from_response

from . import client_fixtures as fakes

CONTRACT = json.loads((Path(__file__).with_name("contract.json")).read_text())
CLIENT_SRC = Path(errors.__file__).parent
FIELDS = CONTRACT["fields"]
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


def required(name: str) -> set[str]:
    return set(FIELDS[name]["required"])


def allowed(name: str) -> set[str]:
    return required(name) | set(FIELDS[name]["optional"])


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


def test_the_status_the_sdk_reads_is_required_by_the_contract():
    sdk_reads = {f.name for f in dataclasses.fields(analysis_module.AnalysisStatus)}
    assert sdk_reads == required("analysis_status_response") & sdk_reads
    assert sdk_reads == required("analysis_status_response")


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("dataset_response", fakes.dataset_record(content_hash="h", name="n")),
        ("analysis_started_response", fakes.started()),
        ("analysis_status_response", fakes.status_record(dataset_id="ds", seed=1, draws=2)),
    ],
)
def test_the_test_fakes_stay_inside_the_contract(name, record):
    assert required(name) <= record.keys() <= allowed(name)


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
