"""Checks of fake service responses against ``contract.json`` and the ``http.md`` route rows.

``FakeService`` and ``local_service`` call ``check_response`` on every JSON body they send, so a
key outside the contract fails the test that triggered it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())
FIELDS = CONTRACT["fields"]
HTTP_REFERENCE = Path(__file__).parents[2] / CONTRACT["http_reference"]["file"]
#: Field groups that describe what the service sends back, as opposed to request bodies.
RESPONSE_FIELDS = (
    "dataset_response",
    "analysis_started_response",
    "analysis_preview_response",
    "analysis_status_response",
    "analysis_list_item",
    "result_response",
    "weekly_response",
    "weekly_row",
    "export_response",
    "export_scenario",
    "result_row",
)
#: Routes whose body wraps a list of records: route -> (wrapper key, record group).
LIST_WRAPPERS = {
    ("GET", "/v1/datasets"): ("datasets", "dataset_response"),
    ("GET", "/v1/analyses"): ("analyses", "analysis_list_item"),
}


def required(name: str) -> set[str]:
    return set(FIELDS[name]["required"])


def allowed(name: str) -> set[str]:
    return required(name) | set(FIELDS[name]["optional"])


def keys_of(name: str, fields: dict | None = None) -> set[str]:
    """Every key of a response field group, whether the contract lists it as required,
    optional or ordered."""
    group = (fields or FIELDS)[name]
    return {key for part in ("required", "optional", "ordered") for key in group.get(part, ())}


def check_fake_record(name: str, record: dict) -> None:
    """Raise ``AssertionError`` when a fake response has a key the contract does not list or
    lacks a required one."""
    extra = record.keys() - keys_of(name)
    assert not extra, (
        f"{name}: keys {sorted(extra)} are not in contract.json; add them there and in http.md"
    )
    missing = required(name) - record.keys() if "required" in FIELDS[name] else set()
    assert not missing, f"{name}: required keys {sorted(missing)} are missing"


def route_template(method: str, path: str) -> tuple[str, str]:
    """``("GET", "/v1/analyses/an_1/result")`` becomes ``("GET", "/v1/analyses/{id}/result")``."""
    parts = path.split("/")  # "", "v1", collection, id, suffix...
    if len(parts) > 3 and parts[3]:
        parts[3] = "{id}"
    return method, "/".join(parts)


def check_response(method: str, path: str, status: int, body: Any) -> None:
    """Raise ``AssertionError`` when a successful JSON response of a contract route has a key
    that ``contract.json`` does not list. Other responses (errors, raw bytes) pass."""
    if status >= 400 or not isinstance(body, dict):
        return
    route = route_template(method, path)
    if route in LIST_WRAPPERS:
        wrapper, group = LIST_WRAPPERS[route]
        assert set(body) <= {wrapper}, f"{route}: unexpected keys {sorted(set(body) - {wrapper})}"
        for record in body.get(wrapper, ()):
            check_unlisted_keys(group, record, route)
        return
    for entry in CONTRACT["routes"]:
        if (entry["method"], entry["path"]) == route and "response" in entry:
            names = entry["response"]
            union = set().union(*(keys_of(name) for name in names))
            extra = body.keys() - union
            assert not extra, (
                f"{route}: keys {sorted(extra)} are not in contract.json; "
                "add them there and in http.md"
            )
            for name in names:
                check_nested(name, body, route)


def check_nested(name: str, body: dict, route: Any) -> None:
    """Check the records under the keys that ``nested`` of a field group names."""
    for key, group in FIELDS[name].get("nested", {}).items():
        for record in body.get(key) or ():
            check_unlisted_keys(group, record, route)


def check_unlisted_keys(group: str, record: dict, route: Any) -> None:
    extra = record.keys() - keys_of(group)
    assert not extra, f"{route}: keys {sorted(extra)} are not in contract.json ({group})"


def http_reference_text() -> str:
    return HTTP_REFERENCE.read_text()


def endpoint_cell(method: str, path: str, reference: str | None = None) -> str:
    """The purpose column of the ``http.md`` endpoint table row of one route."""
    pattern = re.compile(
        rf"^\| `{method} {re.escape(path)}(?:\?[^`]*)?` \| (?P<purpose>.*) \|$", re.MULTILINE
    )
    match = pattern.search(reference if reference is not None else http_reference_text())
    assert match, f"http.md has no endpoint table row for {method} {path}"
    return match["purpose"]


def unnamed_keys(reference: str, fields: dict) -> set[tuple[str, str]]:
    """The (group, key) pairs of ``fields`` that ``reference`` does not name in backticks."""
    return {
        (name, key)
        for name in RESPONSE_FIELDS
        for key in keys_of(name, fields)
        if f"`{key}`" not in reference
    }


def row_drift(reference: str) -> dict[str, tuple[set[str], set[str]]]:
    """For each SDK route with a response in the contract, the keys that its ``http.md`` row
    lacks and the keys the row names that the contract does not list."""
    not_keys = set(CONTRACT["http_reference"]["not_keys"])
    drift = {}
    for route in CONTRACT["routes"]:
        if not route["sdk"] or "response" not in route:
            continue
        expected = set().union(*(keys_of(name) for name in route["response"]))
        named = set(
            re.findall(
                r"`([a-z_][a-z0-9_]*)`", endpoint_cell(route["method"], route["path"], reference)
            )
        )
        named -= not_keys
        if named != expected:
            drift[f"{route['method']} {route['path']}"] = (expected - named, named - expected)
    return drift
