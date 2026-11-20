"""Checks of fake responses against ``contract.json``, the ``http.md`` route rows and the spec.

``FakeService`` and ``local_service`` call ``check_response`` on every JSON body they send, so a
key outside the contract fails the test that triggered it.

``contract.json`` is the part of the API the SDK supports. ``docs/api/openapi-v1.json`` describes
everything the server sends. ``spec_problems`` checks that the first fits the second: every field
of the contract exists in the spec with a compatible type, and the spec may have more (optional
fields and whole routes). It never demands equality.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "docs" / "_scripts"))  # stdlib only: the spec readers of the docs

import http_pages  # noqa: E402

CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())
FIELDS = CONTRACT["fields"]
HTTP_REFERENCE = ROOT / CONTRACT["http_reference"]["file"]
SPEC_FILE = http_pages.SPEC
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


def row_drift(
    reference: str, spec: dict[str, Any] | None = None
) -> dict[str, tuple[set[str], set[str]]]:
    """For each SDK route with a response in the contract, the contract keys that its ``http.md``
    row lacks, and the keys the row names that are not allowed. Allowed: the contract keys of the
    route and, when ``spec`` is given, every key the spec has for the route (the row may name more
    than the SDK reads)."""
    not_keys = set(CONTRACT["http_reference"]["not_keys"])
    operations = http_pages.spec_operations(spec) if spec is not None else {}
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
        allowed = expected | (
            http_pages.all_keys(spec, operations[(route["method"], route["path"])])
            if spec is not None
            else set()
        )
        if not expected <= named or not named <= allowed:
            drift[f"{route['method']} {route['path']}"] = (expected - named, named - allowed)
    return drift


# ---- The contract against the spec ----------------------------------------------------------

#: What a contract type accepts of a spec type. The spec's types are the JSON types, with the
#: formats ``date`` and ``date-time`` as their own words.
ACCEPTS = {
    "string": {"string"},
    "date": {"date"},
    "date-time": {"date-time"},
    "integer": {"integer"},
    "number": {"integer", "number"},
    "boolean": {"boolean"},
    "array": {"array"},
    "object": {"object"},
}


def spec_types(spec: dict[str, Any], schema: dict[str, Any]) -> set[str]:
    """The JSON types a spec schema allows, with ``null`` when it is nullable."""
    if "$ref" in schema:
        return spec_types(spec, http_pages.deref(spec, schema))
    if "anyOf" in schema:
        return set().union(*(spec_types(spec, member) for member in schema["anyOf"]))
    if "const" in schema or "enum" in schema:
        return {"string"}
    kind = schema.get("type", "object" if "properties" in schema else "any")
    if kind == "string" and schema.get("format") in ("date", "date-time"):
        return {schema["format"]}
    return {kind}


def _required(spec: dict[str, Any], schema: dict[str, Any]) -> set[str]:
    return set(http_pages.deref(spec, schema).get("required", ()))


def _accepts(kind: str, found: set[str]) -> bool:
    """Whether a contract type (``number``, ``string|null``) accepts what the spec allows."""
    base, _, nullable = kind.partition("|")
    return found - {"null"} <= ACCEPTS[base] and ("null" not in found or nullable == "null")


def _group_problems(
    spec: dict[str, Any], name: str, candidates: list[dict[str, Any]], where: str
) -> list[str]:
    """Check one field group of the contract against the candidate object schemas where its
    records can be. One candidate must hold the group: every contract key exists, each required
    or ordered key is always sent, and each typed key has a compatible type. A ``typed`` group
    (a record the SDK reads value by value) must give a type to every key."""
    group = FIELDS[name]
    wanted = set(group.get("required", ())) | set(group.get("ordered", ()))
    everything = wanted | set(group.get("optional", ()))
    types = group.get("types", {})
    if group.get("typed") and everything - types.keys():
        return [
            f"{where} {name}: contract.json gives no type to {sorted(everything - types.keys())}"
        ]
    reasons = []
    for candidate in candidates:
        props = http_pages.properties(spec, candidate)
        always = _required(spec, candidate)
        missing = everything - props.keys()
        optional_only = wanted - always
        wrong = [
            f"{key} is {sorted(spec_types(spec, props[key]))}, the contract wants {kind}"
            for key, kind in types.items()
            if key in props and not _accepts(kind, spec_types(spec, props[key]))
        ]
        if not (missing or optional_only or wrong):
            return []
        reasons.append(
            f"missing {sorted(missing)}, not always sent {sorted(optional_only)}, "
            f"wrong type {wrong}"
        )
    return [f"{where} {name}: {'; '.join(reasons) or 'no schema'}"]


def _object_variants(spec: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The object schemas a property can hold (``null`` and other types left out)."""
    return [m for m in http_pages.variants(spec, schema) if "properties" in m]


def _check(
    spec: dict[str, Any], name: str, candidates: list[dict[str, Any]], where: str
) -> list[str]:
    """``_group_problems`` of ``name``, then of the groups under it: ``nested`` (arrays of
    records) and ``objects`` (one record each), by key."""
    problems = _group_problems(spec, name, candidates, where)
    if problems:
        return problems
    group = FIELDS[name]
    for key, child in group.get("nested", {}).items():
        items = [
            item
            for c in candidates
            if key in http_pages.properties(spec, c)
            for item in _items(spec, http_pages.properties(spec, c)[key])
        ]
        problems += _check(spec, child, items, f"{where} {key}[]")
    for key, child in group.get("objects", {}).items():
        found = [
            v
            for c in candidates
            if key in http_pages.properties(spec, c)
            for v in _object_variants(spec, http_pages.properties(spec, c)[key])
        ]
        problems += _check(spec, child, found, f"{where} {key}")
    return problems


def _items(spec: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The item schemas of an array schema, over its ``anyOf`` members."""
    members = schema.get("anyOf") or [schema]
    found = []
    for member in members:
        member = http_pages.deref(spec, member)
        if member.get("type") == "array" and "items" in member:
            found.extend(http_pages.variants(spec, member["items"]))
    return found


def spec_problems(spec: dict[str, Any]) -> list[str]:
    """What in ``contract.json`` the spec does not support. Empty: the contract fits the spec."""
    operations = http_pages.spec_operations(spec)
    problems: list[str] = []
    for route in CONTRACT["routes"]:
        key = (route["method"], route["path"])
        operation = operations.get(key)
        if operation is None:
            problems.append(f"the spec has no route {key[0]} {key[1]}")
            continue
        if not route.get("response"):
            continue
        where = f"{key[0]} {key[1]}"
        bodies = http_pages.success_schemas(spec, operation)
        wrapper = LIST_WRAPPERS.get(key)
        nested_groups = {g for n in route["response"] for g in FIELDS[n].get("nested", {}).values()}
        for name in route["response"]:
            if name in nested_groups:
                continue  # checked as the records of the body that holds them
            if wrapper is not None:
                wrapper_key, group = wrapper
                holders = [b for b in bodies if wrapper_key in http_pages.properties(spec, b)]
                if not holders:
                    problems.append(f"{where}: no body has {wrapper_key}")
                    continue
                items = [
                    item
                    for holder in holders
                    for item in _items(spec, http_pages.properties(spec, holder)[wrapper_key])
                ]
                problems += _check(spec, group, items, where)
                continue
            problems += _check(spec, name, bodies, where)
    problems += _request_problems(spec, operations)
    problems += _error_problems(spec, operations)
    return problems


def _request_problems(spec: dict[str, Any], operations: dict) -> list[str]:
    problems = []
    for request, route in (
        ("dataset_request", ("POST", "/v1/datasets")),
        ("analysis_request", ("POST", "/v1/analyses")),
    ):
        body = operations[route]["requestBody"]["content"]["application/json"]["schema"]
        props = http_pages.properties(spec, body)
        missing = set(FIELDS[request]["optional"]) - props.keys()
        if missing:
            problems.append(f"{route[1]} request: the spec lacks {sorted(missing)}")
    parameters = {p["name"]: p for p in operations[("POST", "/v1/datasets")]["parameters"]}
    pattern = parameters.get("Idempotency-Key", {}).get("schema", {}).get("pattern")
    if pattern != CONTRACT["idempotency_key_pattern"]:
        problems.append(
            f"Idempotency-Key pattern: spec {pattern!r}, contract "
            f"{CONTRACT['idempotency_key_pattern']!r}"
        )
    return problems


def _error_problems(spec: dict[str, Any], operations: dict) -> list[str]:
    documented: set[tuple[int, str]] = set()
    for operation in operations.values():
        for status, _, codes in http_pages.response_errors(spec, operation):
            documented |= {(status, code) for code in codes}
    problems = []
    for error in CONTRACT["errors"]:
        code = error["code"]
        found = any(
            s == error["status"] and (c.startswith(code[:-1]) if code.endswith("*") else c == code)
            for s, c in documented
        )
        if not found:
            problems.append(f"the spec documents no {error['status']} {code}")
    return problems
