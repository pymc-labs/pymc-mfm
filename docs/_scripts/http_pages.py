"""Write the HTTP API pages of the site from the exported spec and the page copy.

``prepare.py`` calls ``generate`` before Quarto renders. Output goes to ``docs/api/``:
an overview, Authentication, Errors, one page for each route and ``_sidebar.yml``.
Generated files are ignored by Git. The routes, the field tables and the error list come
from ``docs/api/openapi-v1.json`` (the exported spec of the service); the prose and the
samples come from ``http_content.py``. ``skills/mfm/reference/http.md`` must agree with the
spec: its "Endpoints" table lists the same routes and each row names the keys the server
sends. If the spec, ``http.md`` and ``http_content.py`` disagree, or the structure of
``http.md`` changes, ``generate`` raises ``HttpDocsError`` and writes nothing.

``docs/api/openapi-v1.json`` is exported from the server code. Do not edit it by hand: a change
there is replaced by the next export, and the server's own checks fail on a file that differs
from what the server builds. To change what the API says, change the server and export the file
again.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import http_content as content
from http_content import Detail, Param

DOCS = Path(__file__).resolve().parent.parent
HTTP_MD = DOCS.parent / "skills" / "mfm" / "reference" / "http.md"
API_DIR = DOCS / "api"
SPEC = API_DIR / "openapi-v1.json"

METHOD_PATH = re.compile(r"^`(GET|POST|PUT|PATCH|DELETE) (/v1/[^`?\s]+)(\?[^`\s]*)?`$")
FIXED_PAGES = (
    ("index.qmd", "Overview"),
    ("authentication.qmd", "Authentication"),
    ("errors.qmd", "Errors"),
)
GROUPS = ("Datasets", "Analyses")


class HttpDocsError(Exception):
    """The spec, http.md and the page copy no longer agree."""


# ---- Reading the spec -----------------------------------------------------------------------

PATH_PARAM = re.compile(r"\{[a-z_]+\}")
STATUS_TEXT = {200: "OK", 201: "Created", 202: "Accepted"}
#: Words in a row of http.md that are not keys of a response.
NOT_KEYS = frozenset({"preview", "not_ready", "plan_failed", "null", "true", "false"})
#: Per route, the arrays whose item keys the row of http.md also names (a list of records).
ITEM_KEYS = {
    ("GET", "/v1/datasets"): "datasets",
    ("GET", "/v1/analyses"): "analyses",
    ("GET", "/v1/analyses/{id}/weekly"): "rows",
    ("GET", "/v1/analyses/{id}/export"): "scenarios",
}
MANY_CODES = 6  # an error with more codes than this points at the Errors page


def load_spec(path: Path = SPEC) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HttpDocsError(f"cannot read the spec {path.name}: {error}") from None
    if not str(spec.get("openapi", "")).startswith("3.") or "paths" not in spec:
        raise HttpDocsError(f"{path.name} is not an OpenAPI 3 document")
    return spec


def route_key(method: str, path: str) -> tuple[str, str]:
    """``("get", "/v1/datasets/{dataset_id}")`` as the pages write it: ``("GET", ".../{id}")``."""
    return method.upper(), PATH_PARAM.sub("{id}", path)


def spec_operations(spec: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    found = {}
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            found[route_key(method, path)] = operation
    return found


def deref(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with a top ``$ref`` followed."""
    while "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[1]
        try:
            schema = spec["components"]["schemas"][name]
        except KeyError:
            raise HttpDocsError(f"the spec names no schema {name}") from None
    return schema


def ref_name(schema: dict[str, Any]) -> str | None:
    return schema["$ref"].rsplit("/", 1)[1] if "$ref" in schema else None


def variants(spec: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The object schemas a body can have: ``anyOf`` members, each with its ``$ref`` followed."""
    members = schema.get("anyOf") or schema.get("oneOf") or [schema]
    return [deref(spec, member) for member in members]


def success_schemas(spec: dict[str, Any], operation: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for status, response in operation["responses"].items():
        if status.startswith("2"):
            body = response.get("content", {}).get("application/json", {}).get("schema")
            if body is not None:
                out.extend(variants(spec, body))
    return out


def properties(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The properties of an object schema. Over an ``anyOf`` (the forms of a request body) the
    properties of all members, the first description of a name winning."""
    target = deref(spec, schema)
    if "anyOf" not in target:
        return target.get("properties", {})
    merged: dict[str, dict[str, Any]] = {}
    for member in variants(spec, target):
        for name, prop in member.get("properties", {}).items():
            merged.setdefault(name, prop)
    return merged


def child_schemas(spec: dict[str, Any], schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The named object schemas under one property schema: itself, its array items, its map
    values and its ``anyOf`` members."""
    found = []
    name = ref_name(schema)
    if name is not None and deref(spec, schema).get("properties"):
        found.append((name, deref(spec, schema)))
    for member in schema.get("anyOf", []):
        found.extend(child_schemas(spec, member))
    for inner in (schema.get("items"), schema.get("additionalProperties")):
        if isinstance(inner, dict):
            found.extend(child_schemas(spec, inner))
    return found


def type_text(spec: dict[str, Any], schema: dict[str, Any]) -> str:
    """A short type for a field: ``string``, ``array of number``, ``string or null``."""
    if "$ref" in schema:
        target = deref(spec, schema)
        if "properties" in target:
            return f"object ({ref_name(schema)})"
        return type_text(spec, target)
    if "anyOf" in schema:
        parts = [type_text(spec, member) for member in schema["anyOf"]]
        null = [p for p in parts if p == "null"]
        rest = [p for p in dict.fromkeys(parts) if p != "null"]
        return " or ".join([*rest, *null])
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return " | ".join(json.dumps(value) for value in schema["enum"])
    kind = schema.get("type")
    if kind == "array":
        return (
            "array of " + type_text(spec, schema.get("items", {}))
            if schema.get("items")
            else "array"
        )
    if kind == "object":
        extra = schema.get("additionalProperties")
        return f"object of {type_text(spec, extra)}" if isinstance(extra, dict) else "object"
    if kind == "string" and schema.get("format") in ("date", "date-time"):
        return schema["format"]
    return str(kind or "any")


def schema_params(
    spec: dict[str, Any], schema: dict[str, Any], *, required_key: str = "required"
) -> tuple[Param, ...]:
    """One ``Param`` for each property of an object schema, in the spec's order. A field is
    required when every form of an ``anyOf`` requires it."""
    members = [m for m in variants(spec, schema) if "properties" in m]
    required = (
        set.intersection(*(set(m.get(required_key, ())) for m in members)) if members else set()
    )
    out = []
    for name, prop in properties(spec, schema).items():
        kind = type_text(spec, prop)
        text = prop.get("description") or deref(spec, prop).get("description") or "-"
        out.append(Param(name, text, kind, name in required))
    return tuple(out)


def error_codes(spec: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """The codes an error schema allows (``error.code``), over its ``anyOf`` members."""
    codes: list[str] = []
    for member in variants(spec, schema):
        detail = properties(spec, member).get("error")
        code = properties(spec, detail).get("code") if detail else None
        if code:
            codes.extend(code.get("enum") or ([code["const"]] if "const" in code else []))
    return list(dict.fromkeys(codes))


def response_errors(
    spec: dict[str, Any], operation: dict[str, Any]
) -> list[tuple[int, str, list[str]]]:
    """``(status, description, codes)`` of each error answer, by status."""
    rows = []
    for status, response in sorted(operation["responses"].items()):
        if status.startswith("2"):
            continue
        body = response.get("content", {}).get("application/json", {}).get("schema", {})
        rows.append((int(status), response.get("description", ""), error_codes(spec, body)))
    return rows


def top_keys(spec: dict[str, Any], route: tuple[str, str], operation: dict[str, Any]) -> set[str]:
    """The keys a row of http.md must name: the top-level keys of every success body, and the item
    keys of the array that ``ITEM_KEYS`` names for the route."""
    keys: set[str] = set()
    for schema in success_schemas(spec, operation):
        found = properties(spec, schema)
        keys |= set(found)
        wanted = ITEM_KEYS.get(route)
        if wanted in found:
            for _, child in child_schemas(spec, found[wanted]):
                keys |= set(child.get("properties", {}))
    return keys


def all_keys(spec: dict[str, Any], operation: dict[str, Any]) -> set[str]:
    """Every key in the success bodies, the request body and the parameters of an operation."""
    seen: set[str] = set()
    keys: set[str] = set()

    def walk(schema: dict[str, Any]) -> None:
        name = ref_name(schema)
        if name is not None:
            if name in seen:
                return
            seen.add(name)
        target = deref(spec, schema)
        for key, prop in target.get("properties", {}).items():
            keys.add(key)
            walk(prop)
        for member in target.get("anyOf", []):
            walk(member)
        for inner in (target.get("items"), target.get("additionalProperties")):
            if isinstance(inner, dict):
                walk(inner)

    for schema in success_schemas(spec, operation):
        walk(schema)
    request = operation.get("requestBody", {}).get("content", {}).get("application/json", {})
    if request.get("schema"):
        walk(request["schema"])
    keys |= {parameter["name"] for parameter in operation.get("parameters", [])}
    return keys


def check_http_md(spec: dict[str, Any], markdown: str) -> None:
    """Raise ``HttpDocsError`` unless ``http.md`` agrees with the spec: the Endpoints table lists
    the spec's routes, and each row names every key the server sends for the route and no key
    the spec does not know."""
    operations = spec_operations(spec)
    rows = table_rows(section_lines(markdown, "Endpoints"), ("Method and path", "Purpose"))
    listed: dict[tuple[str, str], str] = {}
    for cell, purpose in rows:
        found = METHOD_PATH.match(cell.strip("| "))
        if not found:
            raise HttpDocsError(f"cannot read the route {cell!r} in the Endpoints table")
        key = (found.group(1), found.group(2))
        if key in listed:
            raise HttpDocsError("http.md lists a route twice")
        listed[key] = purpose
    for key in operations.keys() - listed.keys():
        raise HttpDocsError(f"the spec has {key[0]} {key[1]}, which http.md does not list")
    for key in listed.keys() - operations.keys():
        raise HttpDocsError(f"http.md lists {key[0]} {key[1]}, which the spec does not have")
    problems = []
    for key, purpose in listed.items():
        named = set(re.findall(r"`([a-z_][a-z0-9_]*)`", purpose)) - NOT_KEYS
        operation = operations[key]
        missing = top_keys(spec, key, operation) - named
        unknown = named - all_keys(spec, operation)
        if missing or unknown:
            problems.append(
                f"{key[0]} {key[1]}: http.md lacks {sorted(missing)}, "
                f"names keys the spec does not have {sorted(unknown)}"
            )
    problems += status_problems(spec, markdown, listed, operations)
    if problems:
        raise HttpDocsError("http.md and the spec disagree:\n" + "\n".join(problems))


SPAN = re.compile(r"`([^`\n]+)`")
PAIR = re.compile(r"^([1-5]\d\d) ([a-z_]+)$")
GATEWAY = {502, 504}  # answered by a proxy before the service: http.md may name them


def error_pairs(spec: dict[str, Any]) -> set[tuple[int, str]]:
    """Every ``(status, code)`` an error answer of the spec can carry."""
    pairs = set()
    for operation in spec_operations(spec).values():
        for status, _, codes in response_errors(spec, operation):
            pairs |= {(status, code) for code in codes}
    return pairs


def top_level_error_keys(spec: dict[str, Any]) -> set[tuple[str, str]]:
    """``(code, key)`` for each error body with a top-level key besides ``error``
    (the 409 ``not_ready`` has ``status``)."""
    found = set()
    for operation in spec_operations(spec).values():
        for status, response in operation["responses"].items():
            schema = response.get("content", {}).get("application/json", {}).get("schema")
            if schema is None or status.startswith("2"):
                continue
            for member in variants(spec, schema):
                extra = set(properties(spec, member)) - {"error"}
                if extra and "error" in properties(spec, member):
                    for code in error_codes(spec, member):
                        found |= {(code, key) for key in extra}
    return found


def pairs_in(text: str) -> set[tuple[int, str]]:
    """Pairs of status and code in ``text``: `` `409 not_ready` `` or `` `409` `not_ready` ``."""
    found = set()
    for span in SPAN.findall(text):
        match = PAIR.match(span)
        if match:
            found.add((int(match.group(1)), match.group(2)))
    for status, code in re.findall(r"`([1-5]\d\d)`\s+`([a-z_]+)`", text):
        found.add((int(status), code))
    return found


ROUTE_SPAN = re.compile(r"`(GET|POST|PUT|PATCH|DELETE) (/v1/[^`?\s]+)(?:\?[^`\s]*)?`")


def endpoint_units(markdown: str) -> list[str]:
    """The table rows and the list items or paragraphs after the table in "Endpoints": each
    names its routes and the errors of those routes."""
    units: list[str] = []
    current: list[str] = []
    for line in section_lines(markdown, "Endpoints"):
        starts = line.startswith(("|", "- "))
        if not line.strip() or starts:
            if current:
                units.append(" ".join(current))
            current = []
        if line.strip() and not line.startswith(("| ---", "| Method")):
            current.append(line.strip())
    if current:
        units.append(" ".join(current))
    return units


def route_pair_problems(
    spec: dict[str, Any],
    markdown: str,
    listed: dict[tuple[str, str], str],
    operations: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    """Where ``http.md`` puts an error next to a route (a row or a list item of "Endpoints"), the
    route must have that status and code in the spec. A row also names the ``409`` codes of its
    route, except ``idempotency_conflict`` (every create call has it)."""
    problems = []
    for unit in endpoint_units(markdown):
        routes = {route_key(m, p) for m, p in ROUTE_SPAN.findall(unit)} & operations.keys()
        for key in routes:
            have = {
                (status, code)
                for status, _, codes in response_errors(spec, operations[key])
                for code in codes
            }
            for status, code in sorted(pairs_in(unit) - have):
                problems.append(
                    f"{key[0]} {key[1]}: http.md says `{status} {code}`, the spec has no such error"
                )
    for key, purpose in listed.items():
        named = pairs_in(purpose)
        for status, _, codes in response_errors(spec, operations[key]):
            for code in codes:
                if status == 409 and code != "idempotency_conflict" and (status, code) not in named:
                    problems.append(f"{key[0]} {key[1]}: the row does not name `{status} {code}`")
    return problems


def code_problems(spec: dict[str, Any], markdown: str) -> list[str]:
    """Every ``422`` code of the spec is named in ``http.md``, and the "Refusal codes" section
    names no other."""
    known = {code for status, code in error_pairs(spec) if status == 422}
    named = {word for span in SPAN.findall(markdown) for word in span.split()}
    problems = [f"http.md does not name the 422 code `{code}`" for code in sorted(known - named)]
    listed = {
        w
        for line in section_lines(markdown, "Refusal codes")
        if line.startswith("- ")
        for w in SPAN.findall(line)
    }
    problems += [
        f"Refusal codes names `{code}`, which is no 422 code of the spec"
        for code in sorted(listed - known)
    ]
    return problems


def status_problems(
    spec: dict[str, Any],
    markdown: str,
    listed: dict[tuple[str, str], str],
    operations: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    """The statuses and error codes of ``http.md`` against the spec: each Endpoints row names the
    success statuses of its route (200 when it names none) and only statuses the route has; every
    ``` `STATUS code` ``` pair anywhere is a pair the spec has; the "Errors and retries" section
    names every error status and every error code of the spec, and the top-level keys of the error
    bodies that have any."""
    problems = []
    for key, purpose in listed.items():
        have = {int(status) for status in operations[key]["responses"]}
        success = {status for status in have if status < 300}
        said = {
            int(number)
            for span in SPAN.findall(purpose)
            for number in re.findall(r"^[1-5]\d\d\b", span)
        }
        if not {n for n in said if n < 300}:
            said.add(200)
        wrong = said - have
        if {n for n in said if n < 300} != success or wrong:
            problems.append(
                f"{key[0]} {key[1]}: http.md says statuses {sorted(said)}, the spec has "
                f"{sorted(have)} (success {sorted(success)})"
            )
    pairs = error_pairs(spec)
    for status, code in sorted(pairs_in(markdown) - pairs):
        problems.append(f"http.md names `{status} {code}`, which the spec does not have")
    problems += route_pair_problems(spec, markdown, listed, operations)
    problems += code_problems(spec, markdown)
    lines = section_lines(markdown, "Errors and retries")
    text = "\n".join(lines)
    words = {w for span in SPAN.findall(text) for w in re.split(r"[\s/]+", span)}
    for status, code in sorted(pairs):
        if str(status) not in words:
            problems.append(f"Errors and retries does not name the status {status}")
        if status != 422 and code not in words:
            problems.append(f"Errors and retries does not name the error code {code} ({status})")
    statuses = {int(w) for w in words if re.fullmatch(r"[45]\d\d", w)}
    for status in sorted(statuses - {s for s, _ in pairs} - GATEWAY):
        problems.append(f"Errors and retries names the status {status}, which the spec lacks")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    for code, key in sorted(top_level_error_keys(spec)):
        if not any(
            code in {w for span in SPAN.findall(p) for w in span.split()}
            and f"`{key}`" in p
            and "top-level" in p
            for p in paragraphs
        ):
            problems.append(
                f"Errors and retries must say that the {code} body has the top-level key `{key}`"
            )
    return problems


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    detail: Detail
    operation: dict[str, Any]

    @property
    def short_path(self) -> str:
        return self.path.removeprefix("/v1")

    @property
    def group(self) -> str:
        return "Datasets" if self.path.startswith("/v1/datasets") else "Analyses"

    @property
    def page(self) -> str:
        return f"{self.detail.slug}.qmd"

    @property
    def verb(self) -> str:
        return f"[{self.method}]{{.mfm-verb .mfm-verb-{self.method.lower()}}}"


# ---- Reading http.md ---------------------------------------------------------------------


def section_lines(markdown: str, heading: str) -> list[str]:
    """The lines under ``## heading``, up to the next ``## `` heading."""
    lines = markdown.splitlines()
    try:
        start = lines.index(f"## {heading}") + 1
    except ValueError:
        raise HttpDocsError(f'http.md has no "## {heading}" section') from None
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def table_rows(lines: list[str], header: tuple[str, str]) -> list[tuple[str, str]]:
    """Rows of the two-column table in ``lines``; the header must match."""
    table = [line for line in lines if line.startswith("|")]
    if len(table) < 3 or tuple(c.strip() for c in table[0].strip("|").split("|")) != header:
        raise HttpDocsError(f"expected a table with columns {header}")
    rows = []
    for line in table[2:]:
        cells = [c.strip() for c in line.strip("|").split(" | ")]
        if len(cells) != 2:
            raise HttpDocsError(f"cannot read the table row: {line}")
        rows.append((cells[0], cells[1]))
    return rows


def parse_routes(markdown: str, spec: dict[str, Any]) -> list[Route]:
    """The routes of the spec, once ``http.md`` agrees with it and the copy covers each one."""
    check_http_md(spec, markdown)
    operations = spec_operations(spec)
    for key in content.DETAILS.keys() - operations.keys():
        raise HttpDocsError(
            f"http_content.py describes {key[0]} {key[1]}, which the spec does not have"
        )
    routes = []
    for key in content.DETAILS:
        if key not in operations:
            continue
        routes.append(Route(key[0], key[1], content.DETAILS[key], operations[key]))
    for key in operations:
        if key not in content.DETAILS:
            raise HttpDocsError(
                f"the spec has {key[0]} {key[1]}, but http_content.py has no page for it"
            )
    return routes


def parse_code_groups(markdown: str) -> list[tuple[str, str]]:
    """``(where, codes)`` for each bullet of the "Refusal codes" section."""
    groups = []
    for line in section_lines(markdown, "Refusal codes"):
        if line.startswith("- "):
            where, _, codes = line[2:].partition(":")
            groups.append((where.strip(), codes.strip()))
    if not groups:
        raise HttpDocsError("the Refusal codes section has no list")
    return groups


def parse_refusals(markdown: str) -> list[tuple[str, str]]:
    rows = table_rows(section_lines(markdown, "Refusals"), ("Code", "Cause"))
    if not rows:
        raise HttpDocsError("the Refusals table is empty")
    return rows


# ---- Markdown pieces -----------------------------------------------------------------------


def front_matter(title: str, description: str) -> str:
    return f"---\ntitle: {json.dumps(title)}\ndescription-meta: {json.dumps(description)}\n---\n\n"


def definition_list(items: list[tuple[str, str]], css_class: str) -> str:
    body = "\n\n".join(f"{term}\n:   {text}" for term, text in items)
    return f"::: {{.{css_class}}}\n{body}\n\n:::\n\n"


def param_term(param: Param) -> str:
    term = f"`{param.name}`"
    if param.kind:
        term += f" [{param.kind}]{{.mfm-type}}"
    if param.required:
        term += " [required]{.mfm-req}"
    return term


def params_section(heading: str, params: tuple[Param, ...]) -> str:
    if not params:
        return ""
    items = [(param_term(p), p.text) for p in params]
    return f"## {heading}\n\n" + definition_list(items, "mfm-params")


def code_block(language: str, code: str) -> str:
    return f"```{language}\n{code}\n```\n\n"


def code_panel(blocks: list[tuple[str, list[tuple[str, str, str]]]]) -> str:
    """One panel: one block for each entry; several (label, language, code) tabs make a tabset."""
    out = ["::: {.mfm-api-code}\n"]
    for title, tabs in blocks:
        out.append(f"::: {{.mfm-code-block}}\n[{title}]{{.mfm-code-title}}\n\n")
        if len(tabs) == 1:
            out.append(code_block(tabs[0][1], tabs[0][2]))
        else:
            out.append("::: {.panel-tabset}\n\n")
            out.extend(f"### {label}\n\n" + code_block(lang, code) for label, lang, code in tabs)
            out.append(":::\n\n")
        out.append(":::\n\n")
    out.append(":::\n")
    return "".join(out)


def two_columns(head: str, text: str, panel: str, extra_class: str = "") -> str:
    """The lead, the sections and the code panel. On a wide screen the panel sits on the right;
    on a narrow one it follows the lead, so a reader meets the sample first."""
    classes = f".mfm-api-grid .{extra_class}" if extra_class else ".mfm-api-grid"
    return (
        f"::: {{{classes}}}\n::: {{.mfm-api-head}}\n\n{head}:::\n\n{panel}"
        f"::: {{.mfm-api-text}}\n\n{text}:::\n:::\n"
    )


def error_items(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Set the status numbers of an error list in the status style."""
    return [
        (
            re.sub(
                r"^([\d, ]+)", lambda m: f"[{m.group(1).strip()}]{{.mfm-status}} ", term
            ).strip(),
            text,
        )
        for term, text in rows
    ]


def sample_tabs(curl: str, python: str) -> list[tuple[str, str, str]]:
    return [("curl", "bash", curl), ("Python", "python", python)]


# ---- Pages ---------------------------------------------------------------------------------


def field_tables(
    spec: dict[str, Any], schemas: list[dict[str, Any]], root_titles: list[str]
) -> str:
    """The fields of the success bodies of a route: one list for each body, then one for each
    object schema under them, each shown once."""
    out, shown = [], set()
    for title, schema in zip(root_titles, schemas):
        params = schema_params(spec, schema)
        if params:
            out.append(
                f"### {title}\n\n"
                + definition_list([(param_term(p), p.text) for p in params], "mfm-params")
            )
        pending = [
            child
            for prop in schema.get("properties", {}).values()
            for child in child_schemas(spec, prop)
        ]
        while pending:
            name, child = pending.pop(0)
            if name in shown:
                continue
            shown.add(name)
            rows = schema_params(spec, child)
            heading = f"### `{name}`" + (
                f"\n\n{child['description']}" if child.get("description") else ""
            )
            out.append(
                heading
                + "\n\n"
                + definition_list([(param_term(p), p.text) for p in rows], "mfm-params")
            )
            pending.extend(
                c
                for prop in child.get("properties", {}).values()
                for c in child_schemas(spec, prop)
            )
    return "".join(out)


def route_inputs(
    route: Route, spec: dict[str, Any]
) -> tuple[tuple[Param, ...], tuple[Param, ...], str]:
    """``(headers, parameters, body title)`` of a route, read from its operation."""
    op = route.operation
    headers = [content.AUTH]
    if "requestBody" in op:
        headers.append(content.JSON_TYPE)
    parameters = [content.ID_PARAM] if "{id}" in route.path else []
    for item in op.get("parameters", []):
        text = (
            item.get("description") or deref(spec, item.get("schema", {})).get("description") or ""
        )
        kind = "query" if item["in"] == "query" else ""
        param = Param(item["name"], text, kind, bool(item.get("required")))
        if item["in"] == "header":
            headers.append(param)
        elif item["in"] == "query":
            parameters.append(param)
    return tuple(headers), tuple(parameters), "Body" if "requestBody" in op else "Parameters"


def error_section(spec: dict[str, Any], route: Route) -> str:
    items = []
    for status, description, codes in response_errors(spec, route.operation):
        if status == 500:
            continue
        text = description
        if codes and len(codes) <= MANY_CODES:
            text += " Code: " + ", ".join(f"`{code}`" for code in codes) + "."
        elif codes:
            text += " See [Errors](errors.qmd) for the codes."
        items.append((f"{status}", text.strip()))
    if not items:
        return ""
    return (
        "## Errors\n\nSee [Errors](errors.qmd) for the body and for retries.\n\n"
        + definition_list(error_items(items), "mfm-errors")
    )


def body_sections(spec: dict[str, Any], schema: dict[str, Any]) -> str:
    """The request body of a route. One table for a plain object. For a body with several forms
    (``anyOf``) one table for each form, with the form's own types, and a line from its
    description saying when to use it: a field that has another type or another rule in each form
    (``history`` is an object in one form and null in another) cannot share a row. The object
    schemas under the fields follow, each shown once."""
    members = [m for m in child_members(spec, schema) if m[1].get("properties")]
    out, shown, pending = [], set(), []
    for index, (name, form) in enumerate(members):
        params = schema_params(spec, form)
        if len(members) == 1:
            out.append(params_section("Body", params))
        else:
            title, _, note = (form.get("description") or name).partition(":")
            out.append(f"## Body: {title.strip().removeprefix('The ')}\n\n")
            out.append(f"Send {note.strip()}\n\n" if note else "")
            out.append(definition_list([(param_term(p), p.text) for p in params], "mfm-params"))
        pending += [c for prop in form["properties"].values() for c in child_schemas(spec, prop)]
        shown.add(name)
    while pending:
        name, child = pending.pop(0)
        if name in shown:
            continue
        shown.add(name)
        rows = schema_params(spec, child)
        heading = f"### `{name}`" + (
            f"\n\n{child['description']}" if child.get("description") else ""
        )
        out.append(
            heading
            + "\n\n"
            + definition_list([(param_term(p), p.text) for p in rows], "mfm-params")
        )
        pending.extend(
            c for prop in child.get("properties", {}).values() for c in child_schemas(spec, prop)
        )
    return "".join(out)


def child_members(spec: dict[str, Any], schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The ``(name, schema)`` of a body: its ``anyOf`` forms by component name, or itself."""
    members = deref(spec, schema).get("anyOf") or [schema]
    return [(ref_name(m) or ref_name(schema) or "Body", deref(spec, m)) for m in members]


def route_page(route: Route, spec: dict[str, Any]) -> str:
    d = route.detail
    head = f"::: {{.mfm-endpoint}}\n{route.verb} `{route.path}`\n:::\n\n{d.lead}\n\n"
    headers, parameters, body_title = route_inputs(route, spec)
    request = route.operation.get("requestBody", {}).get("content", {}).get("application/json", {})
    text = [params_section("Headers", headers)]
    if parameters:
        text.append(params_section("Parameters", parameters))
    if request.get("schema"):
        text.append(body_sections(spec, request["schema"]))
    text.extend(f"## {heading}\n\n{note}\n\n" for heading, note in d.notes)
    statuses = [int(status) for status in route.operation["responses"] if status.startswith("2")]
    for status in statuses:
        response = route.operation["responses"][str(status)]
        text.append(
            f"## Response\n\n[{status}]{{.mfm-status}} {STATUS_TEXT.get(status, '')}\n\n"
            f"{response.get('description', '')}\n\n"
        )
    if d.response_text:
        text.append(f"{d.response_text}\n\n")
    schemas = success_schemas(spec, route.operation)
    if schemas:
        names = [ref_name(m) or "Response" for m in _members(route.operation)]
        text.append(field_tables(spec, schemas, [f"`{n}`" for n in names]))
    text.append(error_section(spec, route))
    blocks = [("Request", sample_tabs(d.curl, d.python))]
    blocks += [
        (title, [(lang.upper() if lang == "json" else lang, lang, code)])
        for title, lang, code in d.extra_blocks
    ]
    if d.response and statuses:
        blocks.append(
            (
                f"Response {statuses[-1]}, illustrative values",
                [("json", "json", d.response)],
            )
        )
    return front_matter(d.title, d.summary) + two_columns(head, "".join(text), code_panel(blocks))


def _members(operation: dict[str, Any]) -> list[dict[str, Any]]:
    """The schema of each success body of an operation, ``anyOf`` members flattened, as written."""
    members = []
    for status, response in operation["responses"].items():
        if status.startswith("2"):
            schema = response.get("content", {}).get("application/json", {}).get("schema")
            if schema is not None:
                members.extend(schema.get("anyOf") or schema.get("oneOf") or [schema])
    return members


def overview_page(routes: list[Route]) -> str:
    head = (
        f"The HTTP API has {len(routes)} routes under `/v1`: upload weekly data, run an analysis "
        "and read the result. The [Python library](../reference/index.qmd) makes the same calls. "
        "Every request needs a key: see [Authentication](authentication.qmd). "
        "For error bodies and retries, see [Errors](errors.qmd).\n\n"
    )
    text = []
    for group in GROUPS:
        items = [
            (
                f"{r.verb} `{r.short_path}`",
                f"[{r.detail.title}]({r.page}){{.mfm-route-link}}\n{r.detail.summary}",
            )
            for r in routes
            if r.group == group
        ]
        text.append(f"## {group} {{#{group.lower()}}}\n\n" + definition_list(items, "mfm-routes"))
    listing = next(r for r in routes if (r.method, r.path) == ("GET", "/v1/datasets")).detail
    panel = code_panel(
        [
            ("Base URL and key", [("shell", "bash", content.ENV_BLOCK)]),
            ("List your datasets", sample_tabs(listing.curl, listing.python)),
        ]
    )
    return front_matter("MFM API", "The HTTP API of MFM: routes, keys and errors.") + two_columns(
        head, "".join(text), panel, "mfm-api-overview"
    )


def authentication_page() -> str:
    head = (
        "Every request carries a key. "
        "Your invitation tells you how to get your service URL and your key.\n\n"
    )
    text = (
        "## Set the base URL and the key\n\n"
        "Keep both in the environment: `MFM_API_URL` holds the base URL "
        "and `MFM_API_KEY` holds the key. "
        "The Python client reads the same two variables.\n\n"
        "## Send the key\n\n"
        "Send `Authorization: Bearer $MFM_API_KEY` with every request. Create calls also send "
        "`Content-Type: application/json`. The body is JSON only.\n\n"
        "## Keep the key private\n\n"
        "Never print the key, and never write the URL or the key into a file. "
        "Never send the key to a `draws_url`.\n\n"
        "## Errors\n\n"
    ) + definition_list(
        error_items(
            [
                ("401", "The key is bad. Stop, and check the key."),
                ("404", "Wrong id, or the id of another user."),
                (
                    "403",
                    "The download of a signed `draws_url` answers `403` when the link has expired. "
                    "Fetch the result again for new links.",
                ),
            ]
        ),
        "mfm-errors",
    )
    request = 'curl -sS "$MFM_API_URL/v1/datasets" \\\n  -H "Authorization: Bearer $MFM_API_KEY"'
    panel = code_panel(
        [
            ("Base URL and key", [("shell", "bash", content.ENV_BLOCK)]),
            (
                "Authorized request",
                sample_tabs(request, content.CLIENT + "\nclient.datasets.list()"),
            ),
        ]
    )
    return front_matter(
        "Authentication", "Send your key as a bearer token with every request."
    ) + two_columns(head, text, panel)


def errors_page(refusals: list[tuple[str, str]], code_groups: list[tuple[str, str]]) -> str:
    head = (
        'An error body looks like `{"error": {"code", "message", "request_id", "field"?}}`. '
        "`request_id` is also the `X-Request-Id` header. Quote it when you ask for help.\n\n"
    )
    text = (
        "## Statuses\n\n"
        + definition_list(error_items(list(content.STATUS_ROWS)), "mfm-errors")
        + "## Retries\n\n"
        "After a timeout or a connection error, repeat the same create call with the same key "
        "and body. This never starts the work twice.\n\n"
        "## Refusals\n\n"
        "The service rejects these before any run, with `422`. Warnings are never refusals. "
        "The service also refuses negative spend and unknown series names.\n\n"
        + definition_list(refusals, "mfm-errors")
        + "## Refusal codes\n\nEvery code of a `422`, by where it arises.\n\n"
        + definition_list(code_groups, "mfm-errors")
        + "## Failed plans\n\nA failed plan is not an HTTP error. The status call answers `200`. "
        "The export and the result give `failure_code` for that plan. To send a failed analysis "
        "again, use a new `Idempotency-Key` (Python client: `fresh=True`). The same key returns "
        "the failed analysis.\n\n" + definition_list(list(content.FAILURE_ROWS), "mfm-errors")
    )
    panel = code_panel(
        [
            ("Error body", [("json", "json", content.ERROR_BODY)]),
            ("Refusal", [("json", "json", content.REFUSAL_BODY)]),
        ]
    )
    return front_matter(
        "Errors", "Error bodies, statuses, retries and refusals of the HTTP API."
    ) + two_columns(head, text, panel)


def sidebar_yaml(routes: list[Route]) -> str:
    """The API sidebar. JSON is valid YAML, so every string goes through ``json.dumps``."""

    def page(href: str, text: str | None = None) -> dict[str, str]:
        return {"href": href} if text is None else {"text": text, "href": href}

    def route_row(route: Route) -> dict[str, str]:
        verb = f'<span class="mfm-verb mfm-verb-{route.method.lower()}">{route.method}</span>'
        breakable = "/" + route.short_path[1:].replace(
            "/", "/<wbr>"
        )  # a long path wraps after a slash
        return page(f"api/{route.page}", f'{verb}<span class="mfm-side-path">{breakable}</span>')

    sections = [
        {
            "section": "HTTP API",
            "contents": [page(f"api/{name}", label) for name, label in FIXED_PAGES],
        }
    ]
    sections += [
        {"section": group, "contents": [route_row(r) for r in routes if r.group == group]}
        for group in GROUPS
    ]
    sections.append(
        {"section": "Python library", "contents": [page("reference/index.qmd", "Python reference")]}
    )
    sidebar = {"website": {"sidebar": [{"id": "api", "style": "floating", "contents": sections}]}}
    return json.dumps(sidebar, indent=2) + "\n"


# ---- Entry point ---------------------------------------------------------------------------


def generate(
    http_md: Path = HTTP_MD, api_dir: Path = API_DIR, spec_path: Path = SPEC
) -> list[Path]:
    """Write every page and the sidebar; return the paths. Nothing is written if parsing fails."""
    markdown = http_md.read_text(encoding="utf-8")
    spec = load_spec(spec_path)
    routes = parse_routes(markdown, spec)
    refusals = parse_refusals(markdown)
    code_groups = parse_code_groups(markdown)
    files = {
        "index.qmd": overview_page(routes),
        "authentication.qmd": authentication_page(),
        "errors.qmd": errors_page(refusals, code_groups),
        "_sidebar.yml": sidebar_yaml(routes),
    }
    files.update({route.page: route_page(route, spec) for route in routes})
    api_dir.mkdir(parents=True, exist_ok=True)
    expected = set(files)
    for stale in [*api_dir.glob("*.qmd"), api_dir / "_sidebar.yml"]:
        if stale.name not in expected and stale.exists():
            stale.unlink()
    for name, text in files.items():
        (api_dir / name).write_text(text, encoding="utf-8")
    return [api_dir / name for name in sorted(files)]
