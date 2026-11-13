"""Write the HTTP API pages of the site from ``skills/mfm/reference/http.md``.

``prepare.py`` calls ``generate`` before Quarto renders. Output goes to ``docs/api/``:
an overview, Authentication, Errors, one page for each route and ``_sidebar.yml``.
Generated files are ignored by Git. The route list comes from the "Endpoints" table of
``http.md``; the copy comes from ``http_content.py``. If the two disagree, or the
structure of ``http.md`` changes, ``generate`` raises ``HttpDocsError`` and writes nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import http_content as content
from http_content import Detail, Param

DOCS = Path(__file__).resolve().parent.parent
HTTP_MD = DOCS.parent / "skills" / "mfm" / "reference" / "http.md"
API_DIR = DOCS / "api"

METHOD_PATH = re.compile(r"^`(GET|POST|PUT|PATCH|DELETE) (/v1/[^`?\s]+)(\?[^`\s]*)?`$")
FIXED_PAGES = (
    ("index.qmd", "Overview"),
    ("authentication.qmd", "Authentication"),
    ("errors.qmd", "Errors"),
)
GROUPS = ("Datasets", "Analyses")


class HttpDocsError(Exception):
    """http.md and the page copy no longer match."""


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    detail: Detail

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


def parse_routes(markdown: str) -> list[Route]:
    rows = table_rows(section_lines(markdown, "Endpoints"), ("Method and path", "Purpose"))
    routes = []
    for cell, _purpose in rows:
        found = METHOD_PATH.match(cell.strip("| "))
        if not found:
            raise HttpDocsError(f"cannot read the route {cell!r} in the Endpoints table")
        method, path = found.group(1), found.group(2)
        detail = content.DETAILS.get((method, path))
        if detail is None:
            raise HttpDocsError(
                f"http.md lists {method} {path}, but http_content.py has no page for it"
            )
        routes.append(Route(method, path, detail))
    order = list(content.DETAILS)
    routes.sort(key=lambda route: order.index((route.method, route.path)))
    listed = {(route.method, route.path) for route in routes}
    if len(listed) != len(routes):
        raise HttpDocsError("http.md lists a route twice")
    for key in content.DETAILS:
        if key not in listed:
            raise HttpDocsError(
                f"http_content.py describes {key[0]} {key[1]}, which http.md does not list"
            )
    return routes


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


def route_page(route: Route) -> str:
    d = route.detail
    head = f"::: {{.mfm-endpoint}}\n{route.verb} `{route.path}`\n:::\n\n{d.lead}\n\n"
    text = [params_section("Headers", d.headers), params_section(d.inputs_title, d.inputs)]
    text.extend(f"## {heading}\n\n{body}\n\n" for heading, body in d.notes)
    if d.status:
        number, _, label = d.status.partition(" ")
        text.append(f"## Response\n\n[{number}]{{.mfm-status}} {label}\n\n{d.response_text}\n\n")
    if d.errors:
        items = error_items([content.ERRORS[key] for key in d.errors])
        text.append("## Errors\n\nSee [Errors](errors.qmd) for the body and for retries.\n\n")
        text.append(definition_list(items, "mfm-errors"))
    blocks = [("Request", sample_tabs(d.curl, d.python))]
    blocks += [
        (title, [(lang.upper() if lang == "json" else lang, lang, code)])
        for title, lang, code in d.extra_blocks
    ]
    if d.response:
        blocks.append((f"Response {d.status}, illustrative values", [("json", "json", d.response)]))
    return front_matter(d.title, d.summary) + two_columns(head, "".join(text), code_panel(blocks))


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
        "Every request carries a key. Create a key in the Console. "
        "The Console also shows the base URL of the service.\n\n"
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


def errors_page(refusals: list[tuple[str, str]]) -> str:
    head = 'An error body looks like `{"error": {"code", "message", "field"?}}`.\n\n'
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


def generate(http_md: Path = HTTP_MD, api_dir: Path = API_DIR) -> list[Path]:
    """Write every page and the sidebar; return the paths. Nothing is written if parsing fails."""
    markdown = http_md.read_text(encoding="utf-8")
    routes = parse_routes(markdown)
    refusals = parse_refusals(markdown)
    files = {
        "index.qmd": overview_page(routes),
        "authentication.qmd": authentication_page(),
        "errors.qmd": errors_page(refusals),
        "_sidebar.yml": sidebar_yaml(routes),
    }
    files.update({route.page: route_page(route) for route in routes})
    api_dir.mkdir(parents=True, exist_ok=True)
    expected = set(files)
    for stale in [*api_dir.glob("*.qmd"), api_dir / "_sidebar.yml"]:
        if stale.name not in expected and stale.exists():
            stale.unlink()
    for name, text in files.items():
        (api_dir / name).write_text(text, encoding="utf-8")
    return [api_dir / name for name in sorted(files)]
