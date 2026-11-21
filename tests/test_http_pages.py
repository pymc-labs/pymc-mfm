"""The HTTP API pages of the docs site are generated from skills/mktfm/reference/http.md."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

DOCS_SCRIPTS = Path(__file__).resolve().parent.parent / "docs" / "_scripts"
sys.path.insert(0, str(DOCS_SCRIPTS))

import http_pages  # noqa: E402

ROUTE_ROW = re.compile(r"^\| `(?:GET|POST|PUT|PATCH|DELETE) /v1/", re.MULTILINE)


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    api_dir = tmp_path_factory.mktemp("api")
    http_pages.generate(http_pages.HTTP_MD, api_dir)
    return api_dir


def test_one_page_per_route_and_every_page_in_the_sidebar(generated):
    route_count = len(ROUTE_ROW.findall(http_pages.HTTP_MD.read_text(encoding="utf-8")))
    assert route_count > 0
    pages = {p.name for p in generated.glob("*.qmd")}
    fixed = {"index.qmd", "authentication.qmd", "errors.qmd"}
    assert len(pages - fixed) == route_count
    sidebar = json.loads((generated / "_sidebar.yml").read_text(encoding="utf-8"))
    sections = sidebar["website"]["sidebar"][0]["contents"]
    listed = {
        item["href"].removeprefix("api/")
        for section in sections
        for item in section["contents"]
        if item["href"].startswith("api/")
    }
    assert listed == pages
    overview = (generated / "index.qmd").read_text(encoding="utf-8")
    assert f"has {route_count} routes" in overview


def test_pages_carry_no_internal_notes_or_hostnames(generated):
    text = "\n".join(p.read_text(encoding="utf-8") for p in generated.glob("*.qmd")).lower()
    for phrase in (
        "confirm before",
        "untested",
        "drift",
        "pymc-labs.com",
        "staging",
        "SKILL.md".lower(),
    ):
        assert phrase not in text


def test_a_route_without_copy_fails_loudly(tmp_path):
    markdown = http_pages.HTTP_MD.read_text(encoding="utf-8")
    extra = markdown.replace(
        "| `POST /v1/datasets` |",
        "| `PUT /v1/datasets/{id}` | Rename. |\n| `POST /v1/datasets` |",
        1,
    )
    source = tmp_path / "http.md"
    source.write_text(extra, encoding="utf-8")
    with pytest.raises(http_pages.HttpDocsError, match="PUT /v1/datasets/{id}"):
        http_pages.generate(source, tmp_path / "api")
    assert not (tmp_path / "api").exists()


def test_a_changed_table_fails_loudly(tmp_path):
    markdown = http_pages.HTTP_MD.read_text(encoding="utf-8")
    source = tmp_path / "http.md"
    source.write_text(
        markdown.replace("| Method and path | Purpose |", "| Route | Meaning |"), encoding="utf-8"
    )
    with pytest.raises(http_pages.HttpDocsError, match="columns"):
        http_pages.generate(source, tmp_path / "api")


# ---- The spec is the source of the routes, fields and errors ---------------------------------

SPEC = http_pages.load_spec()
MARKDOWN = http_pages.HTTP_MD.read_text(encoding="utf-8")


def test_http_md_agrees_with_the_spec():
    http_pages.check_http_md(SPEC, MARKDOWN)


def test_a_key_the_server_sends_but_http_md_lacks_fails():
    row = next(line for line in MARKDOWN.splitlines() if line.startswith("| `POST /v1/analyses` |"))
    shortened = MARKDOWN.replace(row, row.replace("`warning_details`", "warnings"))
    with pytest.raises(http_pages.HttpDocsError, match="warning_details"):
        http_pages.check_http_md(SPEC, shortened)


def test_a_key_the_spec_does_not_know_fails():
    row = next(line for line in MARKDOWN.splitlines() if line.startswith("| `GET /v1/datasets` |"))
    with pytest.raises(http_pages.HttpDocsError, match="dummy_key"):
        http_pages.check_http_md(SPEC, MARKDOWN.replace(row, row + " Also `dummy_key`."))


def test_a_spec_route_http_md_does_not_list_fails():
    spec = copy.deepcopy(SPEC)
    spec["paths"]["/v1/datasets/{dataset_id}"]["put"] = copy.deepcopy(
        spec["paths"]["/v1/datasets/{dataset_id}"]["get"]
    )
    with pytest.raises(http_pages.HttpDocsError, match="PUT /v1/datasets/{id}"):
        http_pages.check_http_md(spec, MARKDOWN)


def test_a_spec_route_without_page_copy_fails(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["paths"]["/v1/datasets/{dataset_id}"]["put"] = copy.deepcopy(
        spec["paths"]["/v1/datasets/{dataset_id}"]["get"]
    )
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    markdown = MARKDOWN.replace(
        "| `DELETE /v1/datasets/{id}` |",
        "| `PUT /v1/datasets/{id}` | Rename `dataset_id`. |\n| `DELETE /v1/datasets/{id}` |",
    )
    source = tmp_path / "http.md"
    source.write_text(markdown, encoding="utf-8")
    with pytest.raises(http_pages.HttpDocsError):
        http_pages.generate(source, tmp_path / "api", path)
    assert not (tmp_path / "api").exists()


def test_field_tables_come_from_the_spec(generated):
    page = (generated / "datasets-create.qmd").read_text(encoding="utf-8")
    for field in ("history_ranges", "warning_details", "content_hash", "outcome_name", "currency"):
        assert f"`{field}`" in page
    listing = (generated / "datasets-list.qmd").read_text(encoding="utf-8")
    assert "`next_cursor`" in listing and "`limit`" in listing and "`cursor`" in listing
    result = (generated / "analyses-result.qmd").read_text(encoding="utf-8")
    assert "`compact`" in result and "`plans`" in result and "`response_curves`" in result
    started = (generated / "analyses-create.qmd").read_text(encoding="utf-8")
    assert "`dataset_id`" in started and "`warnings`" in started


def test_error_lists_come_from_the_spec(generated):
    page = (generated / "analyses-create.qmd").read_text(encoding="utf-8")
    for code in ("idempotency_conflict", "maintenance", "unavailable", "overloaded_user_runs"):
        assert f"`{code}`" in page
    assert "[503]{.mktfm-status}" in page and "[408]{.mktfm-status}" in page


def test_the_idempotency_key_is_optional_on_every_page(generated):
    for name in ("datasets-create.qmd", "analyses-create.qmd"):
        page = (generated / name).read_text(encoding="utf-8")
        line = next(block for block in page.split("\n\n") if block.startswith("`Idempotency-Key`"))
        assert "[required]" not in line and "does not require" in line


def test_every_key_of_a_sample_response_is_a_key_the_server_sends():
    """A stale sample (the status once showed `created_at`) cannot come back through the copy."""
    import http_content

    operations = http_pages.spec_operations(SPEC)
    for key, detail in http_content.DETAILS.items():
        sample = set(re.findall(r'"([a-z_][a-z0-9_]*)"\s*:', detail.response))
        names = {"tv", "search", "holiday"}  # the series of the samples are data, not keys
        assert sample - names <= http_pages.all_keys(SPEC, operations[key]), (key, sample)


# ---- Statuses and error codes ------------------------------------------------------------------


def mutated(old: str, new: str) -> str:
    assert old in MARKDOWN, old
    return MARKDOWN.replace(old, new, 1)


@pytest.mark.parametrize(
    ("what", "old", "new", "expected"),
    [
        ("a 201 written as 200", "`201` with `dataset_id`", "`200` with `dataset_id`", "201"),
        (
            "a success status the row does not name",
            "`202` with `analysis_id`",
            "with `analysis_id`",
            "POST /v1/analyses",
        ),
        (
            "an error code removed",
            " `409 plan_failed`: that\n  plan has no result.",
            "",
            "plan_failed",
        ),
        (
            "an error code the spec lacks",
            "`409 idempotency_conflict`",
            "`409 idempotency_clash`",
            "idempotency_clash",
        ),
        (
            "the not_ready status note dropped",
            " A `409 not_ready` body also has a top-level `status`, the status of the analysis.",
            "",
            "top-level key `status`",
        ),
        (
            "an error status removed",
            "- `413 body_too_large`: the body is too large. Send fewer weeks, scenarios or "
            "curves in one call.\n",
            "",
            "413",
        ),
        ("a status the spec lacks", "`404 not_found`", "`418 not_found`", "418"),
    ],
)
def test_a_status_or_error_code_that_disagrees_with_the_spec_fails(what, old, new, expected):
    with pytest.raises(http_pages.HttpDocsError, match=expected):
        http_pages.check_http_md(SPEC, mutated(old, new))


def test_a_spec_status_that_http_md_lacks_fails():
    spec = copy.deepcopy(SPEC)
    responses = spec["paths"]["/v1/datasets"]["post"]["responses"]
    responses["507"] = copy.deepcopy(responses["413"])
    with pytest.raises(http_pages.HttpDocsError, match="507"):
        http_pages.check_http_md(spec, MARKDOWN)


def test_a_spec_error_code_that_http_md_lacks_fails():
    spec = copy.deepcopy(SPEC)
    schema = spec["components"]["schemas"]["Error410Detail"]["properties"]["code"]
    schema["enum"] = ["gone", "expired"]
    with pytest.raises(http_pages.HttpDocsError, match="expired"):
        http_pages.check_http_md(spec, MARKDOWN)


# ---- Errors per route, both pair formats, every 422 code ---------------------------------------


@pytest.mark.parametrize(
    ("what", "old", "new", "expected"),
    [
        (
            "a route's 409 changed to another route's code",
            "`409 not_ready` until the analysis ends. |\n| `GET /v1/analyses/{id}/weekly",
            "`409 plan_failed` until the analysis ends. |\n| `GET /v1/analyses/{id}/weekly",
            r"GET /v1/analyses/\{id\}/result",
        ),
        (
            "a 409 code a row must name, removed",
            "`409 not_ready` until the analysis ends. `409 plan_failed` when",
            "until the analysis ends. `409 plan_failed` when",
            "not_ready",
        ),
        (
            "an error of another route in the paging bullet",
            "is `422 invalid_input`.",
            "is `422 invalid_limit`.",
            "invalid_limit",
        ),
        (
            "a pair written as two spans, with a code the spec lacks",
            "A bad `limit` is `422 invalid_limit`",
            "A bad `limit` is `422` `invalid_limits`",
            "invalid_limits",
        ),
        (
            "a pair written as two spans, with a code of another route",
            "is `422 invalid_input`.",
            "is `422` `invalid_cursor`.",
            "invalid_cursor",
        ),
        ("a 422 code removed", "`seed_out_of_range`, ", "", "seed_out_of_range"),
        (
            "a 422 code the spec lacks",
            "`invalid_limit`, `invalid_cursor`",
            "`invalid_limit`, `made_up`",
            "made_up",
        ),
    ],
)
def test_an_error_that_disagrees_per_route_or_a_missing_422_code_fails(what, old, new, expected):
    with pytest.raises(http_pages.HttpDocsError, match=expected):
        http_pages.check_http_md(SPEC, mutated(old, new))


def test_a_pair_written_as_two_spans_passes_when_the_spec_has_it():
    text = mutated("is `422 invalid_input`.", "is `422` `invalid_input`.")
    http_pages.check_http_md(SPEC, text)


def test_every_422_code_of_the_spec_is_on_the_errors_page(generated):
    page = (generated / "errors.qmd").read_text(encoding="utf-8")
    codes = SPEC["components"]["schemas"]["Error422Detail"]["properties"]["code"]["enum"]
    assert codes and all(f"`{code}`" in page for code in codes)


def body_section(page: str, title: str) -> str:
    """The text of one ``## Body: ...`` section of a page."""
    start = page.index(f"## Body: {title}")
    end = page.index("\n## ", start + 1)
    return page[start:end]


def test_each_analysis_form_has_its_own_body_table(generated):
    page = (generated / "analyses-create.qmd").read_text(encoding="utf-8")
    stored = body_section(page, "plan form on a stored dataset")
    inline = body_section(page, "plan form on an inline dataset")
    weeks = body_section(page, "weeks form")
    assert "`dataset_id` [string]{.mktfm-type} [required]{.mktfm-req}" in stored
    assert "`plan` [object (PlanBody)]{.mktfm-type} [required]{.mktfm-req}" in stored
    assert "`history` [object (DatasetCreate)]{.mktfm-type} [required]{.mktfm-req}" in inline
    assert (
        "`dataset_id` [null]{.mktfm-type}\n" in inline
        and "[required]" not in inline.split("`dataset_id` [null]")[1].split("\n")[0]
    )
    assert "`context` [object (ContextBody)]{.mktfm-type} [required]{.mktfm-req}" in weeks
    assert not re.search(r"^`(plan|history)` \[", weeks, re.MULTILINE)  # no such field here
    # The objects under the fields follow, so an inline analysis can be built from the page.
    assert "### `DatasetCreate`" in page and "### `PlanBody`" in page
    assert page.index("### `DatasetCreate`") > page.index("## Body: weeks form")
    assert inline.count("Send a `history` and a `plan`.") == 1


def test_a_plain_body_keeps_one_body_table(generated):
    page = (generated / "datasets-create.qmd").read_text(encoding="utf-8")
    assert page.count("## Body") == 1 and "## Body:" not in page
