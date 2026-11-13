"""The HTTP API pages of the docs site are generated from skills/mfm/reference/http.md."""

from __future__ import annotations

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
