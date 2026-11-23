"""docs/_scripts/prepare.py and finish.py: reference links resolve; one <h1> per page."""

from __future__ import annotations

import sys
from pathlib import Path

DOCS_SCRIPTS = Path(__file__).resolve().parent.parent / "docs" / "_scripts"
sys.path.insert(0, str(DOCS_SCRIPTS))

import prepare  # noqa: E402

PAGE = """# client.Analysis { #pymc_mfm.client.Analysis }

| Name | Description |
| --- | --- |
| [curves](#pymc_mfm.client.Analysis.curves) | Response curves. |
| [wait](#pymc_mfm.client.Analysis.wait) | Wait for the analysis. |

See [the class](#pymc_mfm.client.Analysis) and [the guide](../guide/index.qmd).

### wait { #pymc_mfm.client.Analysis.wait }
"""


def test_links_to_missing_sections_become_plain_names() -> None:
    fixed = prepare.unlink_missing_sections(PAGE)
    assert "| curves | Response curves. |" in fixed
    assert "[wait](#pymc_mfm.client.Analysis.wait)" in fixed
    assert "[the class](#pymc_mfm.client.Analysis)" in fixed
    assert "[the guide](../guide/index.qmd)" in fixed


def test_reference_pages_get_their_heading_as_tab_title(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "client.Client.qmd").write_text(
        "# client.Client { #pymc_mfm.client.Client }\n\nBody.\n", encoding="utf-8"
    )
    (tmp_path / "index.qmd").write_text("# Python reference {.doc .doc-index}\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "REFERENCE", tmp_path)
    prepare.add_page_titles()
    assert (tmp_path / "client.Client.qmd").read_text(encoding="utf-8") == (
        '---\npagetitle: "client.Client"\n---\n\n'
        "# client.Client { #pymc_mfm.client.Client }\n\nBody.\n"
    )
    assert (
        (tmp_path / "index.qmd")
        .read_text(encoding="utf-8")
        .startswith('---\npagetitle: "Python reference"\n---\n')
    )


def test_partial_renders_keep_existing_generated_pages(monkeypatch, tmp_path: Path) -> None:
    for folder in ("reference", "api"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "_sidebar.yml").write_text("website: {}\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "REFERENCE", tmp_path / "reference")
    monkeypatch.setattr(prepare, "API", tmp_path / "api")
    monkeypatch.setattr(prepare, "sync_examples", list)
    monkeypatch.delenv("QUARTO_PROJECT_RENDER_ALL", raising=False)

    def fail() -> None:
        raise AssertionError("a partial render regenerated the reference")

    monkeypatch.setattr(prepare, "build_reference", fail)
    prepare.main()  # returns before regenerating anything


def test_finish_leaves_one_h1_per_page(monkeypatch, tmp_path: Path) -> None:
    import finish

    page = tmp_path / "index.html"
    page.write_text(
        '<nav id="TOC"><h2 id="toc-title">On this page</h2></nav>\n'
        '<nav class="quarto-secondary-nav"><a>\n'
        '  <h1 class="quarto-secondary-nav-title">PyMC MFM: Guide</h1>\n'
        "</a></nav>\n"
        '<header><h1 class="title d-none d-lg-block">PyMC MFM: Guide</h1></header>\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("QUARTO_PROJECT_OUTPUT_FILES", f"{page}\n{tmp_path / 'search.json'}")
    finish.main()

    tidy = page.read_text(encoding="utf-8")
    assert tidy.count("<h1") == 1
    assert '<h1 class="title d-none d-lg-block">PyMC MFM: Guide</h1>' in tidy
    assert '<div class="quarto-secondary-nav-title">PyMC MFM: Guide</div>' in tidy
    assert "<h2" not in tidy and '<div id="toc-title">On this page</div>' in tidy


def test_finish_names_the_site_once_in_the_title() -> None:
    import finish

    home = "<title>PyMC MFM: Open-Source Marketing Foundational Models in Python – PyMC MFM</title>"
    guide = "<title>Get started – PyMC MFM</title>"
    assert finish.demote_nav_title(home) == (
        "<title>PyMC MFM: Open-Source Marketing Foundational Models in Python</title>"
    )
    assert finish.demote_nav_title(guide) == guide
