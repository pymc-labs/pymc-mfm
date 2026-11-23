"""The pilot profile of the docs site renders only the tester path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# PyYAML is not a declared dependency of the base install.
yaml = pytest.importorskip("yaml")

DOCS = Path(__file__).resolve().parents[1] / "docs"
sys.path.insert(0, str(DOCS / "_scripts"))

import pilot_post  # noqa: E402
import prepare  # noqa: E402


def load(name: str) -> dict:
    return yaml.safe_load((DOCS / name).read_text(encoding="utf-8"))


def test_pilot_renders_only_the_tester_pages() -> None:
    # Quarto joins the lists of _quarto.yml and a profile, so page lists stay out of the base.
    assert "render" not in load("_quarto.yml")["project"]
    assert load("_quarto.yml")["profile"]["default"] == "full"
    pilot = load("_quarto-pilot.yml")
    assert pilot["project"]["render"] == [
        "index.qmd",
        "guide/index.qmd",
        "guide/python-client.qmd",
        "reference/index.qmd",
    ]
    assert pilot["project"]["output-dir"] != load("_quarto.yml")["project"]["output-dir"]
    menu = [item["text"] for item in pilot["website"]["navbar"]["left"]]
    assert menu == ["Docs", "PyMC Labs"]
    assert pilot["website"]["navbar"]["left"][1]["target"] == "_blank"
    assert pilot["format"]["html"]["toc-expand"] is True
    # The base config has the full site's page lists and navbar items only.
    assert load("_quarto.yml")["website"]["navbar"]["right"][0]["href"] == (
        "guide/index.qmd#get-access"
    )
    assert "api/**" in load("_quarto-full.yml")["project"]["render"]


def test_pilot_summary_links_get_a_target(tmp_path, monkeypatch) -> None:
    page = tmp_path / "client.Analysis.qmd"
    page.write_text(
        "| [table](#a.table) | rows |\n| [wait](#a.wait) | poll |\n"
        "See [table](#a.table).\n\n### wait { #a.wait }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare, "REFERENCE", tmp_path)
    prepare.anchor_summary_links()
    assert page.read_text(encoding="utf-8") == (
        "| [table]{#a.table} | rows |\n| [wait](#a.wait) | poll |\n"
        "See [table](#a.table).\n\n### wait { #a.wait }\n"
    )


def test_reference_pages_get_their_heading_as_tab_title(tmp_path, monkeypatch) -> None:
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


def test_pilot_has_one_sidebar_with_three_flat_entries() -> None:
    (sidebar,) = load("_quarto-pilot.yml")["website"]["sidebar"]
    assert sidebar["id"] == "guide"
    assert sidebar["contents"] == [
        {"text": "Get started", "href": "guide/index.qmd"},
        {"text": "Use the River API from Python", "href": "guide/python-client.qmd"},
        {"text": "Python reference", "href": "reference/index.qmd"},
    ]


def test_full_reference_sidebar_keeps_its_sections(tmp_path, monkeypatch) -> None:
    sidebar = {
        "website": {
            "sidebar": [
                {
                    "contents": [
                        "reference/index.qmd",
                        {
                            "section": "River API client",
                            "contents": ["reference/client.Client.qmd"],
                        },
                    ]
                }
            ]
        }
    }
    (tmp_path / "_sidebar.yml").write_text(yaml.safe_dump(sidebar), encoding="utf-8")
    monkeypatch.setattr(prepare, "REFERENCE", tmp_path)
    prepare.arrange_reference_sidebar()
    entry = yaml.safe_load((tmp_path / "_sidebar.yml").read_text(encoding="utf-8"))["website"][
        "sidebar"
    ][0]
    assert entry["id"] == "reference"
    assert entry["contents"][0] == {
        "section": "Python library",
        "contents": ["reference/index.qmd"],
    }
    assert entry["contents"][1]["section"] == "River API client"


def test_pilot_reference_is_one_page(tmp_path, monkeypatch) -> None:
    (tmp_path / "index.qmd").write_text(
        "# Python reference {.doc .doc-index}\n\n## River API client\n\nThe client.\n\n"
        "| | |\n| --- | --- |\n"
        "| [client.Client](client.Client.qmd#pymc_mfm.client.Client) | The client. |\n"
        "| [client.grid](client.grid.qmd#pymc_mfm.client.grid) | A grid. |\n",
        encoding="utf-8",
    )
    (tmp_path / "client.Client.qmd").write_text(
        "# client.Client { #pymc_mfm.client.Client }\n\n```python\n# a comment\n```\n\n"
        "## Methods\n\n| [get](#pymc_mfm.client.Client.get) | Get. |\n\n"
        "### get { #pymc_mfm.client.Client.get }\n\nBody.\n",
        encoding="utf-8",
    )
    (tmp_path / "client.grid.qmd").write_text(
        "# client.grid { #pymc_mfm.client.grid }\n\nExpand.\n", encoding="utf-8"
    )
    monkeypatch.setattr(prepare, "REFERENCE", tmp_path)
    assert prepare.join_reference_pages() == ["client.Client", "client.grid"]
    assert sorted(path.name for path in tmp_path.glob("*.qmd")) == ["index.qmd"]
    page = (tmp_path / "index.qmd").read_text(encoding="utf-8")
    meta = yaml.safe_load(page.split("---\n")[1])
    assert meta == {
        "pagetitle": "Python reference",
        "mfm-old-pages": ["client.Client", "client.grid"],
    }
    body = page.split("---\n", 2)[2]
    # One h1; summary rows link to anchors on this page; no group heading.
    assert body.startswith("\n# Python reference {.doc .doc-index}\n\nThe client.")
    assert "[client.grid](#pymc_mfm.client.grid)" in body
    # The install and import lines come before the summary table.
    assert body.index("pip install pymc-mfm") < body.index("[client.Client](#")
    assert "River API client" not in body
    # Objects are h2, methods stay h3, labels are plain, code comments are untouched.
    assert "\n## client.Client { #pymc_mfm.client.Client }\n" in body
    assert "\n## client.grid { #pymc_mfm.client.grid }\n" in body
    assert "\n### get { #pymc_mfm.client.Client.get }\n" in body
    assert "\n[Methods]{.mfm-ref-label}\n" in body
    assert "\n# a comment\n" in body
    assert [line for line in body.splitlines() if line.startswith("# ")] == [
        "# Python reference {.doc .doc-index}",
        "# a comment",
    ]


def test_pilot_reference_uses_the_guide_sidebar(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prepare, "REFERENCE", tmp_path)
    prepare.write_pilot_reference_metadata()
    assert (tmp_path / "_sidebar.yml").read_text(encoding="utf-8") == "{}\n"
    meta = yaml.safe_load((tmp_path / "_metadata.yml").read_text(encoding="utf-8"))
    assert meta["sidebar"] == "guide"
    assert meta["body-classes"] == "mfm-ref mfm-ref-one"


def test_old_reference_addresses_redirect_to_their_anchor(tmp_path) -> None:
    (tmp_path / "reference").mkdir()
    pilot_post.write_redirects(tmp_path, ["client.Client", "client.datasets.Datasets"])
    page = (tmp_path / "reference" / "client.datasets.Datasets.html").read_text(encoding="utf-8")
    assert 'content="0; url=index.html#pymc_mfm.client.datasets.Datasets"' in page
    # A link with its own anchor (a method) keeps it.
    assert "location.hash ||" in page
    assert "<title>Python reference \u2013 PyMC MFM</title>" in page


def test_guide_links_to_a_reference_object_reach_its_anchor() -> None:
    lua = (DOCS / "_scripts" / "pilot-links.lua").read_text(encoding="utf-8")
    assert 'reference/index.qmd" .. fragment' in lua
    assert "_scripts/pilot-links.lua" in load("_quarto-pilot.yml")["format"]["html"]["filters"]


def test_only_the_pilot_menu_links_to_the_console() -> None:
    # A path on the pilot host: it is a 404 on any host that does not serve the console.
    pilot_links = [item["href"] for item in load("_quarto-pilot.yml")["website"]["navbar"]["right"]]
    assert pilot_links == ["/console/"]
    for name in ("_quarto.yml", "_quarto-full.yml"):
        right = load(name)["website"]["navbar"].get("right", [])
        assert all("console" not in item["href"] for item in right)


def test_console_links_are_written_back_to_the_host_path(tmp_path) -> None:
    # Quarto writes a path that starts with "/" relative to the page, into the docs folder.
    (tmp_path / "guide").mkdir()
    (tmp_path / "index.html").write_text('<a href="./console/">Sign in</a>', encoding="utf-8")
    (tmp_path / "guide" / "index.html").write_text(
        '<a href="../console/">a</a><a href="../guide/console/">b</a>', encoding="utf-8"
    )
    assert pilot_post.fix_console_links(tmp_path) == 2
    assert (tmp_path / "index.html").read_text(
        encoding="utf-8"
    ) == '<a href="/console/">Sign in</a>'
    assert (tmp_path / "guide" / "index.html").read_text(encoding="utf-8") == (
        '<a href="/console/">a</a><a href="../guide/console/">b</a>'
    )
