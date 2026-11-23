"""The pilot profile of the docs site renders only the tester path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# PyYAML is not a declared dependency of the base install.
yaml = pytest.importorskip("yaml")

DOCS = Path(__file__).resolve().parents[1] / "docs"
sys.path.insert(0, str(DOCS / "_scripts"))

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
        "guide/http-api.qmd",
        "reference/index.qmd",
        "reference/client.*.qmd",
    ]
    assert pilot["project"]["output-dir"] != load("_quarto.yml")["project"]["output-dir"]
    menu = [item["text"] for item in pilot["website"]["navbar"]["left"]]
    assert menu == ["Docs", "Python reference", "PyMC Labs"]
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


def test_reference_sidebar_lists_the_index_once_in_the_pilot(tmp_path, monkeypatch) -> None:
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
    prepare.arrange_reference_sidebar(None)
    contents = yaml.safe_load((tmp_path / "_sidebar.yml").read_text(encoding="utf-8"))["website"][
        "sidebar"
    ][0]["contents"]
    assert contents[0] == "reference/index.qmd"
    assert contents[1]["section"] == "River API client"
