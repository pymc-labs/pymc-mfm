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
