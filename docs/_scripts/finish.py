"""Tidy the rendered pages after Quarto writes them (``project.post-render``).

Quarto copies each page's title into an ``<h1>`` in its phone navigation bar
(``.quarto-secondary-nav``). site.css hides that bar everywhere, but the copy stays in the
HTML, so every page would carry two ``<h1>`` elements and search engines would read the
title twice. The "On this page" label of the outline is an ``<h2>`` that comes before the
page in the HTML, so it would be the page's first heading. Both become a ``<div>`` with the
same class or id (Quarto's scripts and site.css find them that way), leaving the page title
as the only ``<h1>`` and the first heading.

Quarto also ends every ``<title>`` with " – PyMC MFM". A title that already names PyMC MFM
(the home page's) keeps it once: search results show it whole and without the repeat.

Quarto opens each callout title with the callout's type for screen readers
(``<span class="screen-reader-only">Note</span>Separate marginals``), and its search index
runs the two together ("NoteSeparate marginals"), which search results would show. The
index keeps the title alone.

Quarto runs this from ``docs/`` and lists the pages it wrote in
``QUARTO_PROJECT_OUTPUT_FILES`` (one per line); without it, every page in ``_site`` is
tidied.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SITE = DOCS / "_site"
NAV_TITLE = re.compile(r'<h1 class="quarto-secondary-nav-title">(.*?)</h1>', flags=re.DOTALL)
TOC_TITLE = re.compile(r'<h2 id="toc-title">(.*?)</h2>', flags=re.DOTALL)
SITE_NAME = "PyMC MFM"
TITLE = re.compile(r"<title>(.*?) – " + re.escape(SITE_NAME) + r"</title>", flags=re.DOTALL)
CALLOUT_LABEL = re.compile(
    r'<span class="screen-reader-only">([^<]+)</span>(.*?)</div>', flags=re.DOTALL
)
TAG = re.compile(r"<[^>]+>")


def demote_nav_title(page: str) -> str:
    """The navigation bar's title copy and the outline label as ``<div>``s; the rest unchanged."""
    page = NAV_TITLE.sub(r'<div class="quarto-secondary-nav-title">\1</div>', page)
    page = TOC_TITLE.sub(r'<div id="toc-title">\1</div>', page)
    return TITLE.sub(
        lambda m: f"<title>{m.group(1)}</title>" if SITE_NAME in m.group(1) else m.group(0), page
    )


def index_text(fragment: str) -> str:
    """HTML as Quarto's search index writes it: the text, with only ``<`` and ``>`` escaped."""
    return html.unescape(TAG.sub("", fragment)).replace("<", "&lt;").replace(">", "&gt;")


def callout_labels(page: str) -> list[tuple[str, str]]:
    """Each callout's screen-reader label and the first word of its title, as indexed."""
    found = []
    for m in CALLOUT_LABEL.finditer(page):
        words = index_text(m.group(2)).split()
        if words:
            found.append((index_text(m.group(1)), words[0]))
    return found


def unlabel_search(entries: list[dict], labels: dict[str, list[tuple[str, str]]]) -> bool:
    """Drop the callout labels that run into a title in the index entries of each page.

    ``labels`` maps a page path relative to the site (``guide/concepts.html``) to its
    ``callout_labels``. Returns whether any entry changed.
    """
    changed = False
    for entry in entries:
        for label, word in labels.get(entry.get("href", "").split("#", 1)[0], ()):
            text = entry.get("text", "").replace(label + word, word)
            if text != entry.get("text", ""):
                entry["text"] = text
                changed = True
    return changed


def rendered_pages() -> list[Path]:
    listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").strip()
    if not listed:
        return sorted(SITE.rglob("*.html"))
    pages = []
    for line in listed.splitlines():
        path = Path(line.strip())
        if not path.is_absolute():
            path = DOCS / path
        if path.suffix == ".html" and path.is_file():
            pages.append(path)
    return pages


def main() -> None:
    changed = 0
    labels: dict[str, list[tuple[str, str]]] = {}
    for path in rendered_pages():
        page = path.read_text(encoding="utf-8")
        if path.is_relative_to(SITE) and (found := callout_labels(page)):
            labels[path.relative_to(SITE).as_posix()] = found
        tidy = demote_nav_title(page)
        if tidy != page:
            path.write_text(tidy, encoding="utf-8")
            changed += 1
    search = SITE / "search.json"
    if labels and search.is_file():
        entries = json.loads(search.read_text(encoding="utf-8"))
        if unlabel_search(entries, labels):
            search.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"finish: the page title is the first heading ({changed} page(s) tidied)", file=sys.stderr
    )


if __name__ == "__main__":
    main()
