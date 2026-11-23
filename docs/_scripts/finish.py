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

Quarto runs this from ``docs/`` and lists the pages it wrote in
``QUARTO_PROJECT_OUTPUT_FILES`` (one per line); without it, every page in ``_site`` is
tidied.
"""

from __future__ import annotations

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


def demote_nav_title(page: str) -> str:
    """The navigation bar's title copy and the outline label as ``<div>``s; the rest unchanged."""
    page = NAV_TITLE.sub(r'<div class="quarto-secondary-nav-title">\1</div>', page)
    page = TOC_TITLE.sub(r'<div id="toc-title">\1</div>', page)
    return TITLE.sub(
        lambda m: f"<title>{m.group(1)}</title>" if SITE_NAME in m.group(1) else m.group(0), page
    )


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
    for path in rendered_pages():
        page = path.read_text(encoding="utf-8")
        tidy = demote_nav_title(page)
        if tidy != page:
            path.write_text(tidy, encoding="utf-8")
            changed += 1
    print(
        f"finish: the page title is the first heading ({changed} page(s) tidied)", file=sys.stderr
    )


if __name__ == "__main__":
    main()
