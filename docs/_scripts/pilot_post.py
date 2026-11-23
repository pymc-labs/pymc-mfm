"""Finish the pilot site after Quarto renders it (``project.post-render`` of the pilot).

A page whose full-site ``description`` names parts that the pilot leaves out sets
``pilot-description`` in its front matter. This script puts that text in place of the
description in the built page: the meta, Open Graph and X tags and the subtitle.

It also turns the links to the console (``/console/`` in the sources) back into that path.
Quarto treats a path that starts with ``/`` as inside the site and writes it relative to the
page (``../console/``), which would point into ``/docs/``. The console is on the same host,
outside the docs folder.
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent.parent


def front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    return yaml.safe_load(text[4 : text.index("\n---", 4)]) or {}


CONSOLE_LINK = re.compile(r'href="(?:\./|(?:\.\./)+)console/"')


def fix_console_links(site: Path) -> int:
    count = 0
    for page in sorted(site.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        fixed, n = CONSOLE_LINK.subn('href="/console/"', text)
        if n:
            page.write_text(fixed, encoding="utf-8")
            count += n
    return count


def main() -> None:
    site = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", DOCS / "_site-pilot"))
    for source in sorted(DOCS.rglob("*.qmd")):
        if site in source.parents:
            continue
        meta = front_matter(source)
        if "pilot-description" not in meta:
            continue
        page = site / source.relative_to(DOCS).with_suffix(".html")
        if not page.is_file():
            continue
        text = page.read_text(encoding="utf-8")
        old = html.escape(meta["description"], quote=True)
        if old not in text:
            raise SystemExit(f"pilot_post: the description of {source.name} is not in {page}")
        text = text.replace(old, html.escape(meta["pilot-description"], quote=True))
        page.write_text(text, encoding="utf-8")
        print(f"pilot_post: pilot description -> {page.relative_to(site)}", file=sys.stderr)

    print(f"pilot_post: {fix_console_links(site)} link(s) to /console/", file=sys.stderr)


if __name__ == "__main__":
    main()
