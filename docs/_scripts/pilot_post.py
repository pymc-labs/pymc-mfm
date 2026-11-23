"""Finish the pilot site after Quarto renders it (``project.post-render`` of the pilot).

A page whose full-site ``description`` names parts that the pilot leaves out sets
``pilot-description`` in its front matter. This script puts that text in place of the
description in the built page: the meta, Open Graph and X tags and the subtitle.

The Python reference is one page, ``reference/index.html``. For each old per-object address
(``reference/client.X.html``, listed in the ``mfm-old-pages`` key of ``reference/index.qmd``)
this script writes a small redirect page to the matching anchor. Links from the guide pages go
to the anchor directly (``_scripts/pilot-links.lua``).

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


REDIRECT = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Python reference \u2013 PyMC MFM</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="index.html#{anchor}">
<meta http-equiv="refresh" content="0; url=index.html#{anchor}">
<script>location.replace("index.html" + (location.hash || "#{anchor}"));</script>
</head>
<body>
<p>This page moved to <a href="index.html#{anchor}">the Python reference</a>.</p>
</body>
</html>
"""


def old_reference_pages() -> list[str]:
    return front_matter(DOCS / "reference" / "index.qmd").get("mfm-old-pages", [])


def write_redirects(site: Path, names: list[str]) -> None:
    for name in names:
        page = site / "reference" / f"{name}.html"
        page.write_text(REDIRECT.format(anchor=f"pymc_mfm.{name}"), encoding="utf-8")


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

    names = old_reference_pages()
    write_redirects(site, names)
    print(f"pilot_post: {len(names)} redirect page(s) in reference/", file=sys.stderr)
    print(f"pilot_post: {fix_console_links(site)} link(s) to /console/", file=sys.stderr)


if __name__ == "__main__":
    main()
