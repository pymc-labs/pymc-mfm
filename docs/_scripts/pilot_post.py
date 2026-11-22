"""Finish the pilot site after Quarto renders it (``project.post-render`` of the pilot).

A page whose full-site ``description`` names parts that the pilot leaves out sets
``pilot-description`` in its front matter. This script puts that text in place of the
description in the built page: the meta, Open Graph and X tags and the subtitle.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent.parent


def front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    return yaml.safe_load(text[4 : text.index("\n---", 4)]) or {}


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


if __name__ == "__main__":
    main()
