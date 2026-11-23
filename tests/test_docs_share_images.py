"""Share images resolve to one URL on every page of the docs site.

Quarto 1.7 resolves a share-image path without a leading "/" under the folder of each
page, and joins site-url and a "/..." path without adding or removing a slash.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
IMAGE = re.compile(r"^\s+image: (\S+)\s*$", re.MULTILINE)


def share_images(text: str) -> list[str]:
    """`image:` values under `open-graph:` and `twitter-card:` blocks."""
    found = []
    for block in re.split(r"^\s*(?=open-graph:|twitter-card:)", text, flags=re.MULTILINE)[1:]:
        body = block.split("\n", 1)[1]
        indented = re.match(r"((?:[ \t]+\S.*\n?)*)", body).group(1)
        found += IMAGE.findall(indented)
    return found


def test_share_images_start_at_the_site_root_and_exist() -> None:
    config = (DOCS / "_quarto.yml").read_text(encoding="utf-8")
    site_url = re.search(r"^\s+site-url: (\S+)$", config, re.MULTILINE).group(1)
    assert not site_url.endswith("/"), "a trailing slash doubles the slash in share URLs"
    pages = [DOCS / "_quarto.yml", *DOCS.rglob("*.qmd")]
    images = {
        (page.relative_to(DOCS).as_posix(), image)
        for page in pages
        for image in share_images(page.read_text(encoding="utf-8").split("\n---", 1)[0])
    }
    assert ("_quarto.yml", "/assets/img/social/mfm-social-claim.jpg") in images
    for page, image in sorted(images):
        assert image.startswith("/"), f"{page}: share image {image} must start with /"
        assert (DOCS / image.lstrip("/")).is_file(), f"{page}: {image} does not exist"
