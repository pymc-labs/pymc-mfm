"""Lay out the built site for a host that serves the landing page at ``/`` and the site
under ``/<prefix>/`` (by default ``/docs/``), next to the Console at ``/console/``.

    python docs/_scripts/mount.py docs/_site OUT [--prefix docs] [--noindex]

``OUT/<prefix>/`` holds the built site unchanged, except two pages:

- ``OUT/index.html`` is the landing page with ``<base href="/<prefix>/">`` first in its
  ``<head>``, so every relative link and asset on it resolves under ``/<prefix>/``.
- ``OUT/<prefix>/index.html`` sends the browser to ``/``, so the logo on every other page
  leads to the one landing page.

``--noindex`` asks search engines to skip the landing page (staging hosts). Upload ``OUT``
to the root of the host's site bucket. The input is the site Quarto renders; nothing here
changes the page content.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CHARSET = '<meta charset="utf-8">'
REDIRECT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0; url=/">
<link rel="canonical" href="/">
<title>PyMC-MKTFM</title>
</head>
<body>
<p><a href="/">PyMC-MKTFM</a></p>
</body>
</html>
"""


class MountError(Exception):
    """The built site does not have the shape this layout needs."""


def landing_at_root(landing_html: str, prefix: str, noindex: bool) -> str:
    """The landing page, with a base that points its relative URLs at ``/<prefix>/``.

    The base goes right after the charset declaration, before any element with a URL.
    """
    if landing_html.count(CHARSET) != 1:
        raise MountError(f"expected one {CHARSET} in the landing page")
    head = [f'<base href="/{prefix}/">']
    if noindex:
        head.append('<meta name="robots" content="noindex">')
    return landing_html.replace(CHARSET, CHARSET + "\n" + "\n".join(head), 1)


def mount(site: Path, out: Path, prefix: str = "docs", noindex: bool = False) -> None:
    landing = site / "index.html"
    if not landing.is_file():
        raise MountError(f"{landing} is missing; render the site first")
    if out.exists() and any(out.iterdir()):
        raise MountError(f"{out} is not empty")
    prefix = prefix.strip("/")
    shutil.copytree(site, out / prefix, dirs_exist_ok=True)
    root_page = landing_at_root(landing.read_text(encoding="utf-8"), prefix, noindex)
    (out / "index.html").write_text(root_page, encoding="utf-8")
    (out / prefix / "index.html").write_text(REDIRECT, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", type=Path, help="the rendered site (docs/_site)")
    parser.add_argument("out", type=Path, help="an empty or new output folder")
    parser.add_argument(
        "--prefix", default="docs", help="the site folder on the host (default: docs)"
    )
    parser.add_argument(
        "--noindex", action="store_true", help="ask search engines to skip the landing page"
    )
    args = parser.parse_args(argv)
    try:
        mount(args.site, args.out, args.prefix, args.noindex)
    except MountError as error:
        sys.exit(f"mount: {error}")
    print(
        f"mount: / is the landing page; the site is under /{args.prefix.strip('/')}/",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
