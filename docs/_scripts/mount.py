"""Lay out the built site for a host that serves the landing page at ``/`` and the site
under ``/<prefix>/`` (by default ``/docs/``), next to the Console at ``/console/``.

    python docs/_scripts/mount.py docs/_site OUT [--prefix docs] [--noindex]
        [--site-url https://HOST]

``OUT/<prefix>/`` holds the built site, with two pages replaced:

- ``OUT/index.html`` is the landing page with ``<base href="/<prefix>/">`` first in its
  ``<head>``, so every relative link and asset on it resolves under ``/<prefix>/``.
- ``OUT/<prefix>/index.html`` sends the browser to ``/``, so the logo on every other page
  leads to the one landing page.

``OUT/robots.txt`` is the host's crawler policy, the same as pymc-labs.com: search
engines may index every page, AI training crawlers may not. ``--noindex`` (staging hosts)
instead asks every crawler to stay away and marks the landing page noindex.

``--site-url`` is the host's public origin. Quarto writes its own ``site-url``
(``docs/_quarto.yml``) into the built pages: in share-image addresses, and in the script
that opens links to other sites in a new tab. With ``--site-url`` every page names the
served address ``<origin>/<prefix>`` instead, so the site's own links open in the same tab
and link previews find their image. The repository never names the host.

With ``--site-url`` and without ``--noindex``, ``robots.txt`` names the sitemap and
``OUT/sitemap.xml`` lists each page at its served address (the landing page at ``/``).
The sitemap is at the host root so it covers both ``/`` and pages under ``/<prefix>/``.
Quarto's original sitemap uses its own layout. Each page gets a ``<link rel="canonical">``
with its served address unless it already declares one (for example, an old blog address).
``--noindex`` turns both off: a staging host is not for search engines.

Upload ``OUT`` to the root of the host's site bucket. The input is the site Quarto renders;
beyond the addresses above, nothing here changes the page content.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

CHARSET = '<meta charset="utf-8">'
REDIRECT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0; url=/">
<link rel="canonical" href="/">
<title>PyMC MFM</title>
</head>
<body>
<p><a href="/">PyMC MFM</a></p>
</body>
</html>
"""


# The pymc-labs.com policy (EU DSM Directive 2019/790, Art. 4): search yes, AI training no.
AI_TRAINING_CRAWLERS = (
    "Amazonbot",
    "Applebot-Extended",
    "Bytespider",
    "CCBot",
    "ClaudeBot",
    "Google-Extended",
    "GPTBot",
    "meta-externalagent",
)
ROBOTS = (
    "# PyMC MFM, by PyMC Labs: the crawler policy of https://www.pymc-labs.com/robots.txt.\n"
    "\n"
    "# Content signals (EU DSM Directive 2019/790, Art. 4):\n"
    "# search=yes: standard search indexing is permitted\n"
    "# ai-train=no: use for AI/ML training is expressly reserved\n"
    "User-agent: *\n"
    "Content-Signal: search=yes,ai-train=no\n"
    "Allow: /\n"
    "\n"
    "# AI training crawlers: disallow (ai-train=no reservation)\n"
    + "\n".join(f"User-agent: {agent}\nDisallow: /\n" for agent in AI_TRAINING_CRAWLERS)
)
ROBOTS_NOINDEX = "# A staging host: no crawler may index it.\nUser-agent: *\nDisallow: /\n"
LOC = re.compile(r"<loc>([^<]+)</loc>")


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


def with_canonical(page_html: str, url: str, name: str) -> str:
    """The page with a ``<link rel="canonical">`` for ``url``, after the charset declaration.

    A page that already names a canonical address is returned unchanged.
    """
    if 'rel="canonical"' in page_html:
        return page_html
    if page_html.count(CHARSET) != 1:
        raise MountError(f"expected one {CHARSET} in {name}")
    link = f'<link rel="canonical" href="{html.escape(url)}">'
    return page_html.replace(CHARSET, CHARSET + "\n" + link, 1)


def robots(noindex: bool, sitemap: str | None) -> str:
    """The host's ``robots.txt``: closed on staging, otherwise the pymc-labs.com policy."""
    if noindex:
        return ROBOTS_NOINDEX
    if sitemap is None:
        return ROBOTS
    return ROBOTS + f"\nSitemap: {sitemap}\n"


def split_loc(loc: str, site: Path) -> tuple[str, str]:
    """Quarto's ``site-url`` and the page path of one sitemap entry.

    The page's path is the longest tail of the URL that names a file in the built site.
    """
    parts = loc.strip().split("/")
    # parts[:3] is the scheme and host ("https:", "", "host").
    for start in range(3, len(parts)):
        page = "/".join(parts[start:])
        if page and (site / page).is_file():
            return "/".join(parts[:start]), page
    raise MountError(f"sitemap entry {loc.strip()} is not a page of the built site")


def served_sitemap(sitemap_xml: str, site: Path, site_url: str, prefix: str) -> str:
    """Quarto's sitemap with each page at the address this layout serves it from.

    Each page moves under ``<site_url>/<prefix>/``, and the landing page (``index.html``)
    to ``<site_url>/``.
    """

    def move(match: re.Match[str]) -> str:
        _, page = split_loc(match.group(1), site)
        if page == "index.html":
            return f"<loc>{site_url}/</loc>"
        return f"<loc>{site_url}/{prefix}/{page}</loc>"

    return LOC.sub(move, sitemap_xml)


def built_site_url(sitemap_xml: str, site: Path) -> str:
    """The ``site-url`` the site was built with, read from its first sitemap entry."""
    loc = LOC.search(sitemap_xml)
    if loc is None:
        raise MountError("the built sitemap lists no pages")
    return split_loc(loc.group(1), site)[0]


def rebased(text: str, built: str, served: str) -> str:
    """``text`` with the built address replaced by the served one, also in the escaped form
    Quarto's link script matches the site's own links with."""

    def escaped(url: str) -> str:
        return url.replace(".", "\\.").replace("/", "\\/")

    return text.replace(built, served).replace(escaped(built), escaped(served))


def mount(
    site: Path,
    out: Path,
    prefix: str = "docs",
    noindex: bool = False,
    site_url: str | None = None,
) -> None:
    landing = site / "index.html"
    if not landing.is_file():
        raise MountError(f"{landing} is missing; render the site first")
    if out.exists() and any(out.iterdir()):
        raise MountError(f"{out} is not empty")
    prefix = prefix.strip("/")
    if site_url is not None:
        site_url = site_url.rstrip("/")
        parts = urlsplit(site_url)
        origin = parts.scheme in ("https", "http") and parts.netloc
        if not origin or parts.path or parts.query or parts.fragment:
            raise MountError(f"--site-url {site_url} is not an origin such as https://example.com")
    # A staging host is not for search engines: no canonical addresses and no sitemap.
    public = site_url is not None and not noindex
    shutil.copytree(site, out / prefix, dirs_exist_ok=True)
    landing_html = landing.read_text(encoding="utf-8")
    built_sitemap = site / "sitemap.xml"
    if site_url is not None:
        if not built_sitemap.is_file():
            raise MountError(f"{built_sitemap} is missing; set site-url in _quarto.yml")
        built = built_site_url(built_sitemap.read_text(encoding="utf-8"), site)
        served_url = f"{site_url}/{prefix}"
        landing_html = rebased(landing_html, built, served_url)
        for page in sorted((out / prefix).rglob("*")):
            if page.suffix not in (".html", ".xml") or not page.is_file():
                continue
            text = page.read_text(encoding="utf-8")
            moved = rebased(text, built, served_url)
            if moved != text:
                page.write_text(moved, encoding="utf-8")
    if public:
        landing_html = with_canonical(landing_html, f"{site_url}/", "the landing page")
        for page in sorted((out / prefix).rglob("*.html")):
            name = page.relative_to(out / prefix).as_posix()
            page_html = page.read_text(encoding="utf-8")
            served = with_canonical(page_html, f"{site_url}/{prefix}/{name}", name)
            if served != page_html:
                page.write_text(served, encoding="utf-8")
    root_page = landing_at_root(landing_html, prefix, noindex)
    (out / "index.html").write_text(root_page, encoding="utf-8")
    (out / prefix / "index.html").write_text(REDIRECT, encoding="utf-8")

    # Quarto's crawler files point to the unmapped site URLs.
    (out / prefix / "robots.txt").unlink(missing_ok=True)
    (out / prefix / "sitemap.xml").unlink(missing_ok=True)
    sitemap = None
    if public:
        moved = served_sitemap(built_sitemap.read_text(encoding="utf-8"), site, site_url, prefix)
        (out / "sitemap.xml").write_text(moved, encoding="utf-8")
        sitemap = f"{site_url}/sitemap.xml"
    (out / "robots.txt").write_text(robots(noindex, sitemap), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", type=Path, help="the rendered site (docs/_site)")
    parser.add_argument("out", type=Path, help="an empty or new output folder")
    parser.add_argument(
        "--prefix", default="docs", help="the site folder on the host (default: docs)"
    )
    parser.add_argument(
        "--noindex",
        action="store_true",
        help="a staging host: robots.txt closes it and the landing page is noindex",
    )
    parser.add_argument(
        "--site-url",
        help="the host's public origin, such as https://example.com; adds the sitemap and "
        "canonical links",
    )
    args = parser.parse_args(argv)
    try:
        mount(args.site, args.out, args.prefix, args.noindex, args.site_url)
    except MountError as error:
        sys.exit(f"mount: {error}")
    print(
        f"mount: / is the landing page; the site is under /{args.prefix.strip('/')}/",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
