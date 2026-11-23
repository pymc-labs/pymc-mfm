"""The host serves the docs under /docs/, with the landing page also available at /."""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

DOCS_SCRIPTS = Path(__file__).resolve().parent.parent / "docs" / "_scripts"
sys.path.insert(0, str(DOCS_SCRIPTS))

import mount  # noqa: E402

LANDING = """<!DOCTYPE html>
<html lang="en"><head>

<meta charset="utf-8">
<script src="site_libs/quarto-nav/quarto-nav.js"></script>
<link href="site_libs/bootstrap/bootstrap.min.css" rel="stylesheet">
</head>
<body><a href="guide/index.html">Get started</a></body>
</html>
"""
PAGE = '<html><head>\n<meta charset="utf-8">\n<title>Install</title></head></html>'


@pytest.fixture
def site(tmp_path: Path) -> Path:
    built = tmp_path / "_site"
    built.mkdir()
    (built / "index.html").write_text(LANDING, encoding="utf-8")
    for name in ("guide/index.html", "reference/index.html", "guide/python-client.html"):
        page = built / name
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(PAGE, encoding="utf-8")
    return built


def test_landing_at_root_with_base_first_and_site_under_prefix(site: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    mount.mount(site, out, noindex=True)

    root = (out / "index.html").read_text(encoding="utf-8")
    head = root[: root.index("</head>")]
    base = head.index('<base href="/docs/">')
    assert head.index('<meta charset="utf-8">') < base < head.index("site_libs/")
    assert '<meta name="robots" content="noindex">' in head
    assert 'href="guide/index.html"' in root  # content unchanged; the base resolves it

    assert (out / "docs" / "guide" / "index.html").read_text(encoding="utf-8") == PAGE
    assert (out / "docs" / "index.html").read_text(encoding="utf-8") == root
    assert (site / "index.html").read_text(encoding="utf-8") == LANDING  # input untouched


def test_without_noindex_and_other_prefix(site: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    mount.mount(site, out, prefix="/site/")
    root = (out / "index.html").read_text(encoding="utf-8")
    assert '<base href="/site/">' in root and "noindex" not in root
    assert (out / "site" / "guide" / "index.html").is_file()


@pytest.mark.parametrize(
    ("landing", "message"),
    [(None, "missing"), ("<html><head></head></html>", "charset")],
)
def test_refuses_a_site_it_cannot_lay_out(
    tmp_path: Path, landing: str | None, message: str
) -> None:
    built = tmp_path / "_site"
    built.mkdir()
    if landing is not None:
        (built / "index.html").write_text(landing, encoding="utf-8")
    with pytest.raises(mount.MountError, match=message):
        mount.mount(built, tmp_path / "out")


def test_refuses_a_non_empty_output(site: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(mount.MountError, match="not empty"):
        mount.mount(site, out)


QUARTO_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.github.io/pymc-mfm/index.html</loc></url>
  <url><loc>https://example.github.io/pymc-mfm/guide/index.html</loc></url>
  <url><loc>https://example.github.io/pymc-mfm/reference/index.html</loc></url>
  <url><loc>https://example.github.io/pymc-mfm/guide/python-client.html</loc></url>
</urlset>
"""


def test_robots_follow_the_pymc_labs_policy(site: Path, tmp_path: Path) -> None:
    (site / "robots.txt").write_text("Sitemap: https://example.github.io/x\n", encoding="utf-8")
    out = tmp_path / "out"
    mount.mount(site, out)

    robots = (out / "robots.txt").read_text(encoding="utf-8")
    assert "User-agent: *\nContent-Signal: search=yes,ai-train=no\nAllow: /\n" in robots
    for agent in ("GPTBot", "CCBot", "ClaudeBot", "Google-Extended"):
        assert f"User-agent: {agent}\nDisallow: /\n" in robots
    assert "Sitemap:" not in robots  # no public origin given
    assert not (out / "docs" / "robots.txt").exists()  # Quarto's copy names the wrong host


def test_staging_robots_close_the_host(site: Path, tmp_path: Path) -> None:
    (site / "sitemap.xml").write_text(QUARTO_SITEMAP, encoding="utf-8")
    out = tmp_path / "out"
    mount.mount(site, out, noindex=True, site_url="https://staging.example.com")

    robots = (out / "robots.txt").read_text(encoding="utf-8")
    assert "User-agent: *\nDisallow: /\n" in robots
    assert "Allow" not in robots and "Sitemap" not in robots
    assert not (out / "sitemap.xml").exists()
    assert not (out / "docs" / "sitemap.xml").exists()


@pytest.mark.parametrize("prefix", ["docs", "manual"])
def test_site_url_serves_the_sitemap_at_the_mounted_addresses(
    site: Path, tmp_path: Path, prefix: str
) -> None:
    (site / "sitemap.xml").write_text(QUARTO_SITEMAP, encoding="utf-8")
    out = tmp_path / "out"
    mount.mount(site, out, prefix=prefix, site_url="https://mfm.example.com/")

    robots = (out / "robots.txt").read_text(encoding="utf-8")
    docs_url = f"https://mfm.example.com/{prefix}/"
    assert "User-agent: *\nContent-Signal: search=yes,ai-train=no\nAllow: /\n" in robots
    assert robots.endswith(f"\nSitemap: {docs_url}sitemap.xml\n")
    assert not (out / "sitemap.xml").exists()
    sitemap = ElementTree.parse(out / prefix / "sitemap.xml")
    assert [loc.text for loc in sitemap.findall(".//{*}loc")] == [
        docs_url,
        docs_url + "guide/",
        docs_url + "reference/",
        docs_url + "guide/python-client.html",
    ]


def test_site_url_refuses_a_sitemap_page_the_site_lacks(site: Path, tmp_path: Path) -> None:
    (site / "sitemap.xml").write_text(
        QUARTO_SITEMAP.replace("guide/index.html", "gone.html"), encoding="utf-8"
    )
    with pytest.raises(mount.MountError, match="gone.html"):
        mount.mount(site, tmp_path / "out", site_url="https://mfm.example.com")


MOVED = '<head><meta charset="utf-8"><link rel="canonical" href="https://example.org/blog"></head>'


@pytest.fixture
def pages(site: Path) -> Path:
    (site / "sitemap.xml").write_text(QUARTO_SITEMAP, encoding="utf-8")
    (site / "blog").mkdir()
    (site / "blog" / "index.html").write_text(MOVED, encoding="utf-8")
    return site


def test_site_url_names_each_page_at_its_served_address(pages: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    mount.mount(pages, out, site_url="https://example.com/")

    for name, path in (
        ("index.html", "/docs/"),
        ("docs/index.html", "/docs/"),
        ("docs/guide/index.html", "/docs/guide/"),
        ("docs/reference/index.html", "/docs/reference/"),
        ("docs/guide/python-client.html", "/docs/guide/python-client.html"),
    ):
        page = (out / name).read_text(encoding="utf-8")
        assert page.count('rel="canonical"') == 1
        assert f'<link rel="canonical" href="https://example.com{path}">' in page
    home = (out / "docs" / "index.html").read_text(encoding="utf-8")
    assert "Get started" in home and 'http-equiv="refresh"' not in home
    # Redirects to the blog keep their external canonical.
    assert (out / "docs" / "blog" / "index.html").read_text(encoding="utf-8") == MOVED
    assert (pages / "guide" / "index.html").read_text(encoding="utf-8") == PAGE  # input untouched


def test_no_canonical_on_staging_or_without_site_url(pages: Path, tmp_path: Path) -> None:
    mount.mount(pages, tmp_path / "staging", noindex=True, site_url="https://example.com")
    mount.mount(pages, tmp_path / "plain")
    for out in (tmp_path / "staging", tmp_path / "plain"):
        assert "canonical" not in (out / "index.html").read_text(encoding="utf-8")
        assert "canonical" not in (out / "docs" / "index.html").read_text(encoding="utf-8")
        assert (out / "docs" / "guide" / "index.html").read_text(encoding="utf-8") == PAGE


QUARTO_PAGE = (
    '<html><head>\n<meta charset="utf-8">\n'
    '<meta property="og:image" content="https://example.github.io/pymc-mfm/assets/share.jpg">\n'
    r'</head><body><script>new RegExp("https:\/\/example\.github\.io\/pymc-mfm\/")</script>'
    "</body></html>"
)


@pytest.mark.parametrize("noindex", [False, True])
def test_site_url_gives_the_pages_the_served_address(
    pages: Path, tmp_path: Path, noindex: bool
) -> None:
    (pages / "guide" / "index.html").write_text(QUARTO_PAGE, encoding="utf-8")
    out = tmp_path / "out"
    mount.mount(pages, out, noindex=noindex, site_url="https://mfm.example.com")

    guide = (out / "docs" / "guide" / "index.html").read_text(encoding="utf-8")
    assert 'content="https://mfm.example.com/docs/assets/share.jpg"' in guide
    assert r'RegExp("https:\/\/mfm\.example\.com\/docs\/")' in guide
    assert "github.io" not in guide


@pytest.mark.parametrize("site_url", ["example.com", "https://example.com/docs/"])
def test_refuses_a_site_url_that_is_not_an_origin(
    pages: Path, tmp_path: Path, site_url: str
) -> None:
    with pytest.raises(mount.MountError, match="not an origin"):
        mount.mount(pages, tmp_path / "out", site_url=site_url)


def test_refuses_a_page_without_a_charset_when_adding_canonicals(
    pages: Path, tmp_path: Path
) -> None:
    (pages / "guide" / "bare.html").write_text("<p>bare</p>", encoding="utf-8")
    with pytest.raises(mount.MountError, match="guide/bare.html"):
        mount.mount(pages, tmp_path / "out", site_url="https://example.com")
