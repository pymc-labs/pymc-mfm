"""docs/_scripts/mount.py lays out the site with the landing page at / and the rest under /docs/."""

from __future__ import annotations

import sys
from pathlib import Path

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


@pytest.fixture
def site(tmp_path: Path) -> Path:
    built = tmp_path / "_site"
    (built / "guide").mkdir(parents=True)
    (built / "index.html").write_text(LANDING, encoding="utf-8")
    (built / "guide" / "index.html").write_text("<p>guide</p>", encoding="utf-8")
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

    assert (out / "docs" / "guide" / "index.html").read_text(encoding="utf-8") == "<p>guide</p>"
    redirect = (out / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'http-equiv="refresh" content="0; url=/"' in redirect
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
