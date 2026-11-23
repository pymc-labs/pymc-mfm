"""Generate the site's derived pages before Quarto renders (``project.pre-render``).

1. Copies the notebooks in ``examples/`` (top level only; ``examples/dev`` is not
   published) into ``docs/guide/examples/``, so the Guide renders them with their
   stored outputs. The copies are ignored by Git; edit the originals.
2. Writes the Python reference into ``docs/reference/`` from the package docstrings,
   using the ``quartodoc`` section of ``_quarto.yml``, gives each page its name as the
   browser tab title, and gives its sidebar the "Python library" entry first.
3. Writes the HTTP API pages into ``docs/api/`` from ``skills/mfm/reference/http.md``
   (see ``http_pages.py``).

The pilot site (``quarto render docs --profile pilot``, ``_quarto-pilot.yml``) carries
only the client: there this script writes only the "River API client" group of the
reference, gives the attribute links of the class summaries a target, joins the pages of
all classes and functions into the one page ``reference/index.qmd``, and writes an empty
HTTP API sidebar instead of the HTTP API pages. The pilot sidebar is in
``_quarto-pilot.yml``. ``pilot_post.py`` writes a redirect page for each old per-object
address. It copies no notebooks.

Quarto runs this from ``docs/``. Run the build inside the project environment so
quartodoc is importable: ``uv run --group docs quarto render docs``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import http_pages

DOCS = Path(__file__).resolve().parent.parent
EXAMPLES = DOCS.parent / "examples"
GUIDE_EXAMPLES = DOCS / "guide" / "examples"
REFERENCE = DOCS / "reference"
API_SIDEBAR = DOCS / "api" / "_sidebar.yml"
# The quartodoc group that the pilot site keeps.
PILOT_SECTION = "River API client"
# Applies to every generated reference page: the reference sidebar and the API styles.
REFERENCE_METADATA = """\
sidebar: reference
body-classes: mfm-ref
format:
  html:
    css: ../assets/styles/api.css
"""


def sync_examples() -> list[str]:
    GUIDE_EXAMPLES.mkdir(parents=True, exist_ok=True)
    sources = {path.name: path for path in sorted(EXAMPLES.glob("*.ipynb"))}
    for stale in GUIDE_EXAMPLES.glob("*.ipynb"):
        if stale.name not in sources:
            stale.unlink()
    for name, source in sources.items():
        target = GUIDE_EXAMPLES / name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
    return sorted(sources)


def pilot() -> bool:
    """True when Quarto renders the pilot profile (it sets ``QUARTO_PROFILE``)."""
    return "pilot" in os.environ.get("QUARTO_PROFILE", "").split(",")


def build_reference(*, client_only: bool = False) -> None:
    try:
        from quartodoc import Builder
    except ModuleNotFoundError as error:
        raise SystemExit(
            "quartodoc is missing. Build the site in the project environment: "
            "uv run --group docs quarto render docs"
        ) from error
    import yaml

    config = yaml.safe_load((DOCS / "_quarto.yml").read_text(encoding="utf-8"))
    if client_only:
        sections = config["quartodoc"]["sections"]
        config["quartodoc"]["sections"] = [s for s in sections if s["title"] == PILOT_SECTION]
        if not config["quartodoc"]["sections"]:
            raise SystemExit(f"no quartodoc section titled {PILOT_SECTION!r} in _quarto.yml")
        # Pages of an earlier full build would stay in reference/ otherwise.
        for page in REFERENCE.glob("*.qmd"):
            page.unlink()
    Builder.from_quarto_config(config).build()


# quartodoc links each attribute in a class summary to "#<qualified name>" but writes no
# section with that id for attributes, so the link goes nowhere.
SUMMARY_LINK = re.compile(r"\[([^\]]+)\]\(#([\w.]+)\)")
SECTION_ID = re.compile(r"\{ #([\w.]+) \}")


def anchor_summary_links() -> None:
    """Turn summary links that have no target into the target: ``[name]{#id}``.

    The first link to a missing id becomes a span with that id, so other links to it land
    on the summary row. Used by the pilot build only.
    """
    for page in sorted(REFERENCE.glob("*.qmd")):
        text = page.read_text(encoding="utf-8")
        ids = set(SECTION_ID.findall(text))

        def repl(match: re.Match[str]) -> str:
            name, target = match.groups()
            if target in ids:
                return match.group(0)
            ids.add(target)
            return f"[{name}]{{#{target}}}"

        fixed = SUMMARY_LINK.sub(repl, text)
        if fixed != text:
            page.write_text(fixed, encoding="utf-8")


# quartodoc writes no front matter, so Quarto titles a page after its file ("index",
# "client.client"). Each page starts with its heading, "# name { #id }".
PAGE_HEADING = re.compile(r"# (.+?)(?: \{.*\})?")


def add_page_titles() -> None:
    """Give each reference page its heading as the tab title (``pagetitle``).

    ``title`` would draw a second heading above the page's own.
    """
    for page in sorted(REFERENCE.glob("*.qmd")):
        text = page.read_text(encoding="utf-8")
        first = text.split("\n", 1)[0]
        match = PAGE_HEADING.fullmatch(first)
        if text.startswith("---\n") or not match:
            raise SystemExit(f"{page.name}: expected the first line to be a heading, got {first!r}")
        page.write_text(f"---\npagetitle: {json.dumps(match[1])}\n---\n\n{text}", encoding="utf-8")


def arrange_reference_sidebar(first: str = "Python library") -> None:
    """Give the quartodoc sidebar an id and put the index first.

    quartodoc writes the index page first, then one section for each group. The index goes
    under a section called ``first``.
    """
    import yaml

    path = REFERENCE / "_sidebar.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = config["website"]["sidebar"][0]
    index, *groups = entry["contents"]
    if index != "reference/index.qmd":
        raise SystemExit(
            f"quartodoc wrote {index!r} first in the sidebar; expected reference/index.qmd"
        )
    entry["id"] = "reference"
    entry["style"] = "floating"
    entry["contents"] = [{"section": first, "contents": [index]}, *groups]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (REFERENCE / "_metadata.yml").write_text(REFERENCE_METADATA, encoding="utf-8")


# The pilot joins the page of each class and function into reference/index.qmd. The page name
# stays the one h1 and each class or function becomes an h2. Its methods stay h3, so the
# contents list ("On this page", levels 2 and 3) names each class with its methods under it.
# The "Attributes" and "Methods" labels become plain labels (.mfm-ref-label), not headings:
# they would repeat in that list.
SUMMARY_ROW = re.compile(r"\]\(client\.[\w.]+\.qmd#([\w.]+)\)")
HEADING = re.compile(r"(#{1,6}) (.*)")
PILOT_PAGE_METADATA = """\
---
pagetitle: "Python reference"
# Old per-object pages; pilot_post.py writes a redirect page to the anchor of each.
mfm-old-pages: {old_pages}
---

"""


# The top of the one page: how to install and import the client, before the summary table.
PILOT_REFERENCE_START = """\
```bash
pip install pymc-mfm
```

```python
from pymc_mfm.client import Client, amount, curve, grid, scenario
```

The client needs Python 3.11 or newer. A first run is in [Get started](../guide/index.qmd)."""


def demote_headings(text: str) -> str:
    """Make the h1 of an object page an h2, and its h2 labels plain labels."""
    out: list[str] = []
    fenced = False
    for line in text.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        match = None if fenced else HEADING.fullmatch(line)
        if match:
            marks, title = match.groups()
            if len(marks) == 1:
                line = f"## {title}"
            elif len(marks) == 2:
                line = f"[{title}]{{.mfm-ref-label}}"
        out.append(line)
    return "\n".join(out)


def join_reference_pages() -> list[str]:
    """Write every object page of ``reference/`` into ``index.qmd``; return their names.

    The summary table of the old index stays at the top and links to anchors on the page.
    Each object page is removed, so Quarto renders one page.
    """
    index = (REFERENCE / "index.qmd").read_text(encoding="utf-8")
    heading, _, rest = index.partition("\n\n")
    if not heading.startswith("# "):
        raise SystemExit(f"index.qmd: expected a heading first, got {heading!r}")
    # The group heading ("River API client") is dropped: the page is the one group.
    rest = re.sub(r"(?m)^## .*\n\n", "", rest, count=1)
    summary = SUMMARY_ROW.sub(r"](#\1)", rest).strip()
    names = re.findall(r"\]\(#pymc_mfm\.(client\.[\w.]+)\)", summary)
    if not names:
        raise SystemExit("index.qmd: no object rows in the summary table")
    lead, _, table = summary.partition("\n\n")
    parts = [heading, lead, PILOT_REFERENCE_START, table]
    for name in names:
        page = REFERENCE / f"{name}.qmd"
        body = page.read_text(encoding="utf-8")
        if body.startswith("---\n"):
            body = body.split("\n---\n", 1)[1]
        parts.append(demote_headings(body.strip()))
        page.unlink()
    front = PILOT_PAGE_METADATA.format(old_pages=json.dumps(names))
    (REFERENCE / "index.qmd").write_text(front + "\n\n".join(parts) + "\n", encoding="utf-8")
    return names


def write_pilot_reference_metadata() -> None:
    """The reference page uses the guide sidebar of the pilot profile; quartodoc's is empty."""
    (REFERENCE / "_sidebar.yml").write_text("{}\n", encoding="utf-8")
    (REFERENCE / "_metadata.yml").write_text(
        REFERENCE_METADATA.replace("sidebar: reference", "sidebar: guide").replace(
            "body-classes: mfm-ref", "body-classes: mfm-ref mfm-ref-one"
        ),
        encoding="utf-8",
    )


def main() -> None:
    if pilot():
        build_reference(client_only=True)
        anchor_summary_links()
        names = join_reference_pages()
        write_pilot_reference_metadata()
        print(
            f"prepare: pilot: {len(names)} client objects -> reference/index.qmd", file=sys.stderr
        )
        # _quarto.yml names this file in metadata-files; the pilot has no HTTP API pages.
        API_SIDEBAR.parent.mkdir(parents=True, exist_ok=True)
        API_SIDEBAR.write_text("{}\n", encoding="utf-8")
        print("prepare: pilot: no HTTP API pages, no notebooks", file=sys.stderr)
        return
    notebooks = sync_examples()
    print(f"prepare: {len(notebooks)} example notebook(s) -> guide/examples/", file=sys.stderr)
    build_reference()
    add_page_titles()
    arrange_reference_sidebar()
    print("prepare: Python reference -> reference/", file=sys.stderr)
    pages = http_pages.generate()
    print(f"prepare: {len(pages)} HTTP API file(s) -> api/", file=sys.stderr)


if __name__ == "__main__":
    main()
