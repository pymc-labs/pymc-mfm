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
reference, removes the other generated reference pages, gives the attribute links of
the class summaries a target, and writes an empty HTTP API sidebar instead of the HTTP API
pages. It copies no notebooks.

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


def arrange_reference_sidebar(first: str | None = "Python library") -> None:
    """Give the quartodoc sidebar an id and put the index first.

    quartodoc writes the index page first, then one section for each group. The index goes
    under a section called ``first``, or on its own when ``first`` is None.
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
    entry["contents"] = [{"section": first, "contents": [index]} if first else index, *groups]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (REFERENCE / "_metadata.yml").write_text(REFERENCE_METADATA, encoding="utf-8")


def main() -> None:
    if pilot():
        build_reference(client_only=True)
        add_page_titles()
        # The index page is titled "Python reference"; a section of that name would repeat it.
        arrange_reference_sidebar(None)
        anchor_summary_links()
        print("prepare: pilot: client reference -> reference/", file=sys.stderr)
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
