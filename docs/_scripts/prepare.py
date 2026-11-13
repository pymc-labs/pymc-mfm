"""Generate the site's derived pages before Quarto renders (``project.pre-render``).

1. Copies the notebooks in ``examples/`` (top level only; ``examples/dev`` is not
   published) into ``docs/guide/examples/``, so the Guide renders them with their
   stored outputs. The copies are ignored by Git; edit the originals.
2. Writes the Python reference into ``docs/reference/`` from the package docstrings,
   using the ``quartodoc`` section of ``_quarto.yml``, and gives its sidebar the
   "Python library" entry first.
3. Writes the HTTP API pages into ``docs/api/`` from ``skills/mfm/reference/http.md``
   (see ``http_pages.py``).

Quarto runs this from ``docs/``. Run the build inside the project environment so
quartodoc is importable: ``uv run --group docs quarto render docs``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import http_pages

DOCS = Path(__file__).resolve().parent.parent
EXAMPLES = DOCS.parent / "examples"
GUIDE_EXAMPLES = DOCS / "guide" / "examples"
REFERENCE = DOCS / "reference"
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


def build_reference() -> None:
    try:
        from quartodoc import Builder
    except ModuleNotFoundError as error:
        raise SystemExit(
            "quartodoc is missing. Build the site in the project environment: "
            "uv run --group docs quarto render docs"
        ) from error
    Builder.from_quarto_config(str(DOCS / "_quarto.yml")).build()


def arrange_reference_sidebar() -> None:
    """Give the quartodoc sidebar an id and put "Python library" first.

    quartodoc writes the index page first, then one section for each group.
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
    entry["contents"] = [{"section": "Python library", "contents": [index]}, *groups]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (REFERENCE / "_metadata.yml").write_text(REFERENCE_METADATA, encoding="utf-8")


def main() -> None:
    notebooks = sync_examples()
    print(f"prepare: {len(notebooks)} example notebook(s) -> guide/examples/", file=sys.stderr)
    build_reference()
    arrange_reference_sidebar()
    print("prepare: Python reference -> reference/", file=sys.stderr)
    pages = http_pages.generate()
    print(f"prepare: {len(pages)} HTTP API file(s) -> api/", file=sys.stderr)


if __name__ == "__main__":
    main()
