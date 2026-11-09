"""Generate the site's derived pages before Quarto renders (``project.pre-render``).

1. Copies the notebooks in ``examples/`` (top level only; ``examples/dev`` is not
   published) into ``docs/guide/examples/``, so the Guide renders them with their
   stored outputs. The copies are ignored by Git; edit the originals.
2. Writes the API reference into ``docs/reference/`` from the package docstrings,
   using the ``quartodoc`` section of ``_quarto.yml``.

Quarto runs this from ``docs/``. Run the build inside the project environment so
quartodoc is importable: ``uv run --group docs quarto render docs``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
EXAMPLES = DOCS.parent / "examples"
GUIDE_EXAMPLES = DOCS / "guide" / "examples"


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


def main() -> None:
    notebooks = sync_examples()
    print(f"prepare: {len(notebooks)} example notebook(s) -> guide/examples/", file=sys.stderr)
    build_reference()
    print("prepare: API reference -> reference/", file=sys.stderr)


if __name__ == "__main__":
    main()
