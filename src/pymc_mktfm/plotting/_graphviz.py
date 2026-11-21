"""Native Graphviz artifacts, rendered by the system ``dot`` executable."""

from __future__ import annotations

import re
import subprocess
from os import PathLike
from pathlib import Path
from typing import BinaryIO
from xml.etree import ElementTree


def _run_dot(sources: list[str], *, format: str) -> bytes:
    """Render DOT documents in one process without overriding any Graphviz defaults."""
    try:
        result = subprocess.run(
            ["dot", f"-T{format}"],
            input="\n".join(sources).encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Native graph rendering requires Graphviz's 'dot' executable on PATH; "
            "install Graphviz to render PNG or SVG. DOT export does not require it."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Graphviz dot could not render {format}: {detail}") from exc
    if not result.stdout:
        raise ValueError("Graphviz dot produced no graph; source must contain a DOT graph")
    return result.stdout


def _render_svg_batch(sources: list[str]) -> list[str]:
    """Return complete native SVG documents in input order from one ``dot`` process.

    Graphviz writes one XML document per graph. Check both the document count and
    each SVG root so partial or extra output cannot silently select the wrong view.
    The documents retain Graphviz's original XML declaration, doctype and whitespace.
    """
    if not sources:
        return []
    try:
        output = _run_dot(sources, format="svg").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Graphviz dot returned invalid UTF-8 SVG output") from exc
    starts = [match.start() for match in re.finditer(r"(?m)^<\?xml\s", output)]
    if len(starts) != len(sources) or not starts or starts[0] != 0:
        raise RuntimeError(
            f"Graphviz dot returned an unexpected SVG document count; expected {len(sources)}"
        )
    documents = [
        output[start:end] for start, end in zip(starts, [*starts[1:], len(output)], strict=True)
    ]
    for document in documents:
        try:
            root = ElementTree.fromstring(document)
        except ElementTree.ParseError as exc:
            raise RuntimeError("Graphviz dot returned a malformed SVG document") from exc
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise RuntimeError("Graphviz dot returned a document without an SVG root")
    return documents


class GraphvizGraph:
    """Canonical DOT source with notebook PNG display and native PNG/SVG export.

    Rendering requires the system Graphviz ``dot`` executable, not a Python
    Graphviz package. All geometry, typography and styling use ``dot`` defaults.
    This artifact is not a Matplotlib Figure and accepts no Matplotlib options.
    """

    def __init__(self, source: str):
        if not isinstance(source, str):
            raise TypeError("source must be DOT text")
        self.source = source

    def render(self, format: str = "png") -> bytes:
        """Render PNG or vector SVG bytes using native Graphviz defaults."""
        format = format.lower()
        if format == "svg":
            return _render_svg_batch([self.source])[0].encode("utf-8")
        if format != "png":
            raise ValueError("native graph format must be 'png' or 'svg'")
        return _run_dot([self.source], format="png")

    def _repr_png_(self) -> bytes:
        """Display the graph as a native PNG cell result in a notebook."""
        return self.render()

    def savefig(
        self,
        path: str | PathLike[str] | BinaryIO,
        *,
        format: str | None = None,
    ) -> None:
        """Save native PNG/SVG bytes to a path or an open binary file.

        A path's suffix selects the format unless ``format`` is supplied. Paths
        without a suffix and binary files default to PNG. No suffix is appended.
        """
        is_path = isinstance(path, (str, PathLike))
        if format is None:
            format = (Path(path).suffix.removeprefix(".") if is_path else "") or "png"
        rendered = self.render(format=format)
        if is_path:
            Path(path).write_bytes(rendered)
        else:
            path.write(rendered)
