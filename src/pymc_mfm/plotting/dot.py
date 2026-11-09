"""Literal, prediction-only Graphviz DOT export without a Graphviz dependency."""

from ._graphs import _graph_nodes, select_graph


class _DotFormatError(ValueError):
    """A node label contains a character outside XML 1.0's Char production."""


_LABEL_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r"}


def _quoted_label(label):
    """Quote literal text, preserving Graphviz escapes rather than interpreting them."""
    escaped = ['"']
    for character in label:
        codepoint = ord(character)
        if not (
            codepoint in (0x9, 0xA, 0xD)
            or 0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        ):
            raise _DotFormatError(
                "DOT export requires XML 1.0-compatible node labels; "
                f"invalid character U+{codepoint:04X}"
            )
        escaped.append(_LABEL_ESCAPES.get(character, character))
    escaped.append('"')
    return "".join(escaped)


def dot_template(summary):
    """Return a linear-size export template aligned with active edge records.

    The preamble ends with LF; fragments contain no terminal LF; the suffix is
    ``"\\n}\\n"``. Prepend the view comment, then concatenate the preamble,
    nonzero fragments joined by LF, and suffix. Labels are escaped here once,
    never by a browser or per complete-graph rank.
    """
    edges = summary["arrays"]["graphs"]["edges"]
    nodes = {name: f"n{index}" for index, name in enumerate(_graph_nodes(edges))}
    preamble = [
        "// Prediction-only routing topology; supplied graph truth is not exported.",
        "// Average shows marginal edge draw frequencies, not one complete sampled graph.",
        "// Complete graph ranks show binary edges, ordered by count then binary vector.",
        "// Frequencies are empirical router-draw frequencies, not fit or causal scores.",
        "// MAP/fixed modes describe selected/supplied hypotheses, not posterior mass.",
        "digraph {",
        '  graph [rankdir=TB, charset="UTF-8"];',
        '  node [shape=ellipse, color=black, fontcolor=black, fontname="Times-Roman"];',
        '  edge [color=black, fontcolor=black, fontname="Times-Roman"];',
    ]
    for name, identifier in nodes.items():
        preamble.append(f"  {identifier} [label={_quoted_label(name)}];")
    average = []
    individual = []
    for entry in edges:
        source, target = (nodes[name] for name in entry["endpoints"])
        statement = f"  {source} -> {target}"
        probability = float(entry["probability"])
        average.append(f'{statement} [label="{probability:.3g}", probability="{probability!r}"];')
        individual.append(f"{statement};")
    return {
        "preamble": "\n".join(preamble) + "\n",
        "average": average,
        "individual": individual,
        "suffix": "\n}\n",
    }


def graph_to_dot(summary, *, graph=None) -> str:
    """Export Average or a complete graph at its zero-based frequency rank.

    Average includes nonzero marginal edge frequencies as numeric labels and
    exact ``probability`` attributes. A ranked view includes only its binary
    edges, without frequency labels. Every active endpoint node remains, even
    for an empty selection; padding and supplied graph truth are not exported.
    No Graphviz installation is needed. Labels must be valid XML 1.0 text so a
    downstream Graphviz SVG renderer can preserve them without replacement.
    The returned source uses LF separators and one trailing LF.
    """
    selected = select_graph(summary, graph=graph)
    template = dot_template(summary)
    if selected is None:
        view = "// Average\n"
        fragments = template["average"]
        values = (entry["probability"] for entry in summary["arrays"]["graphs"]["edges"])
    else:
        view = f"// Complete graph rank {graph}\n"
        fragments = template["individual"]
        values = selected["graph"]
    body = "\n".join(fragment for fragment, value in zip(fragments, values, strict=True) if value)
    return view + template["preamble"] + body + template["suffix"]
