"""The example notebook runs top to bottom without a notebook kernel."""

import json
import sys
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "diagnostics.ipynb"


@pytest.mark.parametrize("corpus", ["generated", "synthetic"])
def test_diagnostics_notebook_code_cells_run_headless(corpus, tmp_path, monkeypatch):
    # Matplotlib is the only optional dependency of the notebook's path; it ships
    # with the diagnostics and training extras. The generator extra is optional.
    pytest.importorskip("matplotlib")
    if corpus == "generated":
        pytest.importorskip("pymc_generator")
    else:
        # The notebook then builds its own synthetic corpus.
        monkeypatch.setitem(sys.modules, "pymc_generator", None)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{NOTEBOOK.name}:cell-{number}", "exec"), namespace)
    assert namespace["source"].startswith("synthetic") is (corpus == "synthetic")
    manifest = namespace["manifest"]
    assert manifest["status"] == "complete"
    rendered = {
        name for name, panel in manifest["panels"].items() if panel.get("render_status") == "ok"
    }
    assert {"attribution", "window_totals", "cumulative", "temporal", "reliability"} <= rendered
