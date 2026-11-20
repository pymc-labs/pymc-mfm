"""The example notebooks run top to bottom without a notebook kernel."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOK = EXAMPLES / "dev" / "diagnostics.ipynb"
QUICKSTART = EXAMPLES / "river_1_quickstart.ipynb"


def run_code_cells(notebook: Path) -> dict:
    namespace = {"__name__": "__main__"}
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for number, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{notebook.name}:cell-{number}", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "corpus",
    [
        pytest.param("generated", marks=pytest.mark.slow),
        pytest.param("synthetic", marks=pytest.mark.slow),
    ],
)
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
    namespace = run_code_cells(NOTEBOOK)
    assert namespace["source"].startswith("synthetic") is (corpus == "synthetic")
    manifest = namespace["manifest"]
    assert manifest["status"] == "complete"
    rendered = {
        name for name, panel in manifest["panels"].items() if panel.get("render_status") == "ok"
    }
    assert {"attribution", "window_totals", "cumulative", "temporal", "reliability"} <= rendered


@pytest.mark.slow
def test_river_1_quickstart_runs_headless_with_a_stand_in_model(tmp_path, monkeypatch):
    # River-1 is in a private repository, so an untrained model with its control and
    # latent widths stands in: the data, context, inference, charts and response curves
    # still run. The data come from PyMC-Marketing's transformers, which the examples
    # extra installs.
    pytest.importorskip("matplotlib")
    pytest.importorskip("pymc_marketing")
    import pymc_mfm
    from pymc_mfm import MFM, ModelConfig

    requests = []

    def stand_in(repository, **options):
        requests.append((repository, options))
        torch.manual_seed(0)
        config = ModelConfig(
            d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=10, n_demand=1
        )
        return MFM(config).eval()

    monkeypatch.setattr(pymc_mfm, "from_pretrained", stand_in)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    namespace = run_code_cells(QUICKSTART)
    [(repository, options)] = requests
    # The published profile, pinned to a full immutable commit.
    assert (repository, options["profile"]) == ("pymc-labs/mfm", "river-1")
    assert re.fullmatch(r"[0-9a-f]{40}", options["revision"])
    n_weeks, context = namespace["N_WEEKS"], namespace["CONTEXT"]
    assert context == int(0.8 * n_weeks)
    model = namespace["model"]
    # The first chart of the context weeks sampled them once, with the inference's draws.
    history = model._context_decomposition
    for result, weeks in ((history, context), (namespace["query"], n_weeks - context)):
        assert result.components.shape[:2] == (1_000, weeks)
        assert np.isfinite(result.components).all()
    curves = namespace["curves"]
    assert curves.draws.shape == (2, 41, 1_000) and curves.weeks == 8
    assert model._response_curves is curves
