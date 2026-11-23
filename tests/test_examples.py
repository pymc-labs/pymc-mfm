"""The example notebooks run top to bottom without a notebook kernel."""

import json
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
def test_river_1_quickstart_loads_the_pinned_river_1_bundle_and_runs_headless(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from matplotlib.figure import Figure

    import pymc_mfm
    from pymc_mfm import MFM, Model, Result
    from pymc_mfm._product import _RIVER_REVISION
    from pymc_mfm.inference import save_pretrained
    from pymc_mfm.model import ModelConfig

    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "bundle"
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=1
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        save_pretrained(MFM(config).eval(), bundle)

    # The notebook loads River-1 from the Hub; a tiny local bundle stands in for its weights.
    load_bundle = pymc_mfm.from_pretrained
    hub_loads = []

    def from_hub(source, **options):
        hub_loads.append((source, options))
        return load_bundle(bundle)

    monkeypatch.setattr(pymc_mfm, "from_pretrained", from_hub)

    namespace = run_code_cells(QUICKSTART)
    ((source, options),) = hub_loads
    assert source == "pymc-labs/mfm"
    assert (options["profile"], options["revision"]) == ("river-1", _RIVER_REVISION)
    assert isinstance(namespace["model"], Model)
    result = namespace["result"]
    assert isinstance(result, Result)
    tree = result.idata
    assert set(tree.children) == {"context", "query"}

    for window, frame in (("context", namespace["history"]), ("query", namespace["future"])):
        posterior = tree[window]["posterior"]
        assert posterior.sizes["chain"] == 1
        assert posterior.sizes["draw"] == 100
        np.testing.assert_array_equal(posterior["date"].values, frame["week"].to_numpy())
        constants = tree[window]["constant_data"]
        np.testing.assert_array_equal(constants["media"].values, frame[["tv", "search"]].to_numpy())
        np.testing.assert_array_equal(constants["controls"].values, frame[["price"]].to_numpy())
        assert np.isfinite(posterior["channel_contribution"].values).all()

    np.testing.assert_array_equal(
        tree["context"]["observed_data"]["outcome"].values, namespace["history"]["sales"].to_numpy()
    )
    assert "observed_data" not in tree["query"].children

    for name in ("inputs", "fit", "contributions", "shares"):
        figure = namespace[name]
        assert isinstance(figure, Figure) and figure.axes
