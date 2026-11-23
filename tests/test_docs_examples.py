"""The README, Get started and quickstart examples run as published, on both engines.

Each example runs with a tiny bundle at ``./models/my-model``, then with only its
``mfm.load`` line changed to ``engine="hosted"``, against a fake River service.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from client.client_fixtures import (
    FakeService,
    analysis_posts,
    dataset_record,
    marked_example,
    result_record,
    result_rows,
    river_draws,
    route_examples_to,
    started,
    status_record,
)
from matplotlib.figure import Figure

from pymc_mfm import MFM, NotSupported, Result
from pymc_mfm.inference import save_pretrained
from pymc_mfm.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "guide" / "index.qmd"
QUICKSTART = ROOT / "examples" / "river_1_quickstart.ipynb"
SKILL = ROOT / "skills" / "mfm" / "SKILL.md"
BUNDLE_LOAD = 'mfm.load("./models/my-model", **roles)'
HUB_LOAD = 'mfm.load("river-1", **roles)'
HOSTED_LOAD = 'mfm.load("river-1", engine="hosted", **roles)'
SIGNED_URL = "https://files.example.com/baseline.nc?sig=example"
DRAWS = 100  # every example asks for 100 draws
PAGES = [README, GUIDE]
PAGE_IDS = ["README", "Get started"]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """An empty working directory, as a reader starts from."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def lead_example(page: Path) -> str:
    return marked_example(page, "docs-example", "first-analysis")


def notebook_code(notebook: Path) -> str:
    """The notebook's code cells, in order, as one script."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def run(code: str, name: str, namespace: dict | None = None) -> dict:
    namespace = {"__name__": "__main__"} if namespace is None else namespace
    exec(compile(code, name, "exec"), namespace)
    return namespace


def fake_river(directory: Path, history, future) -> FakeService:
    """River for one inference of these frames: their dataset, analysis and draws file."""
    days = history["week"].dt.strftime("%Y-%m-%d")
    dataset = dataset_record()
    dataset["summary"].update(weeks=len(history), start=days.iloc[0], end=days.iloc[-1])
    dataset["warnings"] = []
    draws = river_draws(directory, history["week"], future["week"], draws=DRAWS)
    service = FakeService()
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    service.on("POST", "/v1/datasets", (201, dataset))
    service.on("POST", "/v1/analyses", (202, started()))
    service.on("GET", "/v1/analyses/an_1", (200, status_record()))
    baseline = result_record(result_rows("baseline", "baseline", url=SIGNED_URL))
    service.on("GET", "/v1/analyses/an_1/result", (200, baseline))
    service.on("GET", "/baseline.nc", (200, draws))
    return service


def run_hosted(code: str, name: str, load: str, directory: Path, monkeypatch) -> tuple:
    """Run ``code`` with its ``load`` call on the hosted engine; return its names and River.

    The fake River serves draws for the example's own weeks, so the lines before the
    ``model = ...`` line run first; the rest keeps its line numbers in tracebacks.
    """
    assert code.count(load) == 1, f"{name} must load the model with {load}"
    start = code.index(f"model = {load}")
    namespace = run(code[:start], name)
    service = fake_river(directory, namespace["history"], namespace["future"])
    route_examples_to(service, monkeypatch)
    rest = "\n" * code[:start].count("\n") + code[start:].replace(load, HOSTED_LOAD)
    run(rest, name, namespace)
    return namespace, service


def check_result(namespace: dict) -> None:
    """``result`` holds each window's draws on the example's own weeks and inputs."""
    result = namespace["result"]
    assert isinstance(result, Result) and result is namespace["model"].result
    tree = result.idata
    assert set(tree.children) == {"context", "query"}
    for window, frame in (("context", namespace["history"]), ("query", namespace["future"])):
        posterior = tree[window]["posterior"]
        assert posterior.sizes["draw"] == DRAWS
        np.testing.assert_array_equal(posterior["date"], frame["week"])
        assert np.isfinite(posterior["channel_contribution"]).all()
        constants = tree[window]["constant_data"]
        np.testing.assert_array_equal(constants["media"], frame[["tv", "search"]])
        np.testing.assert_array_equal(constants["controls"], frame[["price"]])
    observed = tree["context"]["observed_data"]["outcome"]
    np.testing.assert_array_equal(observed, namespace["history"]["sales"])


def check_sent(service: FakeService, namespace: dict) -> None:
    """River received the example's history and plan, and its draw count and seed."""
    (upload,) = [service.body(r) for r in service.sent("POST", "/v1/datasets")]
    weeks = namespace["history"]["week"].dt.strftime("%Y-%m-%d").tolist()
    assert upload["dates"] == weeks
    (analysis,) = analysis_posts(service)
    plan = namespace["future"]["week"].dt.strftime("%Y-%m-%d").tolist()
    assert analysis["plan"]["dates"] == plan
    assert (analysis["draws"], analysis["seed"], analysis["explain_history"]) == (DRAWS, 0, True)
    with pytest.raises(NotSupported):
        _ = namespace["model"].network  # it ran on River, not on local weights


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_lead_example_runs_on_a_local_bundle(page, workdir):
    code = lead_example(page)
    assert code.count(BUNDLE_LOAD) == 1, f"{page.name} must load {BUNDLE_LOAD}"
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=1
    )
    torch.manual_seed(0)
    save_pretrained(MFM(config).eval(), workdir / "models" / "my-model")

    namespace = run(code, page.name)
    check_result(namespace)
    assert isinstance(namespace["model"].network, MFM)


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_lead_example_runs_on_hosted_river_with_only_the_load_line_changed(
    page, workdir, monkeypatch
):
    namespace, service = run_hosted(
        lead_example(page), page.name, BUNDLE_LOAD, workdir, monkeypatch
    )
    check_result(namespace)
    check_sent(service, namespace)


def test_quickstart_runs_on_hosted_river_with_only_the_load_line_changed(workdir, monkeypatch):
    code = notebook_code(QUICKSTART)
    namespace, service = run_hosted(code, QUICKSTART.name, HUB_LOAD, workdir, monkeypatch)
    check_result(namespace)
    check_sent(service, namespace)
    for name in ("inputs", "fit", "contributions", "shares"):
        figure = namespace[name]
        assert isinstance(figure, Figure) and figure.axes


def test_docs_install_the_package_without_extras():
    pages = [
        README,
        *ROOT.glob("docs/**/*.qmd"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("examples/*.ipynb"),
        *(path for path in ROOT.glob("skills/**/*") if path.is_file()),
    ]
    assert {README, GUIDE, QUICKSTART, SKILL} <= set(pages)
    # One install has every feature: an extras hint such as pymc-mfm[client] cannot install.
    hinted = [
        str(path.relative_to(ROOT))
        for path in pages
        if "pymc-mfm[" in path.read_text(encoding="utf-8")
    ]
    assert hinted == []
