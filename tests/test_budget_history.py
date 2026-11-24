"""Check that budget history ingestion keeps columns and time aligned."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import save_file

from pymc_mfm.budget.controlled import ControlledBudgetModel
from pymc_mfm.budget.history import read_history


def test_history_preserves_explicit_order_and_optional_controls(tmp_path):
    csv = tmp_path / "history.csv"
    csv.write_text("date,tv,search,sales,promo\n2025-01-01,10,20,40,1\n2025-01-08,12,18,43,0\n")
    history = read_history(csv, channels=["search", "tv"], outcome="sales", controls=["promo"])
    assert history == {
        "channels": ["search", "tv"],
        "spend": [[20, 10], [18, 12]],
        "outcome": [40, 43],
        "controls": [[1], [0]],
    }
    assert read_history(csv, channels=["tv"], outcome="sales")["controls"] == [[], []]


@pytest.mark.parametrize(
    "content",
    [
        "date,tv,sales\n2025-01-01,1,2\n2025-01-15,1,2\n2025-01-22,1,2\n",
        "date,tv,sales\n2025-01-01,1,2\n2025-01-01,1,2\n",
        "date,tv,sales\n2025-01-01,1,2\n2025-01-08,-1,2\n",
        "date,tv,sales\n2025-01-01,1,2\n2025-01-08,NaN,2\n",
        "date,tv,sales\n2025-01-01,1,2\n2025-01-08,,2\n",
        "date,tv,tv,sales\n2025-01-01,1,1,2\n2025-01-08,1,1,2\n",
    ],
)
def test_history_rejects_bad_data_without_filling(tmp_path, content):
    csv = tmp_path / "history.csv"
    csv.write_text(content)
    with pytest.raises(ValueError):
        read_history(csv, channels=["tv"], outcome="sales")


def test_csv_cli_returns_only_feasible_budgets(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text("date,tv,search,sales\n2025-01-01,10,20,40\n2025-01-08,12,18,43\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"budget": 100, "upper_share": [0.1, 1]}))
    checkpoint = tmp_path / "model.safetensors"
    model = ControlledBudgetModel(joint_features=True)
    save_file(
        model.state_dict(),
        str(checkpoint),
        metadata={"architecture": "controlled-budget-v2", "hidden": "64"},
    )
    script = Path(__file__).resolve().parents[1] / "examples/direct_budget/allocate_csv.py"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(checkpoint),
            "--history",
            str(history),
            "--config",
            str(config),
            "--channel",
            "tv",
            "--channel",
            "search",
            "--outcome",
            "sales",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    output = json.loads(run.stdout)
    assert set(output) == {"budget", "allocation"}
    assert output["allocation"]["tv"] <= 10 + 1e-4
    assert sum(output["allocation"].values()) == pytest.approx(100, abs=1e-4)
