"""Load locally stored multi-period allocation weights."""

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from pymc_mfm.budget.planning import MultiPeriodBudgetModel


def load_planning_model(path: str | Path, *, hidden: int = 64) -> MultiPeriodBudgetModel:
    """Load a compatible safetensors checkpoint into evaluation mode."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    if metadata.get("architecture") != "multi-period-budget-v1" or metadata.get("hidden") != str(
        hidden
    ):
        raise ValueError("incompatible multi-period budget checkpoint")
    model = MultiPeriodBudgetModel(hidden=hidden)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()
