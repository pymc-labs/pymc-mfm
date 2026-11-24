"""Load a locally stored complete-schedule decision network."""

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel


def load_schedule_model(path: str | Path, *, hidden: int = 64) -> ScheduleBudgetModel:
    """Load a compatible safe checkpoint without network access."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    if metadata.get("architecture") != "schedule-budget-v1" or metadata.get("hidden") != str(
        hidden
    ):
        raise ValueError("incompatible schedule budget checkpoint")
    model = ScheduleBudgetModel(hidden=hidden)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()
