"""Safely load experimental residual budget checkpoints."""

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from pymc_mfm.budget.residual import ResidualBudgetModel


def load_residual_budget_model(path: str | Path, *, hidden: int = 64):
    """Load a self-contained, explicitly tagged local decision checkpoint."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    if metadata.get("architecture") != "residual-budget-v1" or metadata.get("hidden") != str(
        hidden
    ):
        raise ValueError("incompatible residual-budget checkpoint metadata")
    model = ResidualBudgetModel(hidden=hidden)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()
