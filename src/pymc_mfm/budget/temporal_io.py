"""Safely load order-aware budget-only checkpoints."""

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from pymc_mfm.budget.temporal import TemporalBudgetModel


def load_temporal_budget_model(path: str | Path, *, hidden: int = 64):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    if metadata.get("architecture") != "temporal-budget-v1" or metadata.get("hidden") != str(
        hidden
    ):
        raise ValueError("incompatible temporal-budget checkpoint metadata")
    model = TemporalBudgetModel(hidden=hidden)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()
