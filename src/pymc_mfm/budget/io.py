"""Load local direct-budget research weights without pickle."""

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from pymc_mfm.budget.model import DirectBudgetModel


def load_direct_budget_model(path: str | Path, *, hidden: int = 64) -> DirectBudgetModel:
    """Load a local safetensors model into evaluation mode.

    The caller must use the same architecture as training. No network request,
    automatic weight download or pickle deserialization occurs.
    """
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    if metadata.get("architecture") != "direct-budget-v2" or metadata.get("hidden") != str(hidden):
        raise ValueError("incompatible direct-budget checkpoint metadata")
    model = DirectBudgetModel(hidden=hidden)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()


def load_controlled_budget_model(path: str | Path, *, hidden: int = 64):
    """Load the distinct controlled-budget research architecture."""
    from pymc_mfm.budget.controlled import ControlledBudgetModel

    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    if metadata.get("architecture") not in (
        "controlled-budget-v1",
        "controlled-budget-v2",
    ) or metadata.get("hidden") != str(hidden):
        raise ValueError("incompatible controlled-budget checkpoint metadata")
    model = ControlledBudgetModel(
        hidden=hidden, joint_features=metadata["architecture"] == "controlled-budget-v2"
    )
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()


def load_budget_model(path: str | Path):
    """Load either explicitly tagged local direct-budget architecture."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    if metadata.get("architecture") == "direct-budget-v2":
        return load_direct_budget_model(path)
    if metadata.get("architecture") in ("controlled-budget-v1", "controlled-budget-v2"):
        return load_controlled_budget_model(path)
    raise ValueError("unknown budget checkpoint architecture")
