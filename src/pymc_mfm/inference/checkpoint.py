"""Local model artifacts; transport and training bookkeeping are separate."""

from pathlib import Path

import torch

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM


def save_model(model: MFM, path: str | Path) -> None:
    """Save the complete trainable model and its construction configuration.

    This artifact contains all five trained module groups, not just inference
    weights. Optimizer state and run bookkeeping are not part of this format.
    No hosted service or authentication is involved.
    """
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, path)


def load_model(path: str | Path, *, device: str | torch.device = "cpu") -> MFM:
    """Strictly load a canonical artifact in evaluation mode.

    Only tensor and primitive payloads are deserialized. The configuration
    version defines architecture and preprocessing semantics; malformed or
    incomplete artifacts are rejected rather than inferred.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.keys() != {"config", "state_dict"}:
        raise ValueError("model artifact must contain exactly config and state_dict")
    config = ModelConfig.from_dict(payload["config"])
    # Avoid allocating/randomizing a second set of weights just to overwrite it.
    with torch.device("meta"):
        model = MFM(config)
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    return model.to(device).eval()
