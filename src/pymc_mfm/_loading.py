"""``pymc_mfm.from_pretrained``: choose the hosted service or a local model.

This module imports no PyTorch. ``api_key=`` selects the hosted service; without it
the call goes to :func:`pymc_mfm.inference.hub.from_pretrained` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymc_mfm.hosted import HostedMFM
    from pymc_mfm.model import MFM


def from_pretrained(
    path_or_repo: str | Path,
    *,
    revision: str | None = None,
    profile: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    device: Any = "cpu",
    api_key: str | None = None,
    base_url: str | None = None,
) -> MFM | HostedMFM:
    """Load a model: a local folder, a Hugging Face profile, or the hosted service.

    Without ``api_key`` this is :func:`pymc_mfm.inference.hub.from_pretrained` and
    returns an ``MFM`` that runs in this process (needs PyTorch). The environment
    variable ``MFM_API_KEY`` never selects the hosted service.

    With ``api_key`` it returns a :class:`pymc_mfm.hosted.HostedMFM` that runs the
    model on the hosted service through :mod:`pymc_mfm.client`; no PyTorch needed.
    ``path_or_repo`` must be ``"pymc-labs/mfm"`` and ``profile`` ``"river-1"`` (or
    omitted). ``base_url`` defaults to ``MFM_API_URL``. ``revision``, ``cache_dir``,
    ``local_files_only``, ``token`` and ``device`` belong to local loading and raise
    ``ValueError`` here; ``token`` keeps its Hugging Face meaning.
    """
    if api_key is None:
        if base_url is not None:
            raise ValueError("base_url applies only with api_key (the hosted service)")
        import pymc_mfm

        local = pymc_mfm._import_local("from_pretrained")
        return local(
            path_or_repo,
            revision=revision,
            profile=profile,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            device=device,
        )
    from pymc_mfm.hosted import HostedMFM

    return HostedMFM.connect(
        path_or_repo,
        api_key=api_key,
        base_url=base_url,
        revision=revision,
        profile=profile,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
        device=device,
    )
