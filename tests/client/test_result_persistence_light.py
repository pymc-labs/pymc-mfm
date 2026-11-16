"""A saved result loads, saves and reloads with no PyTorch.

The light CI job runs this file without PyTorch. With PyTorch installed it still checks
the same round trip; only the install-hint test needs the absence of PyTorch.
"""

import importlib.util
import subprocess
import sys
import textwrap
from dataclasses import fields

import numpy as np
import pytest

from pymc_mfm.inference.result import DecompositionResult

NO_TORCH = importlib.util.find_spec("torch") is None


def numpy_result() -> DecompositionResult:
    """Built from NumPy alone: three weekly dates, four components, three edges."""
    draws = np.arange(5 * 3 * 4, dtype=np.float64).reshape(5, 3, 4) / 7
    draws[..., 3] = 0.0  # the padded component carries nothing
    return DecompositionResult(
        components=draws,
        graphs=np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1]]),
        component_names=("baseline", "channel:search", "control:weather", "latent"),
        component_families=("baseline", "channel", "control", "latent"),
        component_mask=np.array([True, True, True, False]),
        edge_names=("cy:channel:search->sales", "zb:control:weather->outcome", "u:latent->sales"),
        edge_mask=np.array([True, False, True]),
        channel_names=("search",),
        control_names=("weather",),
        time=np.array(["2024-07-08", "2024-07-15", "2024-07-22"], dtype="datetime64[D]"),
        window="query",
        mode="model_average",
        units="EUR",
        provenance={"sales_scale": 12.5, "note": "café", "nested": {"seed": 7, "ids": [1, 2]}},
        context_length=4,
    )


def test_save_and_load_round_trip_without_torch(tmp_path):
    result = numpy_result()
    path = result.save(tmp_path / "saved")
    replay = DecompositionResult.load(path)
    for item in fields(DecompositionResult):
        if not item.init:
            continue
        expected, actual = getattr(result, item.name), getattr(replay, item.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype, item.name
        else:
            assert actual == expected, item.name
    assert replay.time.dtype == np.dtype("datetime64[D]")
    assert DecompositionResult.load(replay.save(tmp_path / "again")).provenance == result.provenance


def test_lazy_submodules_need_no_torch():
    code = textwrap.dedent(
        """
        import sys
        import pymc_mfm.inference as inference
        import pymc_mfm.inference.result
        assert inference.result.DecompositionResult is inference.DecompositionResult
        assert inference._datatree is not None
        assert inference._digest is not None
        import pymc_mfm.diagnostics as diagnostics
        assert diagnostics.accessors is not None
        import pymc_mfm.training as training
        assert training._metric_tables is not None
        assert "torch" not in sys.modules
        """
    )
    if not NO_TORCH:
        code = code.replace('assert "torch" not in sys.modules', "")
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


@pytest.mark.skipif(not NO_TORCH, reason="PyTorch is installed")
@pytest.mark.parametrize(
    "module, name",
    [
        ("pymc_mfm.inference", "decompose"),
        ("pymc_mfm.inference", "from_pretrained"),
        ("pymc_mfm.inference", "api"),
        ("pymc_mfm.diagnostics", "EvaluationArchive"),
        ("pymc_mfm.training", "load_checkpoint"),
        ("pymc_mfm", "decompose"),
    ],
)
def test_torch_names_give_the_install_hint(module, name):
    module_object = __import__(module, fromlist=["_"])
    with pytest.raises(ImportError, match=r'pip install "pymc-mfm\[local\]"') as caught:
        getattr(module_object, name)
    assert not isinstance(caught.value, ModuleNotFoundError)
    assert str(caught.value).startswith(f"{module}.{name} runs the model")
