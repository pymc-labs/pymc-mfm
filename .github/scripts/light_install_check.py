"""Check a light install: no PyTorch is installed, imported or needed for the client."""

import importlib.util
import sys

import pymc_mfm
from pymc_mfm import client  # noqa: F401

assert importlib.util.find_spec("torch") is None, "torch is installed in the light environment"
assert "torch" not in sys.modules, "importing pymc_mfm imported torch"
try:
    pymc_mfm.decompose
except ImportError as error:
    assert 'pip install "pymc-mfm[local]"' in str(error), error
else:
    raise SystemExit("pymc_mfm.decompose loaded without torch")
print("light install ok")
