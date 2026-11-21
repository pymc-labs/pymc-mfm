"""Check a light install: no PyTorch is installed, imported or needed for the client."""

import importlib.util
import sys

import pymc_mktfm
from pymc_mktfm import client  # noqa: F401

assert importlib.util.find_spec("torch") is None, "torch is installed in the light environment"
assert "torch" not in sys.modules, "importing pymc_mktfm imported torch"
try:
    pymc_mktfm.MFM
except ImportError as error:
    assert 'pip install "pymc-mktfm[local]"' in str(error), error
else:
    raise SystemExit("pymc_mktfm.MFM loaded without torch")
print("light install ok")
