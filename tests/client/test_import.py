"""The light install must not import PyTorch, and local names must explain how to get it."""

import subprocess
import sys
import textwrap


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False
    )


def test_importing_the_package_and_client_does_not_import_torch():
    result = run(
        """
        import sys
        import pymc_mfm
        from pymc_mfm import client
        from pymc_mfm.client import Client
        assert "torch" not in sys.modules, "torch was imported"
        assert "safetensors" not in sys.modules, "safetensors was imported"
        """
    )
    assert result.returncode == 0, result.stderr


def test_local_names_explain_the_extra_when_torch_is_missing():
    result = run(
        """
        import sys
        sys.modules["torch"] = None  # makes "import torch" fail, as in a light install
        sys.modules["safetensors"] = None
        import pymc_mfm
        for name in ("decompose", "MFM"):
            try:
                getattr(pymc_mfm, name)
            except ImportError as error:
                assert 'pip install "pymc-mfm[local]"' in str(error), error
            else:
                raise SystemExit(f"{name} loaded without torch")
        # from_pretrained is a torch-free dispatcher; only its local branch needs torch
        try:
            pymc_mfm.from_pretrained("models/river-1")
        except ImportError as error:
            assert 'pip install "pymc-mfm[local]"' in str(error), error
        else:
            raise SystemExit("local from_pretrained loaded without torch")
        try:
            pymc_mfm.nope
        except AttributeError:
            pass
        else:
            raise SystemExit("unknown name did not raise AttributeError")
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_dir_lists_the_lazy_names():
    result = run(
        """
        import pymc_mfm
        assert {"MFM", "decompose", "from_pretrained"} <= set(dir(pymc_mfm))
        """
    )
    assert result.returncode == 0, result.stderr
