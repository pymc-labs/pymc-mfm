"""The light install must not import PyTorch, and local names must explain how to get it."""

import subprocess
import sys
import textwrap

import pytest


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False
    )


def test_importing_the_package_and_client_does_not_import_torch():
    result = run(
        """
        import sys
        import pymc_mktfm
        from pymc_mktfm import client
        from pymc_mktfm.client import Client
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
        import pymc_mktfm
        for name in ("MFM", "from_pretrained"):
            try:
                getattr(pymc_mktfm, name)
            except ImportError as error:
                assert str(error).startswith(f"pymc_mktfm.{name} runs the model"), error
                assert 'pip install "pymc-mktfm[local]"' in str(error), error
            else:
                raise SystemExit(f"{name} loaded without torch")
        try:
            pymc_mktfm.nope
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
        import pymc_mktfm
        assert {"MFM", "from_pretrained"} <= set(dir(pymc_mktfm))
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "pymc_mktfm.inference._datatree",
        "pymc_mktfm.inference.curves",
        "pymc_mktfm.inference.result",
        "pymc_mktfm.plotting",
        "pymc_mktfm.plotting.charts",
    ],
)
def test_result_tree_curves_and_charts_import_without_torch(module):
    loaded = run(
        f"""
        import sys
        import {module}
        assert "torch" not in sys.modules, "torch was imported"
        assert "safetensors" not in sys.modules, "safetensors was imported"
        """
    )
    assert loaded.returncode == 0, loaded.stderr + loaded.stdout
    blocked = run(
        f"""
        import sys
        sys.modules["torch"] = None  # makes "import torch" fail, as in a light install
        sys.modules["safetensors"] = None
        import {module}
        import pymc_mktfm.inference
        try:
            pymc_mktfm.inference.decompose
        except ImportError as error:
            assert 'pip install "pymc-mktfm[local]"' in str(error), error
        else:
            raise SystemExit("decompose loaded without torch")
        """
    )
    assert blocked.returncode == 0, blocked.stderr + blocked.stdout


def test_top_level_exports_and_the_light_names_load_without_torch():
    result = run(
        """
        import sys
        sys.modules["torch"] = None
        sys.modules["safetensors"] = None
        import pymc_mktfm
        from pymc_mktfm.plotting import charts
        assert pymc_mktfm.__all__ == [
            "MFM", "from_pretrained", "list_profiles", "Plots", "set_backend",
            "save_idata", "load_idata", "validate_idata",
        ], pymc_mktfm.__all__
        assert pymc_mktfm.Plots is charts.Plots
        assert pymc_mktfm.set_backend is charts.set_backend
        assert {"Plots", "set_backend"} <= set(dir(pymc_mktfm))
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


DEPRECATED = {
    "ModelConfig": ("pymc_mktfm.model", "pymc_mktfm.model.ModelConfig"),
    "DecompositionResult": ("pymc_mktfm.inference", "pymc_mktfm.inference.DecompositionResult"),
    "decompose": ("pymc_mktfm.inference", "pymc_mktfm.inference.decompose"),
    "observation_batch": ("pymc_mktfm.inference", "pymc_mktfm.inference.observation_batch"),
    "save_pretrained": ("pymc_mktfm.inference", "pymc_mktfm.inference.save_pretrained"),
    "push_to_hub": ("pymc_mktfm.inference", "pymc_mktfm.inference.push_to_hub"),
}


@pytest.mark.parametrize("name", DEPRECATED)
def test_deprecated_top_level_names_warn_and_return_the_subpackage_object(name):
    if name != "DecompositionResult":  # the others load PyTorch; see the no-torch test below
        pytest.importorskip("torch")
    module, replacement = DEPRECATED[name]
    result = run(
        f"""
        import importlib, warnings
        import pymc_mktfm
        assert {name!r} not in dir(pymc_mktfm) and {name!r} not in pymc_mktfm.__all__
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from pymc_mktfm import {name}
        found = [w for w in caught if w.category is DeprecationWarning]
        assert len(found) == 1, [str(w.message) for w in found]
        assert {replacement!r} in str(found[0].message), found[0].message
        assert found[0].filename == "<string>", found[0].filename  # the caller, not pymc_mktfm
        assert {name} is getattr(importlib.import_module({module!r}), {name!r})
        assert {name!r} not in dir(pymc_mktfm)
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize("name", [name for name in DEPRECATED if name != "DecompositionResult"])
def test_deprecated_top_level_names_explain_the_extra_when_torch_is_missing(name):
    result = run(
        f"""
        import sys, warnings
        sys.modules["torch"] = None
        sys.modules["safetensors"] = None
        import pymc_mktfm
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                pymc_mktfm.{name}
            except ImportError as error:
                assert 'pip install "pymc-mktfm[local]"' in str(error), error
            else:
                raise SystemExit("{name} loaded without torch")
        assert any(w.category is DeprecationWarning for w in caught), caught
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_star_import_skips_the_six_deprecated_names_without_a_warning():
    pytest.importorskip("torch")
    result = run(
        f"""
        import warnings
        import pymc_mktfm
        namespace = {{}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec("from pymc_mktfm import *", namespace)
        assert not caught, [str(w.message) for w in caught]
        names = set(namespace) - {{"__builtins__"}}
        assert names == set(pymc_mktfm.__all__), names
        for name in {sorted(DEPRECATED)!r}:
            assert name not in namespace and name not in dir(pymc_mktfm), name
            try:
                exec(name, namespace)
            except NameError:
                pass
            else:
                raise SystemExit(f"{{name}} was in the star import")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pymc_mktfm.decompose
        assert [w.category for w in caught] == [DeprecationWarning], caught
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_star_import_without_torch_explains_the_extra():
    result = run(
        """
        import sys
        sys.modules["torch"] = None
        sys.modules["safetensors"] = None
        try:
            exec("from pymc_mktfm import *", {})
        except ImportError as error:
            assert 'pip install "pymc-mktfm[local]"' in str(error), error
        else:
            raise SystemExit("star import worked without torch")
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_a_name_that_loaded_stays_silent_after_the_warning_filter_changes():
    result = run(
        """
        import warnings
        import pymc_mktfm
        with warnings.catch_warnings(record=True) as hidden:
            warnings.simplefilter("ignore")
            pymc_mktfm.DecompositionResult  # the first access warns, and a filter hides it
        assert not hidden, hidden
        with warnings.catch_warnings(record=True) as later:
            warnings.simplefilter("always")
            pymc_mktfm.DecompositionResult
            from pymc_mktfm import DecompositionResult
        assert not later, [str(w.message) for w in later]
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_a_failed_load_is_not_cached_and_warns_again():
    result = run(
        """
        import sys, warnings
        sys.modules["torch"] = None
        sys.modules["safetensors"] = None
        import pymc_mktfm
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                try:
                    pymc_mktfm.decompose
                except ImportError as error:
                    assert 'pip install "pymc-mktfm[local]"' in str(error), error
                else:
                    raise SystemExit("decompose loaded without torch")
        assert [w.category for w in caught] == [DeprecationWarning] * 2, caught
        """
    )
    assert result.returncode == 0, result.stderr + result.stdout
