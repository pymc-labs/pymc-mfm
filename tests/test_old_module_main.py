"""``python -m pymc_mfm...`` runs the same commands as ``python -m pymc_mktfm...``."""

import subprocess
import sys

import pytest

pytest.importorskip("torch")


def run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("command", ["data", "training"])
def test_old_name_command_runs_like_the_new_one(command):
    old = run(f"pymc_mfm.{command}", "--help")
    new = run(f"pymc_mktfm.{command}", "--help")
    assert old.returncode == 0, old.stderr
    assert new.returncode == 0, new.stderr
    assert old.stdout == new.stdout
    assert "usage:" in old.stdout


def test_old_name_package_without_main_still_fails_cleanly():
    result = run("pymc_mfm")
    assert result.returncode != 0
    assert "No module named pymc_mfm.__main__" in result.stderr
