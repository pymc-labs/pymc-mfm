"""Tracked files stay inside the same top-level allowlist the sdist is built from."""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture(scope="module")
def tracked_top_level() -> set[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    try:
        toplevel = Path(_git("rev-parse", "--show-toplevel").strip()).resolve()
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout (for example an unpacked sdist)")
    if toplevel != ROOT:
        pytest.skip("tests are not at the root of their own git checkout")
    return {path.split("/", 1)[0] for path in _git("ls-files").splitlines() if path}


def test_tracked_files_are_inside_the_sdist_allowlist(tracked_top_level):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    allowed = set(config["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"])
    assert tracked_top_level <= allowed, sorted(tracked_top_level - allowed)
