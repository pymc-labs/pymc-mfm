"""The release gate in scripts/check_release_version.py, without network access."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release_version.py"
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="needs a Git checkout")


def gate():
    spec = importlib.util.spec_from_file_location("check_release_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_problems(version, releases, tag=None):
    return gate().release_problems(version, releases, tag)


def test_the_simple_index_marks_files_yanked_with_or_without_a_reason():
    index = {
        "versions": ["0.1.2", "0.1.3", "1.0.0-rc1"],
        "files": [
            {"filename": "pymc_mfm-0.1.2-py3-none-any.whl", "yanked": "Published by mistake"},
            {"filename": "pymc_mfm-0.1.2.tar.gz", "yanked": ""},
            {"filename": "pymc_mfm-0.0.1-py3-none-any.whl", "yanked": False},
            {"filename": "pymc_mfm-0.0.1.tar.gz"},
            {"filename": "pymc_mfm-1.0.0rc1.tar.gz", "yanked": True},
            {"filename": "pymc-mfm-0.0.1.win32.exe", "yanked": False},
        ],
    }
    releases = gate().releases_from_index(index)
    assert {version: [f["yanked"] for f in found] for version, found in releases.items()} == {
        "0.1.2": [True, True],
        "0.1.3": [],
        "1.0.0rc1": [True],
        "0.0.1": [False, False],
    }
    assert release_problems("0.0.1", releases, "v0.0.1") == [
        "PyPI already has 0.0.1; PyPI never accepts a version twice"
    ]
    assert release_problems("0.0.2", releases, "v0.0.2") == []


def files(yanked=False):
    return [{"filename": "x.whl", "yanked": yanked}, {"filename": "x.tar.gz", "yanked": yanked}]


@pytest.mark.parametrize(
    ("version", "releases", "tag", "message"),
    [
        # The client-only 0.1.2 shadows 0.0.1 until it is yanked.
        ("0.0.1", {"0.1.2": files()}, "v0.0.1", "newer releases that are not yanked: 0.1.2"),
        ("0.0.1", {"0.1.2": files()}, None, "yank them"),
        ("0.0.1", {"0.1.2": files(yanked=True)}, "v0.0.1", None),
        # A release whose files were all deleted installs nothing.
        ("0.0.1", {"0.1.2": []}, "v0.0.1", None),
        ("0.0.1", {}, "v0.0.1", None),
        ("0.0.1", {"0.0.1": files(yanked=True)}, "v0.0.1", "already has 0.0.1"),
        ("0.0.1", {}, "v0.0.2", "tag the release v0.0.1"),
        ("0.0.1", {}, "0.0.1", "tag the release v0.0.1"),
        ("0.0.1.dev0", {}, "v0.0.1.dev0", "development or local"),
        ("0.0.1+g1234", {}, None, "development or local"),
        # pip skips pre-releases by default, so a newer one shadows only a pre-release.
        ("0.0.1", {"0.1.0a1": files()}, "v0.0.1", None),
        ("0.1.0a1", {"0.1.0a2": files()}, "v0.1.0a1", "0.1.0a2"),
        ("0.0.2", {"0.0.1": files()}, "v0.0.2", None),
    ],
)
def test_release_gate(version, releases, tag, message):
    problems = release_problems(version, releases, tag)
    if message is None:
        assert problems == []
    else:
        assert len(problems) == 1 and message in problems[0], problems
