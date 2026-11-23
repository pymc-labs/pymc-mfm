"""Refuse a release version that its tag does not name or that PyPI would not install.

Run before publishing. The version comes from ``pyproject.toml``. It fails when:

- ``--tag`` is given and is not ``v<version>``;
- the version is a development or local version;
- PyPI already has the version (PyPI never accepts the same version twice);
- PyPI has a newer release that is not yanked. ``pip install pymc-mfm`` and
  ``uv add pymc-mfm`` would keep installing that release. A newer pre-release
  counts only when the version being released is a pre-release too.

``--warn-only`` prints the problems as GitHub warnings and exits 0, for dry runs; it
also only warns when PyPI cannot be read. Needs ``packaging`` and network access to
pypi.org, whose PEP 691 simple index lists every file and its yank status.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
SIMPLE_INDEX = "https://pypi.org/simple/{name}/"


def release_problems(version: str, releases: dict[str, list[dict]], tag: str | None) -> list[str]:
    """Why ``version`` must not be released, given PyPI's ``releases`` and the git ``tag``.

    ``releases`` maps each published version to its files; a file is yanked when its
    ``yanked`` field is true.
    """
    try:
        candidate = Version(version)
    except InvalidVersion:
        return [f"{version!r} is not a PEP 440 version"]
    problems = []
    if tag is not None and tag != f"v{version}":
        problems.append(f"tag {tag} does not name version {version}; tag the release v{version}")
    if candidate.is_devrelease or candidate.local is not None:
        problems.append(f"{version} is a development or local version; release a final version")
    published = {}
    for text, files in releases.items():
        try:
            published[Version(text)] = files
        except InvalidVersion:
            continue
    if candidate in published:
        problems.append(f"PyPI already has {version}; PyPI never accepts a version twice")
    shadowing = sorted(
        str(other)
        for other, files in published.items()
        if other > candidate
        and (candidate.is_prerelease or not other.is_prerelease)
        and any(not file.get("yanked", False) for file in files)
    )
    if shadowing:
        problems.append(
            f"PyPI has newer releases that are not yanked: {', '.join(shadowing)}. "
            "pip and uv would keep installing them; yank them on PyPI first"
        )
    return problems


def releases_from_index(index: dict) -> dict[str, list[dict]]:
    """Files by normalized version from a PEP 691 project page, with a boolean ``yanked``."""

    def normalized(text: str) -> str:
        try:
            return str(Version(text))
        except InvalidVersion:
            return text

    releases: dict[str, list[dict]] = {normalized(text): [] for text in index.get("versions", [])}
    for file in index["files"]:
        filename = file["filename"]
        try:
            if filename.endswith(".whl"):
                version = parse_wheel_filename(filename)[1]
            else:
                version = parse_sdist_filename(filename)[1]
        except (InvalidSdistFilename, InvalidWheelFilename):
            continue
        # PEP 691: false, true, or the reason text, which may be empty.
        yanked = file.get("yanked", False) is not False
        releases.setdefault(str(version), []).append({"filename": filename, "yanked": yanked})
    return releases


def pypi_releases(name: str) -> dict[str, list[dict]]:
    """The project's files on PyPI by version, or nothing when the project does not exist."""
    request = urllib.request.Request(
        SIMPLE_INDEX.format(name=name),
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return releases_from_index(json.load(response))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", help="the git tag being released, for example v0.0.1")
    parser.add_argument("--warn-only", action="store_true", help="report problems, exit 0")
    args = parser.parse_args(argv)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    try:
        releases = pypi_releases(project["name"])
    except (
        OSError,
        http.client.HTTPException,
        AttributeError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        releases = None
        problems = [f"cannot read {project['name']}'s releases from PyPI: {error!r}"]
    if releases is not None:
        problems = release_problems(project["version"], releases, args.tag)
    for problem in problems:
        print(f"::warning::{problem}" if args.warn_only else f"error: {problem}", file=sys.stderr)
    if not problems:
        print(f"{project['name']} {project['version']} can be released")
    return 0 if args.warn_only or not problems else 1


if __name__ == "__main__":
    sys.exit(main())
