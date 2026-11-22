#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the ``pymc-mfm`` client package (wheel and sdist) from this repository, client only.

    build_client_package.py --out <dir> [--version 0.1.0] [--manifest <file>]
                            [--check] [--check-python 3.x ...] [--allow-known-defects]

The package holds only the hosted client, copied from this checkout at its current commit:

- the files of the reviewed allowlist in ``package.toml`` (``src/pymc_mfm/client/``);
- a generated ``pymc_mfm/__init__.py`` (the library's root module is not shipped: it advertises
  names the package does not hold) and a generated ``pymc_mfm/_build.py`` (commit and version).

Name, licence, extras and summary come from ``package.toml`` and ``LICENSE``; dependency bounds and
extras from this repository's ``pyproject.toml``; the page text from ``README.template.md``.
The build fails when

- a file of ``client/`` is not in the allowlist, or an allowlisted file is missing;
- a shipped module imports a ``pymc_mfm`` module outside the shipped set (imports under
  ``if TYPE_CHECKING`` do not count; names loaded from strings are not seen, ``--check`` catches them);
- the commit still has a known defect (see ``known_defects``) unless ``--allow-known-defects``,
  which is for local dry runs only: the publish workflow never passes it;
- any shipped file, the README, the metadata or an archive member names GitHub, Hugging Face, a
  private repository or an internal host;
- a shipped message recommends an extra the metadata lacks, or an archive holds an unexpected file.

``--manifest`` writes the release manifest (each archive and each file with size and sha256); the same
text goes to ``$GITHUB_STEP_SUMMARY`` when that is set. ``--check`` also builds a second time and
compares the sha256 of both archives, runs ``twine check --strict``, and installs the wheel and the
sdist in fresh virtual environments (``uv``; one per ``--check-python``, default the lowest allowed
Python), imports every name of ``pymc_mfm.client.__all__``, and asserts that no ``torch`` and no
module outside the shipped set was loaded and that a name that is not shipped gives the clear
ImportError.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = "pymc_mfm"
# Python minors the package is tested on (classifiers); every one at or above requires-python is listed.
TESTED_PYTHONS = ("3.11", "3.12", "3.13")
# Text that must never reach the public package (README, metadata, source). The private names are
# assembled from pieces so that this file does not name them itself.
FORBIDDEN = re.compile(
    "|".join(
        [
            r"github\.com",
            r"githubusercontent",
            r"huggingface",
            r"hf\.co",
            r"pymc-labs/pymc-mfm",
            "pymc-mfm" + "-private",
            "pymc-mkt" + "fm",
            r"\.run\." + "app",
            "river" + "-web",
            "river" + "-api",
        ]
    ),
    re.IGNORECASE,
)


class PackageError(Exception):
    pass


# ---- which files go in -------------------------------------------------------------------


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(tree: ast.AST):
    """Every import node that can run, in or out of functions; ``if TYPE_CHECKING`` bodies skipped."""
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                stack.extend(child.orelse)  # the else branch runs
                continue
            if isinstance(child, ast.Import | ast.ImportFrom):
                yield child
            else:
                stack.append(child)


def module_name(rel: str) -> str:
    """``pymc_mfm/client/_http.py`` -> ``pymc_mfm.client._http``; ``.../__init__.py`` -> the package."""
    parts = rel.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def library_files(package_root: Path, cfg: dict) -> list[str]:
    """The allowlisted files (``pymc_mfm/...``), after checking the tree holds them and nothing else in client/."""
    allow = list(cfg["files"]["allowlist"])
    missing = [rel for rel in allow if not (package_root.parent / rel).is_file()]
    if missing:
        raise PackageError(f"allowlisted files are missing from the source tree: {missing}")
    in_client = {
        p.relative_to(package_root.parent).as_posix()
        for p in (package_root / "client").rglob("*.py")
    }
    new = sorted(in_client - set(allow))
    if new:
        raise PackageError(
            f"client/ has files that are not in the allowlist (packaging/client/package.toml): {new}. "
            "Review them and add them, or move them out of client/."
        )
    return allow


def check_imports(stage_src: Path, rels: list[str]) -> None:
    """Every ``pymc_mfm`` module that a shipped file imports at run time is shipped (or the generated root)."""
    shipped = {module_name(rel) for rel in rels} | {"pymc_mfm", "pymc_mfm._build"}
    for rel in rels:
        name = module_name(rel)
        here = name if rel.endswith("/__init__.py") else name.rpartition(".")[0]
        tree = ast.parse((stage_src / rel).read_text(encoding="utf-8"), rel)
        for node in _runtime_imports(tree):
            found: list[tuple[str, list[str]]] = []
            if isinstance(node, ast.Import):
                found = [(a.name, []) for a in node.names if a.name.split(".")[0] == "pymc_mfm"]
            else:
                if node.level:
                    base = here.split(".")
                    if node.level - 1 > len(base) - 1:
                        raise PackageError(f"{rel}: relative import goes above pymc_mfm")
                    base = base[: len(base) - (node.level - 1)]
                    module = ".".join([*base, *(node.module.split(".") if node.module else [])])
                else:
                    module = node.module or ""
                if module.split(".")[0] == "pymc_mfm":
                    found = [(module, [a.name for a in node.names])]
            for module, names in found:
                if module not in shipped:
                    raise PackageError(
                        f"{name} imports {module}, which the package does not ship (allowlist in "
                        "packaging/client/package.toml). Move the code into client/, or review and add the file."
                    )
                for imported in names:
                    # the generated root holds __version__ and the client; any other name is not shipped
                    if module == "pymc_mfm" and imported not in ("__version__", "client"):
                        raise PackageError(
                            f"{name} imports {imported} from pymc_mfm, which the generated root does not hold"
                        )


# ---- known defects -----------------------------------------------------------------------

# Two defects of the client are fixed in open pull requests. A release must not carry them, so the
# build refuses a commit that has either. After both pull requests are merged the checks pass and
# this section can stay (they keep guarding against a revert) or be removed.
RETENTION_PHRASES = re.compile(r"moves it later|can move later", re.IGNORECASE)

# A fake service: the baseline has failed while two other runs are still active. wait() must keep
# polling until every run has ended (3 status requests); only then does .table work.
WAIT_GUARD_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import httpx
from pymc_mfm.client import Client

calls = {"status": 0}
RUNS = [
    dict(status="failed", runs_total=3, runs_done=0, runs_failed=1, runs_cancelled=0),
    dict(status="failed", runs_total=3, runs_done=0, runs_failed=1, runs_cancelled=0),
    dict(status="failed", runs_total=3, runs_done=0, runs_failed=1, runs_cancelled=2),
]
ROW = {"scenario_id": "baseline", "component": "outcome", "mean": 1.0, "p05": 0.5, "p95": 1.5,
       "draws_url_expires_at": None}

def handler(request):
    if request.url.path == "/v1/analyses/an_1":
        record = RUNS[min(calls["status"], len(RUNS) - 1)]
        calls["status"] += 1
        return httpx.Response(
            200, json={**record, "estimate_seconds": 1.0, "expires_at": "2030-01-08T00:00:00Z"}
        )
    if request.url.path == "/v1/analyses/an_1/result":
        if calls["status"] < len(RUNS):  # the service answers 409 while runs are active
            body = {"error": {"code": "not_ready", "message": "not finished"}, "status": "failed"}
            return httpx.Response(409, json=body)
        return httpx.Response(200, json={"table": [ROW], "labels": {}})
    return httpx.Response(404, json={"error": {"code": "not_found", "message": "x"}})

real = httpx.Client
httpx.Client = lambda **kw: real(**{**kw, "transport": httpx.MockTransport(handler)})
analysis = Client("https://api.example.com", api_key="k").get("an_1").wait(
    poll=0.01, timeout=10, announce=False
)
early = calls["status"] < len(RUNS)
rows = None if early else len(analysis.table)
print(json.dumps({"early": early, "status_calls": calls["status"], "rows": rows}))
"""


def _guard_python() -> list[str]:
    """A Python command that has numpy, pandas and httpx: this one, else a throwaway uv environment."""
    if all(importlib.util.find_spec(name) for name in ("numpy", "pandas", "httpx")):
        return [sys.executable, "-I"]
    uv = shutil.which("uv")
    if uv is None:
        raise PackageError("the wait() check needs numpy, pandas and httpx, or uv to fetch them")
    withs = [arg for name in ("numpy", "pandas", "httpx") for arg in ("--with", name)]
    return [uv, "run", "--no-project", "--python", "3.13", *withs, "python", "-I"]


def known_defects(stage_src: Path, rels: list[str]) -> list[tuple[int, str]]:
    """The known defects this client still has, as (library pull request number, message)."""
    found: list[tuple[int, str]] = []
    with tempfile.TemporaryDirectory(prefix="client-guard-") as tmp:
        probe = Path(tmp)
        for rel in rels:
            (probe / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage_src / rel, probe / rel)
        # The real root module reads the installed metadata; this probe has none.
        (probe / PACKAGE / "__init__.py").write_text('__version__ = "0"\n', encoding="utf-8")
        done = subprocess.run(
            [*_guard_python(), "-c", WAIT_GUARD_SCRIPT, str(probe)],
            capture_output=True,
            text=True,
            cwd=tmp,
        )
    if done.returncode:
        raise PackageError(f"the wait() check could not run against the client:\n{done.stderr}")
    result = json.loads(done.stdout)
    if result["early"] or result["rows"] != 1:
        found.append(
            (
                142,
                "Analysis.wait() returns while runs are still active, so .table can raise NotReady "
                "right after it (client/analysis.py)",
            )
        )
    for rel in rels:
        text = " ".join((stage_src / rel).read_text(encoding="utf-8").split())
        hit = RETENTION_PHRASES.search(text)
        if hit:
            found.append(
                (
                    134,
                    f"{rel} says the expiry moves later on use ({hit.group(0)!r}); the service keeps it "
                    "7 days after upload and does not renew it",
                )
            )
            break
    return found


def known_defects_message(defects: list[tuple[int, str]]) -> str:
    lines = [f"  - library PR #{number}: {text}" for number, text in defects]
    numbers = " and ".join(f"#{n}" for n in sorted({n for n, _ in defects}))
    return (
        "this commit still has known defects that a release must not carry:\n"
        + "\n".join(lines)
        + f"\nMerge {numbers} first, then build from the new main. "
        "(--allow-known-defects exists for local dry runs only; never publish from such a build.)"
    )


# ---- metadata ----------------------------------------------------------------------------


def _expand_extra(
    optional: dict[str, list[str]], name: str, seen: tuple[str, ...] = ()
) -> list[str]:
    if name in seen or name not in optional:
        raise PackageError(f"the library has no extra {name!r} (or it loops)")
    out: list[str] = []
    for requirement in optional[name]:
        match = re.fullmatch(r"pymc-mfm\[([^\]]+)\]", requirement.strip())
        if match:
            for inner in match.group(1).split(","):
                out.extend(_expand_extra(optional, inner.strip(), (*seen, name)))
        else:
            out.append(requirement)
    return list(dict.fromkeys(out))


def minimum_python(requires_python: str) -> str:
    match = re.search(r">=\s*(3\.\d+)", requires_python)
    if not match:
        raise PackageError(f"cannot read the lowest Python from {requires_python!r}")
    return match.group(1)


def classifiers(requires_python: str) -> list[str]:
    low = int(minimum_python(requires_python).split(".")[1])
    versions = [v for v in TESTED_PYTHONS if int(v.split(".")[1]) >= low]
    return [
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        *(f"Programming Language :: Python :: {v}" for v in versions),
    ]


def check_version(version: str) -> None:
    if not re.fullmatch(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", version):
        raise PackageError(f"version {version!r} is not a public PEP 440 version (no local +part)")


def config() -> dict:
    return tomllib.loads((HERE / "package.toml").read_text(encoding="utf-8"))


def extras_of(lib: dict, cfg: dict) -> dict[str, list[str]]:
    optional = lib["project"].get("optional-dependencies", {})
    return {name: _expand_extra(optional, name) for name in cfg["extras"]["names"]}


def pyproject_text(lib: dict, cfg: dict, version: str, requires_python: str) -> str:
    project = lib["project"]
    if project["name"] != "pymc-mfm" or cfg["package"]["name"] != "pymc-mfm":
        raise PackageError(
            "the distribution name must be pymc-mfm (the client reads its version from it)"
        )

    def quote(items: list[str]) -> str:
        return "".join(f'    "{item}",\n' for item in items)

    extras = "".join(f"{name} = [\n{quote(deps)}]\n" for name, deps in extras_of(lib, cfg).items())
    return (
        "[project]\n"
        f'name = "{cfg["package"]["name"]}"\n'
        f'version = "{version}"\n'
        f'description = "{cfg["package"]["description"]}"\n'
        'readme = "README.md"\n'
        f'requires-python = "{requires_python}"\n'
        f'license = "{cfg["license"]["spdx"]}"\n'
        f'license-files = ["{cfg["license"]["file"]}"]\n'
        f'authors = [{{ name = "{cfg["package"]["author"]}" }}]\n'
        f"classifiers = [\n{quote(classifiers(requires_python))}]\n"
        f"dependencies = [\n{quote(project['dependencies'])}]\n\n"
        f"[project.optional-dependencies]\n{extras}\n"
        '[build-system]\nrequires = ["hatchling>=1.27"]\nbuild-backend = "hatchling.build"\n\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/pymc_mfm"]\n'
    )


def readme_text(cfg: dict, requires_python: str, version: str) -> str:
    template = (HERE / "README.template.md").read_text(encoding="utf-8")
    return template.replace(
        "{requires_python}", minimum_python(requires_python) + " or newer"
    ).replace("{version}", version)


ROOT_INIT = '''"""PyMC Marketing Foundational Models: the hosted client.

This distribution ships the hosted client only. Import from ``pymc_mfm.client``::

    from pymc_mfm.client import Client
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version

__version__ = version("pymc-mfm")
__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "client":  # `import pymc_mfm` then `pymc_mfm.client`
        return import_module(f"{__name__}.client")
    if name.startswith("_"):  # introspection (__path__, __wrapped__, ...) is not a library name
        raise AttributeError(name)
    raise ImportError(
        f"pymc-mfm {__version__} ships the hosted client only; `{name}` is part of the full library, "
        "which is not published. Import from pymc_mfm.client."
    )
'''


def root_init_text() -> str:
    """The package's own ``pymc_mfm/__init__.py``: the version, the client, and a clear error for library names."""
    return ROOT_INIT


def build_file_text(library_sha: str, version: str, defects: list[tuple[int, str]]) -> str:
    numbers = "".join(f"{number}, " for number, _ in defects)
    return (
        '"""Where this package came from (generated by build_client_package.py)."""\n\n'
        f'LIBRARY_COMMIT = "{library_sha}"\nPACKAGE_VERSION = "{version}"\n'
        "# Library pull requests whose fix this build lacks. A release build has none.\n"
        f"KNOWN_DEFECTS = ({numbers})\n"
    )


def shipped_extras(stage_src: Path, rels: list[str]) -> set[str]:
    """Every extra that a shipped file recommends: ``pymc-mfm[draws]`` or ``pymc-mfm[client,plot]`` in its text."""
    found: set[str] = set()
    for rel in rels:
        text = (stage_src / rel).read_text(encoding="utf-8")
        for group in re.findall(r"pymc-mfm\[([^\]]+)\]", text):
            found.update(part.strip() for part in group.split(","))
    return found


def git_commit(src: Path) -> str:
    """The checked-out commit of ``src``; fails when the files the package is built from are changed."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(src),
                "status",
                "--porcelain",
                "--",
                "pyproject.toml",
                "src/pymc_mfm/client",
                "packaging/client",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PackageError(
            f"cannot read the commit of {src} (pass --library-sha): {error}"
        ) from error
    if dirty:
        raise PackageError(f"files the package is built from have uncommitted changes:\n{dirty}")
    return commit


# ---- build -------------------------------------------------------------------------------


def expected_sources(rels: list[str]) -> set[str]:
    """Archive names of the package's source files, without the dist-info."""
    return {*rels, "pymc_mfm/__init__.py", "pymc_mfm/_build.py"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(
    src: Path,
    library_sha: str,
    version: str,
    out: Path,
    *,
    requires_python: str | None = None,
    cfg: dict | None = None,
    allow_known_defects: bool = False,
) -> tuple[Path, Path, list[tuple[int, str]]]:
    """Build both archives into ``out``; return the wheel, the sdist and the known defects allowed through."""
    if not re.fullmatch(r"[0-9a-f]{40}", library_sha):
        raise PackageError(f"library sha must be the full 40-hex commit id, not {library_sha!r}")
    check_version(version)
    cfg = cfg or config()
    package_root = src / "src" / PACKAGE
    rels = library_files(package_root, cfg)
    lib = tomllib.loads((src / "pyproject.toml").read_text(encoding="utf-8"))
    requires = requires_python or cfg["package"]["requires_python"]
    pyproject = pyproject_text(lib, cfg, version, requires)
    readme = readme_text(cfg, requires, version)
    uv = shutil.which("uv")
    if uv is None:
        raise PackageError("uv is not installed")
    with tempfile.TemporaryDirectory(prefix="client-package-") as tmp:
        stage = Path(tmp)
        stage_src = stage / "src"
        for rel in rels:
            target = stage_src / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package_root.parent / rel, target)
        check_imports(stage_src, rels)
        defects = known_defects(stage_src, rels)
        if defects and not allow_known_defects:
            raise PackageError(known_defects_message(defects))
        if defects:
            print(
                "WARNING (--allow-known-defects): " + known_defects_message(defects),
                file=sys.stderr,
            )
        missing = sorted(shipped_extras(stage_src, rels) - set(cfg["extras"]["names"]))
        if missing:
            raise PackageError(
                f"shipped messages recommend extras the package does not provide: {missing}"
            )
        (stage_src / PACKAGE / "__init__.py").write_text(root_init_text(), encoding="utf-8")
        (stage_src / PACKAGE / "_build.py").write_text(
            build_file_text(library_sha, version, defects), encoding="utf-8"
        )
        for name, text in (("pyproject.toml", pyproject), ("README.md", readme)):
            (stage / name).write_text(text, encoding="utf-8")
        shutil.copyfile(HERE / cfg["license"]["file"], stage / cfg["license"]["file"])
        # Nothing that goes public may name a private place: every file of the stage.
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            hit = FORBIDDEN.search(path.read_text(encoding="utf-8"))
            if hit:
                raise PackageError(
                    f"{path.relative_to(stage)} contains {hit.group(0)!r}, which must not be public"
                )
        out.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(
            [uv, "build", "--sdist", "--wheel", "--out-dir", str(out), str(stage)],
            capture_output=True,
            text=True,
        )
        if done.returncode:
            raise PackageError(f"uv build failed:\n{done.stdout}{done.stderr}")
    wheels, sdists = sorted(out.glob("pymc_mfm-*.whl")), sorted(out.glob("pymc_mfm-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise PackageError(
            f"expected one wheel and one sdist in {out}, found {[p.name for p in wheels + sdists]}"
        )
    check_archives(
        wheels[0], sdists[0], expected_sources(rels), version, cfg, set(extras_of(lib, cfg))
    )
    return wheels[0], sdists[0], defects


# ---- checks ------------------------------------------------------------------------------


def check_archives(
    wheel: Path, sdist: Path, sources: set[str], version: str, cfg: dict, extras: set[str]
) -> None:
    """The archive file lists are exactly the expected ones, no member names a private place, every extra exists."""
    dist_info = f"pymc_mfm-{version}.dist-info"
    wanted_wheel = (
        sources
        | {f"{dist_info}/{n}" for n in ("METADATA", "WHEEL", "RECORD")}
        | {f"{dist_info}/licenses/{cfg['license']['file']}"}
    )
    with zipfile.ZipFile(wheel) as z:
        names = {n for n in z.namelist() if not n.endswith("/")}
        if names != wanted_wheel:
            raise PackageError(
                f"wheel files differ: extra {sorted(names - wanted_wheel)}, missing {sorted(wanted_wheel - names)}"
            )
        meta = z.read(f"{dist_info}/METADATA").decode()
        for name in sorted(names):
            hit = FORBIDDEN.search(z.read(name).decode("utf-8", "replace"))
            if hit:
                raise PackageError(f"wheel member {name} contains {hit.group(0)!r}")
    provided = set(re.findall(r"^Provides-Extra: (.+)$", meta, re.M))
    if provided != extras:
        raise PackageError(
            f"METADATA extras {sorted(provided)} differ from package.toml {sorted(extras)}"
        )
    root = f"pymc_mfm-{version}"
    wanted_sdist = {f"{root}/src/{n}" for n in sources} | {
        f"{root}/{n}" for n in ("PKG-INFO", "pyproject.toml", "README.md", cfg["license"]["file"])
    }
    with tarfile.open(sdist) as t:
        members = {m.name for m in t.getmembers() if m.isfile()}
        if members != wanted_sdist:
            raise PackageError(
                f"sdist files differ: extra {sorted(members - wanted_sdist)}, missing {sorted(wanted_sdist - members)}"
            )
        for name in sorted(members):
            hit = FORBIDDEN.search(t.extractfile(name).read().decode("utf-8", "replace"))
            if hit:
                raise PackageError(f"sdist member {name} contains {hit.group(0)!r}")


def manifest_text(
    wheel: Path, sdist: Path, library_sha: str, version: str, defects: list[tuple[int, str]]
) -> str:
    """The release manifest: the two archives and every file in them, each with its size and sha256."""
    lines = [
        f"pymc-mfm {version}",
        f"library commit {library_sha}",
        "known defects: "
        + (
            "PRESENT, local dry run only, do not publish: " + ", ".join(f"#{n}" for n, _ in defects)
            if defects
            else "none"
        ),
        "",
        "archives",
    ]
    for path in (wheel, sdist):
        lines.append(f"  {sha256(path.read_bytes())}  {path.stat().st_size:>7} bytes  {path.name}")
    lines += ["", f"files in {wheel.name}"]
    with zipfile.ZipFile(wheel) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if not info.filename.endswith("/"):
                lines.append(
                    f"  {sha256(z.read(info))}  {info.file_size:>7} bytes  {info.filename}"
                )
    lines += ["", f"files in {sdist.name}"]
    with tarfile.open(sdist) as t:
        for member in sorted((m for m in t.getmembers() if m.isfile()), key=lambda m: m.name):
            lines.append(
                f"  {sha256(t.extractfile(member).read())}  {member.size:>7} bytes  {member.name}"
            )
    return "\n".join(lines) + "\n"


IMPORT_CHECK = """
import json, sys
import pymc_mfm
import pymc_mfm.client
import pymc_mfm.client as client
for name in client.__all__:
    getattr(client, name)
    exec(f"from pymc_mfm.client import {name}")
exec("from pymc_mfm.client import *")
import pymc_mfm._build as build
try:
    pymc_mfm.MFM
    refusal = None
except ImportError as error:
    refusal = str(error)
loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'pymc_mfm')
print(json.dumps({'torch': 'torch' in sys.modules, 'loaded': loaded, 'commit': build.LIBRARY_COMMIT,
                  'version': pymc_mfm.__version__, 'refusal': refusal, 'names': len(client.__all__)}))
"""


def check_install(
    artifact: Path, python: str, sources: set[str], library_sha: str, version: str
) -> None:
    """Install one artifact into a fresh venv, import every public name, check what was loaded."""
    uv = shutil.which("uv")
    with tempfile.TemporaryDirectory(prefix="client-venv-") as tmp:
        venv = Path(tmp) / "venv"
        for command in (
            [uv, "venv", "--python", python, str(venv)],
            [uv, "pip", "install", "--python", str(venv / "bin" / "python"), str(artifact)],
        ):
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode:
                raise PackageError(
                    f"{' '.join(command[:3])} failed for {artifact.name} on Python {python}:\n{done.stdout}{done.stderr}"
                )
        done = subprocess.run(
            [str(venv / "bin" / "python"), "-I", "-c", IMPORT_CHECK],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if done.returncode:
            raise PackageError(
                f"import failed for {artifact.name} on Python {python}:\n{done.stderr}"
            )
        result = json.loads(done.stdout)
    allowed = {module_name(n) for n in sources}
    stray = sorted(set(result["loaded"]) - allowed)
    if result["torch"] or stray:
        raise PackageError(
            f"{artifact.name} on Python {python}: torch loaded={result['torch']}, modules outside the list: {stray}"
        )
    if result["commit"] != library_sha or result["version"] != version:
        raise PackageError(
            f"{artifact.name}: build record {result['commit']} {result['version']} differs"
        )
    if not result["refusal"] or "ships the hosted client only" not in result["refusal"]:
        raise PackageError(
            f"{artifact.name}: a library name does not give the clear ImportError ({result['refusal']!r})"
        )


def check_reproducible(wheel: Path, sdist: Path, rebuild: tuple[Path, Path]) -> None:
    for first, second in zip((wheel, sdist), rebuild, strict=True):
        if sha256(first.read_bytes()) != sha256(second.read_bytes()):
            raise PackageError(f"two builds of {first.name} differ: the build is not reproducible")


def run_checks(
    wheel: Path, sdist: Path, sources: set[str], library_sha: str, version: str, pythons: list[str]
) -> None:
    runner = shutil.which("twine")
    command = (
        [runner, "check", "--strict", str(wheel), str(sdist)]
        if runner
        else [
            shutil.which("uv") or "uv",
            "tool",
            "run",
            "twine",
            "check",
            "--strict",
            str(wheel),
            str(sdist),
        ]
    )
    done = subprocess.run(command, capture_output=True, text=True)
    print(done.stdout + done.stderr, end="")
    if done.returncode:
        raise PackageError("twine check failed")
    for python in pythons:
        for artifact in (wheel, sdist):
            check_install(artifact, python, sources, library_sha, version)
            print(
                f"installed {artifact.name} on Python {python}: every public name imports, no torch, no stray module"
            )


def main(argv: list[str] | None = None) -> int:
    cfg = config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--src", type=Path, default=REPO, help="the checkout to build from (default: this one)"
    )
    parser.add_argument(
        "--library-sha",
        help="record this commit (default: the checkout's HEAD, which must be clean)",
    )
    parser.add_argument("--version", default=cfg["package"]["default_version"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="write the release manifest here (not into --out: publish uploads that)",
    )
    parser.add_argument("--requires-python", help="override package.toml (tests use it)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="reproducibility, twine check, and installs in fresh venvs",
    )
    parser.add_argument(
        "--check-python",
        action="append",
        default=[],
        help="a Python to install on (default: the lowest allowed)",
    )
    parser.add_argument(
        "--allow-known-defects",
        action="store_true",
        help="LOCAL DRY RUNS ONLY: build although the commit has a known defect. The publish workflow never passes this.",
    )
    args = parser.parse_args(argv)
    try:
        sha = args.library_sha or git_commit(args.src)
        wheel, sdist, defects = build(
            args.src,
            sha,
            args.version,
            args.out,
            requires_python=args.requires_python,
            cfg=cfg,
            allow_known_defects=args.allow_known_defects,
        )
        text = manifest_text(wheel, sdist, sha, args.version, defects)
        if args.check:
            with tempfile.TemporaryDirectory(prefix="client-rebuild-") as tmp:
                again = build(
                    args.src,
                    sha,
                    args.version,
                    Path(tmp),
                    requires_python=args.requires_python,
                    cfg=cfg,
                    allow_known_defects=args.allow_known_defects,
                )
                check_reproducible(wheel, sdist, again[:2])
            print("reproducible: two builds gave the same sha256 for the wheel and the sdist")
            low = minimum_python(args.requires_python or cfg["package"]["requires_python"])
            run_checks(
                wheel,
                sdist,
                expected_sources(cfg["files"]["allowlist"]),
                sha,
                args.version,
                args.check_python or [low],
            )
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(text, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
                summary.write(f"### Release manifest\n\n```\n{text}```\n")
        print(text, end="")
    except PackageError as error:
        print(f"build_client_package: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
