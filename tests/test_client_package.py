# ruff: noqa: E501
"""The public ``pymc-mfm`` client package (``packaging/client/build_client_package.py``).

Fast tests run the builder on a tiny fake source tree: the reviewed allowlist, the failure when a
shipped file imports a module that is not shipped, the generated root module, the extras, the
generated metadata, the release manifest, the text that must not reach PyPI, and the guard that
refuses a commit with a known defect (with fake services and stub clients). One slow test builds the
real client from this checkout and installs both archives in fresh virtual environments. Building needs
``uv``. The sdist does not ship ``packaging/``, so the module skips there.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "packaging" / "client"
BUILDER = CLIENT_DIR / "build_client_package.py"

pytestmark = pytest.mark.skipif(not BUILDER.is_file(), reason="needs a Git checkout")

if BUILDER.is_file():
    spec = importlib.util.spec_from_file_location("build_client_package", BUILDER)
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["build_client_package"] = pkg
    spec.loader.exec_module(pkg)

SHA = "a" * 40
needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")

LIB_PYPROJECT = """
[project]
name = "pymc-mfm"
version = "9.9.9"
requires-python = ">=3.13"
dependencies = ["numpy>=2.1,<3", "httpx>=0.27,<1"]

[project.optional-dependencies]
draws = ["xarray>=2024.10"]
plot = ["matplotlib>=3.9"]
client = ["pymc-mfm[draws,plot]"]
local = ["torch>=2.6"]
"""

FAKE_FILES = ["pymc_mfm/client/__init__.py", "pymc_mfm/client/_client.py"]

# A stub client with the fixed wait() (polls until every run has ended) or the unfixed one
# (returns as soon as the analysis is done or failed). Same calls as the real client.
STUB_CLIENT = """
import httpx

class NotReady(Exception):
    pass

class Analysis:
    def __init__(self, http):
        self._http = http
    def wait(self, timeout=None, poll=2.0, announce=True):
        while True:
            record = self._http.get("/v1/analyses/an_1").json()
            ended = record["runs_done"] + record["runs_failed"] + record["runs_cancelled"]
            if CONDITION:
                return self
    @property
    def table(self):
        response = self._http.get("/v1/analyses/an_1/result")
        if response.status_code == 409:
            raise NotReady()
        return response.json()["table"]

class Client:
    def __init__(self, url, api_key=None):
        self._http = httpx.Client(base_url=url)
    def get(self, analysis_id):
        return Analysis(self._http)
"""
FIXED_CONDITION = 'ended >= record["runs_total"]'
UNFIXED_CONDITION = 'record["status"] in ("done", "failed")'


def fake_cfg() -> dict:
    return {**pkg.config(), "files": {"allowlist": FAKE_FILES}}


def fake_library(root: Path, client_extra: str = "") -> Path:
    """A tiny library: root __init__ with a lazy export, helpers, a client, a model package."""
    src = root / "src" / "pymc_mfm"
    (src / "client").mkdir(parents=True)
    (src / "model").mkdir()
    (root / "pyproject.toml").write_text(LIB_PYPROJECT)
    (src / "__init__.py").write_text(
        '"""Library root."""\nfrom ._lazy import lazy\n__version__ = "0"\n'
    )
    (src / "_lazy.py").write_text("def lazy():\n    return 1\n")
    (src / "client" / "__init__.py").write_text(
        "from ._client import Client\nfrom pymc_mfm import __version__\n__all__ = ['Client']\n"
    )
    stub = STUB_CLIENT.replace("CONDITION", FIXED_CONDITION)
    (src / "client" / "_client.py").write_text(stub + client_extra)
    (src / "model" / "__init__.py").write_text("import torch\nclass MFM: ...\n")
    return root


def stage_with_client(tmp_path: Path, client_extra: str = "") -> Path:
    lib = fake_library(tmp_path / "lib", client_extra)
    stage = tmp_path / "stage"
    for rel in FAKE_FILES:
        (stage / "src" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lib / "src" / rel, stage / "src" / rel)
    return stage


def stub_stage(tmp_path: Path, condition: str, docstring: str = "") -> Path:
    stage = tmp_path / "stub" / "src"
    (stage / "pymc_mfm" / "client").mkdir(parents=True)
    (stage / "pymc_mfm" / "client" / "__init__.py").write_text(
        f'"""{docstring}"""\n' + STUB_CLIENT.replace("CONDITION", condition)
    )
    return stage


# ---- which files go in ------------------------------------------------------------------------


def test_only_the_allowlisted_files_are_copied_and_a_new_client_file_stops_the_build(tmp_path):
    root = fake_library(tmp_path) / "src" / "pymc_mfm"
    assert pkg.library_files(root, fake_cfg()) == FAKE_FILES
    (root / "client" / "_new.py").write_text("x = 1\n")
    with pytest.raises(pkg.PackageError, match="not in the allowlist"):
        pkg.library_files(root, fake_cfg())
    (root / "client" / "_new.py").unlink()
    (root / "client" / "_client.py").unlink()
    with pytest.raises(pkg.PackageError, match="missing"):
        pkg.library_files(root, fake_cfg())


def test_the_real_allowlist_is_exactly_the_client_files_of_this_checkout():
    allowlist = pkg.config()["files"]["allowlist"]
    on_disk = sorted(
        p.relative_to(ROOT / "src").as_posix() for p in (ROOT / "src/pymc_mfm/client").rglob("*.py")
    )
    assert sorted(allowlist) == on_disk and len(allowlist) == 11
    assert not any("_lazy" in f or "_deprecated" in f for f in allowlist)


@pytest.mark.parametrize(
    "extra",
    [
        "from pymc_mfm.model import MFM\n",  # absolute import of a module that is not shipped
        "from ..model import MFM\n",  # relative
        "def f():\n    import pymc_mfm.model\n",  # inside a function: still runs
        "from pymc_mfm import model\n",  # a name the generated root does not hold
        "from pymc_mfm import MFM\n",
        "from pymc_mfm import _lazy\n",  # a helper that is not shipped
    ],
)
def test_the_build_fails_loudly_when_a_shipped_file_imports_what_is_not_shipped(tmp_path, extra):
    stage = stage_with_client(tmp_path, extra)
    with pytest.raises(pkg.PackageError, match="does not ship|does not hold"):
        pkg.check_imports(stage / "src", FAKE_FILES)


def test_version_and_client_imports_and_type_checking_imports_are_allowed(tmp_path):
    stage = stage_with_client(
        tmp_path,
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pymc_mfm.model import MFM\n",
    )
    pkg.check_imports(stage / "src", FAKE_FILES)


# ---- the guard against known defects ------------------------------------------------------------


def test_the_guard_flags_a_wait_that_returns_while_runs_are_active(tmp_path):
    unfixed = stub_stage(tmp_path / "a", UNFIXED_CONDITION)
    found = pkg.known_defects(unfixed, ["pymc_mfm/client/__init__.py"])
    assert [number for number, _ in found] == [142]
    fixed = stub_stage(tmp_path / "b", FIXED_CONDITION)
    assert pkg.known_defects(fixed, ["pymc_mfm/client/__init__.py"]) == []


@pytest.mark.parametrize(
    "text",
    [
        "The service moves it later each time you use the dataset.",
        "It can move later when the dataset is used\n    again.",
    ],
)
def test_the_guard_flags_docstrings_that_promise_the_expiry_moves_later(tmp_path, text):
    stage = stub_stage(tmp_path, FIXED_CONDITION, text)
    found = pkg.known_defects(stage, ["pymc_mfm/client/__init__.py"])
    assert [number for number, _ in found] == [134]


def test_a_commit_with_a_known_defect_does_not_build_and_the_message_names_both_pull_requests():
    message = pkg.known_defects_message([(142, "wait"), (134, "docs")])
    assert "#134 and #142" in message and "--allow-known-defects" in message
    assert "library PR #142: wait" in message


@needs_uv
def test_the_build_refuses_a_defective_client_unless_the_dry_run_flag_is_given(tmp_path):
    lib = fake_library(tmp_path / "lib", "# It can move later when used.\n")
    cfg = fake_cfg()
    with pytest.raises(pkg.PackageError, match="known defects"):
        pkg.build(lib, SHA, "0.1.0", tmp_path / "out", cfg=cfg)
    assert not (tmp_path / "out").exists()
    _, _, allowed = pkg.build(
        lib, SHA, "0.1.0", tmp_path / "out", cfg=cfg, allow_known_defects=True
    )
    assert sorted(number for number, _ in allowed) == [134]


# ---- generated files, metadata, text ------------------------------------------------------------


def test_the_generated_root_module_gives_the_version_the_client_and_a_clear_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    namespace: dict = {"__name__": "pymc_mfm"}
    exec(compile(pkg.root_init_text(), "root", "exec"), namespace)
    assert namespace["__version__"] == "9.9.9"
    with pytest.raises(
        ImportError,
        match=r"pymc-mfm 9\.9\.9 ships the hosted client only; `MFM` is part of the full library, "
        r"which is not published",
    ):
        namespace["__getattr__"]("MFM")
    with pytest.raises(AttributeError):  # introspection names are not library names
        namespace["__getattr__"]("__wrapped__")
    assert "local" not in pkg.root_init_text()


def test_the_build_record_names_the_commit_the_version_and_the_missing_fixes():
    namespace: dict = {}
    exec(pkg.build_file_text(SHA, "0.1.0", [(142, "x")]), namespace)
    assert (namespace["LIBRARY_COMMIT"], namespace["PACKAGE_VERSION"]) == (SHA, "0.1.0")
    assert namespace["KNOWN_DEFECTS"] == (142,)
    release: dict = {}
    exec(pkg.build_file_text(SHA, "0.1.0", []), release)
    assert release["KNOWN_DEFECTS"] == ()


def test_the_generated_metadata_names_the_package_and_provides_every_extra():
    lib = pkg.tomllib.loads(LIB_PYPROJECT)
    text = pkg.pyproject_text(lib, pkg.config(), "0.1.0", ">=3.11")
    data = pkg.tomllib.loads(text)["project"]
    assert (data["name"], data["version"], data["requires-python"]) == (
        "pymc-mfm",
        "0.1.0",
        ">=3.11",
    )
    assert data["license"] == "Apache-2.0" and data["license-files"] == ["LICENSE"]
    assert data["dependencies"] == ["numpy>=2.1,<3", "httpx>=0.27,<1"]
    assert data["optional-dependencies"] == {
        "draws": ["xarray>=2024.10"],
        "plot": ["matplotlib>=3.9"],
        "client": ["xarray>=2024.10", "matplotlib>=3.9"],
    }
    assert [
        c for c in data["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
    ] == [f"Programming Language :: Python :: 3.{n}" for n in (11, 12, 13)]
    assert "License ::" not in text


def test_the_real_dependencies_come_from_this_repositorys_pyproject():
    lib = pkg.tomllib.loads((ROOT / "pyproject.toml").read_text())
    data = pkg.tomllib.loads(pkg.pyproject_text(lib, pkg.config(), "0.1.0", ">=3.11"))["project"]
    assert data["dependencies"] == lib["project"]["dependencies"]
    assert {d.split(">")[0] for d in data["dependencies"]} == {
        "numpy",
        "pandas",
        "packaging",
        "httpx",
    }
    assert data["optional-dependencies"]["client"] == [
        *lib["project"]["optional-dependencies"]["draws"],
        *lib["project"]["optional-dependencies"]["plot"],
    ]


def test_every_extra_that_a_shipped_message_names_is_provided(tmp_path):
    names = set(pkg.config()["extras"]["names"])
    for rel, extra in {"a.py": "draws", "b.py": "plot"}.items():
        (tmp_path / rel).write_text(f"msg = 'pip install \"pymc-mfm[{extra}]\"'\n")
        assert pkg.shipped_extras(tmp_path, [rel]) == {extra} <= names
    (tmp_path / "m.py").write_text('x = "pymc-mfm[client,local]"\n')
    # `local` is not provided: the build refuses it
    assert pkg.shipped_extras(tmp_path, ["m.py"]) == {"client", "local"}


def test_the_real_client_names_no_extra_the_package_lacks():
    names = set(pkg.config()["extras"]["names"])
    named = pkg.shipped_extras(ROOT / "src", pkg.config()["files"]["allowlist"])
    assert named <= names, named - names


@pytest.mark.parametrize("version", ["0.1.0+abc1234", "v0.1", "", "0.1.0 "])
def test_a_version_pypi_rejects_is_refused(version):
    with pytest.raises(pkg.PackageError):
        pkg.check_version(version)


def test_the_page_text_has_the_licence_sentence_with_the_generated_version_and_no_private_name():
    plain = pkg.readme_text(pkg.config(), ">=3.11", "0.2.3")
    assert plain.startswith(
        "# pymc-mfm\n\nPython client for PyMC Marketing Foundational Models, "
        "the hosted service from PyMC Labs."
    )
    assert "pip install pymc-mfm\n" in plain and "Python 3.11 or newer" in plain
    assert (
        "Connect page of the console" in plain and "MFM_API_URL" in plain and "MFM_API_KEY" in plain
    )
    assert (
        "The client code and documentation distributed in pymc-mfm 0.2.3 are licensed under Apache-2.0. "
        "This grant excludes the hosted service, model weights, training and inference code. "
        "Future releases may use different licences."
    ) in plain
    assert "0.1.0" not in plain
    assert "{version}" not in plain and "{requires_python}" not in plain
    assert "Copyright 2026 The PyMC Labs Developers" in plain
    assert not pkg.FORBIDDEN.search(plain) and "http" not in plain and "@" not in plain


def test_the_page_example_is_the_documented_first_example_and_only_uses_the_shipped_api():
    text = pkg.readme_text(pkg.config(), ">=3.11", "0.1.0")
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    for call in (
        "Client()",
        "client.datasets.create(",
        "client.analyze(",
        "analysis.wait()",
        "analysis.table",
    ):
        assert call in code
    import ast

    ast.parse(code)
    imported = re.search(r"from pymc_mfm\.client import (.+)", code).group(1)
    namespace: dict = {}
    exec(f"from pymc_mfm.client import {imported}", namespace)  # every imported name exists


def test_the_licence_file_is_the_unmodified_apache_text():
    text = (CLIENT_DIR / "LICENSE").read_text().lstrip("\n")
    assert text.startswith(
        "                                 Apache License\n"
        "                           Version 2.0, January 2004"
    )
    assert (
        len((CLIENT_DIR / "LICENSE").read_bytes()) == 11358
        and "END OF TERMS AND CONDITIONS" in text
    )


def test_the_forbidden_words_catch_the_private_places():
    for bad in (
        "https://github.com/some/repo",
        "raw.githubusercontent.com/x",
        "huggingface.co/some/model",
        "pymc-labs/pymc-mfm",
        "pymc-mfm" + "-private",
        "x.run." + "app",
        "river" + "-api",
    ):
        assert pkg.FORBIDDEN.search(bad), bad


# ---- a fake library, built ---------------------------------------------------------------------


@needs_uv
def test_a_fake_library_builds_a_wheel_and_an_sdist_with_exactly_the_expected_files(tmp_path):
    lib = fake_library(tmp_path / "lib")
    cfg = fake_cfg()
    wheel, sdist, defects = pkg.build(lib, SHA, "0.1.0", tmp_path / "out", cfg=cfg)
    assert defects == []
    assert wheel.name == "pymc_mfm-0.1.0-py3-none-any.whl" and sdist.name == "pymc_mfm-0.1.0.tar.gz"
    licence = (CLIENT_DIR / "LICENSE").read_bytes()
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        assert {"pymc_mfm/__init__.py", "pymc_mfm/_build.py", *FAKE_FILES} <= names
        assert not any(n.startswith("pymc_mfm/model") or n == "pymc_mfm/_lazy.py" for n in names)
        init = z.read("pymc_mfm/__init__.py").decode()
        assert "Library root" not in init and "hosted client only" in init
        meta = z.read("pymc_mfm-0.1.0.dist-info/METADATA").decode()
        assert "License-Expression: Apache-2.0" in meta and "License-File: LICENSE" in meta
        assert {"draws", "plot", "client"} == set(re.findall(r"^Provides-Extra: (.+)$", meta, re.M))
        assert z.read("pymc_mfm-0.1.0.dist-info/licenses/LICENSE") == licence
        assert f'LIBRARY_COMMIT = "{SHA}"' in z.read("pymc_mfm/_build.py").decode()
        sums = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names if not n.endswith("/")}
    with tarfile.open(sdist) as t:
        assert t.extractfile("pymc_mfm-0.1.0/LICENSE").read() == licence
        assert "pymc_mfm-0.1.0/README.md" in t.getnames()
    manifest = pkg.manifest_text(wheel, sdist, SHA, "0.1.0", defects)
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() in manifest
    assert "known defects: none" in manifest
    assert all(f"{digest}" in manifest and name in manifest for name, digest in sums.items())
    again = pkg.build(lib, SHA, "0.1.0", tmp_path / "out2", cfg=cfg)
    pkg.check_reproducible(wheel, sdist, again[:2])


@needs_uv
def test_a_private_name_in_a_shipped_file_stops_the_build(tmp_path):
    lib = fake_library(tmp_path / "lib", "URL = 'https://github.com/some/repo'\n")
    with pytest.raises(pkg.PackageError, match="must not be public"):
        pkg.build(lib, SHA, "0.1.0", tmp_path / "out", cfg=fake_cfg())


@needs_uv
def test_the_cli_exits_1_with_the_reason_when_the_allowlist_does_not_match(tmp_path, capsys):
    lib = fake_library(tmp_path / "lib", "from pymc_mfm.model import MFM\n")
    code = pkg.main(["--src", str(lib), "--library-sha", SHA, "--out", str(tmp_path / "out")])
    assert code == 1 and capsys.readouterr().err  # the real allowlist is not in this fake tree


# ---- this checkout, built ----------------------------------------------------------------------


@needs_uv
@pytest.mark.slow  # builds twice, installs both archives into fresh venvs on 3.11, 3.12, 3.13 (needs PyPI): about 30 s
@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_this_checkout_builds_installs_and_imports_every_client_name(tmp_path):
    out, manifest = tmp_path / "out", tmp_path / "MANIFEST.txt"
    pythons = [arg for p in ("3.11", "3.12", "3.13") for arg in ("--check-python", p)]
    done = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--library-sha",
            SHA,
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--allow-known-defects",
            "--check",
            *pythons,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )  # --allow-known-defects: the guard is tested above; this test must pass on a commit that has the defects
    assert done.returncode == 0, done.stdout + done.stderr
    assert "reproducible" in done.stdout and done.stdout.count("every public name imports") == 6
    assert manifest.read_text().startswith("pymc-mfm 0.1.0\n")
