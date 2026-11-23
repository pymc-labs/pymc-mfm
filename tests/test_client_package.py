"""The public ``pymc-mfm`` client package (``packaging/client/build_client_package.py``).

Fast tests run the builder on a tiny fake source tree: the reviewed allowlist, the failure when
a shipped file imports a module that is not shipped, the generated root module, the extras, the
generated metadata, the release manifest, the text that must not reach PyPI, and the guard that
refuses a commit with a known defect (with stub clients and fake services). One slow test builds
the real client from this checkout and installs both archives in fresh virtual environments.
Building needs ``uv``. The sdist does not ship ``packaging/``, so the module skips there.
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
import tomllib
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
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
# sha256 of the unmodified Apache License 2.0 text published by the Apache Software Foundation.
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

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
GOOD_DOCSTRING = "The API keeps a dataset 7 days after upload and does not extend it."


def fake_cfg() -> dict:
    return {**pkg.config(), "files": {"allowlist": FAKE_FILES}}


def fake_library(root: Path, client_extra: str = "") -> Path:
    """A tiny library: root __init__ with a lazy export, helpers, a client, a model package."""
    src = root / "src" / "pymc_mfm"
    (src / "client").mkdir(parents=True)
    (src / "model").mkdir()
    (root / "pyproject.toml").write_text(LIB_PYPROJECT)
    (src / "__init__.py").write_text('"""Library root."""\nfrom ._lazy import lazy\n')
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


STUB_FILES = ["pymc_mfm/client/__init__.py"]

# ---- which files go in ------------------------------------------------------------------------


def test_only_the_allowlisted_files_are_copied_and_a_new_client_file_stops_the_build(tmp_path):
    root = fake_library(tmp_path) / "src" / "pymc_mfm"
    assert pkg.library_files(root, fake_cfg()) == FAKE_FILES
    (root / "client" / "_new.py").write_text("x = 1\n")
    with pytest.raises(pkg.PackageError, match=r"not in the allowlist .*\['pymc_mfm/client/_new"):
        pkg.library_files(root, fake_cfg())
    (root / "client" / "_new.py").unlink()
    (root / "client" / "_client.py").unlink()
    with pytest.raises(pkg.PackageError, match=r"missing .*\['pymc_mfm/client/_client.py'\]"):
        pkg.library_files(root, fake_cfg())


def test_a_file_in_client_that_is_not_python_stops_the_build_because_it_would_not_ship(tmp_path):
    root = fake_library(tmp_path) / "src" / "pymc_mfm"
    (root / "client" / "defaults.json").write_text("{}")
    (root / "client" / "__pycache__").mkdir()
    (root / "client" / "__pycache__" / "x.pyc").write_bytes(b"\0")  # build output is ignored
    with pytest.raises(pkg.PackageError, match=r"not Python .*\['pymc_mfm/client/defaults.json'\]"):
        pkg.library_files(root, fake_cfg())


def test_the_real_allowlist_is_exactly_the_client_files_of_this_checkout():
    allowlist = pkg.config()["files"]["allowlist"]
    client = ROOT / "src/pymc_mfm/client"
    on_disk = sorted(
        p.relative_to(ROOT / "src").as_posix()
        for p in client.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    assert sorted(allowlist) == on_disk and len(allowlist) == 11
    assert not any("_lazy" in f or "_deprecated" in f for f in allowlist)


@pytest.mark.parametrize(
    "extra",
    [
        "from pymc_mfm.model import MFM\n",  # absolute import of a module that is not shipped
        "from ..model import MFM\n",  # relative
        "from .. import model\n",  # relative, a name the generated root does not hold
        "import pymc_mfm.model as model\n",
        "def f():\n    import pymc_mfm.model\n",  # inside a function: still runs
        "from pymc_mfm import model\n",
        "from pymc_mfm import MFM\n",
        "from pymc_mfm import _lazy\n",  # a helper that is not shipped
    ],
)
def test_the_build_fails_loudly_when_a_shipped_file_imports_what_is_not_shipped(tmp_path, extra):
    stage = stage_with_client(tmp_path, extra)
    with pytest.raises(pkg.PackageError, match="does not ship|does not hold"):
        pkg.check_imports(stage / "src", FAKE_FILES)


@pytest.mark.parametrize(
    "extra",
    [
        'import importlib\nMODULE = importlib.import_module("pymc_mfm.model")\n',
        'MODULE = __import__("pymc_mfm.model")\n',
        'LAZY = {"MFM": "pymc_mfm.model"}\n',
    ],
)
def test_a_module_named_only_by_a_string_is_found_too(tmp_path, extra):
    stage = stage_with_client(tmp_path, extra)
    with pytest.raises(pkg.PackageError, match=r"names the module 'pymc_mfm.model' in a string"):
        pkg.check_imports(stage / "src", FAKE_FILES)


def test_allowed_imports_pass_and_a_docstring_may_name_a_module(tmp_path):
    stage = stage_with_client(
        tmp_path,
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from pymc_mfm.model import MFM\n"
        "from importlib import import_module\n"
        'CLIENT = import_module("pymc_mfm.client")\n'
        "def f():\n    'pymc_mfm.model is not shipped'\n",
    )
    pkg.check_imports(stage / "src", FAKE_FILES)


# ---- the guard against known defects ------------------------------------------------------------


def test_the_guard_flags_a_wait_that_returns_while_runs_are_active(tmp_path):
    unfixed = stub_stage(tmp_path / "a", UNFIXED_CONDITION, GOOD_DOCSTRING)
    found = pkg.known_defects(unfixed, STUB_FILES)
    assert [number for number, _ in found] == [142]
    assert "Analysis.wait() returns while runs are still active" in found[0][1]
    fixed = stub_stage(tmp_path / "b", FIXED_CONDITION, GOOD_DOCSTRING)
    assert pkg.known_defects(fixed, STUB_FILES) == []


@pytest.mark.parametrize(
    "text",
    [
        "The service moves it later each time you use the dataset.",
        "It can move later when the dataset is used\n    again.",
        "Each time you use a dataset, the expiry is pushed back.",
        "The service extends the deadline whenever the dataset is used.",
        "Using a dataset renews it for another 7 days.",
    ],
)
def test_the_guard_flags_docstrings_that_promise_the_expiry_moves_later(tmp_path, text):
    stage = stub_stage(tmp_path, FIXED_CONDITION, text)
    found = pkg.known_defects(stage, STUB_FILES)
    assert [number for number, _ in found] == [134]
    assert found[0][1].startswith("pymc_mfm/client/__init__.py says the expiry moves later")


def test_the_guard_does_not_flag_the_wording_that_denies_renewal(tmp_path):
    stage = stub_stage(tmp_path, FIXED_CONDITION, GOOD_DOCSTRING + " It never renews it on use.")
    assert pkg.known_defects(stage, STUB_FILES) == []


def test_the_guard_reports_both_defects_when_a_client_has_both(tmp_path):
    stage = stub_stage(tmp_path, UNFIXED_CONDITION, "It can move later when used.")
    assert [number for number, _ in pkg.known_defects(stage, STUB_FILES)] == [142, 134]


def test_the_refusal_names_each_pull_request_and_the_dry_run_flag():
    message = pkg.known_defects_message([(142, "wait"), (134, "docs")])
    assert message == (
        "this commit still has known defects that a release must not carry:\n"
        "  - library PR #142: wait\n"
        "  - library PR #134: docs\n"
        "Merge #134 and #142 first, then build from the new main. (--allow-known-defects "
        "exists for local dry runs only; never publish from such a build.)"
    )


@needs_uv
def test_the_build_refuses_a_defective_client_unless_the_dry_run_flag_is_given(tmp_path):
    lib = fake_library(tmp_path / "lib", "# It can move later when used.\n")
    cfg = fake_cfg()
    with pytest.raises(pkg.PackageError, match=r"(?s)known defects.*PR #134"):
        pkg.build(lib, SHA, "0.1.0", tmp_path / "out", cfg=cfg)
    assert not (tmp_path / "out").exists()  # refused before anything was built
    _, _, allowed = pkg.build(
        lib, SHA, "0.1.0", tmp_path / "out", cfg=cfg, allow_known_defects=True
    )
    assert [number for number, _ in allowed] == [134]


def test_the_publish_workflow_never_passes_a_dry_run_flag_and_gates_publish_on_main():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((ROOT / ".github/workflows/publish-sdk.yml").read_text())
    commands = "\n".join(
        step["run"] for job in data["jobs"].values() for step in job["steps"] if "run" in step
    )
    assert "build_client_package.py" in commands
    assert "--allow-known-defects" not in commands and "--library-sha" not in commands
    publish = data["jobs"]["publish"]
    assert publish["if"] == "${{ inputs.dry_run == false && github.ref == 'refs/heads/main' }}"
    assert publish["environment"] == "pypi" and publish["permissions"] == {"id-token": "write"}
    assert "environment" not in data["jobs"]["build"] and "permissions" not in data["jobs"]["build"]
    assert data["permissions"] == {"contents": "read"}
    assert list(data[True]) == ["workflow_dispatch"]  # YAML reads the key `on` as True
    action = [s for s in publish["steps"] if "pypi-publish" in s.get("uses", "")][0]
    assert action["with"] == {"attestations": False}
    assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}( #.*)?", action["uses"])


# ---- inputs the builder must refuse -------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.1.0", "0.1", "1.0.0rc1", "0.1.0.dev0", "2.3.4.post1"])
def test_a_canonical_public_version_is_accepted(version):
    pkg.check_version(version)


@pytest.mark.parametrize(
    "version", ["0.1.0+abc1234", "v0.1.0", "", "0.1.0 ", "01.0", "0.1.0-rc1", "1.0.0RC1", "latest"]
)
def test_a_version_pypi_would_change_or_reject_is_refused(version):
    with pytest.raises(pkg.PackageError, match="not a canonical public PEP 440 version"):
        pkg.check_version(version)


@pytest.mark.parametrize("sha", ["", "abc", "A" * 40, "a" * 39, "main"])
def test_a_commit_id_that_is_not_40_lowercase_hex_is_refused(tmp_path, sha):
    with pytest.raises(pkg.PackageError, match="40-hex"):
        pkg.build(fake_library(tmp_path), sha, "0.1.0", tmp_path / "out", cfg=fake_cfg())


@needs_uv
def test_a_non_empty_output_directory_is_refused_so_an_old_archive_is_never_mixed_in(tmp_path):
    out = tmp_path / "dist"
    out.mkdir()
    (out / "pymc_mfm-0.0.9-py3-none-any.whl").write_bytes(b"old")
    with pytest.raises(pkg.PackageError, match="is not empty"):
        pkg.build(fake_library(tmp_path / "lib"), SHA, "0.1.0", out, cfg=fake_cfg())
    assert [p.name for p in out.iterdir()] == ["pymc_mfm-0.0.9-py3-none-any.whl"]


@needs_git
def test_a_checkout_with_changed_or_untracked_build_inputs_is_refused(tmp_path):
    def git(*args: str) -> str:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    git("init", "-q")
    for rel in (
        "pyproject.toml",
        "src/pymc_mfm/client/__init__.py",
        "packaging/client/x",
        "README.md",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("v1\n")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    assert pkg.git_commit(tmp_path) == git("rev-parse", "HEAD")
    (tmp_path / "README.md").write_text("changed, not a build input\n")
    assert pkg.git_commit(tmp_path) == git("rev-parse", "HEAD")
    (tmp_path / "src/pymc_mfm/client/__init__.py").write_text("v2\n")
    with pytest.raises(pkg.PackageError, match=r"uncommitted changes:\n M src/pymc_mfm/client"):
        pkg.git_commit(tmp_path)
    git("checkout", "--", ".")
    (tmp_path / "packaging/client/new.py").write_text("x\n")  # untracked
    with pytest.raises(pkg.PackageError, match=r"uncommitted changes:\n\?\? packaging/client/new"):
        pkg.git_commit(tmp_path)
    (tmp_path / "packaging/client/new.py").unlink()
    (tmp_path / "pyproject.toml").write_text("v2\n")
    with pytest.raises(pkg.PackageError, match=r" M pyproject.toml"):
        pkg.git_commit(tmp_path)


@needs_uv
def test_a_shipped_message_that_names_a_missing_extra_stops_the_build(tmp_path):
    lib = fake_library(tmp_path / "lib", 'HINT = "pip install pymc-mfm[local]"\n')
    with pytest.raises(pkg.PackageError, match=r"extras the package does not provide: \['local'\]"):
        pkg.build(lib, SHA, "0.1.0", tmp_path / "out", cfg=fake_cfg())


# ---- generated files, metadata, text ------------------------------------------------------------


def test_the_generated_root_module_gives_the_version_the_client_and_a_clear_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    namespace: dict = {"__name__": "pymc_mfm"}
    exec(compile(pkg.root_init_text(), "root", "exec"), namespace)
    assert namespace["__version__"] == "9.9.9"
    with pytest.raises(ImportError) as error:
        namespace["__getattr__"]("MFM")
    assert str(error.value) == (
        "pymc-mfm 9.9.9 ships the hosted client only; `MFM` is part of the full library, "
        "which is not published. Import from pymc_mfm.client."
    )
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
    lib = tomllib.loads(LIB_PYPROJECT)
    text = pkg.pyproject_text(lib, pkg.config(), "0.1.0", ">=3.11")
    data = tomllib.loads(text)["project"]
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
    languages = [
        c for c in data["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
    ]
    assert languages == [f"Programming Language :: Python :: 3.{n}" for n in (11, 12, 13)]
    assert "License ::" not in text


def test_the_real_dependencies_come_from_this_repositorys_pyproject():
    lib = tomllib.loads((ROOT / "pyproject.toml").read_text())
    data = tomllib.loads(pkg.pyproject_text(lib, pkg.config(), "0.1.0", ">=3.11"))["project"]
    assert data["dependencies"] == lib["project"]["dependencies"]
    assert {d.split(">")[0] for d in data["dependencies"]} == {
        "numpy",
        "pandas",
        "packaging",
        "httpx",
    }
    extras = lib["project"]["optional-dependencies"]
    assert data["optional-dependencies"]["draws"] == extras["draws"]
    assert data["optional-dependencies"]["plot"] == extras["plot"]
    assert data["optional-dependencies"]["client"] == [*extras["draws"], *extras["plot"]]


def test_every_extra_that_a_shipped_message_names_is_found(tmp_path):
    for rel, text in {
        "a.py": 'm = "pip install \\"pymc-mfm[draws]\\""',
        "b.py": "m = 'pymc_mfm[plot]'",
        "c.py": 'm = "pymc-mfm[client, local]"',
    }.items():
        (tmp_path / rel).write_text(text + "\n")
    names = pkg.shipped_extras(tmp_path, ["a.py", "b.py", "c.py"])
    assert names == {"draws", "plot", "client", "local"}
    assert names - set(pkg.config()["extras"]["names"]) == {"local"}  # the build refuses it


def test_the_real_client_names_no_extra_the_package_lacks():
    names = set(pkg.config()["extras"]["names"])
    named = pkg.shipped_extras(ROOT / "src", pkg.config()["files"]["allowlist"])
    assert named and named <= names, named - names


def test_the_page_text_has_the_licence_sentence_with_the_generated_version_and_no_private_name():
    plain = pkg.readme_text(pkg.config(), ">=3.11", "0.2.3")
    assert plain.startswith(
        "# pymc-mfm\n\nPython client for PyMC MFM (marketing foundational models), "
        "the hosted service from PyMC Labs."
    )
    assert "```bash\npip install pymc-mfm\n```" in plain
    assert plain.index("It needs Python 3.11 or newer.") < plain.index("pip install pymc-mfm\n")
    assert "Connect page of the console. That page also\nshows your service URL." in plain
    assert "pilot invitation" not in plain
    assert 'export MFM_API_URL="<your service URL>"\nexport MFM_API_KEY="<your API key>"' in plain
    assert (
        'import os\n\nos.environ["MFM_API_URL"] = "<your service URL>"\n'
        'os.environ["MFM_API_KEY"] = "<your API key>"'
    ) in plain
    assert plain.split("## Licence\n\n")[1].split("\n\n") == [
        "The client code and documentation distributed in pymc-mfm 0.2.3 are licensed under "
        "Apache-2.0. This grant excludes the hosted service, model weights, training and "
        "inference code. Future releases may use different licences. Use of the hosted service "
        "is under its own terms.",
        "Copyright 2026 The PyMC Labs Developers\n",
    ]
    help_line = (
        "If something fails, reply to your invitation. Include the full error text, "
        "the time, and the dataset or analysis id if you have one."
    )
    assert f"## Help\n\n{help_line}\n" in plain
    assert (
        "Use sample or synthetic data only. The service deletes uploaded data 7 days after upload."
    ) in plain
    assert "internal data" not in plain
    assert "{version}" not in plain and "{requires_python}" not in plain
    assert not pkg.FORBIDDEN.search(plain) and "http" not in plain and "@" not in plain


def test_the_licence_file_is_the_unmodified_apache_text():
    assert hashlib.sha256((CLIENT_DIR / "LICENSE").read_bytes()).hexdigest() == APACHE_2_SHA256


def test_the_forbidden_words_catch_the_private_places():
    for bad in (
        "https://github.com/some/repo",
        "raw.githubusercontent.com/x",
        "huggingface.co/some/model",
        "hf.co/x",
        "pymc-labs/pymc-mfm",
        "pymc-mfm" + "-private",
        "x.run." + "app",
        "river" + "-api",
    ):
        assert pkg.FORBIDDEN.search(bad), bad
    assert not pkg.FORBIDDEN.search("pip install pymc-mfm")


def test_the_venv_python_path_is_the_one_of_the_platform():
    path = pkg.venv_python(Path("v"))
    assert path == (Path("v/Scripts/python.exe") if pkg.os.name == "nt" else Path("v/bin/python"))


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
        assert names == {
            "pymc_mfm/__init__.py",
            "pymc_mfm/_build.py",
            *FAKE_FILES,
            "pymc_mfm-0.1.0.dist-info/METADATA",
            "pymc_mfm-0.1.0.dist-info/WHEEL",
            "pymc_mfm-0.1.0.dist-info/RECORD",
            "pymc_mfm-0.1.0.dist-info/licenses/LICENSE",
        }
        init = z.read("pymc_mfm/__init__.py").decode()
        assert init == pkg.root_init_text()
        meta = z.read("pymc_mfm-0.1.0.dist-info/METADATA").decode()
        assert "License-Expression: Apache-2.0" in meta and "License-File: LICENSE" in meta
        assert "Classifier: License" not in meta
        assert {"draws", "plot", "client"} == set(re.findall(r"^Provides-Extra: (.+)$", meta, re.M))
        assert z.read("pymc_mfm-0.1.0.dist-info/licenses/LICENSE") == licence
        build_record: dict = {}
        exec(z.read("pymc_mfm/_build.py").decode(), build_record)
        assert (build_record["LIBRARY_COMMIT"], build_record["KNOWN_DEFECTS"]) == (SHA, ())
        sums = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names}
    with tarfile.open(sdist) as t:
        assert t.extractfile("pymc_mfm-0.1.0/LICENSE").read() == licence
        assert "pymc_mfm-0.1.0/README.md" in t.getnames()
    manifest = pkg.manifest_text(wheel, sdist, SHA, "0.1.0", defects)
    assert manifest.splitlines()[:3] == [
        "pymc-mfm 0.1.0",
        f"library commit {SHA}",
        "known defects: none",
    ]
    wheel_lines = manifest.split("files in pymc_mfm-0.1.0-py3-none-any.whl\n")[1].split("\n\n")[0]
    listed = {line.split()[-1]: line.split()[0] for line in wheel_lines.splitlines()}
    assert listed == sums
    assert (
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.stat().st_size:>7} bytes"
        in manifest
    )
    again = pkg.build(lib, SHA, "0.1.0", tmp_path / "out2", cfg=cfg)
    pkg.check_reproducible(wheel, sdist, again[:2])


def test_two_archives_that_differ_are_reported_as_not_reproducible(tmp_path):
    for name, data in (("a.whl", b"1"), ("a.tar.gz", b"2"), ("b.whl", b"1"), ("b.tar.gz", b"3")):
        (tmp_path / name).write_bytes(data)
    with pytest.raises(pkg.PackageError, match="two builds of a.tar.gz differ"):
        pkg.check_reproducible(
            tmp_path / "a.whl", tmp_path / "a.tar.gz", (tmp_path / "b.whl", tmp_path / "b.tar.gz")
        )


@needs_uv
def test_a_private_name_in_a_shipped_file_stops_the_build(tmp_path):
    lib = fake_library(tmp_path / "lib", "URL = 'https://github.com/some/repo'\n")
    with pytest.raises(pkg.PackageError, match=r"contains 'github.com', which must not be public"):
        pkg.build(lib, SHA, "0.1.0", tmp_path / "out", cfg=fake_cfg())


@needs_uv
def test_the_cli_exits_1_with_the_reason_when_the_allowlist_does_not_match(tmp_path, capsys):
    lib = fake_library(tmp_path / "lib", "from pymc_mfm.model import MFM\n")
    code = pkg.main(["--src", str(lib), "--library-sha", SHA, "--out", str(tmp_path / "out")])
    assert code == 1  # the real allowlist is not in this fake tree
    assert capsys.readouterr().err.startswith("build_client_package: allowlisted files are missing")


# ---- this checkout, built ----------------------------------------------------------------------


@needs_uv
@needs_git
@pytest.mark.slow  # builds twice, installs both archives on 3.11-3.13 (needs PyPI): about 30 s
def test_this_checkout_builds_installs_and_imports_every_client_name(tmp_path):
    out, manifest = tmp_path / "out", tmp_path / "MANIFEST.txt"
    pythons = [arg for p in ("3.11", "3.12", "3.13") for arg in ("--check-python", p)]
    # --allow-known-defects: the guard is tested above; this test must also pass on a commit
    # that still has a known defect.
    command = [sys.executable, str(BUILDER), "--library-sha", SHA, "--out", str(out)]
    command += ["--manifest", str(manifest), "--allow-known-defects", "--check", *pythons]
    done = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "reproducible" in done.stdout and done.stdout.count("every public name imports") == 6
    assert manifest.read_text().startswith("pymc-mfm 0.1.0\n")
