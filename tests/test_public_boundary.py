"""Tests for scripts/check_public_boundary.py, using throwaway Git repositories."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_public_boundary.py"

# The sdist ships tests/ but not scripts/; the check runs from a Git checkout.
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="needs a Git checkout")

# Seeds are assembled at run time so this file never matches the rules it tests.
FAKE_AWS_KEY = "AK" + "IA" + "QWERTYUIOPASDFGH"
FAKE_HF_TOKEN = "hf" + "_" + "aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
PRIVATE_LINK = "." + "private"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def commit(repo: Path, message: str = "change") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def write(repo: Path, name: str, data: str | bytes) -> None:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data)
    else:
        target.write_bytes(data)


def check(repo: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-I", str(SCRIPT)]
    if repo is not None:
        command += ["--repo", str(repo)]
    return subprocess.run(command + list(args), capture_output=True, text=True, check=False)


def rules_of(result: subprocess.CompletedProcess[str]) -> set[str]:
    return {line.split()[0] for line in result.stdout.splitlines()}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    write(root, "README.md", "hello\n")
    commit(root, "base")
    return root


# ----------------------------------------------------------- negative controls


def test_clean_repo_passes(repo: Path) -> None:
    write(repo, "src/app.py", "def f():\n    return 1\n")
    commit(repo)
    result = check(repo, "--tree")
    assert result.returncode == 0, result.stdout


def test_tracked_symlink_fires(repo: Path) -> None:
    (repo / "link").symlink_to("README.md")
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "symlink link " in result.stdout


def test_weights_suffix_fires(repo: Path) -> None:
    write(repo, "ab/model.safetensors", b"x")
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "weights-suffix ab/model.safetensors" in result.stdout


def test_large_file_fires(repo: Path) -> None:
    write(repo, "blob.dat", os.urandom(2 * 1024 * 1024))
    commit(repo)
    assert "large-file blob.dat" in check(repo).stdout


def test_embedded_payload_fires_in_json(repo: Path) -> None:
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    write(repo, "data.json", f'{{"blob": "{payload}"}}\n')
    commit(repo)
    assert "embedded-payload data.json:1" in check(repo).stdout


def test_wrapped_base64_lines_fire(repo: Path) -> None:
    row = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9w"
    lines = ",\n".join(f'"{row}\\n"' for _ in range(100))
    write(repo, "wrapped.json", f'{{"lines": [\n{lines}\n]}}\n')
    commit(repo)
    assert "embedded-payload wrapped.json" in check(repo).stdout


def test_secrets_fire_and_are_masked(repo: Path) -> None:
    write(repo, "config.txt", f"key = {FAKE_AWS_KEY}\ntoken = {FAKE_HF_TOKEN}\n")
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "secret config.txt:1 AWS access key AKIA***" in result.stdout
    assert "secret config.txt:2 Hugging Face token hf_a***" in result.stdout
    assert FAKE_AWS_KEY not in result.stdout + result.stderr
    assert FAKE_HF_TOKEN not in result.stdout + result.stderr


def test_service_hostnames_fire_and_are_masked(repo: Path) -> None:
    # Assembled at run time so this file never matches the rule it tests.
    staging = "mfm-staging." + "pymc-labs" + ".com"
    preview = "docs-preview." + "fly" + ".dev"
    write(repo, "docs.yml", f"image: https://{staging}/social.jpg\nsite: {preview}\n")
    write(repo, "ok.md", "See https://www.pymc-labs.com and write to team@pymc-labs.com.\n")
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "secret docs.yml:1 service hostname mfm-***" in result.stdout
    assert "secret docs.yml:2 service hostname docs***" in result.stdout
    assert "ok.md" not in result.stdout
    assert staging not in result.stdout + result.stderr


def test_private_path_reference_fires(repo: Path) -> None:
    write(repo, "run.sh", f"python {PRIVATE_LINK}/x.py\n")
    commit(repo)
    assert "private-path run.sh:1" in check(repo).stdout


def test_private_link_as_tracked_path_fires(repo: Path) -> None:
    write(repo, f"{PRIVATE_LINK}/note.txt", "x\n")
    commit(repo)
    assert "private-path" in check(repo).stdout


def test_deny_file_hits_do_not_echo_pattern(repo: Path, tmp_path: Path) -> None:
    secret_word = "zebrafish" + "-internal"
    write(repo, "notes.md", f"line one\nsee {secret_word} here\n")
    commit(repo)
    deny = tmp_path / "deny.txt"
    deny.write_text("# comment\n\n(?i)ZEBRAFISH-internal\n")
    result = check(repo, "--deny-file", str(deny))
    assert result.returncode == 1
    assert "deny notes.md:2 deny-file line 3" in result.stdout
    assert "zebrafish" not in (result.stdout + result.stderr).lower()


def test_bad_deny_regex_is_usage_error(repo: Path, tmp_path: Path) -> None:
    deny = tmp_path / "deny.txt"
    deny.write_text("(unclosed\n")
    assert check(repo, "--deny-file", str(deny)).returncode == 2


def test_usage_errors(repo: Path, tmp_path: Path) -> None:
    assert check(repo, "--allow", str(tmp_path / "missing.toml")).returncode == 2
    assert check(repo, "--range", "nodots").returncode == 2
    assert check(repo, "--tree", "--staged").returncode == 2


# ------------------------------------------------------------------ git modes


def test_staged_mode_reads_the_index(repo: Path) -> None:
    write(repo, "m.pt", b"x")
    git(repo, "add", "m.pt")
    assert check(repo, "--tree").returncode == 0
    assert "weights-suffix m.pt" in check(repo, "--staged").stdout


def test_working_tree_is_not_read(repo: Path) -> None:
    write(repo, "README.md", f"python {PRIVATE_LINK}/x.py\n")
    assert check(repo, "--tree").returncode == 0


def test_range_catches_weight_added_then_deleted(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "model.safetensors", b"x")
    commit(repo, "add weights")
    git(repo, "rm", "-q", "model.safetensors")
    head = commit(repo, "remove weights")
    assert check(repo, "--tree").returncode == 0
    result = check(repo, "--range", f"{base}..{head}")
    assert result.returncode == 1
    assert "weights-suffix model.safetensors" in result.stdout


def test_range_ignores_blobs_outside_the_range(repo: Path) -> None:
    write(repo, "old.pt", b"x")
    base = commit(repo, "old weights")
    write(repo, "new.py", "x = 1\n")
    head = commit(repo, "code")
    assert check(repo, "--range", f"{base}..{head}").returncode == 0


def test_range_scans_commit_messages_with_deny_rules(repo: Path, tmp_path: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "a.py", "x = 1\n")
    head = commit(repo, "touches zebrafish" + "-internal cluster")
    deny = tmp_path / "deny.txt"
    deny.write_text("zebrafish-internal\n")
    result = check(repo, "--range", f"{base}..{head}", "--deny-file", str(deny))
    assert result.returncode == 1
    assert f"deny commit:{head[:7]} deny-file line 1" in result.stdout
    assert "zebrafish" not in result.stdout


def test_commit_message_secret_is_found_in_all_refs(repo: Path) -> None:
    write(repo, "a.py", "x = 1\n")
    head = commit(repo, f"oops {FAKE_AWS_KEY}")
    result = check(repo, "--all-refs")
    assert f"secret commit:{head[:7]} AWS access key AKIA***" in result.stdout


def test_ordinary_commit_message_passes(repo: Path, tmp_path: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "a.py", "x = 1\n")
    head = commit(repo, "Add the .private link docs\n\nNo paths here.")
    deny = tmp_path / "deny.txt"
    deny.write_text("zebrafish-internal\n")
    assert check(repo, "--range", f"{base}..{head}", "--deny-file", str(deny)).returncode == 0


def test_all_refs_finds_blob_deleted_on_a_side_branch(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "side")
    write(repo, "ab/model.safetensors", b"x")
    commit(repo, "weights")
    git(repo, "rm", "-q", "-r", "ab")
    commit(repo, "drop")
    git(repo, "checkout", "-q", "main")
    assert check(repo, "--tree").returncode == 0
    result = check(repo, "--all-refs")
    assert result.returncode == 1
    assert "weights-suffix ab/model.safetensors" in result.stdout
    assert "refs/heads/side" in result.stdout


# -------------------------------------------------------------------- archives


def make_tar(path: Path, mode: str) -> None:
    with tarfile.open(path, mode) as archive:
        data = b"weights"
        info = tarfile.TarInfo("pkg-1/model.pt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("pkg-1/alias")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        archive.addfile(link)


def test_tar_gz_archive_members_are_checked(tmp_path: Path) -> None:
    archive = tmp_path / "pkg-1.tar.gz"
    make_tar(archive, "w:gz")
    result = check(None, "--archive", str(archive))
    assert result.returncode == 1
    assert "weights-suffix pkg-1.tar.gz!pkg-1/model.pt" in result.stdout
    assert "symlink pkg-1.tar.gz!pkg-1/alias" in result.stdout


def test_zip_archive_members_are_checked(tmp_path: Path) -> None:
    archive = tmp_path / "pkg-1-py3-none-any.whl"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pkg/model.pt", b"weights")
        link = zipfile.ZipInfo("pkg/alias")
        link.external_attr = 0o120777 << 16
        bundle.writestr(link, "../outside")
        bundle.writestr("pkg/ok.py", "x = 1\n")
    result = check(None, "--archive", str(archive))
    assert result.returncode == 1
    assert "weights-suffix" in result.stdout and "model.pt" in result.stdout
    assert "symlink" in result.stdout and "alias" in result.stdout
    assert "ok.py" not in result.stdout


def test_clean_archive_passes(tmp_path: Path) -> None:
    archive = tmp_path / "clean.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pkg/ok.py", "x = 1\n")
    assert check(None, "--archive", str(archive)).returncode == 0


# ------------------------------------------------------------ legitimate cases


def test_legitimate_content_passes(repo: Path) -> None:
    write(repo, "src/a.py", "token = 'hf_notForLogsOrArtifacts'\n")
    write(repo, "docs.md", f"Use the {PRIVATE_LINK} link when you work in a paired checkout.\n")
    write(repo, "data.json", '{"hash": "' + "ab12" * 16 + '"}\n')
    write(repo, "sums.txt", "\n".join(hashlib.sha256(bytes([n])).hexdigest() for n in range(80)))
    commit(repo)
    result = check(repo)
    assert result.returncode == 0, result.stdout


def write_allow(repo: Path, body: str) -> Path:
    path = repo / "allow.toml"
    path.write_text(body)
    return path


def allow_entry(sha: str, extra: str = "") -> str:
    return (
        '[[allow]]\npath = "model.pt"\nrules = ["weights-suffix"]\n'
        f'reason = "test fixture"\nsha256 = "{sha}"\n{extra}'
    )


def test_allowlisted_file_passes(repo: Path) -> None:
    write(repo, "model.pt", b"tiny")
    commit(repo)
    allow = write_allow(repo, allow_entry(hashlib.sha256(b"tiny").hexdigest()))
    result = check(repo, "--allow", str(allow))
    assert result.returncode == 0, result.stdout
    assert "unused" not in result.stderr


def test_changed_content_voids_allowlist_entry(repo: Path) -> None:
    write(repo, "model.pt", b"tiny")
    commit(repo)
    allow = write_allow(repo, allow_entry(hashlib.sha256(b"other").hexdigest()))
    result = check(repo, "--allow", str(allow))
    assert result.returncode == 1
    assert "sha256 mismatch" in result.stdout
    assert "unused allowlist entry model.pt" in result.stderr


def test_unused_entry_warns_without_failing(repo: Path) -> None:
    allow = write_allow(repo, allow_entry("0" * 64))
    result = check(repo, "--allow", str(allow))
    assert result.returncode == 0
    assert "unused allowlist entry model.pt" in result.stderr


def test_max_bytes_limits_large_file_entry(repo: Path) -> None:
    write(repo, "big.dat", "x = 1\n" * 400_000)
    commit(repo)
    entry = '[[allow]]\npath = "big.dat"\nrules = ["large-file"]\nreason = "r"\nmax_bytes = {}\n'
    assert check(repo, "--allow", str(write_allow(repo, entry.format(3_000_000)))).returncode == 0
    assert check(repo, "--allow", str(write_allow(repo, entry.format(1_500_000)))).returncode == 1


def test_pending_entry_fails_only_with_release(repo: Path) -> None:
    write(repo, "model.pt", b"tiny")
    commit(repo)
    sha = hashlib.sha256(b"tiny").hexdigest()
    allow = write_allow(repo, allow_entry(sha, 'pending = "confirm provenance"\n'))
    assert check(repo, "--allow", str(allow)).returncode == 0
    result = check(repo, "--allow", str(allow), "--release")
    assert result.returncode == 1
    assert "pending model.pt confirm provenance" in result.stdout


def test_bad_allowlist_entry_is_usage_error(repo: Path) -> None:
    allow = write_allow(repo, '[[allow]]\npath = "x"\nrules = ["nope"]\nreason = "r"\n')
    assert check(repo, "--allow", str(allow)).returncode == 2


# ------------------------------------------------------------ merges (review P1)


def merge_with_extra_file(repo: Path, name: str, data: str | bytes) -> str:
    """Create a merge commit whose tree adds `name`, a change neither parent has."""
    git(repo, "checkout", "-q", "-b", "side")
    write(repo, "side.txt", "side\n")
    commit(repo, "side work")
    git(repo, "checkout", "-q", "main")
    write(repo, "main.txt", "main\n")
    commit(repo, "main work")
    git(repo, "merge", "-q", "--no-ff", "--no-commit", "side")
    write(repo, name, data)
    return commit(repo, "merge")


def test_range_catches_token_added_by_merge_then_deleted(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    merge_with_extra_file(repo, "evil.txt", f"key = {FAKE_AWS_KEY}\n")
    git(repo, "rm", "-q", "evil.txt")
    head = commit(repo, "remove it")
    assert check(repo, "--tree").returncode == 0
    result = check(repo, "--range", f"{base}..{head}")
    assert result.returncode == 1
    assert "secret evil.txt:1 AWS access key AKIA***" in result.stdout


def test_range_with_clean_merge_passes(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    head = merge_with_extra_file(repo, "fine.txt", "ok\n")
    assert check(repo, "--range", f"{base}..{head}").returncode == 0


def test_all_refs_catches_merge_only_symlink_of_known_blob(repo: Path) -> None:
    write(repo, "a.txt", "README.md")
    commit(repo, "regular file with the same bytes as a link target")
    git(repo, "checkout", "-q", "-b", "side")
    write(repo, "side.txt", "side\n")
    commit(repo, "side work")
    git(repo, "checkout", "-q", "main")
    write(repo, "main.txt", "main\n")
    commit(repo, "main work")
    git(repo, "merge", "-q", "--no-ff", "--no-commit", "side")
    (repo / "a.txt").unlink()
    (repo / "a.txt").symlink_to("README.md")
    commit(repo, "merge")
    result = check(repo, "--all-refs")
    assert "symlink a.txt " in result.stdout


# ------------------------------------------------- private link component (P1)


def test_nested_private_component_fires_in_tree_and_sdist_tar(repo: Path, tmp_path: Path) -> None:
    write(repo, f"tests/{PRIVATE_LINK}/customer.csv", "a,b\n")
    commit(repo)
    assert f"private-path tests/{PRIVATE_LINK}/customer.csv" in check(repo).stdout
    archive = tmp_path / "pkg-1.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        data = b"a,b\n"
        info = tarfile.TarInfo(f"pkg-1/tests/{PRIVATE_LINK}/customer.csv")
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    result = check(None, "--archive", str(archive))
    assert result.returncode == 1
    assert f"private-path pkg-1.tar.gz!pkg-1/tests/{PRIVATE_LINK}/customer.csv" in result.stdout


def test_similar_directory_names_pass(repo: Path) -> None:
    write(repo, "private/notes.txt", "x\n")
    write(repo, f"{PRIVATE_LINK}ish/notes.txt", "x\n")
    commit(repo)
    assert check(repo).returncode == 0


# ------------------------------------------------------ wide text and binary (P1)


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le"])
def test_wide_text_is_decoded_for_secret_private_path_and_deny(
    repo: Path, tmp_path: Path, codec: str
) -> None:
    text = f"key = {FAKE_AWS_KEY}\nrun {PRIVATE_LINK}/x.py\nzebrafish-internal\n"
    write(repo, "wide.txt", text.encode(codec))
    commit(repo)
    deny = tmp_path / "deny.txt"
    deny.write_text("zebrafish-internal\n")
    result = check(repo, "--deny-file", str(deny))
    assert "secret wide.txt:1 AWS access key AKIA***" in result.stdout
    assert "private-path wide.txt:2" in result.stdout
    assert "deny wide.txt:3 deny-file line 1" in result.stdout
    assert "binary" not in result.stdout


def test_clean_wide_text_passes(repo: Path) -> None:
    write(repo, "wide.txt", "hello world\n".encode("utf-16"))
    write(repo, "wide-le.txt", "hello world\n".encode("utf-16-le"))
    commit(repo)
    assert check(repo).returncode == 0


def test_opaque_binary_needs_an_allowlist_entry(repo: Path) -> None:
    payload = bytes([0, 1, 2, 3, 255, 0, 7, 9] * 8)
    write(repo, "logo.bin2", payload)
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "binary logo.bin2 not plain UTF-8" in result.stdout
    sha = hashlib.sha256(payload).hexdigest()
    entry = (
        f'[[allow]]\npath = "logo.bin2"\nrules = ["binary"]\nreason = "image"\nsha256 = "{sha}"\n'
    )
    assert check(repo, "--allow", str(write_allow(repo, entry))).returncode == 0
    changed = entry.replace(sha, "1" * 64)
    assert check(repo, "--allow", str(write_allow(repo, changed))).returncode == 1


# ------------------------------------------------- allowlist source (P1)

DEFAULT_ALLOW_PATH = ".github/public-boundary-allow.toml"


def default_allow_text() -> str:
    return allow_entry(hashlib.sha256(b"tiny").hexdigest())


def test_working_tree_only_allowlist_does_not_suppress_staged_finding(repo: Path) -> None:
    write(repo, "model.pt", b"tiny")
    git(repo, "add", "model.pt")
    write(repo, DEFAULT_ALLOW_PATH, default_allow_text())
    assert check(repo, "--staged").returncode == 1
    git(repo, "add", DEFAULT_ALLOW_PATH)
    assert check(repo, "--staged").returncode == 0


def test_tree_reads_allowlist_from_head(repo: Path) -> None:
    write(repo, "model.pt", b"tiny")
    commit(repo)
    write(repo, DEFAULT_ALLOW_PATH, default_allow_text())
    assert check(repo, "--tree").returncode == 1
    commit(repo, "add allowlist")
    assert check(repo, "--tree").returncode == 0
    (repo / DEFAULT_ALLOW_PATH).write_text("")
    assert check(repo, "--tree").returncode == 0


def test_range_reads_allowlist_from_range_end(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "model.pt", b"tiny")
    head = commit(repo)
    write(repo, DEFAULT_ALLOW_PATH, default_allow_text())
    assert check(repo, "--range", f"{base}..{head}").returncode == 1
    head = commit(repo, "add allowlist")
    assert check(repo, "--range", f"{base}..{head}").returncode == 0


# ------------------------------------------------------------------- real repo


def test_real_repository_tree_is_clean() -> None:
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or Path(probe.stdout.strip()).resolve() != REPO_ROOT:
        pytest.skip("not a git checkout")
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--tree"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_archive_mode_outside_repo_uses_allowlist_beside_script(tmp_path):
    """--archive run from a non-repo directory still applies the repo's allowlist."""
    notebook = "examples/river_1_quickstart.ipynb"
    if not (REPO_ROOT / notebook).is_file():
        pytest.skip("notebook not present")
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(REPO_ROOT / notebook, notebook)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--archive", str(archive)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_binary_without_nul_bytes_needs_an_allowlist_entry(repo: Path) -> None:
    """Invalid UTF-8 without NUL bytes is not readable text."""
    write(repo, "blob.dat", bytes(range(128, 256)) * 8)
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "binary blob.dat" in result.stdout


def test_ascii_token_after_wide_header_is_still_found(repo: Path) -> None:
    """Decoding wide text must not hide a token present in the raw bytes."""
    header = "log header line\n".encode("utf-16-le")
    write(repo, "mixed.log", header + ("\n" + FAKE_HF_TOKEN + "\n").encode())
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "secret mixed.log" in result.stdout


def test_latin_text_in_utf8_passes(repo: Path) -> None:
    write(repo, "notes.md", "Café, naïve, Zürich: résumé\n".encode())
    commit(repo)
    assert check(repo).returncode == 0


@pytest.mark.parametrize(
    "payload",
    [b"\x0a\x0bAlice Smith\x10\x00", "\x0bAlice\x01".encode("utf-16")],
    ids=["utf8-control", "utf16-control"],
)
def test_binary_records_with_control_bytes_need_an_entry(repo: Path, payload: bytes) -> None:
    write(repo, "record.pb", payload)
    commit(repo)
    result = check(repo)
    assert result.returncode == 1
    assert "binary record.pb" in result.stdout


def test_text_with_tabs_and_ansi_colours_passes(repo: Path) -> None:
    write(repo, "log.txt", "a\tb\r\n\x1b[31mred\x1b[0m\f\n")
    commit(repo)
    assert check(repo).returncode == 0


# ------------------------------------------------------- gitlinks and submodules

POINTED_SHA = "0123456789abcdef0123456789abcdef01234567"


def stage_gitlink(repo: Path, path: str = "sub") -> None:
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{POINTED_SHA},{path}")


def commit_index(repo: Path, message: str = "gitlink") -> str:
    """Commit the index as it is; `commit()` would run `git add -A` and drop the gitlink."""
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def test_staged_gitlink_fires_without_full_sha(repo: Path) -> None:
    stage_gitlink(repo)
    result = check(repo, "--staged")
    assert result.returncode == 1
    assert "gitlink sub " in result.stdout
    assert POINTED_SHA[8:] not in result.stdout


def test_committed_gitlink_fires_in_tree_range_and_all_refs(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    stage_gitlink(repo)
    head = commit_index(repo)
    for args in (("--tree",), ("--range", f"{base}..{head}"), ("--all-refs",)):
        result = check(repo, *args)
        assert result.returncode == 1, args
        assert "gitlink sub " in result.stdout, args


def test_gitlink_found_in_all_refs_after_it_is_removed(repo: Path) -> None:
    stage_gitlink(repo)
    commit_index(repo)
    git(repo, "rm", "-q", "--cached", "sub")
    commit_index(repo, "remove gitlink")
    assert check(repo, "--tree").returncode == 0
    assert "gitlink sub " in check(repo, "--all-refs").stdout


def test_gitmodules_file_fires_at_any_depth(repo: Path) -> None:
    write(repo, ".gitmodules", '[submodule "x"]\n\tpath = x\n')
    write(repo, "deep/dir/.gitmodules", "\n")
    commit(repo)
    result = check(repo, "--tree")
    assert result.returncode == 1
    assert "gitlink .gitmodules " in result.stdout
    assert "gitlink deep/dir/.gitmodules " in result.stdout


def test_gitmodules_in_archive_fires(tmp_path: Path) -> None:
    archive = tmp_path / "pkg-1.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("pkg-1/vendor/.gitmodules")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"\n"))
    result = check(None, "--archive", str(archive))
    assert result.returncode == 1
    assert "gitlink pkg-1.tar.gz!pkg-1/vendor/.gitmodules" in result.stdout


def test_file_named_like_gitmodules_passes(repo: Path) -> None:
    write(repo, "docs/gitmodules.md", "notes on submodules\n")
    write(repo, "my.gitmodules.txt", "x\n")
    commit(repo)
    result = check(repo, "--tree")
    assert result.returncode == 0, result.stdout


def test_gitlink_rule_can_be_allowlisted(repo: Path) -> None:
    stage_gitlink(repo)
    commit_index(repo)
    allow = write_allow(repo, '[[allow]]\npath = "sub"\nrules = ["gitlink"]\nreason = "r"\n')
    result = check(repo, "--allow", str(allow))
    assert result.returncode == 0, result.stdout


# ------------------------------------------------------------------- notebooks


def png_chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body).to_bytes(4, "big")
    return len(body).to_bytes(4, "big") + kind + body + crc


def make_png(*extra: tuple[bytes, bytes], size: int = 64) -> bytes:
    """A valid size x 1 grey PNG with random pixels and optional extra chunks before the data."""
    header = size.to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    pixels = zlib.compress(b"\0" + os.urandom(size))
    chunks = [(b"IHDR", header), *extra, (b"IDAT", pixels), (b"IEND", b"")]
    return b"\x89PNG\r\n\x1a\n" + b"".join(png_chunk(kind, body) for kind, body in chunks)


def notebook(outputs: list[dict], cell_type: str = "code", **cell_extra: object) -> str:
    cell = {"cell_type": cell_type, "metadata": {}, "source": ["x = 1\n"], **cell_extra}
    if cell_type == "code":
        cell |= {"execution_count": 1, "outputs": outputs}
    document = {"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(document, indent=1)


def image_output(mime: str, data: bytes | str, **more: object) -> dict:
    value = data if isinstance(data, str) else base64.b64encode(data).decode()
    return {"output_type": "display_data", "metadata": {}, "data": {mime: value, **more}}


def check_notebook_file(repo: Path, text: str, *args: str) -> subprocess.CompletedProcess[str]:
    write(repo, "nb.ipynb", text)
    commit(repo)
    return check(repo, *args)


def check_pinned_notebook(
    repo: Path, text: str, sha: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Check a notebook against an allowlist entry that pins `sha` (default: its own bytes)."""
    write(repo, "nb.ipynb", text)
    commit(repo)
    digest = sha or hashlib.sha256(text.encode()).hexdigest()
    entry = (
        '[[allow]]\npath = "nb.ipynb"\nrules = ["embedded-payload"]\n'
        f'reason = "test fixture"\nsha256 = "{digest}"\n'
    )
    return check(repo, "--allow", str(write_allow(repo, entry)))


def test_notebook_with_real_png_output_passes(repo: Path) -> None:
    text = notebook(
        [image_output("image/png", make_png(size=6000), **{"text/plain": ["<Figure>"]})]
    )
    result = check_pinned_notebook(repo, text)
    assert result.returncode == 0, result.stdout


def test_notebook_png_split_into_lines_passes(repo: Path) -> None:
    encoded = base64.b64encode(make_png(size=6000)).decode()
    lines = [encoded[i : i + 76] + "\n" for i in range(0, len(encoded), 76)]
    document = json.loads(notebook([image_output("image/png", "")]))
    document["cells"][0]["outputs"][0]["data"]["image/png"] = lines
    result = check_pinned_notebook(repo, json.dumps(document))
    assert result.returncode == 0, result.stdout


def test_valid_image_notebook_without_pin_fires(repo: Path) -> None:
    text = notebook([image_output("image/png", make_png(size=600))] * 2)
    result = check_notebook_file(repo, text)
    assert result.returncode == 1
    assert "embedded-payload nb.ipynb 2 image outputs: pin reviewed bytes" in result.stdout


def test_notebook_entry_without_sha_does_not_cover_the_pin(repo: Path) -> None:
    write(repo, "nb.ipynb", notebook([image_output("image/png", make_png())]))
    commit(repo)
    entry = '[[allow]]\npath = "nb.ipynb"\nrules = ["embedded-payload"]\nreason = "test"\n'
    result = check(repo, "--allow", str(write_allow(repo, entry)))
    assert result.returncode == 1
    assert "needs an allowlist sha256" in result.stdout


def test_notebook_pin_voids_when_bytes_change(repo: Path) -> None:
    pinned = hashlib.sha256(notebook([image_output("image/png", make_png())]).encode())
    changed = notebook([image_output("image/png", make_png())])
    result = check_pinned_notebook(repo, changed, pinned.hexdigest())
    assert result.returncode == 1
    assert "sha256 mismatch" in result.stdout


def test_pinned_notebook_still_reports_malformed_images(repo: Path) -> None:
    text = notebook([image_output("image/png", make_png() + os.urandom(50))])
    result = check_pinned_notebook(repo, text)
    assert result.returncode == 1
    assert "malformed PNG" in result.stdout
    assert "cannot be allowlisted" in result.stdout


def test_pin_text_in_notebook_metadata_cannot_unlock_a_payload(repo: Path) -> None:
    """Only the structured image-pin finding is coverable; diagnostic text never decides."""
    document = json.loads(notebook([image_output("image/png", make_png())]))
    payload = base64.b64encode(os.urandom(6000)).decode()
    document["metadata"]["image outputs: pin reviewed bytes with sha256 in the allowlist"] = payload
    result = check_pinned_notebook(repo, json.dumps(document))
    assert result.returncode == 1, result.stdout
    assert "cannot be allowlisted" in result.stdout


def test_pinned_notebook_passes_in_history_scans(repo: Path) -> None:
    """The image pin must survive the commit context added by --range and --all-refs."""
    text = notebook([image_output("image/png", make_png())])
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "nb.ipynb", text)
    commit(repo)
    digest = hashlib.sha256(text.encode()).hexdigest()
    entry = (
        '[[allow]]\npath = "nb.ipynb"\nrules = ["embedded-payload"]\n'
        f'reason = "test fixture"\nsha256 = "{digest}"\n'
    )
    allow = str(write_allow(repo, entry))
    for mode in (["--range", f"{base}..HEAD"], ["--all-refs"]):
        result = check(repo, *mode, "--allow", allow)
        assert result.returncode == 0, (mode, result.stdout)


def test_notebook_jpeg_magic_with_junk_fires(repo: Path) -> None:
    text = notebook([image_output("image/jpeg", b"\xff\xd8\xff" + os.urandom(9600))])
    assert "malformed JPEG" in check_pinned_notebook(repo, text).stdout


def test_notebook_gif_magic_with_junk_fires(repo: Path) -> None:
    text = notebook([image_output("image/gif", b"GIF89a" + os.urandom(9600))])
    assert "malformed GIF" in check_pinned_notebook(repo, text).stdout


def make_jpeg() -> bytes:
    frame = b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    scan = b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\x12\xff\x00\x34"
    return b"\xff\xd8" + b"\xff\xe0\x00\x04hi" + frame + scan + b"\xff\xd9"


def make_gif() -> bytes:
    header = b"GIF89a" + b"\x01\x00\x01\x00\x80\x00\x00" + bytes(6)
    frame = b"\x2c" + bytes(4) + b"\x01\x00\x01\x00\x00" + b"\x02\x02\x44\x01\x00"
    return header + frame + b"\x3b"


def test_notebook_minimal_jpeg_and_gif_pass_when_pinned(repo: Path) -> None:
    text = notebook(
        [image_output("image/jpeg", make_jpeg()), image_output("image/gif", make_gif())]
    )
    result = check_pinned_notebook(repo, text)
    assert result.returncode == 0, result.stdout


def test_notebook_jpeg_without_eoi_or_with_tail_fires(repo: Path) -> None:
    for image in (make_jpeg()[:-2], make_jpeg() + b"tail", make_gif()[:-1], make_gif() + b"x"):
        mime = "image/jpeg" if image[:2] == b"\xff\xd8" else "image/gif"
        text = notebook([image_output(mime, image)])
        assert "malformed" in check_pinned_notebook(repo, text).stdout


def test_notebook_png_without_ihdr_fires(repo: Path) -> None:
    body = zlib.compress(os.urandom(300))
    chunks = png_chunk(b"IDAT", body) + png_chunk(b"IEND", b"")
    text = notebook([image_output("image/png", b"\x89PNG\r\n\x1a\n" + chunks)])
    result = check_pinned_notebook(repo, text)
    assert "first chunk is not IHDR" in result.stdout


def test_notebook_png_idat_size_mismatch_fires(repo: Path) -> None:
    header = (10).to_bytes(4, "big") + (10).to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    idat = zlib.compress(b"\0" + os.urandom(500))
    parts = [(b"IHDR", header), (b"IDAT", idat), (b"IEND", b"")]
    image = b"\x89PNG\r\n\x1a\n" + b"".join(png_chunk(k, b) for k, b in parts)
    text = notebook([image_output("image/png", image)])
    assert "IDAT size does not match IHDR" in check_pinned_notebook(repo, text).stdout


def test_notebook_interlaced_png_fires(repo: Path) -> None:
    header = (4).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 0, 0, 0, 1])
    idat = zlib.compress(b"\0" + bytes(4))
    parts = [(b"IHDR", header), (b"IDAT", idat), (b"IEND", b"")]
    image = b"\x89PNG\r\n\x1a\n" + b"".join(png_chunk(k, b) for k, b in parts)
    text = notebook([image_output("image/png", image)])
    assert "interlaced PNG" in check_pinned_notebook(repo, text).stdout


def test_real_notebook_passes_with_its_pin(tmp_path: Path) -> None:
    notebook_name = "examples/river_1_quickstart.ipynb"
    allow = REPO_ROOT / ".github" / "public-boundary-allow.toml"
    if not (REPO_ROOT / notebook_name).is_file() or not allow.is_file():
        pytest.skip("notebook not present")
    archive = tmp_path / "nb.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(REPO_ROOT / notebook_name, notebook_name)
    result = check(None, "--archive", str(archive), "--allow", str(allow))
    assert result.returncode == 0, result.stdout


def test_notebook_non_image_bytes_in_image_field_fire(repo: Path) -> None:
    text = notebook([image_output("image/png", bytes(range(256)) * 40)])
    result = check_notebook_file(repo, text)
    assert result.returncode == 1
    assert "embedded-payload nb.ipynb decoded data is not a image/png file" in result.stdout


def test_notebook_oversize_image_fires(repo: Path) -> None:
    text = notebook([image_output("image/png", make_png(size=3 * 1024 * 1024))])
    assert "limit 2097152" in check_notebook_file(repo, text).stdout


def test_notebook_png_with_trailing_data_fires(repo: Path) -> None:
    text = notebook([image_output("image/png", make_png() + os.urandom(5000))])
    assert "malformed PNG" in check_notebook_file(repo, text).stdout


def test_notebook_png_with_unknown_chunk_fires(repo: Path) -> None:
    text = notebook([image_output("image/png", make_png((b"prVt", os.urandom(300))))])
    assert "unknown PNG chunk" in check_notebook_file(repo, text).stdout


def test_notebook_base64_in_text_output_fires(repo: Path) -> None:
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    text = notebook([image_output("text/plain", payload)])
    result = check_notebook_file(repo, text)
    assert result.returncode == 1
    assert "embedded-payload nb.ipynb base64 run" in result.stdout


def test_notebook_base64_in_markdown_cell_fires(repo: Path) -> None:
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    text = notebook([], cell_type="markdown")
    document = json.loads(text)
    document["cells"][0]["source"] = [payload]
    assert "embedded-payload" in check_notebook_file(repo, json.dumps(document)).stdout


def test_notebook_png_in_markdown_data_slot_fires(repo: Path) -> None:
    document = json.loads(notebook([image_output("image/png", make_png(size=6000))]))
    document["cells"][0]["cell_type"] = "markdown"
    result = check_notebook_file(repo, json.dumps(document))
    assert "embedded-payload" in result.stdout


def test_notebook_attachment_fires(repo: Path) -> None:
    attachments = {"a.png": {"image/png": base64.b64encode(make_png()).decode()}}
    text = notebook([], cell_type="markdown", attachments=attachments)
    result = check_notebook_file(repo, text)
    assert result.returncode == 1
    assert "embedded-payload nb.ipynb cell attachments" in result.stdout


def test_notebook_svg_output_is_scanned_as_text(repo: Path) -> None:
    svg = f"<svg><text>{FAKE_AWS_KEY}</text></svg>"
    result = check_notebook_file(repo, notebook([image_output("image/svg+xml", svg)]))
    assert "secret nb.ipynb" in result.stdout


def test_notebook_png_text_chunk_is_scanned_with_deny_rules(repo: Path, tmp_path: Path) -> None:
    word = "zebrafish" + "-internal"
    chunks = [
        (b"tEXt", b"Source\0/srv/" + word.encode()),
        (b"zTXt", b"Comment\0\0" + zlib.compress(f"see {PRIVATE_LINK}/x".encode())),
        (b"iTXt", b"Title\0\0\0\0\0" + FAKE_AWS_KEY.encode()),
    ]
    text = notebook([image_output("image/png", make_png(*chunks))])
    deny = tmp_path / "deny.txt"
    deny.write_text("zebrafish-internal\n")
    write(repo, "nb.ipynb", text)
    commit(repo)
    assert check(repo).stdout.count("secret nb.ipynb") == 1
    result = check(repo, "--deny-file", str(deny))
    assert "deny nb.ipynb" in result.stdout
    assert "private-path nb.ipynb" in result.stdout
    assert "zebrafish" not in (result.stdout + result.stderr)


def test_notebook_escaped_text_is_still_scanned(repo: Path) -> None:
    escaped = "".join(f"\\u{ord(c):04x}" for c in FAKE_AWS_KEY)
    text = notebook([]).replace('"x = 1\\n"', f'"{escaped}"')
    assert "secret nb.ipynb" in check_notebook_file(repo, text).stdout


@pytest.mark.parametrize("text", ["{not json", '{"nbformat": 3, "cells": []}', "[]"])
def test_malformed_notebook_fires(repo: Path, text: str) -> None:
    result = check_notebook_file(repo, text)
    assert result.returncode == 1
    assert "embedded-payload nb.ipynb unparseable notebook" in result.stdout


def test_non_notebook_json_with_image_keeps_payload_rule(repo: Path) -> None:
    text = notebook([image_output("image/png", make_png(size=6000))])
    write(repo, "plot.json", text)
    commit(repo)
    assert "embedded-payload plot.json" in check(repo).stdout


def test_notebook_in_archive_is_checked(tmp_path: Path) -> None:
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    text = notebook([image_output("text/plain", payload)])
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pkg/nb.ipynb", text)
    assert "embedded-payload" in check(None, "--archive", str(archive)).stdout


# --------------------------------------------------------------- foreign history


def add_unrelated_root_merge(repo: Path) -> str:
    git(repo, "checkout", "-q", "--orphan", "foreign")
    write(repo, "other.txt", "from elsewhere\n")
    commit(repo, "foreign root")
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-q", "-b", "feature")
    git(repo, "merge", "-q", "--allow-unrelated-histories", "-m", "merge", "foreign")
    return git(repo, "rev-parse", "HEAD")


def test_range_with_unrelated_root_fires(repo: Path) -> None:
    base = git(repo, "rev-parse", "main")
    head = add_unrelated_root_merge(repo)
    result = check(repo, "--range", f"{base}..{head}")
    assert result.returncode == 1
    assert "foreign-history commit:" in result.stdout


def test_range_with_whole_new_history_fires(repo: Path) -> None:
    base = git(repo, "rev-parse", "main")
    git(repo, "checkout", "-q", "--orphan", "fresh")
    write(repo, "a.txt", "x\n")
    head = commit(repo, "new root")
    assert "foreign-history" in check(repo, "--range", f"{base}..{head}").stdout


def test_allowed_root_accepts_listed_root_only(repo: Path) -> None:
    base = git(repo, "rev-parse", "main")
    git(repo, "checkout", "-q", "--orphan", "fresh")
    write(repo, "a.txt", "x\n")
    head = commit(repo, "new root")
    root = git(repo, "rev-list", "--max-parents=0", head)
    result = check(repo, "--range", f"{base}..{head}", "--allowed-root", root)
    assert result.returncode == 0, result.stdout
    other = check(repo, "--range", f"{base}..{head}", "--allowed-root", "0" * 40)
    assert "foreign-history" in other.stdout


def test_allowed_root_needs_range_and_full_sha(repo: Path) -> None:
    assert check(repo, "--tree", "--allowed-root", "0" * 40).returncode == 2
    assert check(repo, "--range", "HEAD..HEAD", "--allowed-root", "abc").returncode == 2


def test_range_on_ordinary_branch_has_no_foreign_history(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "a.txt", "x\n")
    head = commit(repo)
    assert check(repo, "--range", f"{base}..{head}").returncode == 0


def test_existing_second_root_in_base_does_not_fire(repo: Path) -> None:
    add_unrelated_root_merge(repo)
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--ff-only", "feature")
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "a.txt", "x\n")
    head = commit(repo)
    assert check(repo, "--range", f"{base}..{head}").returncode == 0
