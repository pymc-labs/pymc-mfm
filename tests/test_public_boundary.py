"""Tests for scripts/check_public_boundary.py, using throwaway Git repositories."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import zipfile
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
    write(repo, "big.dat", os.urandom(2 * 1024 * 1024))
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
