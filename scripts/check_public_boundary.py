#!/usr/bin/env python3
"""Check that content headed for the public repo carries nothing that belongs in private.

Stdlib only, no network. Reads what Git tracks (blobs through git plumbing), never the
working tree. Only generic rules live here; internal names come in through --deny-file.

Modes (pick one):
  --tree          tracked files at HEAD (default)
  --staged        the index
  --range A..B    every blob added or modified in the commits A..B, plus the net diff,
                  and a check that A..B brings no new root commit (foreign history)
  --archive PATH  members of wheels, zips and tar(.gz) files
  --all-refs      every blob reachable from all refs (read-only publication audit)

Output: one line per finding, `<rule> <path>[:<line>] <short reason>`.
Exit codes: 0 clean, 1 violations, 2 usage or configuration error.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import codecs
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_ALLOW = ".github/public-boundary-allow.toml"
SYMLINK_MODE = "120000"
SUBMODULE_MODE = "160000"
ZERO_OID_CHARS = frozenset("0")
LARGE_FILE_BYTES = 1024 * 1024
PAYLOAD_RUN_BYTES = 4096
WIDE_SNIFF_BYTES = 8192
SCAN_LIMIT_BYTES = 32 * 1024 * 1024
MAX_FINDINGS_PER_RULE_AND_FILE = 10
GITMODULES_NAME = ".gitmodules"
RULES = (
    "symlink",
    "gitlink",
    "weights-suffix",
    "large-file",
    "embedded-payload",
    "secret",
    "private-path",
    "deny",
    "binary",
    "foreign-history",
)
WEIGHT_SUFFIXES = (
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".onnx",
    ".gguf",
    ".pkl",
    ".pickle",
    ".joblib",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".nc",
    ".msgpack",
)

# Patterns are assembled from pieces so this file never matches its own rules.
# The end of a hostname: no further letter, digit, hyphen or DNS label after it.
_HOST_END = rb"(?![A-Za-z0-9-]|\.[A-Za-z0-9-])"
_NO_WORD_BEFORE = rb"(?<![A-Za-z0-9_])"
_PEM_DASHES = b"-" * 5
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private key header",
        re.compile(_PEM_DASHES + rb"BEGIN (?:[A-Z]+ )*PRIVATE" + rb" KEY(?: BLOCK)?" + _PEM_DASHES),
    ),
    ("AWS access key", re.compile(_NO_WORD_BEFORE + rb"AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(_NO_WORD_BEFORE + rb"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("GitHub token", re.compile(_NO_WORD_BEFORE + rb"github_pat_[A-Za-z0-9_]{22,}")),
    ("Hugging Face token", re.compile(_NO_WORD_BEFORE + rb"hf_[A-Za-z0-9]{34,}")),
    ("Anthropic key", re.compile(_NO_WORD_BEFORE + rb"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI key", re.compile(_NO_WORD_BEFORE + rb"sk-[A-Za-z0-9_-]{32,}")),
    ("Modal token", re.compile(_NO_WORD_BEFORE + rb"(?:ak|as)-[A-Za-z0-9]{20,}")),
    ("Slack token", re.compile(_NO_WORD_BEFORE + rb"xox[abprs]-[A-Za-z0-9-]{10,}")),
    # Service hostnames: any pymc-labs.com host except exactly www.pymc-labs.com, and hosts on
    # the platforms that serve apps and previews. Public docs must not name one. A match must
    # end at a full DNS label, so a longer name such as x.pymc-labs.com.example.org is not one.
    (
        "service hostname",
        re.compile(
            rb"(?<![A-Za-z0-9_.-])(?!www\.pymc-labs\.com" + _HOST_END + rb")(?:[A-Za-z0-9-]+\.)+"
            rb"(?:pymc-labs\.com|fly\.dev|run\.app|workers\.dev|modal\.run)" + _HOST_END,
            re.IGNORECASE,
        ),
    ),
)
PRIVATE_PATH_PATTERN = re.compile(rb"\.private" + rb"/[^\s]")
BASE64_LINE_PATTERN = re.compile(rb"[A-Za-z0-9+/]+={0,2}")
WRAPPED_LINE_MIN_BYTES = 20
PRIVATE_LINK_NAME = ".private"
BASE64_RUN_PATTERN = re.compile(rb"[A-Za-z0-9+/]{%d,}={0,2}" % PAYLOAD_RUN_BYTES)

NOTEBOOK_SUFFIX = ".ipynb"
NOTEBOOK_IMAGE_MAGIC: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}
NOTEBOOK_IMAGE_MAX_BYTES = 2 * 1024 * 1024
PNG_TEXT_INFLATE_LIMIT = 1024 * 1024
PNG_PIXEL_INFLATE_LIMIT = 64 * 1024 * 1024
PNG_SAMPLE_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
PNG_BIT_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
# A parser cannot rule out hidden data in a valid image, so every notebook with image
# outputs yields one finding with this reason. Only an allowlist entry with `sha256` covers it.
NOTEBOOK_PIN_REASON = "pin reviewed bytes with sha256 in the allowlist"
PNG_CHUNKS = frozenset(
    "IHDR PLTE IDAT IEND tRNS cHRM gAMA iCCP sBIT sRGB bKGD hIST pHYs sPLT tIME "
    "tEXt zTXt iTXt eXIf acTL fcTL fdAT".split()
)


class UsageError(Exception):
    """Bad arguments, bad configuration or a failing Git call (exit code 2)."""


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int | None
    reason: str
    # Set only on the one finding a notebook gets for its image outputs. Only
    # this finding can be allowlisted (by sha256); text never decides that.
    notebook_image_pin: bool = False

    def render(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{self.rule} {location} {self.reason}"


@dataclass
class Item:
    """One file to check: a Git blob or an archive member."""

    path: str
    mode: str
    size: int
    reader: Callable[[], bytes] | None
    match_path: str = ""
    is_archive: bool = False
    where: str = ""
    _content: bytes | None = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def content(self) -> bytes | None:
        """File bytes, or None when the file is a symlink, unreadable or over the scan limit."""
        if not self._loaded:
            self._loaded = True
            readable = self.reader is not None and self.size <= SCAN_LIMIT_BYTES
            if readable and self.mode != SYMLINK_MODE:
                self._content = self.reader()
        return self._content

    def sha256(self) -> str | None:
        data = self.content()
        return None if data is None else hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- Git plumbing


def run_git(repo: str, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise UsageError(f"git {' '.join(args[:2])} failed: {detail}")
    return result.stdout


def find_repo_root() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, check=False
    )
    return result.stdout.decode().strip() if result.returncode == 0 else None


class GitObjects:
    """Read sizes and contents of Git objects through two long-lived cat-file processes."""

    def __init__(self, repo: str) -> None:
        self._repo = repo
        self._check: subprocess.Popen[bytes] | None = None
        self._batch: subprocess.Popen[bytes] | None = None

    def _start(self, mode: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            ["git", "-C", self._repo, "cat-file", mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _ask(process: subprocess.Popen[bytes], oid: str) -> bytes:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(oid.encode() + b"\n")
        process.stdin.flush()
        return process.stdout.readline()

    def info(self, oid: str) -> tuple[str, int] | None:
        """Return (type, size) for an object, or None when it is missing."""
        if self._check is None:
            self._check = self._start("--batch-check")
        fields = self._ask(self._check, oid).split()
        if len(fields) != 3:
            return None
        return fields[1].decode(), int(fields[2])

    def read(self, oid: str) -> bytes:
        if self._batch is None:
            self._batch = self._start("--batch")
        fields = self._ask(self._batch, oid).split()
        if len(fields) != 3:
            raise UsageError(f"cannot read object {oid}")
        size = int(fields[2])
        assert self._batch.stdout is not None
        data = self._batch.stdout.read(size)
        self._batch.stdout.read(1)
        return data

    def close(self) -> None:
        for process in (self._check, self._batch):
            if process is not None and process.stdin is not None:
                process.stdin.close()
                process.wait()

    def blob_item(self, path: str, mode: str, oid: str, where: str = "") -> Item | None:
        if mode == SUBMODULE_MODE:
            # A gitlink has no blob to read; report it by path.
            return Item(path=path, mode=mode, size=0, reader=None, match_path=path, where=where)
        info = self.info(oid)
        if info is None or info[0] != "blob":
            return None
        return Item(
            path=path,
            mode=mode,
            size=info[1],
            reader=lambda: self.read(oid),
            match_path=path,
            where=where,
        )


def tree_items(objects: GitObjects, repo: str) -> Iterator[Item]:
    listing = run_git(repo, "ls-tree", "-r", "-z", "HEAD")
    for record in filter(None, listing.split(b"\0")):
        meta, _, raw_path = record.partition(b"\t")
        mode, _, oid = meta.decode().split()
        item = objects.blob_item(raw_path.decode(errors="replace"), mode, oid)
        if item is not None:
            yield item


def staged_items(objects: GitObjects, repo: str) -> Iterator[Item]:
    listing = run_git(repo, "ls-files", "-s", "-z")
    for record in filter(None, listing.split(b"\0")):
        meta, _, raw_path = record.partition(b"\t")
        mode, oid, _stage = meta.decode().split()
        item = objects.blob_item(raw_path.decode(errors="replace"), mode, oid)
        if item is not None:
            yield item


@dataclass(frozen=True)
class RawChange:
    path: str
    mode: str
    oid: str
    where: str


def parse_raw_log(output: bytes) -> Iterator[RawChange]:
    """Parse `git log --raw -z` with a leading `@@<commit> <ref>@@` header per commit."""
    where = ""
    pending: list[str] | None = None
    for token in output.split(b"\0"):
        token = token.lstrip(b"\n")
        while token.startswith(b"@@"):
            header, _, token = token.partition(b"\n")
            commit, _, ref = header.strip(b"@").decode().partition(" ")
            where = f"commit {commit[:7]}" + (f" ({ref})" if ref else "")
        if not token:
            continue
        if pending is None:
            if token.startswith(b":"):
                pending = token.decode().split()
            continue
        _old_mode, new_mode, _old_oid, new_oid, _status = pending[:5]
        pending = None
        if set(new_oid) != ZERO_OID_CHARS:
            yield RawChange(token.decode(errors="replace"), new_mode, new_oid, where)


def log_changes(repo: str, *revision_args: str) -> list[RawChange]:
    output = run_git(
        repo,
        "log",
        "--source",
        "-m",
        "--format=@@%H %S@@",
        "--raw",
        "--no-abbrev",
        "--no-renames",
        "-z",
        *revision_args,
    )
    return list(parse_raw_log(output))


def changes_to_items(objects: GitObjects, changes: Iterable[RawChange]) -> Iterator[Item]:
    seen: set[tuple[str, str, str]] = set()
    for change in changes:
        key = (change.path, change.mode, change.oid)
        if key in seen:
            continue
        seen.add(key)
        item = objects.blob_item(change.path, change.mode, change.oid, change.where)
        if item is not None:
            yield item


def range_items(objects: GitObjects, repo: str, revision_range: str) -> Iterator[Item]:
    if ".." not in revision_range:
        raise UsageError("--range needs the form A..B")
    changes = log_changes(repo, revision_range)
    # `git log -m` diffs merges against each parent; the net change catches the rest.
    start, _, end = revision_range.partition("..")
    net = run_git(repo, "diff", "--raw", "--no-abbrev", "--no-renames", "-z", start, end)
    changes += parse_raw_log(net)
    yield from changes_to_items(objects, changes)


def foreign_history_findings(
    repo: str, revision_range: str, allowed_roots: frozenset[str] = frozenset()
) -> list[Finding]:
    """Flag root commits inside A..B: they come from a history that A does not contain.

    Roots in `allowed_roots` (full SHAs) are accepted: the caller has authorised them.
    """
    roots = run_git(repo, "rev-list", "--max-parents=0", revision_range).decode().split()
    return [
        Finding("foreign-history", f"commit:{root[:7]}", None, "new root commit: unrelated history")
        for root in roots
        if root not in allowed_roots
    ]


def commit_message_findings(
    repo: str, deny: list[DenyPattern], *revision_args: str
) -> list[Finding]:
    """Apply the secret, private-path and deny rules to commit messages."""
    output = run_git(repo, "log", "--format=%x01%H%x00%B", *revision_args)
    findings = []
    for record in filter(None, output.split(b"\x01")):
        sha, _, message = record.partition(b"\0")
        path = f"commit:{sha.decode()[:7]}"
        findings += check_text_rules(path, message, with_lines=False)
        for entry in deny:
            if entry.byte_pattern.search(message):
                findings.append(Finding("deny", path, None, f"deny-file line {entry.line_number}"))
    return findings


def all_refs_items(objects: GitObjects, repo: str) -> Iterator[Item]:
    changes = log_changes(repo, "--all")
    seen = {(change.path, change.oid) for change in changes}
    yield from changes_to_items(objects, changes)
    # Blobs that only appear through merge resolutions or tags: add by object listing.
    listing = run_git(repo, "rev-list", "--objects", "--all").decode(errors="replace")
    for line in listing.splitlines():
        oid, _, path = line.partition(" ")
        if (path, oid) in seen or not path:
            continue
        item = objects.blob_item(path, "100644", oid, "rev-list")
        if item is not None:
            yield item


# -------------------------------------------------------------------- archives


def _bounded(read: Callable[[], bytes]) -> Callable[[], bytes]:
    return read


def zip_items(path: Path) -> Iterator[Item]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            unix_type = (info.external_attr >> 16) & 0o170000
            mode = SYMLINK_MODE if unix_type == 0o120000 else "100644"
            yield Item(
                path=f"{path.name}!{info.filename}",
                mode=mode,
                size=info.file_size,
                reader=lambda info=info: archive.read(info),
                match_path=info.filename,
                is_archive=True,
            )


def tar_items(path: Path) -> Iterator[Item]:
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isdir():
                continue
            if member.issym() or member.islnk():
                mode = SYMLINK_MODE
            elif member.isfile():
                mode = "100644"
            else:
                continue
            yield Item(
                path=f"{path.name}!{member.name}",
                mode=mode,
                size=member.size,
                reader=lambda member=member: _read_tar_member(archive, member),
                match_path=member.name,
                is_archive=True,
            )


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    return b"" if handle is None else handle.read()


def archive_items(paths: Iterable[str]) -> Iterator[Item]:
    for name in paths:
        path = Path(name)
        if not path.is_file():
            raise UsageError(f"archive not found: {name}")
        try:
            opener = zip_items if zipfile.is_zipfile(path) else tar_items
            yield from opener(path)
        except (tarfile.TarError, zipfile.BadZipFile, OSError) as error:
            raise UsageError(f"cannot read archive {name}: {error}") from error


# ----------------------------------------------------------------------- rules


def line_of(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def mask(text: bytes) -> str:
    return text[:4].decode(errors="replace") + "***"


def check_gitlink_rule(item: Item, name: str) -> list[Finding]:
    """Flag submodule pointers (mode 160000) and `.gitmodules` files at any depth."""
    findings = []
    if item.mode == SUBMODULE_MODE:
        findings.append(
            Finding("gitlink", item.path, None, "gitlink (nested repository or submodule)")
        )
    if name.replace("\\", "/").rsplit("/", 1)[-1].casefold() == GITMODULES_NAME:
        findings.append(Finding("gitlink", item.path, None, ".gitmodules file"))
    return findings


def check_path_rules(item: Item) -> list[Finding]:
    findings = []
    name = item.match_path or item.path
    if item.mode == SYMLINK_MODE:
        findings.append(Finding("symlink", item.path, None, "symlink is not allowed"))
    findings += check_gitlink_rule(item, name)
    if name.lower().endswith(WEIGHT_SUFFIXES):
        suffix = "." + name.rsplit(".", 1)[-1].lower()
        findings.append(Finding("weights-suffix", item.path, None, f"{suffix} file"))
    if item.size > LARGE_FILE_BYTES:
        findings.append(
            Finding("large-file", item.path, None, f"{item.size} bytes, limit {LARGE_FILE_BYTES}")
        )
    parts = name.replace("\\", "/").split("/")
    if PRIVATE_LINK_NAME in (part.casefold() for part in parts):
        findings.append(
            Finding("private-path", item.path, None, "path has a private link component")
        )
    return findings


def _mostly_nul(sample: bytes) -> bool:
    return len(sample) >= 4 and sample.count(0) >= 0.9 * len(sample)


# Control bytes that do not occur in text: everything below 0x20 except tab, newline,
# form feed, carriage return and escape (terminal colours), plus DEL.
CONTROL_BYTES = re.compile(rb"[\x00-\x08\x0b\x0e-\x1a\x1c-\x1f\x7f]")


def decode_wide_text(data: bytes) -> bytes | None:
    """Return UTF-8 bytes for UTF-16 or UTF-32 text (BOM or NUL-interleaved ASCII).

    Returns None when the content is not wide text. Raises UnicodeDecodeError when it looks
    wide but does not decode cleanly.
    """
    boms = (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    )
    for bom, codec in boms:
        if data.startswith(bom):
            return data.decode(codec).encode()
    head = data[:WIDE_SNIFF_BYTES]
    for codec, width, ascii_offset in (
        ("utf-32-le", 4, 0),
        ("utf-32-be", 4, 3),
        ("utf-16-le", 2, 0),
        ("utf-16-be", 2, 1),
    ):
        filler = b"".join(head[i::width] for i in range(width) if i != ascii_offset)
        if _mostly_nul(filler) and head[ascii_offset::width].count(0) < len(head) // width // 2:
            return data.decode(codec).encode()
    return None


def inspect_content(data: bytes) -> tuple[list[bytes], bool]:
    """Return (byte strings to scan, True when the content is not readable text).

    Readable text is strict UTF-8, or UTF-16/UTF-32 that decodes cleanly. Everything else
    is opaque and needs an allowlist entry. The raw bytes are always scanned as well, so
    decoding never hides a pattern that is present in the raw file.
    """
    try:
        wide = decode_wide_text(data)
    except UnicodeDecodeError:
        return [data], True
    if wide is not None:
        return [wide, data], CONTROL_BYTES.search(wide) is not None
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return [data], True
    return [data], CONTROL_BYTES.search(data) is not None


def check_content_rules(item: Item, data: bytes, notebook: bool = False) -> list[Finding]:
    """Content rules. A parsed notebook gets its payload check from `check_notebook`."""
    texts, opaque = inspect_content(data)
    findings = []
    if opaque:
        findings.append(
            Finding(
                "binary", item.path, None, "not plain UTF-8, UTF-16 or UTF-32 text: cannot be read"
            )
        )
    elif not notebook:
        findings += check_embedded_payload(item.path, texts[0])
    for text in texts:
        findings += check_text_rules(item.path, text)
    return findings


def _looks_encoded(run: bytes) -> bool:
    """True for a run with enough variety to be base64, and not plain hex."""
    return len(set(run)) >= 16 and not re.fullmatch(rb"[0-9a-fA-F]+", run)


def wrapped_base64_runs(data: bytes) -> Iterator[tuple[int, int]]:
    """Yield (first line, size) for consecutive lines that are base64 chunks.

    Catches payloads that are wrapped at 76 columns or stored as a JSON array of strings.
    """
    start, parts = 0, []
    for number, raw in enumerate(data.split(b"\n") + [b""], start=1):
        chunk = raw.strip(b' \t\r,"').removesuffix(b"\\n").removesuffix(b"\\r")
        if len(chunk) >= WRAPPED_LINE_MIN_BYTES and BASE64_LINE_PATTERN.fullmatch(chunk):
            if not parts:
                start = number
            parts.append(chunk)
            continue
        joined = b"".join(parts)
        if len(joined) >= PAYLOAD_RUN_BYTES and _looks_encoded(joined):
            yield start, len(joined)
        parts = []


def check_embedded_payload(path: str, data: bytes) -> list[Finding]:
    findings = []
    flagged_lines = set()
    for run in BASE64_RUN_PATTERN.finditer(data):
        if _looks_encoded(run.group()):
            line = line_of(data, run.start())
            flagged_lines.add(line)
            reason = f"base64 run of {len(run.group())} bytes"
            findings.append(Finding("embedded-payload", path, line, reason))
    for line, size in wrapped_base64_runs(data):
        if line not in flagged_lines:
            reason = f"wrapped base64 lines totalling {size} bytes"
            findings.append(Finding("embedded-payload", path, line, reason))
    return findings


def check_text_rules(path: str, data: bytes, with_lines: bool = True) -> list[Finding]:
    """Secret and private-path rules, shared by file content and commit messages."""
    findings = []
    claimed: set[int] = set()

    def where(offset: int) -> int | None:
        return line_of(data, offset) if with_lines else None

    for label, pattern in SECRET_PATTERNS:
        for found in pattern.finditer(data):
            if found.start() in claimed:
                continue
            claimed.add(found.start())
            shown = label if label == "private key header" else f"{label} {mask(found.group())}"
            findings.append(Finding("secret", path, where(found.start()), shown))
    for found in PRIVATE_PATH_PATTERN.finditer(data):
        reason = "reference to a path inside the private link"
        findings.append(Finding("private-path", path, where(found.start()), reason))
    return findings


@dataclass(frozen=True)
class DenyPattern:
    line_number: int
    pattern: re.Pattern[str]
    byte_pattern: re.Pattern[bytes]


def load_deny_file(path: Path) -> list[DenyPattern]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise UsageError(f"cannot read deny file: {error.strerror}") from error
    patterns = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(
                DenyPattern(
                    number,
                    re.compile(line, re.IGNORECASE),
                    re.compile(line.encode(), re.IGNORECASE),
                )
            )
        except re.error:
            raise UsageError(f"deny-file line {number} is not a valid regex") from None
    return patterns


def check_deny_rules(item: Item, data: bytes | None, deny: list[DenyPattern]) -> list[Finding]:
    findings = []
    for entry in deny:
        reason = f"deny-file line {entry.line_number}"
        if entry.pattern.search(item.path):
            findings.append(Finding("deny", item.path, None, reason + " (in path)"))
        if data is not None:
            for found in entry.byte_pattern.finditer(data):
                findings.append(Finding("deny", item.path, line_of(data, found.start()), reason))
    return findings


# ------------------------------------------------------------------- notebooks


def is_notebook(item: Item) -> bool:
    return (item.match_path or item.path).lower().endswith(NOTEBOOK_SUFFIX)


def png_chunks(image: bytes) -> list[tuple[bytes, bytes]]:
    """Split a PNG into chunks. Raises ValueError unless it ends exactly at IEND."""
    chunks, offset = [], 8
    while offset + 12 <= len(image):
        length = int.from_bytes(image[offset : offset + 4], "big")
        kind = image[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(image):
            raise ValueError("truncated chunk")
        body = image[offset + 8 : offset + 8 + length]
        if zlib.crc32(kind + body) != int.from_bytes(image[end - 4 : end], "big"):
            raise ValueError("bad chunk checksum")
        chunks.append((kind, body))
        offset = end
        if kind == b"IEND":
            break
    if not chunks or chunks[-1][0] != b"IEND" or offset != len(image):
        raise ValueError("data after IEND or missing IEND")
    return chunks


def _inflate(blob: bytes) -> bytes:
    inflater = zlib.decompressobj()
    text = inflater.decompress(blob, PNG_TEXT_INFLATE_LIMIT)
    if inflater.unconsumed_tail:
        raise ValueError("text chunk inflates past the limit")
    return text


def png_text_chunks(chunks: list[tuple[bytes, bytes]]) -> list[bytes]:
    """Keyword and text of every tEXt, zTXt and iTXt chunk. Raises ValueError or zlib.error."""
    texts = []
    for kind, body in chunks:
        if kind == b"tEXt":
            texts.append(body.replace(b"\0", b" "))
        elif kind == b"zTXt":
            keyword, _, rest = body.partition(b"\0")
            texts.append(keyword + b" " + _inflate(rest[1:]))
        elif kind == b"iTXt":
            keyword, _, rest = body.partition(b"\0")
            compressed = rest[:1] == b"\x01"
            # Skip flag, method, language tag and translated keyword.
            parts = rest[2:].split(b"\0", 2)
            text = parts[2] if len(parts) == 3 else b""
            texts.append(keyword + b" " + (_inflate(text) if compressed else text))
    return texts


def png_pixel_stream_size(header: bytes) -> int:
    """Bytes of filtered scanlines that a non-interlaced IHDR implies. Raises ValueError."""
    if len(header) != 13:
        raise ValueError("bad IHDR length")
    width = int.from_bytes(header[0:4], "big")
    height = int.from_bytes(header[4:8], "big")
    depth, colour, method, filtering, interlace = header[8:13]
    if width == 0 or height == 0 or width > 2**31 - 1 or height > 2**31 - 1:
        raise ValueError("bad image size")
    if depth not in PNG_BIT_DEPTHS.get(colour, ()) or method != 0 or filtering != 0:
        raise ValueError("bad IHDR fields")
    if interlace != 0:
        raise ValueError("interlaced PNG")
    row_bytes = (width * PNG_SAMPLE_CHANNELS[colour] * depth + 7) // 8
    size = height * (1 + row_bytes)
    if size > PNG_PIXEL_INFLATE_LIMIT:
        raise ValueError("image data past the size limit")
    return size


def check_png_pixel_stream(chunks: list[tuple[bytes, bytes]]) -> None:
    """The first chunk is IHDR and the IDAT data inflates to exactly the implied size."""
    if chunks[0][0] != b"IHDR":
        raise ValueError("first chunk is not IHDR")
    expected = png_pixel_stream_size(chunks[0][1])
    stream = b"".join(body for kind, body in chunks if kind == b"IDAT")
    if not stream:
        raise ValueError("no IDAT chunk")
    inflater = zlib.decompressobj()
    pixels = inflater.decompress(stream, expected + 1)
    if len(pixels) != expected or not inflater.eof or inflater.unused_data:
        raise ValueError("IDAT size does not match IHDR")


def inspect_png(image: bytes) -> tuple[list[bytes], str | None]:
    """Return (metadata texts to scan, problem or None)."""
    try:
        chunks = png_chunks(image)
        unknown = sorted({k.decode("latin-1") for k, _ in chunks} - PNG_CHUNKS)
        if unknown:
            return [], f"unknown PNG chunk {unknown[0]!r}"
        check_png_pixel_stream(chunks)
        return png_text_chunks(chunks), None
    except (ValueError, zlib.error) as error:
        return [], f"malformed PNG ({error})"


JPEG_NO_LENGTH_MARKERS = frozenset([0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)])
JPEG_RESTART_MARKERS = frozenset(range(0xD0, 0xD8))
JPEG_FRAME_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg_marker_at(image: bytes, offset: int) -> tuple[int, int]:
    """(marker, offset after it). Raises ValueError unless a marker starts here."""
    if offset >= len(image) or image[offset] != 0xFF:
        raise ValueError("expected a marker")
    while offset < len(image) and image[offset] == 0xFF:
        offset += 1
    if offset >= len(image) or image[offset] == 0:
        raise ValueError("bad marker")
    return image[offset], offset + 1


def _skip_jpeg_scan(image: bytes, offset: int) -> int:
    """Offset of the first marker after entropy-coded data."""
    while offset + 1 < len(image):
        following = image[offset + 1]
        if image[offset] == 0xFF and following != 0 and following not in JPEG_RESTART_MARKERS:
            return offset
        offset += 1
    raise ValueError("no marker after scan data")


def inspect_jpeg(image: bytes) -> str | None:
    """Walk the marker sequence from SOI through SOS to the final EOI. Return a problem or None."""
    try:
        marker, offset = _jpeg_marker_at(image, 0)
        if marker != 0xD8:
            raise ValueError("no SOI")
        frames = scans = 0
        while True:
            marker, offset = _jpeg_marker_at(image, offset)
            if marker == 0xD9:
                break
            if marker in JPEG_NO_LENGTH_MARKERS:
                continue
            length = int.from_bytes(image[offset : offset + 2], "big")
            if length < 2 or offset + length > len(image):
                raise ValueError("bad segment length")
            offset += length
            frames += marker in JPEG_FRAME_MARKERS
            if marker == 0xDA:
                if not frames:
                    raise ValueError("scan before frame header")
                scans += 1
                offset = _skip_jpeg_scan(image, offset)
        if not scans or offset != len(image):
            raise ValueError("missing scan or data after EOI")
    except ValueError as error:
        return f"malformed JPEG ({error})"
    return None


def _skip_gif_sub_blocks(image: bytes, offset: int) -> int:
    while True:
        if offset >= len(image):
            raise ValueError("truncated block")
        size = image[offset]
        offset += 1 + size
        if size == 0:
            return offset


def _gif_color_table_size(flags: int) -> int:
    return 3 * 2 ** ((flags & 7) + 1) if flags & 0x80 else 0


def inspect_gif(image: bytes) -> str | None:
    """Walk header, screen descriptor and blocks to the trailer. Return a problem or None."""
    try:
        if len(image) < 13:
            raise ValueError("truncated header")
        offset = 13 + _gif_color_table_size(image[10])
        frames = 0
        while offset < len(image) and image[offset] != 0x3B:
            kind = image[offset]
            if kind == 0x21:
                offset = _skip_gif_sub_blocks(image, offset + 2)
            elif kind == 0x2C:
                if offset + 10 > len(image):
                    raise ValueError("truncated image descriptor")
                offset += 10 + _gif_color_table_size(image[offset + 9]) + 1
                offset = _skip_gif_sub_blocks(image, offset)
                frames += 1
            else:
                raise ValueError("unknown block")
        if offset != len(image) - 1 or not frames:
            raise ValueError("missing trailer or image")
    except ValueError as error:
        return f"malformed GIF ({error})"
    return None


def inspect_image(mime: str, image: bytes) -> tuple[list[bytes], str | None]:
    """Return (texts to scan, problem or None) for a decoded notebook image."""
    if mime == "image/png":
        return inspect_png(image)
    flaw = inspect_jpeg(image) if mime == "image/jpeg" else inspect_gif(image)
    return [image], flaw


def decode_notebook_image(mime: str, value: object) -> tuple[bytes | None, str | None]:
    """Decode an image output. Return (image bytes, None) or (None, problem)."""
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        value = "".join(value)
    if not isinstance(value, str):
        return None, "image output is not a string"
    try:
        image = base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        return None, "image output is not valid base64"
    if not image.startswith(NOTEBOOK_IMAGE_MAGIC[mime]):
        return None, f"decoded data is not a {mime} file"
    if len(image) > NOTEBOOK_IMAGE_MAX_BYTES:
        return None, f"image of {len(image)} bytes, limit {NOTEBOOK_IMAGE_MAX_BYTES}"
    return image, None


def image_slot(path: tuple[str | int, ...], cell_types: dict[int, str]) -> str | None:
    """The image MIME type when `path` is cells[i].outputs[j].data[mime] of a code cell."""
    if len(path) != 6 or path[0] != "cells" or path[2] != "outputs" or path[4] != "data":
        return None
    index, mime = path[1], path[5]
    is_code = isinstance(index, int) and cell_types.get(index) == "code"
    return mime if is_code and mime in NOTEBOOK_IMAGE_MAGIC else None


def json_path(path: tuple[str | int, ...]) -> str:
    return "".join(f"[{part!r}]" if isinstance(part, str) else f"[{part}]" for part in path)


def walk_strings(document: object) -> Iterator[tuple[tuple[str | int, ...], str | list[str]]]:
    """Yield (path, value) for each string, and each list of strings, in a JSON document.

    Iterative so that a deeply nested notebook cannot exhaust the stack.
    """
    stack: list[tuple[tuple[str | int, ...], object]] = [((), document)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, str):
            yield path, node
        elif isinstance(node, list):
            if node and all(isinstance(part, str) for part in node):
                yield path, node
            else:
                stack += [((*path, i), child) for i, child in reversed(list(enumerate(node)))]
        elif isinstance(node, dict):
            for key, child in node.items():
                yield (*path, f"key:{key}"), str(key)
                stack.append(((*path, key), child))


def notebook_cell_types(document: dict) -> dict[int, str]:
    return {
        index: cell.get("cell_type", "")
        for index, cell in enumerate(document["cells"])
        if isinstance(cell, dict)
    }


def parse_notebook(data: bytes) -> dict | None:
    try:
        document = json.loads(data)
    except (ValueError, RecursionError):
        return None
    valid = isinstance(document, dict) and document.get("nbformat") == 4
    return document if valid and isinstance(document.get("cells"), list) else None


def check_notebook(path: str, data: bytes, deny: list[DenyPattern]) -> list[Finding]:
    """Payload, metadata, secret and deny rules for an nbformat 4 notebook.

    Base64 is allowed only as the value of a code-cell image output whose bytes decode to
    a structurally valid image of that type within the size cap. Every other string is
    scanned as ordinary text. A parser cannot rule out hidden data in a valid image, so a
    notebook with image outputs also yields one pin finding that only an allowlist entry
    with `sha256` covers. Malformed-image and other notebook findings are never covered.
    """
    document = parse_notebook(data)
    if document is None:
        findings = [Finding("embedded-payload", path, None, "unparseable notebook")]
        return findings + check_embedded_payload(path, data)
    findings: list[Finding] = []
    extra_texts: list[bytes] = []
    cell_types = notebook_cell_types(document)
    image_outputs = 0
    for index, cell in enumerate(document["cells"]):
        if isinstance(cell, dict) and cell.get("attachments"):
            reason = f"cell attachments at cells[{index}]"
            findings.append(Finding("embedded-payload", path, None, reason))
    for where, value in walk_strings(document):
        mime = image_slot(where, cell_types)
        if mime is not None:
            image, problem = decode_notebook_image(mime, value)
            if problem is not None:
                findings.append(
                    Finding("embedded-payload", path, None, f"{problem} at {json_path(where)}")
                )
            elif image is not None:
                image_outputs += 1
                texts, flaw = inspect_image(mime, image)
                extra_texts += texts
                if flaw:
                    findings.append(
                        Finding("embedded-payload", path, None, f"{flaw} at {json_path(where)}")
                    )
            continue
        text = "".join(value).encode() if isinstance(value, list) else value.encode()
        for found in check_embedded_payload(path, text):
            reason = f"{found.reason} at {json_path(where)}"
            findings.append(Finding(found.rule, path, None, reason))
        if text not in data:
            extra_texts.append(text)  # JSON escapes hid it from the raw scan
    if image_outputs:
        reason = f"{image_outputs} image outputs: {NOTEBOOK_PIN_REASON}"
        findings.append(Finding("embedded-payload", path, None, reason, notebook_image_pin=True))
    for text in extra_texts:
        findings += check_text_rules(path, text, with_lines=False)
        findings += [
            Finding("deny", path, None, f"deny-file line {entry.line_number} (in notebook content)")
            for entry in deny
            if entry.byte_pattern.search(text)
        ]
    return findings


def cap_findings(findings: list[Finding]) -> list[Finding]:
    """Keep the first few findings per rule and file; summarise the rest on one line."""
    kept: list[Finding] = []
    counts: dict[tuple[str, str], int] = {}
    for finding in findings:
        key = (finding.rule, finding.path)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= MAX_FINDINGS_PER_RULE_AND_FILE:
            kept.append(finding)
    for (rule, path), count in counts.items():
        if count > MAX_FINDINGS_PER_RULE_AND_FILE:
            extra = count - MAX_FINDINGS_PER_RULE_AND_FILE
            kept.append(Finding(rule, path, None, f"(+{extra} more)"))
    return kept


def check_item(item: Item, deny: list[DenyPattern]) -> list[Finding]:
    findings = check_path_rules(item)
    data = item.content()
    texts: list[bytes | None] = [None]
    if data is not None:
        notebook = is_notebook(item)
        findings += check_content_rules(item, data, notebook)
        if notebook:
            findings += check_notebook(item.path, data, deny)
        texts = list(inspect_content(data)[0])
    for index, text in enumerate(texts):
        deny_hits = check_deny_rules(item, text, deny)
        # Path matches are the same for every decoding; report them once.
        findings += deny_hits if index == 0 else [f for f in deny_hits if f.line is not None]
    unique = list(dict.fromkeys(findings))
    return cap_findings(unique)


# ------------------------------------------------------------------- allowlist


@dataclass
class AllowEntry:
    path: str
    rules: frozenset[str]
    reason: str
    sha256: str | None = None
    max_bytes: int | None = None
    pending: str | None = None
    used: bool = False


ALLOWED_KEYS = {"path", "rules", "reason", "sha256", "max_bytes", "pending"}


def parse_allow_entry(index: int, raw: dict) -> AllowEntry:
    label = f"allowlist entry {index}"
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise UsageError(f"{label}: unknown key {sorted(unknown)[0]}")
    for key in ("path", "reason"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise UsageError(f"{label}: {key} must be a non-empty string")
    rules = raw.get("rules")
    if not isinstance(rules, list) or not rules or not set(rules) <= set(RULES):
        raise UsageError(f"{label}: rules must be a non-empty list of {', '.join(RULES)}")
    digest = raw.get("sha256")
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise UsageError(f"{label}: sha256 must be 64 lowercase hex characters")
    max_bytes = raw.get("max_bytes")
    if max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes <= 0):
        raise UsageError(f"{label}: max_bytes must be a positive integer")
    pending = raw.get("pending")
    if pending is not None and (not isinstance(pending, str) or not pending.strip()):
        raise UsageError(f"{label}: pending must be a non-empty string")
    return AllowEntry(raw["path"], frozenset(rules), raw["reason"], digest, max_bytes, pending)


def load_allowlist(path: Path, required: bool) -> list[AllowEntry]:
    if not path.is_file():
        if required:
            raise UsageError(f"allowlist not found: {path}")
        return []
    try:
        return parse_allowlist(path.read_text(encoding="utf-8"), str(path))
    except OSError as error:
        raise UsageError(f"cannot read allowlist {path}: {error.strerror}") from error


def parse_allowlist(text: str, label: str) -> list[AllowEntry]:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise UsageError(f"cannot parse allowlist {label}: {error}") from error
    entries = document.get("allow", [])
    if not isinstance(entries, list):
        raise UsageError("allowlist: 'allow' must be an array of tables")
    return [parse_allow_entry(index, raw) for index, raw in enumerate(entries, start=1)]


def entry_covers(entry: AllowEntry, item: Item, finding: Finding) -> str | None:
    """Return None when the entry covers the finding, else a short note why it does not."""
    name = item.match_path or item.path
    same_path = name == entry.path or (item.is_archive and name.endswith("/" + entry.path))
    if not same_path or finding.rule not in entry.rules:
        return ""
    if is_notebook(item) and finding.rule == "embedded-payload":
        if not finding.notebook_image_pin:
            return " (notebook payload findings cannot be allowlisted)"
        if entry.sha256 is None:
            return " (notebook image pin needs an allowlist sha256)"
    if entry.sha256 is not None and item.sha256() != entry.sha256:
        return " (allowlist sha256 mismatch)"
    if finding.rule == "large-file" and entry.max_bytes is not None:
        if item.size > entry.max_bytes:
            return f" (over allowlist max_bytes {entry.max_bytes})"
    return None


def apply_allowlist(item: Item, findings: list[Finding], allow: list[AllowEntry]) -> list[Finding]:
    remaining = []
    for finding in findings:
        note = ""
        covered = False
        for entry in allow:
            verdict = entry_covers(entry, item, finding)
            if verdict is None:
                entry.used = covered = True
                break
            note = note or verdict
        if not covered:
            remaining.append(replace(finding, reason=finding.reason + note))
    return remaining


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tree", action="store_true", help="tracked files at HEAD (default)")
    modes.add_argument("--staged", action="store_true", help="files in the index")
    modes.add_argument("--range", metavar="A..B", help="blobs added or changed in A..B")
    modes.add_argument("--archive", nargs="+", metavar="PATH", help="wheel, zip or tar members")
    modes.add_argument("--all-refs", action="store_true", help="every blob in every ref")
    parser.add_argument("--allow", metavar="FILE", help=f"allowlist (default {DEFAULT_ALLOW})")
    parser.add_argument("--deny-file", metavar="FILE", help="extra regexes, one per line")
    parser.add_argument(
        "--allowed-root",
        metavar="SHA",
        action="append",
        default=[],
        help="with --range: accept this root commit (full SHA, repeatable)",
    )
    parser.add_argument("--release", action="store_true", help="pending allowlist entries fail")
    parser.add_argument("--repo", metavar="DIR", help="repository (default: git toplevel of cwd)")
    return parser


def collect_items(args: argparse.Namespace, repo: str | None, objects: GitObjects | None):
    if args.archive:
        return archive_items(args.archive)
    assert repo is not None and objects is not None
    if args.staged:
        return staged_items(objects, repo)
    if args.range:
        return range_items(objects, repo, args.range)
    if args.all_refs:
        return all_refs_items(objects, repo)
    return tree_items(objects, repo)


def default_allowlist(repo: str | None) -> Path | None:
    """The checked repo's allowlist, else the one beside this script (archives, mirrors)."""
    candidates = [Path(repo) / DEFAULT_ALLOW] if repo else []
    candidates.append(Path(__file__).resolve().parents[1] / DEFAULT_ALLOW)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    print(f"warning: no {DEFAULT_ALLOW} found; checking without exceptions", file=sys.stderr)
    return None


def git_allowlist_revision(args: argparse.Namespace) -> str:
    """The Git object prefix the default allowlist is read from: index, HEAD or range end."""
    if args.staged:
        return ":"
    if args.range:
        end = args.range.partition("..")[2].lstrip(".")
        return (end or "HEAD") + ":"
    return "HEAD:"


def read_git_file(repo: str, spec: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", repo, "cat-file", "blob", spec], capture_output=True, check=False
    )
    return result.stdout.decode(errors="replace") if result.returncode == 0 else None


def beside_script_allowlist() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / DEFAULT_ALLOW
    return candidate if candidate.is_file() else None


def select_allowlist(args: argparse.Namespace, repo: str | None) -> list[AllowEntry]:
    """Load exceptions. Git modes read committed or staged content, never the working tree."""
    if args.allow:
        return load_allowlist(Path(args.allow), required=True)
    if args.archive or repo is None:
        path = default_allowlist(repo)
        return load_allowlist(path, required=False) if path else []
    text = read_git_file(repo, f"{git_allowlist_revision(args)}{DEFAULT_ALLOW}")
    if text is not None:
        return parse_allowlist(text, f"{DEFAULT_ALLOW} (from Git)")
    own_repo = Path(__file__).resolve().parents[1] == Path(repo).resolve()
    beside = None if own_repo else beside_script_allowlist()
    if args.all_refs and beside is not None:
        return load_allowlist(beside, required=False)
    print(f"warning: no {DEFAULT_ALLOW} in Git; checking without exceptions", file=sys.stderr)
    return []


def allowed_roots(args: argparse.Namespace) -> frozenset[str]:
    roots = [root.lower() for root in args.allowed_root]
    for root in roots:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", root):
            raise UsageError("--allowed-root needs a full commit SHA")
    if roots and not args.range:
        raise UsageError("--allowed-root needs --range")
    return frozenset(roots)


def run(args: argparse.Namespace) -> int:
    repo = args.repo or find_repo_root()
    if repo is None and not args.archive:
        raise UsageError("not inside a git repository (use --repo)")
    roots = allowed_roots(args)
    allow = select_allowlist(args, repo)
    deny = load_deny_file(Path(args.deny_file)) if args.deny_file else []
    objects = GitObjects(repo) if repo else None
    findings: list[Finding] = []
    checked = 0
    if repo and (args.range or args.all_refs):
        revisions = ("--all",) if args.all_refs else (args.range,)
        findings += commit_message_findings(repo, deny, *revisions)
    if repo and args.range:
        findings += foreign_history_findings(repo, args.range, roots)
    try:
        for item in collect_items(args, repo, objects):
            checked += 1
            raw = check_item(item, deny)
            if item.where:
                raw = [replace(f, reason=f"{f.reason} [{item.where}]") for f in raw]  # keeps flags
            findings += apply_allowlist(item, raw, allow)
    finally:
        if objects is not None:
            objects.close()
    if args.release:
        findings += [
            Finding("pending", entry.path, None, entry.pending) for entry in allow if entry.pending
        ]
    for finding in sorted(set(findings), key=lambda f: (f.path, f.line or 0, f.rule, f.reason)):
        print(finding.render())
    if not (args.archive or args.range):
        for entry in allow:
            if not entry.used:
                print(f"warning: unused allowlist entry {entry.path}", file=sys.stderr)
    print(
        f"public-boundary: {checked} file(s) checked, {len(findings)} finding(s)", file=sys.stderr
    )
    return 1 if findings else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
