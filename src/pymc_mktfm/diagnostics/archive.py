"""Replayable multi-world evaluation archives: saved, hash-bound evaluator captures.

An archive persists the per-world conditional captures that
:func:`pymc_mktfm.training.evaluate_rows` passes to its ``result_sink``, so that
cross-world reports (:func:`~pymc_mktfm.diagnostics.sbc_report`,
:func:`~pymc_mktfm.diagnostics.conditional_sbc_report`,
:func:`~pymc_mktfm.diagnostics.strata_summary`) are built from saved draws without any
model call. :meth:`EvaluationArchive.create` returns an
:class:`EvaluationArchiveWriter` whose ``add`` is a drop-in ``result_sink``; its
``close`` publishes the archive atomically, and :meth:`EvaluationArchive.load`
verifies it before anything is used::

    <archive>/
      manifest.json      strict JSON: format_version, kind, identity, entries, archive_sha256
      entries/
        0000/            one world window, in archive order; a saved DecompositionResult
          result.json
          arrays.npz
          truth.npz      truth [T,D], graph_truth [L], observed [T]; float64, no pickles
        0001/
        ...

**Identity and call plan.** ``manifest.json`` declares the evaluated model (weight
and configuration digests), the corpora (input digests, generator seeds, origin and
verified prior hashes; never paths) and the evaluation: the worlds ``(corpus_index,
row)`` with their cell, context length and encoded-input digest, the settings, and the
call plan ``calls: [{"worlds", "seed"}]``: consecutive blocks of the worlds with
consecutive seeds, each declared as one ``evaluate_rows`` call, in order. Each
manifest entry records the SHA-256 and size of its three files, and
``archive_sha256`` digests the rest of the manifest.

**Archive order** is the order in which those calls feed their sink: call by call;
inside a call, worlds sorted stably by context length (their batch order), each
world's ``query`` window immediately followed by its ``context`` window. ``add``
checks each capture against the declaration: its world and encoded inputs, its
call's seed, the leg, ``n_samples``, mode, units, head, context length and model
configuration, and its place in archive order; ``close`` needs every declared world
window. Captures record neither the ``heads`` they were sampled with, nor the
weights, nor which worlds shared a call, so ``add`` cannot check the declared
``heads``, ``model.state_sha256`` or call partition: an archive replays from its
identity when exactly the declared calls of the declared model filled it, as
:func:`pymc_mktfm.training.evaluate_into_archive`, the trainer and the CLI do.

**Verification.** ``load`` reads every file once through one bounded reader that
refuses symlinked archive files and entry directories, never blocks, and checks the
recorded size before reading a byte. Every byte used is size- and hash-checked
against the manifest and decoded from exactly those bytes; every manifest field,
shape and cross-check is validated before anything is used. A failure raises
:class:`ArchiveIntegrityError` naming the offending path relative to the archive.
Later reads (``record``, ``records``, iteration, the converters) re-verify what they
read. This is integrity, not authenticity: a rewrite that also re-signs the manifest
consistently is caught only where it breaks a cross-check, so callers who must pin an
archive compare ``archive_sha256``.

**Ids and cells.** A corpus is labelled ``shard:<seed>`` when its generator seed is
declared and ``corpus:<index>`` otherwise. World ids are ``<label>/row:<row>`` and
the independence key of a world is its cell ``(label, cell_id)``.

**Converters** take one window, never pool windows and call no model.
:meth:`~EvaluationArchive.effect_rows` feeds ``strata_summary`` with every world;
:meth:`~EvaluationArchive.sbc_inputs` (per-family window totals,
:data:`WINDOW_TOTAL_FAMILIES`) and :meth:`~EvaluationArchive.conditional_sbc_inputs`
keep the first world of each cell in archive order, the rule
``conditional_sbc_report`` applies to record order. ``effect_rows`` imports the
training metrics at call time; reading, verification and the SBC converters import no
training code.

**Size, memory and time.** Measured for one world window at A/B geometry (K=15, M=6,
J=1: D=26 components, L=373 edge slots, 104 weeks) with an untrained model; trained
draws and other worlds compress differently. The draws take
``S × (T_window × D + L) × 8`` bytes raw, of which deflate keeps ≈25–40%, and
``result.json`` is ≈69 kB regardless of S. A world takes ≈0.42 MB at S=32 and
≈1.19 MB at S=128, so at S=128 16 worlds take ≈19 MB, 320 worlds ≈380 MB and the
1,280 worlds of ``SHARD_PLANS["evaluation"]`` ≈1.5 GB. Inference memory, not archive
size, limits a call. ``evaluate_rows`` puts every requested world of one context
length into one batch, and on CPU the A/B architecture with the default heads peaks
at ≈1.7 GB RSS per world in a batch at context 52 and ≈3 GB at context 78, almost
independent of the number of draws. Batches and calls run one after another, so
``worlds_per_call=N`` bounds a call's peak at ≈0.2 GB + 3 GB × N on 104-week shards:
≈12 GB for the CLI default of 4. Wall time on CPU is ≈0.4–0.6 s per world and draw
(128 worlds × 128 draws ≈2 h). ``load`` verifies and decodes at ≈7 s/GB (≈10 s for
1.5 GB), each converter re-verifies what it reads at the same rate, and writing
decodes each entry once as a self-check and hashes it once more before publishing
(≈0.5 s/GB). Verification and iteration hold one world window in memory;
``conditional_sbc_inputs`` holds the independent worlds of one window (e.g. 320
cells × 128 draws × 78 weeks × 26 components ≈665 MB of components plus ≈120 MB of
graph draws, ≈790 MB raw for context). Keep archives under ``artifacts/`` or
``runs/``: ``*.npz`` is git-ignored everywhere, but the JSON files only inside ignored
directories such as these.

**Privacy.** Archives hold simulated worlds with their generating truth. ``add``
accepts only results whose encoded inputs equal a declared corpus world. Build
identities with :func:`pymc_mktfm.training.archive_identity` (or use the trainer,
:func:`pymc_mktfm.training.evaluate_into_archive` or the CLI), never by hand; results of
``decompose`` on observed data then cannot enter an archive with truth. The manifest
holds hashes, counts, seeds and context lengths (no paths, arrays or credentials),
but the entries hold draws, truth and input summaries, so treat an archive like its
corpora. Archives whose identity comes from ``archive_identity`` (the trainer,
``evaluate_into_archive`` and the CLI), and the SBC cards built from them, are marked
non-sensitive only when every referenced corpus has a hash-bound synthetic sidecar.
This is stricter than the per-world training reports, which trust the caller's
``data_origin``. ``load`` cannot verify a ``prior_sha256``, so a hand-built identity
is trusted as written. The library never uploads archives.
"""

from __future__ import annotations

import copy
import errno
import io
import json
import os
import shutil
import stat
import tempfile
import weakref
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mktfm.inference._artifact_schema import (
    SHA256_PATTERN,
    exact_keys,
    parse_json,
    validate_profile,
)
from pymc_mktfm.inference._digest import digest_bytes, json_bytes
from pymc_mktfm.inference.result import DecompositionResult, _checked_evidence, _require_text

from .strata import result_effect_rows

if TYPE_CHECKING:
    from xarray import DataTree

ARCHIVE_FORMAT_VERSION = 1
ARCHIVE_KIND = "evaluation_archive"
#: Declared worlds per archive, and declared corpora: at both limits manifest.json stays
#: ≤ ≈7.5 MiB, under the strict reader's 8 MiB limit.
ARCHIVE_MAX_WORLDS = 4096
#: Quantities of ``EvaluationArchive.sbc_inputs``: per-family window totals.
WINDOW_TOTAL_FAMILIES = ("baseline", "channel", "control", "latent", "indirect")
_WINDOWS = ("query", "context")
#: pymc_mktfm.training.evaluation.DECOMPOSITION_HEADS (not imported: diagnostics never imports
#: training).
_HEADS = ("joint", "conditional")
#: Seeds torch.manual_seed accepts. Its CPU generator seeds mt19937 from the low 32 bits, so an
#: archive's call seeds must be consecutive: up to ARCHIVE_MAX_WORLDS consecutive seeds stay
#: distinct modulo 2**32 (CPU) and 2**64 (CUDA, MPS). Checked against ``start``/``stop``: ``in``
#: scans the whole range for a seed that is not an int.
_TORCH_SEEDS = range(-(1 << 63), 1 << 64)
_MANIFEST_LIMIT = 8 << 20
#: Largest finite float64; a recorded ``sales_scale`` above it cannot divide the draws.
_FLOAT64_MAX = float(np.finfo(np.float64).max)
_FILES = ("result.json", "arrays.npz", "truth.npz")  # read/check order
_EVIDENCE = ("truth", "graph_truth", "observed")
_CAPTURE_KEYS = (
    "capture",
    "data_split",
    "corpus_index",
    "row",
    "world_id",
    "seed",
    "data_sha256",
    "head",
    "inference_config",
)
_MANIFEST_KEYS = frozenset({"format_version", "kind", "identity", "entries", "archive_sha256"})
_IDENTITY_KEYS = frozenset({"model", "corpora", "evaluation"})
_MODEL_KEYS = frozenset(
    {
        "state_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "bundle_sha256",
        "profile",
        "source_step",
    }
)
_CORPUS_KEYS = frozenset(
    {
        "shard_seed",
        "n_worlds",
        "actual_inputs_sha256",
        "content_sha256",
        "source_sha256",
        "data_origin",
        "prior_sha256",
    }
)
_EVALUATION_KEYS = frozenset(
    {"worlds", "calls", "leg", "heads", "n_samples", "flow_steps", "mode", "units", "device"}
)
_WORLD_KEYS = frozenset({"corpus_index", "row", "cell_id", "context_length", "data_sha256"})
_CALL_KEYS = frozenset({"worlds", "seed"})
_ENTRY_KEYS = frozenset({"world", "window", "files"})
_RECORD_KEYS = frozenset({"sha256", "size"})


class ArchiveIntegrityError(ValueError):
    """An evaluation archive failed verification at ``path``, relative to ``archive``.

    ``archive`` is the absolute, resolved archive root and ``path`` the POSIX path
    of the offending file or directory relative to it (``"manifest.json"``,
    ``"entries/0003/truth.npz"``, ...). It pickles with its type, attributes and text.
    """

    def __init__(self, archive: Path, path: str, problem: str) -> None:
        super().__init__(archive, path, problem)  # all three, so that it unpickles
        self.archive, self.path, self.problem = Path(archive), path, problem

    def __str__(self) -> str:
        return f"{self.archive / self.path}: {self.problem}"


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One verified world window of an archive: metadata only, no arrays.

    ``index`` is its position in archive order (its manifest entry and directory);
    ``world`` indexes ``identity["evaluation"]["worlds"]`` and ``call`` the declared
    call holding it, whose draw ``seed`` it records. ``world_id`` is
    ``<label>/row:<row>`` and ``cell`` the independence key ``(label, cell_id)``,
    where ``label`` is ``shard:<shard_seed>`` or ``corpus:<corpus_index>``. ``path``
    is the absolute entry directory.
    """

    index: int
    world: int
    window: str
    call: int
    seed: int
    corpus_index: int
    row: int
    shard_seed: int | None
    cell_id: int
    context_length: int
    data_sha256: str
    world_id: str
    cell: tuple[str, int]
    path: Path


class _ReadError(ValueError):
    """The problem of one bounded read."""


class _EntryError(ValueError):
    """A decoding problem of one entry; ``file`` names its file (``""``: the entry itself)."""

    def __init__(self, file: str, problem: str) -> None:
        super().__init__(file, problem)
        self.file, self.problem = file, problem

    def __str__(self) -> str:
        return self.problem


def _read_bounded(path: Path, size: int | None = None, *, limit: int | None = None) -> bytes:
    """All bytes of the regular file ``path``; the only way any archive file is read.

    A symbolic link (``O_NOFOLLOW``, the last component only), a FIFO, a device, a
    file whose size differs from the recorded ``size`` or exceeds ``limit`` is refused
    from ``fstat`` before any byte is read; the open never blocks (``O_NONBLOCK``) and
    at most size + 1 bytes are read. Raises :class:`_ReadError` with the problem.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if not nofollow and os.path.islink(path):
        raise _ReadError("is a symbolic link")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _ReadError("is a symbolic link") from exc
        if exc.errno == errno.ENOENT:
            raise _ReadError("missing") from exc
        raise _ReadError(f"unreadable: {type(exc).__name__}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _ReadError("is not a regular file")
        if size is not None and info.st_size != size:
            raise _ReadError("SHA-256 or size differs from manifest.json")
        if limit is not None and info.st_size > limit:
            raise _ReadError("exceeds the 8 MiB format limit")
        blocks = []
        remaining = info.st_size + 1
        while remaining > 0:
            block = os.read(descriptor, min(remaining, 1 << 20))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
    except OSError as exc:
        raise _ReadError(f"unreadable: {type(exc).__name__}: {exc}") from exc
    finally:
        os.close(descriptor)
    data = b"".join(blocks)
    if len(data) != info.st_size:
        raise _ReadError("changed while being read")
    return data


def _corpus_label(index: int, shard_seed: int | None) -> str:
    return f"corpus:{index}" if shard_seed is None else f"shard:{shard_seed}"


def _not_new_or_empty(path: Path, exc: OSError | None = None) -> FileExistsError:
    reason = "" if exc is None else f" ({exc.strerror or exc})"
    return FileExistsError(f"evaluation archive needs a new or empty directory: {path}{reason}")


def _mount_point(path: Path) -> FileExistsError:
    return FileExistsError(
        f"{path} is a mount point or on another filesystem than its parent; "
        "pass a new subdirectory of it"
    )


def _require_new_or_empty(path: Path) -> None:
    """Refuse a symbolic link, a file or a non-empty directory at ``path``.

    A path that cannot even be inspected (too long, unsearchable) is refused alike,
    naming the reason, where Python reports the error (3.13). Python 3.14 reports such a
    path as missing instead; ``create`` then refuses it when its staging directory,
    name probe or path-length check fails, still before any inference.
    """
    try:
        occupied = path.is_symlink() or (
            path.exists() and (not path.is_dir() or any(path.iterdir()))
        )
    except OSError as exc:
        raise _not_new_or_empty(path, exc) from exc
    if occupied:
        raise _not_new_or_empty(path)


def _archive_target(out_dir: str | Path) -> Path:
    """The resolved new or empty directory an archive will be published at.

    ``out_dir`` is made absolute with its ``..`` kept and its final component must not
    be a symbolic link; its parent is created through that spelling, then the path is
    resolved as the OS resolves it. The current directory and its parents (by file
    identity) and an existing file or non-empty directory are refused with
    ``FileExistsError``, as is a path that cannot be inspected where Python reports
    the error (see :func:`_require_new_or_empty`).
    """
    raw = Path(out_dir).expanduser().absolute()
    try:
        linked = raw.is_symlink()
    except OSError as exc:
        raise _not_new_or_empty(raw, exc) from exc
    if linked:
        raise _not_new_or_empty(raw)
    try:
        raw.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # a dangling or looping link, or a file, in the parent path
        raise _not_new_or_empty(raw, exc) from exc
    path = Path(os.path.realpath(raw))
    try:
        exists = path.exists()
    except OSError as exc:
        raise _not_new_or_empty(path, exc) from exc
    if exists:
        try:
            cwd = os.getcwd()
        except FileNotFoundError:  # a deleted directory cannot be the target
            cwd = None
        if cwd is not None and any(
            os.path.samefile(path, folder) for folder in (cwd, *Path(cwd).parents)
        ):
            raise FileExistsError(
                "evaluation archive cannot replace the current directory or one of its "
                f"parents: {path}"
            )
        _require_new_or_empty(path)
    return path


def _probe_name(staging: Path, path: Path) -> None:
    """Create and remove a directory named like the target inside ``staging``.

    Only the name is tried, relative to the open staging directory where the platform
    allows it, on the target's filesystem. A name the filesystem cannot create (too
    long, outside its encoding, reserved characters) is then refused before any
    inference, not by ``close``'s ``os.replace`` after it.
    """
    try:
        if {os.mkdir, os.rmdir} <= os.supports_dir_fd:
            descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.mkdir(path.name, dir_fd=descriptor)
                os.rmdir(path.name, dir_fd=descriptor)
            finally:
                os.close(descriptor)
        else:
            (staging / path.name).mkdir()
            (staging / path.name).rmdir()
    except OSError as exc:
        raise _not_new_or_empty(path, exc) from exc


def _path_limit(directory: Path) -> int | None:
    """``PC_PATH_MAX`` of the filesystem holding ``directory``; ``None`` where it is unknown.

    Some filesystems (macOS exFAT and FAT32) do not answer for a directory although the
    kernel still enforces a limit on every path; the root's answer is that limit.
    """
    if not hasattr(os, "pathconf") or "PC_PATH_MAX" not in os.pathconf_names:
        return None
    for where in (directory, os.sep):
        try:
            limit = os.pathconf(where, "PC_PATH_MAX")
        except (OSError, ValueError):
            continue
        if limit > 0:
            return limit
    return None


def _check_path_lengths(path: Path, staging: Path, n_entries: int) -> None:
    """Refuse a target whose deepest published or staged file would reach ``PC_PATH_MAX``.

    The deepest published file is ``entries/NNNN/result.json``. While staging,
    ``DecompositionResult.save`` writes each entry in a temporary sibling
    ``entries/.result-XXXXXXXX/result/`` (tempfile names have 8 characters) before
    renaming it, which is deeper still under the staging directory.
    """
    limit = _path_limit(staging)
    if limit is None:
        return
    published = path / "entries" / f"{n_entries - 1:04d}" / "result.json"
    staged = staging / "entries" / (".result-" + "x" * 8) / "result" / "result.json"
    longest = max(len(os.fsencode(published)), len(os.fsencode(staged)))
    if longest >= limit:
        raise FileExistsError(
            f"evaluation archive path {path} is too long for its filesystem: its files need "
            f"paths of {longest} bytes, which must stay below {limit}; pass a shorter target"
        )


def _remove_staging(staging: Path, owner: int) -> None:
    """Remove a writer's staging directory, only in the process that created the writer.

    A forked child inherits the writer and its finalizer; run at the child's exit, the
    finalizer would otherwise delete the parent's open staging directory.
    """
    if os.getpid() == owner:
        shutil.rmtree(staging, ignore_errors=True)


def _check_digest(value: object, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        alternative = " or null" if nullable else ""
        raise ValueError(f"{label} must be a full lowercase SHA-256{alternative}")


def _check_text(value: object, label: str) -> None:
    """Nonempty text that encodes as UTF-8, as every capture's provenance must."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    _require_text(value, label)


def _check_identity(identity: Mapping[str, Any]) -> None:
    """Validate an archive identity, raising ``ValueError`` that names the field path."""
    exact_keys(identity, _IDENTITY_KEYS, "identity")
    model = exact_keys(identity["model"], _MODEL_KEYS, "identity.model")
    for key in ("state_sha256", "config_sha256"):
        _check_digest(model[key], f"identity.model.{key}")
    for key in ("checkpoint_sha256", "bundle_sha256"):
        _check_digest(model[key], f"identity.model.{key}", nullable=True)
    try:
        validate_profile(model["profile"])
    except ValueError as exc:
        raise ValueError(f"identity.model.{exc}") from None
    step = model["source_step"]
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("identity.model.source_step must be an integer >= 0 or null")

    corpora = identity["corpora"]
    if not isinstance(corpora, list) or not corpora:
        raise ValueError("identity.corpora must be a nonempty list")
    if len(corpora) > ARCHIVE_MAX_WORLDS:
        raise ValueError(
            f"identity.corpora holds {len(corpora)} corpora; an archive declares at most "
            f"{ARCHIVE_MAX_WORLDS}"
        )
    inputs: dict[str, int] = {}
    seeds: dict[int, int] = {}
    for index, corpus in enumerate(corpora):
        label = f"identity.corpora[{index}]"
        exact_keys(corpus, _CORPUS_KEYS, label)
        seed = corpus["shard_seed"]
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError(f"{label}.shard_seed must be an integer >= 0 or null")
        if type(corpus["n_worlds"]) is not int or corpus["n_worlds"] < 1:
            raise ValueError(f"{label}.n_worlds must be a positive integer")
        _check_digest(corpus["actual_inputs_sha256"], f"{label}.actual_inputs_sha256")
        for key in ("content_sha256", "source_sha256", "prior_sha256"):
            _check_digest(corpus[key], f"{label}.{key}", nullable=True)
        if corpus["data_origin"] not in ("synthetic", "unknown"):
            raise ValueError(f"{label}.data_origin must be 'synthetic' or 'unknown'")
        if corpus["data_origin"] == "synthetic" and corpus["prior_sha256"] is None:
            raise ValueError(f"{label}.data_origin 'synthetic' needs a verified prior_sha256")
        first = inputs.setdefault(corpus["actual_inputs_sha256"], index)
        if first != index:
            raise ValueError(
                f"{label}.actual_inputs_sha256 repeats identity.corpora[{first}]; an archive "
                "needs distinct corpora"
            )
        if seed is not None and (first := seeds.setdefault(seed, index)) != index:
            raise ValueError(f"{label}.shard_seed {seed} repeats identity.corpora[{first}]")

    evaluation = exact_keys(identity["evaluation"], _EVALUATION_KEYS, "identity.evaluation")
    worlds = evaluation["worlds"]
    if not isinstance(worlds, list) or not worlds:
        raise ValueError("identity.evaluation.worlds must be a nonempty list")
    if len(worlds) > ARCHIVE_MAX_WORLDS:
        raise ValueError(
            f"identity.evaluation.worlds holds {len(worlds)} worlds; an archive declares at "
            f"most {ARCHIVE_MAX_WORLDS}"
        )
    declared: dict[tuple[int, int], int] = {}
    for index, world in enumerate(worlds):
        label = f"identity.evaluation.worlds[{index}]"
        exact_keys(world, _WORLD_KEYS, label)
        corpus = world["corpus_index"]
        if type(corpus) is not int or not 0 <= corpus < len(corpora):
            raise ValueError(f"{label}.corpus_index must index identity.corpora")
        row, n_worlds = world["row"], corpora[corpus]["n_worlds"]
        if type(row) is not int or not 0 <= row < n_worlds:
            raise ValueError(f"{label}.row must lie in [0, {n_worlds}) of corpus {corpus}")
        if type(world["cell_id"]) is not int:
            raise ValueError(f"{label}.cell_id must be an integer")
        if type(world["context_length"]) is not int or world["context_length"] < 1:
            raise ValueError(f"{label}.context_length must be a positive integer")
        _check_digest(world["data_sha256"], f"{label}.data_sha256")
        if (first := declared.setdefault((corpus, row), index)) != index:
            raise ValueError(
                f"{label} repeats identity.evaluation.worlds[{first}] (corpus {corpus} row {row})"
            )

    calls = evaluation["calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("identity.evaluation.calls must be a nonempty list")
    covered = []
    for index, call in enumerate(calls):
        label = f"identity.evaluation.calls[{index}]"
        exact_keys(call, _CALL_KEYS, label)
        members = call["worlds"]
        if (
            not isinstance(members, list)
            or not members
            or any(type(member) is not int for member in members)
        ):
            raise ValueError(f"{label}.worlds must be a nonempty list of world indices")
        covered.extend(members)
        seed = call["seed"]
        if type(seed) is not int:
            raise ValueError(f"{label}.seed must be an integer")
        if index == 0 and not _TORCH_SEEDS.start <= seed < _TORCH_SEEDS.stop:
            raise ValueError(f"{label}.seed {seed} is outside the torch seed range [-2**63, 2**64)")
        if index and seed != calls[0]["seed"] + index:
            raise ValueError(f"{label}.seed {seed} must be calls[0].seed + {index}")
    if covered != list(range(len(worlds))):
        raise ValueError(
            f"identity.evaluation.calls must split worlds 0..{len(worlds) - 1} into "
            "consecutive blocks in order"
        )
    if not _TORCH_SEEDS.start <= (seed := calls[-1]["seed"]) < _TORCH_SEEDS.stop:
        raise ValueError(
            f"identity.evaluation.calls[{len(calls) - 1}].seed {seed} is outside the torch "
            "seed range [-2**63, 2**64)"
        )

    _check_text(evaluation["leg"], "identity.evaluation.leg")
    heads = evaluation["heads"]
    if (
        not isinstance(heads, list)
        or heads != [head for head in _HEADS if head in heads]
        or "conditional" not in heads
    ):
        raise ValueError(
            f"identity.evaluation.heads must list distinct heads of {list(_HEADS)} in that "
            "order, including 'conditional'"
        )
    for key in ("n_samples", "flow_steps"):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"identity.evaluation.{key} must be a positive integer")
    if evaluation["mode"] != "model_average":
        raise ValueError(
            "identity.evaluation.mode must be 'model_average': archives hold evaluator captures"
        )
    for key in ("units", "device"):
        _check_text(evaluation[key], f"identity.evaluation.{key}")


def _archive_order(identity: Mapping[str, Any]) -> list[tuple[int, str]]:
    """``(world, window)`` in the order the declared calls feed their ``result_sink``.

    Call by call; inside a call, worlds sorted stably by context length (the batch
    order of ``batches_for_rows``), each world's query window then its context window.
    """
    evaluation = identity["evaluation"]
    worlds = evaluation["worlds"]
    order = []
    for call in evaluation["calls"]:
        for world in sorted(call["worlds"], key=lambda member: worlds[member]["context_length"]):
            order.extend((world, window) for window in _WINDOWS)
    return order


def _world_lookup(identity: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    """The declared world index of each ``(corpus_index, row)``."""
    return {
        (world["corpus_index"], world["row"]): index
        for index, world in enumerate(identity["evaluation"]["worlds"])
    }


def _call_index(evaluation: Mapping[str, Any], world: int) -> int:
    """The declared call holding ``world``; calls are consecutive blocks in order."""
    calls = evaluation["calls"]
    low, high = 0, len(calls) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if calls[middle]["worlds"][0] <= world:
            low = middle
        else:
            high = middle - 1
    return low


def _capture_world(
    result: DecompositionResult,
    identity: Mapping[str, Any],
    lookup: Mapping[tuple[int, int], int],
) -> int:
    """The declared world of an evaluator capture, after checking it against the identity.

    Shared by ``add``, the writer's read-back and the reader. Each problem raises a
    ``ValueError`` describing it.
    """
    provenance = result.provenance
    missing = [key for key in _CAPTURE_KEYS if key not in provenance]
    if missing:
        raise ValueError(
            "only evaluator captures of declared corpus worlds enter an evaluation archive: "
            f"this result lacks {missing}"
        )
    corpus, row = provenance["corpus_index"], provenance["row"]
    world = lookup.get((corpus, row)) if type(corpus) is int and type(row) is int else None
    if world is None:
        raise ValueError(f"corpus {corpus!r} row {row!r} is not a declared world of this archive")
    evaluation = identity["evaluation"]
    declared = evaluation["worlds"][world]
    seed = evaluation["calls"][_call_index(evaluation, world)]["seed"]
    inference = provenance["inference_config"]
    model = inference.get("model") if isinstance(inference, dict) else None
    configuration = digest_bytes(json_bytes(model)) if isinstance(model, dict) else None
    for name, actual, expected in (
        ("world_id", provenance["world_id"], f"corpus:{corpus}/row:{row}"),
        ("data_sha256", provenance["data_sha256"], declared["data_sha256"]),
        ("data_split", provenance["data_split"], evaluation["leg"]),
        ("seed", provenance["seed"], seed),
        ("head", provenance["head"], "conditional"),
        ("mode", result.mode, evaluation["mode"]),
        ("n_samples", result.n_samples, evaluation["n_samples"]),
        ("units", result.units, evaluation["units"]),
        ("context_length", result.context_length, declared["context_length"]),
        ("model configuration digest", configuration, identity["model"]["config_sha256"]),
    ):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{name} {actual!r} differs from the archive's {expected!r}")
    scale = provenance.get("sales_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not 0 < scale <= _FLOAT64_MAX
    ):
        raise ValueError("sales_scale must be a finite positive number")
    edges = result.graphs.shape[1]
    families = provenance.get("edge_families")
    if (
        not isinstance(families, list)
        or len(families) != edges
        or not all(isinstance(family, str) and family for family in families)
    ):
        raise ValueError("edge_families must name one family per edge slot")
    endpoints = provenance.get("edge_endpoints")
    if (
        not isinstance(endpoints, list)
        or len(endpoints) != edges
        or not all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(node, str) and node for node in pair)
            for pair in endpoints
        )
    ):
        raise ValueError("edge_endpoints must hold one [source, target] pair per edge slot")
    return world


def _problem(exc: Exception) -> str:
    """An exception as a problem description; a plain ``ValueError`` reads as it is."""
    return str(exc) if type(exc) is ValueError else f"{type(exc).__name__}: {exc}"


def _decode_entry(
    data: Mapping[str, bytes],
    identity: Mapping[str, Any],
    lookup: Mapping[tuple[int, int], int],
    world: int,
    window: str,
    index: int,
) -> tuple[DecompositionResult, np.ndarray, np.ndarray, np.ndarray]:
    """Decode entry ``index`` from exactly the bytes read, checked as the reader checks it.

    Returns ``(result, truth, graph_truth, observed)`` with read-only float64 evidence.
    Any failure raises :class:`_EntryError` naming the file (``""``: the entry).
    """
    try:
        payload = parse_json(data["result.json"])
    except Exception as exc:
        raise _EntryError("result.json", f"not strict JSON: {type(exc).__name__}: {exc}") from exc
    try:
        DecompositionResult._checked_payload(payload)
    except Exception as exc:
        raise _EntryError("result.json", f"not a saved result: {_problem(exc)}") from exc
    try:
        result = DecompositionResult._from_saved(payload, data["arrays.npz"])
    except Exception as exc:
        raise _EntryError("", f"saved result does not load: {type(exc).__name__}: {exc}") from exc
    try:
        found = _capture_world(result, identity, lookup)
    except Exception as exc:
        raise _EntryError("result.json", _problem(exc)) from exc
    worlds = identity["evaluation"]["worlds"]
    if found != world:
        held, expected = worlds[found], worlds[world]
        raise _EntryError(
            "result.json",
            f"holds corpus {held['corpus_index']} row {held['row']}; manifest.json "
            f"entries[{index}] declares corpus {expected['corpus_index']} row {expected['row']}",
        )
    if result.window != window:
        raise _EntryError(
            "result.json",
            f"holds the {result.window} window; manifest.json entries[{index}] declares {window}",
        )
    try:
        stored = np.load(io.BytesIO(data["truth.npz"]), allow_pickle=False)
        if not isinstance(stored, np.lib.npyio.NpzFile):
            raise ValueError("is not an npz archive")
        with stored:
            if sorted(stored.files) != sorted(_EVIDENCE):
                raise ValueError(f"must hold exactly {sorted(_EVIDENCE)}")
            arrays = [stored[name] for name in _EVIDENCE]
        evidence = _checked_evidence(result, *arrays, stored=True)
    except Exception as exc:
        raise _EntryError("truth.npz", _problem(exc)) from exc
    for array in evidence.values():
        array.setflags(write=False)
    return result, evidence["truth"], evidence["graph_truth"], evidence["observed"]


def _check_entries(entries: object, identity: Mapping[str, Any]) -> None:
    """Validate the manifest entries and that they follow archive order exactly."""
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    n_worlds = len(identity["evaluation"]["worlds"])
    for index, entry in enumerate(entries):
        label = f"entries[{index}]"
        exact_keys(entry, _ENTRY_KEYS, label)
        if type(entry["world"]) is not int or not 0 <= entry["world"] < n_worlds:
            raise ValueError(f"{label}.world must index identity.evaluation.worlds")
        if entry["window"] not in _WINDOWS:
            raise ValueError(f"{label}.window must be 'query' or 'context'")
        files = exact_keys(entry["files"], frozenset(_FILES), f"{label}.files")
        for name in _FILES:
            record = exact_keys(files[name], _RECORD_KEYS, f"{label}.files[{name!r}]")
            _check_digest(record["sha256"], f"{label}.files[{name!r}].sha256")
            if type(record["size"]) is not int or record["size"] < 1:
                raise ValueError(f"{label}.files[{name!r}].size must be a positive integer")
    order = _archive_order(identity)
    if len(entries) != len(order):
        raise ValueError(
            f"entries hold {len(entries)} world windows; the identity declares {len(order)}"
        )
    for index, (entry, (world, window)) in enumerate(zip(entries, order, strict=True)):
        if (entry["world"], entry["window"]) != (world, window):
            raise ValueError(
                f"entries[{index}] is world {entry['world']} {entry['window']}; the declared "
                f"calls capture world {world} {window} there"
            )


def _check_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Validate a parsed manifest.json; every failure names ``manifest.json``."""

    def invalid(problem: str) -> ArchiveIntegrityError:
        return ArchiveIntegrityError(root, "manifest.json", problem)

    version = manifest.get("format_version")
    if type(version) is not int or version != ARCHIVE_FORMAT_VERSION:  # before the key set
        raise invalid(
            f"unsupported evaluation archive format_version {version!r}; this release reads "
            f"version {ARCHIVE_FORMAT_VERSION}"
        )
    try:
        exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    except ValueError as exc:
        raise invalid(str(exc)) from exc
    if manifest["kind"] != ARCHIVE_KIND:
        raise invalid(f"kind must be {ARCHIVE_KIND!r}")
    digest = manifest["archive_sha256"]
    body = {key: value for key, value in manifest.items() if key != "archive_sha256"}
    if (
        not isinstance(digest, str)
        or SHA256_PATTERN.fullmatch(digest) is None
        or digest != digest_bytes(json_bytes(body))
    ):
        raise invalid("archive_sha256 does not match the manifest contents")
    try:
        _check_identity(manifest["identity"])
        _check_entries(manifest["entries"], manifest["identity"])
    except ValueError as exc:
        raise invalid(str(exc)) from exc


def _archive_entries(
    root: Path, identity: Mapping[str, Any], manifest_entries: list[dict[str, Any]]
) -> tuple[ArchiveEntry, ...]:
    """The :class:`ArchiveEntry` of every verified manifest entry, under ``root``."""
    evaluation, corpora = identity["evaluation"], identity["corpora"]
    entries = []
    for index, item in enumerate(manifest_entries):
        world = evaluation["worlds"][item["world"]]
        call = _call_index(evaluation, item["world"])
        corpus, row, cell_id = world["corpus_index"], world["row"], world["cell_id"]
        shard_seed = corpora[corpus]["shard_seed"]
        label = _corpus_label(corpus, shard_seed)
        entries.append(
            ArchiveEntry(
                index=index,
                world=item["world"],
                window=item["window"],
                call=call,
                seed=evaluation["calls"][call]["seed"],
                corpus_index=corpus,
                row=row,
                shard_seed=shard_seed,
                cell_id=cell_id,
                context_length=world["context_length"],
                data_sha256=world["data_sha256"],
                world_id=f"{label}/row:{row}",
                cell=(label, cell_id),
                path=root / "entries" / f"{index:04d}",
            )
        )
    return tuple(entries)


def _lstat_mode(root: Path, relative: str) -> int:
    """``st_mode`` of ``root / relative``, never following it: a symbolic link is refused."""
    try:
        mode = os.lstat(root / relative).st_mode
    except FileNotFoundError:
        raise ArchiveIntegrityError(root, relative, "missing") from None
    except OSError as exc:
        raise ArchiveIntegrityError(
            root, relative, f"unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    if stat.S_ISLNK(mode):
        raise ArchiveIntegrityError(root, relative, "is a symbolic link")
    return mode


def _check_folder(root: Path, relative: str, layout: Mapping[str, Any]) -> None:
    """The visible names of one directory are exactly ``layout``'s, of the right kinds.

    ``layout`` maps each name to ``None`` (a regular file) or the layout of a
    subdirectory, which is checked depth first in sorted order.
    """
    try:
        present = {name for name in os.listdir(root / relative) if not name.startswith(".")}
    except OSError as exc:
        raise ArchiveIntegrityError(
            root, relative or ".", f"unreadable: {type(exc).__name__}: {exc}"
        ) from exc
    for name in sorted(present.union(layout)):
        path = f"{relative}/{name}" if relative else name
        if name not in layout:
            raise ArchiveIntegrityError(root, path, "is not listed in manifest.json")
        if name not in present:
            raise ArchiveIntegrityError(root, path, "missing")
        mode = _lstat_mode(root, path)
        if layout[name] is None:
            if not stat.S_ISREG(mode):
                raise ArchiveIntegrityError(root, path, "is not a regular file")
        elif not stat.S_ISDIR(mode):
            raise ArchiveIntegrityError(root, path, "is not a directory")
        else:
            _check_folder(root, path, layout[name])


def _entries_layout(n_entries: int) -> dict[str, Any]:
    """The ``_check_folder`` layout of ``entries/``: ``n_entries`` directories of ``_FILES``."""
    files = dict.fromkeys(_FILES)
    return {f"{index:04d}": files for index in range(n_entries)}


def _check_tree(root: Path, n_entries: int) -> None:
    """Before any entry is read: the archive holds exactly its listed files, and no link.

    Hidden names (a leading ``.``, such as ``.DS_Store``) are ignored everywhere.
    """
    _check_folder(root, "", {"manifest.json": None, "entries": _entries_layout(n_entries)})


def _check_entry_dir(root: Path, index: int) -> None:
    """``lstat`` ``entries/`` and ``entries/NNNN`` before any file of entry ``index`` opens."""
    for relative in ("entries", f"entries/{index:04d}"):
        if not stat.S_ISDIR(_lstat_mode(root, relative)):
            raise ArchiveIntegrityError(root, relative, "is not a directory")


def _read_files(root: Path, index: int, files: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    """The bytes of entry ``index``'s files, each size- and hash-checked against ``files``.

    The entry directories are checked first; then each file is read through the
    bounded reader and checked before the next is read, so the first differing file
    is the one named.
    """
    _check_entry_dir(root, index)
    folder = f"entries/{index:04d}"
    data = {}
    for name in _FILES:
        record = files[name]
        try:
            content = _read_bounded(root / folder / name, record["size"])
        except _ReadError as exc:
            raise ArchiveIntegrityError(root, f"{folder}/{name}", str(exc)) from exc
        if digest_bytes(content) != record["sha256"]:
            raise ArchiveIntegrityError(
                root, f"{folder}/{name}", "SHA-256 or size differs from manifest.json"
            )
        data[name] = content
    return data


def _read_entry(
    root: Path,
    entry: ArchiveEntry,
    files: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    lookup: Mapping[tuple[int, int], int],
) -> tuple[DecompositionResult, np.ndarray, np.ndarray, np.ndarray]:
    """Read, check and decode one entry against its manifest ``files`` records.

    :func:`_read_files` reads and checks the files; the same bytes are decoded.
    """
    data = _read_files(root, entry.index, files)
    try:
        return _decode_entry(data, identity, lookup, entry.world, entry.window, entry.index)
    except _EntryError as exc:
        folder = f"entries/{entry.index:04d}"
        relative = f"{folder}/{exc.file}" if exc.file else folder
        raise ArchiveIntegrityError(root, relative, exc.problem) from exc


def _check_staged(staging: Path, entries: list[dict[str, Any]]) -> None:
    """The staging directory still holds exactly the entries ``add`` recorded, unchanged.

    The same checks as ``load``'s, before the manifest exists: the tree, then each
    file's size and hash. Raises :class:`ArchiveIntegrityError` relative to ``staging``.
    """
    _check_folder(staging, "", {"entries": _entries_layout(len(entries))})
    for index, item in enumerate(entries):
        _read_files(staging, index, item["files"])


def _check_window(window: object) -> None:
    if window not in _WINDOWS:
        raise ValueError("window must be 'query' or 'context'")


def _sbc_provenance(archive: EvaluationArchive, window: str, prior: object) -> dict[str, Any]:
    """The SBC provenance of one window: dataset, prior, model, settings and seeds."""
    identity = archive._identity
    if prior is None:
        prior = archive.prior
    if prior is None:
        corpora = identity["corpora"]
        labels = [
            _corpus_label(index, corpora[index]["shard_seed"])
            for index in archive._referenced
            if corpora[index]["prior_sha256"] is None
        ]
        raise ValueError(
            f"{archive.path}: SBC needs a prior description; {', '.join(labels)} record no "
            "verified generator prior; pass prior=..."
        )
    model, evaluation = identity["model"], identity["evaluation"]
    return {
        "independent_worlds": True,
        "prior": prior,
        "data_sha256": archive.dataset_sha256,
        "synthetic_training_evaluation": archive.synthetic,
        "archive_sha256": archive.archive_sha256,
        "state_sha256": model["state_sha256"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "data_split": evaluation["leg"],
        "head": "conditional",
        "mode": evaluation["mode"],
        "window": window,
        "units": evaluation["units"],
        "n_samples": evaluation["n_samples"],
        "evaluation_seeds": [call["seed"] for call in evaluation["calls"]],
        "independence": "first world of each (corpus, cell_id) in archive order",
    }


def family_window_totals(result: DecompositionResult, truth) -> tuple[np.ndarray, np.ndarray]:
    """Per-family window totals of one world window: ``(draws [S, Q], totals [Q])``, float64.

    ``Q`` follows :data:`WINDOW_TOTAL_FAMILIES`. A family's total sums its active
    components (``component_mask``) over the window, per draw and in ``truth``
    (natural ``[time, component]``, aligned with ``result.components``), so padded
    truth never enters and a family without an active component gives exact zeros.
    These quantities exist for every world, whatever its window length or active
    entities, and their ranks do not depend on its ``sales_scale``. Pure NumPy.
    """
    components = np.asarray(result.components, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if target.shape != components.shape[1:]:
        raise ValueError("truth must align with result.components [time, component]")
    active = np.asarray(result.component_mask, dtype=bool)
    families = np.asarray(result.component_families)
    draws = np.zeros((components.shape[0], len(WINDOW_TOTAL_FAMILIES)))
    totals = np.zeros(len(WINDOW_TOTAL_FAMILIES))
    for index, family in enumerate(WINDOW_TOTAL_FAMILIES):
        columns = np.flatnonzero(active & (families == family))
        draws[:, index] = components[:, :, columns].sum(axis=(1, 2))
        totals[index] = target[:, columns].sum()
    return draws, totals


class EvaluationArchiveWriter:
    """Stages verified evaluator captures and publishes them as one archive.

    Built only by :meth:`EvaluationArchive.create`. ``add`` takes the four
    ``result_sink`` arguments, writes the world window into a hidden staging
    directory next to the target and reads it back through the archive reader before
    accepting it. ``close`` checks completeness, re-checks every staged file against
    what ``add`` recorded and publishes with one ``os.replace``, so a published archive
    passes :meth:`EvaluationArchive.load`; nothing appears at the target before.
    States: ``open``; ``closed`` after ``close``; ``aborted`` after ``abort`` or an
    exception leaving a ``with`` block; ``failed`` after an error while writing or
    publishing. The staging directory is removed on abort and failure, and when a
    writer that was never closed is garbage collected or the interpreter exits, only
    ever by the process that created the writer (never by a forked child). A refused
    capture changes nothing. No random numbers are drawn.

    ``add`` verifies that a capture is an evaluator capture (it carries the capture
    keys) of a declared world with that world's encoded inputs (``data_sha256``) and
    context length, its call's seed, the declared leg, ``n_samples``, mode and units,
    the conditional head and the declared model configuration; that its edge
    provenance and evidence are well formed; and that it comes strictly later in
    archive order. Captures record neither the ``heads`` they were sampled with, nor
    the weights, nor which worlds shared an ``evaluate_rows`` call, so ``add`` cannot
    verify the declared ``heads``, ``model.state_sha256`` or call partition. An
    archive therefore replays from its identity only when exactly the declared calls
    of the declared model filled it, as :func:`pymc_mktfm.training.evaluate_into_archive`,
    the trainer and the CLI do.
    """

    def __init__(self, path: Path, identity: dict[str, Any], staging: Path) -> None:
        self._path = path
        self._identity = identity
        self._staging = staging
        self._lookup = _world_lookup(identity)
        self._order = _archive_order(identity)
        self._positions = {item: index for index, item in enumerate(self._order)}
        self._last = -1
        self._added: set[tuple[int, str]] = set()
        self._entries: list[dict[str, Any]] = []
        self._state = "open"
        self._archive_sha256: str | None = None
        # Removes the staging directory of a writer dropped, or alive at exit, before it
        # is closed, in this process only; detached once the archive is published.
        self._cleanup = weakref.finalize(self, _remove_staging, staging, os.getpid())

    @property
    def path(self) -> Path:
        """The final archive directory, resolved as :meth:`EvaluationArchive.load` resolves it."""
        return self._path

    @property
    def archive_sha256(self) -> str | None:
        """The published manifest's ``archive_sha256``; ``None`` until ``close`` succeeds."""
        return self._archive_sha256

    def _fail(self) -> None:
        self._state = "failed"
        self._cleanup()

    def add(self, result: DecompositionResult, truth, graph_truth, observed) -> None:
        """Archive one evaluator capture; signature-compatible with ``ResultSink``.

        Checks the capture as the class describes and refuses it without changing
        anything: ``TypeError`` for a non-result, ``ValueError`` for a capture that
        does not belong to the archive (an undeclared world or other inputs, other
        settings or model configuration, a repeat, a capture out of archive order,
        invalid evidence). The ``heads``, weights and call partition it was sampled
        with are not recorded and cannot be checked. Any failure while writing fails
        the writer.
        """
        if self._state != "open":
            raise ValueError(
                f"evaluation archive {self._path} is {self._state}; only an open archive "
                "accepts results"
            )
        if not isinstance(result, DecompositionResult):
            raise TypeError(f"add() takes a DecompositionResult, got {type(result).__name__}")
        provenance, window = result.provenance, result.window
        prefix = (
            f"cannot archive corpus {provenance.get('corpus_index')} row "
            f"{provenance.get('row')} {window}: "
        )
        try:
            world = _capture_world(result, self._identity, self._lookup)
        except ValueError as exc:
            if any(key not in provenance for key in _CAPTURE_KEYS):
                raise  # the observation-only refusal stands alone
            raise ValueError(f"{prefix}{exc}") from None
        key = (world, window)
        if key in self._added:
            raise ValueError(f"{prefix}this world window was already added")
        position = self._positions[key]
        if position <= self._last:
            last_world, last_window = self._order[self._last]
            previous = self._identity["evaluation"]["worlds"][last_world]
            raise ValueError(
                f"{prefix}it was captured out of the declared world order (after corpus "
                f"{previous['corpus_index']} row {previous['row']} {last_window}); evaluate "
                "the declared calls in order with exactly their declared rows"
            )
        evidence = _checked_evidence(result, truth, graph_truth, observed, stored=False)
        index = len(self._entries)
        folder = self._staging / "entries" / f"{index:04d}"
        try:
            result.save(folder)
            np.savez_compressed(folder / "truth.npz", **evidence)
            data = {}
            for name in _FILES:
                try:
                    data[name] = _read_bounded(folder / name)
                except _ReadError as exc:
                    raise ValueError(
                        f"{prefix}entries/{index:04d}/{name} would not pass the archive "
                        f"reader: {exc}"
                    ) from exc
            try:
                _decode_entry(data, self._identity, self._lookup, world, window, index)
            except _EntryError as exc:
                relative = f"entries/{index:04d}" + (f"/{exc.file}" if exc.file else "")
                raise ValueError(
                    f"{prefix}{relative} would not pass the archive reader: {exc.problem}"
                ) from exc
        except BaseException:
            self._fail()
            raise
        files = {
            name: {"sha256": digest_bytes(content), "size": len(content)}
            for name, content in data.items()
        }
        self._entries.append({"world": world, "window": window, "files": files})
        self._added.add(key)
        self._last = position

    def close(self) -> Path:
        """Publish the complete archive atomically and return its path; idempotent.

        An incomplete archive, staged files that changed or vanished since ``add``
        recorded them (``ValueError`` naming the file), a target that is no longer new or
        empty, or a failed ``os.replace`` (``FileExistsError`` naming a non-empty target or
        a mount point) fails the writer; nothing is published then.
        """
        if self._state == "closed":
            return self._path
        if self._state != "open":
            raise ValueError(
                f"evaluation archive {self._path} is {self._state}; nothing was published"
            )
        missing = [item for item in self._order if item not in self._added]
        if missing:
            self._fail()
            world, window = missing[0]
            first = self._identity["evaluation"]["worlds"][world]
            raise ValueError(
                f"evaluation archive {self._path} is incomplete: {len(missing)} of "
                f"{len(self._order)} world windows were never added (first: corpus "
                f"{first['corpus_index']} row {first['row']} {window}); nothing was published"
            )
        path = self._path
        try:
            try:
                _check_staged(self._staging, self._entries)
            except ArchiveIntegrityError as exc:
                raise ValueError(
                    f"evaluation archive {path} cannot be published: its staged files changed "
                    f"after add recorded them ({exc.path}: {exc.problem}); nothing was published"
                ) from exc
            manifest = {
                "format_version": ARCHIVE_FORMAT_VERSION,
                "kind": ARCHIVE_KIND,
                "identity": self._identity,
                "entries": self._entries,
            }
            manifest["archive_sha256"] = digest_bytes(json_bytes(manifest))
            data = json_bytes(manifest)
            if len(data) > _MANIFEST_LIMIT:
                raise ValueError(
                    "evaluation archive manifest would exceed the 8 MiB limit of the strict "
                    "reader; split the evaluation"
                )
            (self._staging / "manifest.json").write_bytes(data)
            _require_new_or_empty(path)
            try:
                os.replace(self._staging, path)
            except OSError as exc:
                if exc.errno in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR):
                    raise _not_new_or_empty(path) from exc
                if exc.errno in (errno.EXDEV, errno.EBUSY):  # macOS, Linux
                    raise _mount_point(path) from exc
                raise
            self._cleanup.detach()  # published: nothing is left to remove
        except BaseException:
            self._fail()
            raise
        self._state = "closed"
        self._archive_sha256 = manifest["archive_sha256"]
        return path

    def abort(self) -> None:
        """Discard an open writer's staging directory; a no-op in any other state."""
        if self._state == "open":
            self._state = "aborted"
            self._cleanup()

    def __enter__(self) -> EvaluationArchiveWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Abort on an exception (never suppressed); otherwise close an open writer.

        A clean exit from a failed writer raises: its failure was swallowed in the block.
        """
        if exc_type is not None:
            self.abort()
        elif self._state == "open":
            self.close()
        elif self._state == "failed":
            raise ValueError(f"evaluation archive {self._path} failed; nothing was published")


class EvaluationArchive:
    """A loaded, verified, read-only evaluation archive (see the module docstring).

    Only :meth:`load` builds one; :meth:`create` returns a writer. It keeps the
    verified manifest, never arrays: every read re-reads and re-verifies its entries,
    one at a time, against that manifest.
    """

    def __init__(
        self, root: Path, manifest: dict[str, Any], entries: tuple[ArchiveEntry, ...]
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._identity = manifest["identity"]
        self._entries = entries
        self._lookup = _world_lookup(self._identity)
        self._referenced = sorted(
            {world["corpus_index"] for world in self._identity["evaluation"]["worlds"]}
        )

    @classmethod
    def create(cls, out_dir: str | Path, identity: Mapping[str, Any]) -> EvaluationArchiveWriter:
        """A writer for a new archive at ``out_dir``, declared by ``identity``.

        Everything is checked before any inference: the identity (JSON values only,
        ``ValueError`` naming the field path) and the target (``FileExistsError``: a
        symlink, the current directory or a parent of it, an existing file or
        non-empty directory, a path that cannot be inspected, a parent the staging
        directory cannot be created in, a name the filesystem cannot create, a path
        too long for the archive's files, a mount point or another filesystem). The
        target is resolved as the OS resolves it, so
        ``writer.path == EvaluationArchive.load(writer.path).path``; its parent is
        created. Use the writer as a context manager: it closes on success and aborts
        on an exception. Fill it by running exactly the declared ``evaluate_rows`` calls
        of the declared model, as :func:`pymc_mktfm.training.evaluate_into_archive` does:
        ``add`` cannot check the heads, weights or call partition behind a capture, and
        only such an archive replays from its identity.
        """
        try:
            normalized = json.loads(json_bytes(identity))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"archive identity must hold only JSON values: {exc}") from exc
        _check_identity(normalized)
        path = _archive_target(out_dir)
        # A bounded prefix keeps the staging name legal for every legal target name.
        label = os.fsencode(path.name)[:64].decode("utf-8", "ignore")
        try:
            staging = Path(tempfile.mkdtemp(prefix=f".{label}-", dir=path.parent))
        except OSError as exc:  # a parent it cannot be created in, or a path too long
            raise _not_new_or_empty(path, exc) from exc
        try:
            _probe_name(staging, path)
            _check_path_lengths(path, staging, 2 * len(normalized["evaluation"]["worlds"]))
            (staging / "entries").mkdir()
            if path.exists() and (
                os.stat(staging).st_dev != os.stat(path).st_dev or os.path.ismount(path)
            ):
                raise _mount_point(path)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return EvaluationArchiveWriter(path, normalized, staging)

    @classmethod
    def load(cls, path: str | Path) -> EvaluationArchive:
        """Verify the archive at ``path`` (a symlinked root is followed once) and return it.

        Every file is read once through the bounded reader, size- and hash-checked
        against the manifest and decoded from the same bytes; every manifest field,
        cross-check and shape is validated, one entry in memory at a time. Raises
        ``FileNotFoundError`` without an archive directory or manifest and
        :class:`ArchiveIntegrityError` naming the offending path otherwise.
        """
        root = Path(os.path.realpath(Path(path).expanduser()))
        try:
            mode = os.stat(root).st_mode
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                raise ArchiveIntegrityError(
                    root, ".", f"unreadable: {type(exc).__name__}: {exc}"
                ) from exc
            mode = 0
        if not stat.S_ISDIR(mode):
            raise FileNotFoundError(f"{root}: not an evaluation archive directory")
        try:
            os.lstat(root / "manifest.json")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{root / 'manifest.json'}: evaluation archive manifest not found"
            ) from None
        except OSError as exc:
            raise ArchiveIntegrityError(
                root, "manifest.json", f"unreadable: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            data = _read_bounded(root / "manifest.json", limit=_MANIFEST_LIMIT)
        except _ReadError as exc:
            raise ArchiveIntegrityError(root, "manifest.json", str(exc)) from exc
        try:
            manifest = parse_json(data)
        except Exception as exc:
            raise ArchiveIntegrityError(
                root, "manifest.json", f"not strict JSON: {type(exc).__name__}: {exc}"
            ) from exc
        _check_manifest(root, manifest)
        identity = manifest["identity"]
        entries = _archive_entries(root, identity, manifest["entries"])
        _check_tree(root, len(entries))
        lookup = _world_lookup(identity)
        for entry, item in zip(entries, manifest["entries"], strict=True):
            _read_entry(root, entry, item["files"], identity, lookup)
        return cls(root, manifest, entries)

    @property
    def path(self) -> Path:
        """The absolute, resolved archive root."""
        return self._root

    @property
    def identity(self) -> dict[str, Any]:
        """A copy of the verified identity: model, corpora and evaluation."""
        return copy.deepcopy(self._identity)

    @property
    def archive_sha256(self) -> str:
        """The verified manifest's self-digest; compare it to pin an archive."""
        return self._manifest["archive_sha256"]

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        """Every world window, in archive order."""
        return self._entries

    @property
    def dataset_sha256(self) -> str:
        """Digest of the sorted input digests of the corpora the declared worlds come from.

        Independent of the model, of which of their worlds were evaluated and of the
        corpus order; the ``data_sha256`` of the SBC inputs.
        """
        corpora = self._identity["corpora"]
        return digest_bytes(
            json_bytes(sorted(corpora[index]["actual_inputs_sha256"] for index in self._referenced))
        )

    @property
    def synthetic(self) -> bool:
        """Whether every referenced corpus is declared synthetic (hence SBC cards non-sensitive)."""
        corpora = self._identity["corpora"]
        return all(corpora[index]["data_origin"] == "synthetic" for index in self._referenced)

    @property
    def prior(self) -> dict[str, Any] | None:
        """The declared generator priors of the referenced corpora; ``None`` unless all have one."""
        corpora = self._identity["corpora"]
        if any(corpora[index]["prior_sha256"] is None for index in self._referenced):
            return None
        return {
            "name": "verified_generation_priors",
            "corpus_prior_hashes": {
                _corpus_label(index, corpora[index]["shard_seed"]): corpora[index]["prior_sha256"]
                for index in self._referenced
            },
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[DecompositionResult, np.ndarray, np.ndarray, np.ndarray]]:
        """Every ``(result, truth, graph_truth, observed)`` in archive order, re-verified."""
        return self.records()

    def records(
        self, window: str | None = None
    ) -> Iterator[tuple[DecompositionResult, np.ndarray, np.ndarray, np.ndarray]]:
        """The records of ``window`` (``None``: both), lazily and in archive order."""
        if window is not None:
            _check_window(window)
        return (
            self.record(entry.index)
            for entry in self._entries
            if window is None or entry.window == window
        )

    def record(self, index: int) -> tuple[DecompositionResult, np.ndarray, np.ndarray, np.ndarray]:
        """Entry ``index`` as ``(result, truth, graph_truth, observed)``, re-verified.

        The entry directories are checked and its three files read, size- and
        hash-checked against the manifest verified at ``load`` and decoded from the
        same bytes; the evidence arrays are read-only float64.
        """
        entry = self._entries[index]
        files = self._manifest["entries"][entry.index]["files"]
        return _read_entry(self._root, entry, files, self._identity, self._lookup)

    def independent_entries(self, window: str) -> tuple[ArchiveEntry, ...]:
        """The first entry of ``window`` of each cell ``(label, cell_id)``, in archive order.

        The rule ``conditional_sbc_report`` applies to record order: earlier calls
        first, then shorter context windows, then the requested order.
        """
        _check_window(window)
        seen = set()
        selected = []
        for entry in self._entries:
            if entry.window == window and entry.cell not in seen:
                seen.add(entry.cell)
                selected.append(entry)
        return tuple(selected)

    def _cells(self, window: str) -> tuple[ArchiveEntry, ...]:
        selected = self.independent_entries(window)
        if len(selected) < 2:
            raise ValueError(
                f"{self._root}: {window} SBC needs at least two independent generator cells; "
                f"the archive has {len(selected)} across "
                f"{len(self._identity['evaluation']['worlds'])} worlds"
            )
        return selected

    def effect_rows(self, window: str) -> list[dict[str, Any]]:
        """:func:`~pymc_mktfm.diagnostics.result_effect_rows` of every world of ``window``.

        In archive order, each row's ``world`` set to the archive world id; feed them
        to ``strata_summary``. Imports the training metrics at call time.
        """
        _check_window(window)
        rows = []
        for entry in self._entries:
            if entry.window != window:
                continue
            result, truth, graph_truth, _ = self.record(entry.index)
            rows.extend(
                {**row, "world": entry.world_id}
                for row in result_effect_rows(result, truth, graph_truth)
            )
        return rows

    def sbc_inputs(self, window: str, *, prior: object = None) -> dict[str, Any]:
        """Keyword arguments of :func:`~pymc_mktfm.diagnostics.sbc_report` for ``window``.

        One world per cell (:meth:`independent_entries`), its
        :func:`family_window_totals` as the ``Q`` quantities; ``prior`` defaults to
        :attr:`prior`. Needs two cells and a prior, checked before any read.
        """
        _check_window(window)
        selected = self._cells(window)
        provenance = _sbc_provenance(self, window, prior)
        shape = (len(selected), self._identity["evaluation"]["n_samples"])
        samples = np.empty((*shape, len(WINDOW_TOTAL_FAMILIES)))
        truth = np.empty((len(selected), len(WINDOW_TOTAL_FAMILIES)))
        for row, entry in enumerate(selected):
            result, target, _, _ = self.record(entry.index)
            samples[row], truth[row] = family_window_totals(result, target)
        return {
            "samples": samples,
            "truth": truth,
            "world_ids": [entry.world_id for entry in selected],
            "quantity_names": list(WINDOW_TOTAL_FAMILIES),
            "provenance": provenance,
        }

    def conditional_sbc_inputs(self, window: str, *, prior: object = None) -> dict[str, Any]:
        """Keyword arguments of :func:`~pymc_mktfm.diagnostics.conditional_sbc_report`.

        The records of one world per cell of ``window`` with their ids and cells, the
        same report as from every record of the window; ``prior`` defaults to
        :attr:`prior`. Needs two cells and a prior, checked before any read.
        """
        _check_window(window)
        selected = self._cells(window)
        provenance = _sbc_provenance(self, window, prior)
        return {
            "records": tuple(self.record(entry.index) for entry in selected),
            "world_ids": [entry.world_id for entry in selected],
            "cell_ids": [entry.cell for entry in selected],
            "provenance": provenance,
        }

    def to_datatree(self, window: str) -> DataTree:
        """Every world of ``window`` as one ArviZ ``xarray.DataTree``; needs ``pymc-mktfm[arviz]``.

        Export only: the archive stays the evidence and MKTFM never reads a tree back. A
        ``world`` dimension in archive order (coordinate: the world ids) joins the
        worlds' draws, truth and observations on the union of their time labels and
        active components; ``NaN`` marks a time label or component a world does not
        have. Active edges are compacted to ``world_edge`` pairs (``<world_id>|<edge>``).
        One tree per window: context and query are separate marginals. Reads and
        re-verifies the window's entries once and holds the window in memory.
        This call imports xarray and arviz-base lazily.
        """
        _check_window(window)
        from pymc_mktfm.inference._datatree import archive_datatree

        return archive_datatree(self, window)
