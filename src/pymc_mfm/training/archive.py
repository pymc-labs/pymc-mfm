"""Identities of evaluation archives, derived from what is evaluated, and the evaluation itself.

An evaluation archive (:class:`pymc_mfm.diagnostics.EvaluationArchive`) keeps
the conditional captures of :func:`~pymc_mfm.training.evaluation.evaluate_rows`
under an ``identity`` that declares what produced them. :func:`archive_identity`
derives it from the evaluated model, the corpora and the request, as the
trainer and the training CLI do:

* ``model``: ``state_sha256``, the digest of the weights actually evaluated, and
  ``config_sha256``, the digest of the configuration every capture records under
  ``inference_config.model``. A loaded bundle with unchanged weights also gives
  its ``bundle_sha256``, source checkpoint, step and profile; for other models
  the caller may state the checkpoint, step and profile.
* ``corpora``: one record per corpus, in the order passed to ``evaluate_rows``:
  the digest of its actual inputs (:func:`corpus_identity`), its file digest, its
  generator shard seed and prior hash, and ``data_origin``, which is
  ``"synthetic"`` only when the corpus record's prior sidecar is bound to the
  corpus file (:func:`verified_prior_sha256`). No paths are recorded.
* ``evaluation``: the requested worlds in order, each with its corpus, row, cell,
  context length and encoded-input digest; the settings; and the call plan:
  consecutive blocks of the worlds, each evaluated by one ``evaluate_rows`` call,
  whose seeds are consecutive.

An identity declares at most ``ARCHIVE_MAX_WORLDS`` (4096) worlds and as many
corpora, which keeps the manifest within the strict reader's limit.
:func:`archive_identity` checks the request before it hashes the weights or any
corpus or encodes any world. Only the checks that need those digests come after
them: a stated checkpoint, step or profile that conflicts with a loaded bundle,
and corpora with equal inputs or equal shard seeds.

:func:`evaluate_into_archive` derives the identity of a request and runs
exactly the calls it declares, with the archive writer as their
``result_sink``: the archive keeps the evaluator's own draws, and running the
declared calls again replays them.

Deriving an identity encodes the requested worlds but samples nothing and draws
no random number. Nothing here imports :mod:`pymc_mfm.diagnostics` at module
scope, so importing :mod:`pymc_mfm.training` loads no diagnostics code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pymc_mfm.data.observations import observation_identity
from pymc_mfm.inference._artifact_schema import (
    SHA256_PATTERN,
    digest_bytes,
    json_bytes,
    validate_profile,
)
from pymc_mfm.inference.checkpoint import _canonical_model, _state_sha256, _validate_state
from pymc_mfm.inference.result import _require_text
from pymc_mfm.model.flow import FLOW_STEPS_SAMPLE
from pymc_mfm.model.model import MFM
from pymc_mfm.training._provenance import corpus_identity, verified_prior_sha256
from pymc_mfm.training.evaluation import DECOMPOSITION_HEADS, _selected_heads, evaluate_rows
from pymc_mfm.training.stream import batches_for_rows

#: Seeds ``torch.manual_seed`` accepts; every call seed of an archive lies in it. Checked with
#: ``start <= seed < stop``, never ``in``, which scans the whole range for a non-int seed.
_TORCH_SEEDS = range(-(1 << 63), 1 << 64)


def archive_identity(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    rows: Iterable[tuple[int, int]],
    *,
    n_samples: int,
    seed: int,
    leg: str,
    device: str | torch.device,
    heads: Sequence[str] = DECOMPOSITION_HEADS,
    worlds_per_call: int | None = None,
    corpus_provenance: Sequence[Mapping[str, Any]] | None = None,
    shard_seeds: Sequence[int | None] | None = None,
    checkpoint_sha256: str | None = None,
    source_step: int | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """The identity of an evaluation archive of ``model`` on ``rows`` of ``corpora``.

    ``rows`` are distinct ``(corpus_index, row)`` pairs, read once and kept in
    the given order, on which the draws depend. The identity declares the
    evaluation as one ``evaluate_rows(model, corpora, call_rows,
    n_samples=n_samples, seed=seed + k, device=device, leg=leg, heads=heads)``
    per call ``k``: a single call over every world when ``worlds_per_call`` is
    ``None``, otherwise consecutive blocks of at most ``worlds_per_call``
    worlds. Every call seed must lie in torch's seed range ``[-2**63, 2**64)``,
    and ``heads`` must include ``"conditional"``, the captured head. On CPU, with
    the same thread count, libraries and hardware, the declared calls reproduce
    the draws bit for bit.

    ``corpus_provenance`` holds one record per corpus, as
    :func:`pymc_mfm.data.recipes.provenance` returns it. Its ``sha256`` file
    digest is recorded, and a prior sidecar bound to that digest gives the prior
    hash, the shard seed and ``data_origin: "synthetic"``; every other corpus is
    ``"unknown"``, whatever its record claims. ``shard_seeds`` (one non-negative
    int or ``None`` per corpus) replace the sidecar seeds. The corpora must hold
    distinct inputs and distinct non-null seeds.

    ``checkpoint_sha256``, ``source_step`` and ``profile`` describe a model that
    records none, e.g. a training checkpoint's file digest, step and profile. A
    loaded bundle with unchanged weights records its own, which they must equal.

    The request is checked before anything is hashed or encoded: the rows (at
    least one, distinct, at most ``ARCHIVE_MAX_WORLDS`` = 4096, inside the
    corpora, of corpora with a ``cell_id`` array), the corpora (at most as many,
    with one ``corpus_provenance`` record and one ``shard_seeds`` seed each when
    given, the seeds non-negative ints or ``None``), the settings (``leg`` must
    be nonempty valid Unicode text and ``device`` a torch device) and the format
    of ``checkpoint_sha256``, ``source_step`` and ``profile``. Only the checks
    that need digests come later: a stated value that conflicts with a loaded
    bundle, once the weights are hashed, and corpora with equal inputs or equal
    non-null shard seeds, once the corpora are hashed.

    Returns JSON values only. Raises ``ValueError`` when any of the above fails.
    """
    rows = [(int(corpus_index), int(row)) for corpus_index, row in rows]
    _check_rows(corpora, rows)
    _check_corpus_arguments(corpora, corpus_provenance, shard_seeds)
    settings = _evaluation_settings(
        len(rows),
        leg=leg,
        heads=heads,
        seed=seed,
        n_samples=n_samples,
        device=device,
        worlds_per_call=worlds_per_call,
    )
    return {
        "model": _model_identity(
            model, checkpoint_sha256=checkpoint_sha256, source_step=source_step, profile=profile
        ),
        "corpora": _corpus_records(corpora, corpus_provenance, shard_seeds=shard_seeds),
        "evaluation": {"worlds": _world_records(corpora, rows), **settings},
    }


def evaluate_into_archive(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    rows: Iterable[tuple[int, int]],
    out_dir: str | Path,
    *,
    n_samples: int,
    seed: int,
    leg: str,
    device: str | torch.device = "cpu",
    heads: Sequence[str] = DECOMPOSITION_HEADS,
    worlds_per_call: int | None = None,
    corpus_provenance: Sequence[Mapping[str, Any]] | None = None,
    shard_seeds: Sequence[int | None] | None = None,
    checkpoint_sha256: str | None = None,
    source_step: int | None = None,
    profile: str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Evaluate ``model`` on ``rows`` of ``corpora`` straight into a new archive at ``out_dir``.

    The identity is :func:`archive_identity` of the same request, which
    documents every keyword, and the evaluation is exactly the one it declares:
    one ``evaluate_rows`` call per declared call, in order, with the declared
    settings, the call's seed (``seed + k`` for call ``k``) and the archive
    writer as ``result_sink``. ``model`` is moved to ``device``. The archive
    therefore holds the evaluator's own conditional draws, and running the
    declared calls again replays them: on CPU bit for bit, given the same
    thread count, libraries and hardware.

    ``evaluate_rows`` puts every requested world of one context length into one
    batch, so the peak memory of a call grows with its worlds.
    ``worlds_per_call`` splits ``rows`` into consecutive calls of at most that
    many worlds; ``None`` makes one call, as ``evaluate_rows`` itself does.

    The request is checked as :func:`archive_identity` documents: all of it
    before anything is hashed or encoded, except a conflict with a loaded bundle
    and corpora with equal inputs or shard seeds, which need the digests. The
    identity (``ValueError``) and the target (``FileExistsError`` unless it is a
    new or empty directory) are checked before any inference. A world window that
    cannot be captured (``ValueError`` naming the first), a capture the archive
    refuses or any other failure publishes nothing and leaves no staging
    directory behind.

    Returns ``(path, records)``: the published archive directory, for
    :meth:`pymc_mfm.diagnostics.EvaluationArchive.load`, and the
    ``evaluate_rows`` record of each call, in order.
    """
    rows = [(int(corpus_index), int(row)) for corpus_index, row in rows]
    identity = archive_identity(
        model,
        corpora,
        rows,
        n_samples=n_samples,
        seed=seed,
        leg=leg,
        device=device,
        heads=heads,
        worlds_per_call=worlds_per_call,
        corpus_provenance=corpus_provenance,
        shard_seeds=shard_seeds,
        checkpoint_sha256=checkpoint_sha256,
        source_step=source_step,
        profile=profile,
    )
    return _evaluate_identity(model, corpora, identity, out_dir, device=device)


def _model_identity(
    model: MFM,
    *,
    checkpoint_sha256: str | None = None,
    source_step: int | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """The ``identity.model`` block: digests of the evaluated weights and configuration.

    ``state_sha256`` always comes from ``model.state_dict()``, so it names the
    weights actually evaluated. A model loaded from a bundle whose weights still
    match the load-time digest of ``model.artifact_metadata`` (not the editable
    ``model.provenance``) records that bundle's ``bundle_sha256``, source
    checkpoint, ``source_step`` and ``profile``; any other model records
    ``None`` for them. Explicit values fill ``None`` fields and must equal
    recorded ones. Draws no random numbers.
    """
    if checkpoint_sha256 is not None and (
        not isinstance(checkpoint_sha256, str) or not SHA256_PATTERN.fullmatch(checkpoint_sha256)
    ):
        raise ValueError("checkpoint_sha256 must be a full lowercase SHA-256")
    if source_step is not None and (type(source_step) is not int or source_step < 0):
        raise ValueError("source_step must be a non-negative integer")
    validate_profile(profile)
    state = model.state_dict()
    state_sha256 = _state_sha256(state, _validate_state(_canonical_model(model.config), state))
    recorded: dict[str, Any] = dict.fromkeys(
        ("checkpoint_sha256", "bundle_sha256", "profile", "source_step")
    )
    loaded = getattr(model, "artifact_metadata", None)
    if isinstance(loaded, Mapping) and loaded.get("state_sha256") == state_sha256:
        lineage, manifest = loaded["provenance"], loaded["manifest"]
        recorded = {
            "checkpoint_sha256": lineage.get("source_checkpoint_sha256")
            or manifest["files"]["model.safetensors"]["sha256"],
            "bundle_sha256": manifest["bundle_sha256"],
            "profile": loaded["profile"],
            "source_step": lineage.get("source_step"),
        }
    stated = (
        ("checkpoint_sha256", checkpoint_sha256),
        ("source_step", source_step),
        ("profile", profile),
    )
    for key, value in stated:
        if value is None:
            continue
        if recorded[key] is None:
            recorded[key] = value
        elif value != recorded[key]:
            raise ValueError(
                f"{key} {value!r} conflicts with the loaded bundle's {recorded[key]!r}"
            )
    return {
        "state_sha256": state_sha256,
        "config_sha256": digest_bytes(json_bytes(model.config.to_dict())),
        **recorded,
    }


def _corpus_records(
    corpora: Sequence[Mapping[str, Any]],
    corpus_provenance: Sequence[Mapping[str, Any]] | None = None,
    *,
    shard_seeds: Sequence[int | None] | None = None,
    identities: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The ``identity.corpora`` records, one per corpus, in order.

    ``identities`` are precomputed :func:`corpus_identity` values, as the trainer
    keeps them. ``actual_inputs_sha256`` is the digest of that identity, which
    exists even when its ``content_sha256`` is ``None``. ``data_origin`` is
    ``"synthetic"`` only with a verified prior sidecar, so a bare claim in a
    record is recorded as ``"unknown"``. Explicit ``shard_seeds`` win over
    sidecar seeds. The arguments are checked first, before any corpus is hashed
    (:func:`_check_corpus_arguments`); then corpora with equal inputs are
    refused, then equal non-null shard seeds. No paths are recorded.
    """
    _check_corpus_arguments(corpora, corpus_provenance, shard_seeds, identities)
    records = []
    for index, corpus in enumerate(corpora):
        record = {} if corpus_provenance is None else corpus_provenance[index]
        identity = corpus_identity(corpus) if identities is None else identities[index]
        source = record.get("sha256")
        prior_sha256 = verified_prior_sha256(record)
        if shard_seeds is None:
            shard_seed = _sidecar_seed(record)
        else:
            shard_seed = None if shard_seeds[index] is None else int(shard_seeds[index])
        records.append(
            {
                "shard_seed": shard_seed,
                "n_worlds": int(np.asarray(corpus["is_val"]).shape[0]),
                "actual_inputs_sha256": digest_bytes(json_bytes(identity)),
                "content_sha256": identity["content_sha256"],
                "source_sha256": (
                    source if isinstance(source, str) and SHA256_PATTERN.fullmatch(source) else None
                ),
                "data_origin": "unknown" if prior_sha256 is None else "synthetic",
                "prior_sha256": prior_sha256,
            }
        )
    inputs: dict[str, int] = {}
    for index, entry in enumerate(records):
        first = inputs.setdefault(entry["actual_inputs_sha256"], index)
        if first != index:
            raise ValueError(
                f"corpora {first} and {index} hold the same inputs; an archive needs distinct "
                "corpora"
            )
    seeds: dict[int, int] = {}
    for index, entry in enumerate(records):
        seed = entry["shard_seed"]
        if seed is not None and seeds.setdefault(seed, index) != index:
            raise ValueError(f"corpora {seeds[seed]} and {index} share shard seed {seed}")
    return records


def _world_records(
    corpora: Sequence[Mapping[str, Any]], rows: Sequence[tuple[int, int]]
) -> list[dict[str, Any]]:
    """The ``identity.evaluation.worlds`` records of normalized ``rows``, in order.

    A world's ``data_sha256`` is the ``observation_identity`` of its encoded
    inputs and its ``context_length`` the width of its context window, exactly
    as its captures record them; neither depends on the device or on the other
    worlds of its batch. ``cell_id`` is ``corpus["cell_id"][row]``. The rows are
    checked before anything is encoded (:func:`_check_rows`).
    """
    _check_rows(corpora, rows)
    batches = batches_for_rows(corpora, rows, device="cpu")
    data_sha256: dict[tuple[int, int], str] = {}
    context_length: dict[tuple[int, int], int] = {}
    for batch, group in batches:
        for world, key in enumerate(group):
            data_sha256[key] = observation_identity(batch, world=world)
            context_length[key] = int(batch["ctx_spend_z"].shape[1])
    return [
        {
            "corpus_index": int(corpus_index),
            "row": int(row),
            "cell_id": int(np.asarray(corpora[corpus_index]["cell_id"])[row]),
            "context_length": context_length[(corpus_index, row)],
            "data_sha256": data_sha256[(corpus_index, row)],
        }
        for corpus_index, row in rows
    ]


def _evaluation_block(
    worlds: list[dict[str, Any]],
    *,
    leg: str,
    heads: Sequence[str],
    seed: int,
    n_samples: int,
    device: str | torch.device,
    worlds_per_call: int | None = None,
) -> dict[str, Any]:
    """The ``identity.evaluation`` block: ``worlds`` and their :func:`_evaluation_settings`."""
    return {
        "worlds": worlds,
        **_evaluation_settings(
            len(worlds),
            leg=leg,
            heads=heads,
            seed=seed,
            n_samples=n_samples,
            device=device,
            worlds_per_call=worlds_per_call,
        ),
    }


def _evaluation_settings(
    n_worlds: int,
    *,
    leg: str,
    heads: Sequence[str],
    seed: int,
    n_samples: int,
    device: str | torch.device,
    worlds_per_call: int | None = None,
) -> dict[str, Any]:
    """The ``identity.evaluation`` block of ``n_worlds`` worlds without them, checked.

    Needs only the request, so it can refuse one before anything is hashed or
    encoded. Call ``k`` evaluates the ``k``-th consecutive block of at most
    ``worlds_per_call`` worlds (all of them when ``None``) with seed ``seed + k``,
    and every call seed must lie in torch's seed range ``[-2**63, 2**64)``. The
    heads are recorded in canonical order and must include ``"conditional"``.
    ``leg``, which every capture records, must be nonempty valid Unicode text,
    and ``device`` one that ``torch.device`` accepts. ``flow_steps`` and
    ``device`` are informational: the draws depend on the device and the thread
    count.
    """
    heads = list(_selected_heads(heads))
    if "conditional" not in heads:
        raise ValueError("archives capture conditional draws; select the conditional head")
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not isinstance(leg, str) or not leg:
        raise ValueError("leg must be a nonempty string")
    _require_text(leg, "leg")
    if worlds_per_call is not None and (
        isinstance(worlds_per_call, bool)
        or not isinstance(worlds_per_call, Integral)
        or worlds_per_call < 1
    ):
        raise ValueError("worlds_per_call must be a positive integer or None")
    size = n_worlds if worlds_per_call is None else int(worlds_per_call)
    calls = [
        {"worlds": list(range(start, min(start + size, n_worlds))), "seed": seed + call}
        for call, start in enumerate(range(0, n_worlds, size))
    ]
    for call, block in enumerate(calls):
        if not _TORCH_SEEDS.start <= block["seed"] < _TORCH_SEEDS.stop:
            raise ValueError(
                f"evaluation seed {block['seed']} of call {call} is outside the torch seed range "
                "[-2**63, 2**64)"
            )
    try:
        device_name = str(torch.device(device))
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"device {device!r} is not a torch device: {exc}") from exc
    return {
        "calls": calls,
        "leg": str(leg),
        "heads": heads,
        "n_samples": int(n_samples),
        "flow_steps": FLOW_STEPS_SAMPLE,
        "mode": "model_average",
        "units": "outcome",
        "device": device_name,
    }


def _check_rows(corpora: Sequence[Mapping[str, Any]], rows: Sequence[tuple[int, int]]) -> None:
    """Refuse normalized ``rows`` of ``corpora`` without encoding anything.

    In order: no rows, more than ``ARCHIVE_MAX_WORLDS``, a repeat, a corpus index
    or a row outside the corpora (a corpus holds ``len(is_val)`` worlds) and a
    corpus without a ``cell_id`` array.
    """
    from pymc_mfm.diagnostics.archive import ARCHIVE_MAX_WORLDS

    if not rows:
        raise ValueError("archive rows must name at least one world")
    if len(rows) > ARCHIVE_MAX_WORLDS:
        raise ValueError(
            f"an evaluation archive holds at most {ARCHIVE_MAX_WORLDS} worlds; "
            f"{len(rows)} were selected"
        )
    seen: set[tuple[int, int]] = set()
    for corpus_index, row in rows:
        if (corpus_index, row) in seen:
            raise ValueError(f"archive rows must be distinct; ({corpus_index}, {row}) repeats")
        seen.add((corpus_index, row))
    sizes: dict[int, int] = {}
    for corpus_index, row in rows:
        if not 0 <= corpus_index < len(corpora):
            raise ValueError(f"corpus index {corpus_index} is outside [0, {len(corpora)})")
        if corpus_index not in sizes:
            sizes[corpus_index] = int(np.asarray(corpora[corpus_index]["is_val"]).shape[0])
        if not 0 <= row < sizes[corpus_index]:
            raise ValueError(
                f"world index {row} is outside [0, {sizes[corpus_index]}) of corpus {corpus_index}"
            )
    for corpus_index in sizes:
        if "cell_id" not in corpora[corpus_index]:
            raise ValueError(f"corpus {corpus_index} has no cell_id array")


def _check_corpus_arguments(
    corpora: Sequence[Mapping[str, Any]],
    corpus_provenance: Sequence[Mapping[str, Any]] | None = None,
    shard_seeds: Sequence[int | None] | None = None,
    identities: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Refuse the per-corpus arguments without hashing any corpus.

    In order: more than ``ARCHIVE_MAX_WORLDS`` corpora, ``corpus_provenance``,
    ``shard_seeds`` or ``identities`` without one item per corpus, and a shard
    seed that is neither a non-negative integer nor ``None``.
    """
    from pymc_mfm.diagnostics.archive import ARCHIVE_MAX_WORLDS

    if len(corpora) > ARCHIVE_MAX_WORLDS:
        raise ValueError(
            f"an evaluation archive declares at most {ARCHIVE_MAX_WORLDS} corpora; "
            f"{len(corpora)} were given"
        )
    for name, given, item in (
        ("corpus_provenance", corpus_provenance, "record"),
        ("shard_seeds", shard_seeds, "seed"),
        ("identities", identities, "identity"),
    ):
        if given is not None and len(given) != len(corpora):
            raise ValueError(f"{name} needs one {item} per corpus")
    for value in () if shard_seeds is None else shard_seeds:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError(f"shard seed must be a non-negative integer or None, got {value!r}")


def _sidecar_seed(record: Mapping[str, Any]) -> int | None:
    """The generator seed of a corpus record's verified prior sidecar, else ``None``.

    Only a sidecar bound to the corpus file (:func:`verified_prior_sha256`)
    counts, and only a non-negative int seed.
    """
    if verified_prior_sha256(record) is None:
        return None
    seed = record["prior_sidecar"]["prior"].get("seed")
    return seed if type(seed) is int and seed >= 0 else None


def _evaluate_identity(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    out_dir: str | Path,
    *,
    device: str | torch.device,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run the evaluation ``identity`` declares into a new archive at ``out_dir``.

    The archive is created first, so the identity and the target are checked
    before any inference. Then ``model`` is moved to ``device`` and each
    declared call runs, in order, as ``evaluate_rows`` of its worlds with the
    declared settings, its seed and the writer as ``result_sink``. A capture
    failure raises ``ValueError`` naming the first failed world window; on any
    exception the writer aborts, so neither the target nor a staging directory
    remains. Returns the archive path and the record of each call.
    """
    from pymc_mfm.diagnostics.archive import EvaluationArchive

    evaluation = identity["evaluation"]
    worlds = evaluation["worlds"]
    records = []
    with EvaluationArchive.create(out_dir, identity) as writer:
        model.to(device)
        for call in evaluation["calls"]:
            rows = [
                (worlds[world]["corpus_index"], worlds[world]["row"]) for world in call["worlds"]
            ]
            record = evaluate_rows(
                model,
                corpora,
                rows,
                n_samples=evaluation["n_samples"],
                seed=call["seed"],
                device=device,
                leg=evaluation["leg"],
                heads=tuple(evaluation["heads"]),
                result_sink=writer.add,
            )
            failures = record["result_capture"]["failures"]
            if failures:
                first = failures[0]
                raise ValueError(
                    f"{len(failures)} world windows could not be captured, so no archive was "
                    f"written; first: corpus {first['corpus_index']} row {first['row']} "
                    f"{first['window']}: {first['error_type']}: {first['error']}"
                )
            records.append(record)
    return writer.path, records
