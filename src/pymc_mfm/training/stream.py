"""Epoch-based consumption of in-memory schema-4 corpora as training batches.

A single ``np.random.default_rng(seed)`` drives every permutation and weighted
bucket choice. Each epoch permutes the selected split, groups rows by context
length, and draws a batchable bucket in proportion to its remaining rows.
Rows that cannot fill a batch are dropped at the next reshuffle.

``seen_rows`` returns the first distinct consumed rows. ``holdout_rows`` uses
an independent RNG, so evaluation probes do not perturb training order.
``PrefetchingStream`` produces the same ordered batches on one background
thread and re-raises producer errors to the consumer.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pymc_mfm.data.corpus import SUPPORTED_CORPUS_SCHEMAS, corpus_batch

SPLITS = ("train", "holdout", "all")

Row = tuple[int, int]


@dataclass(frozen=True)
class BatchProvenance:
    """Which worlds a training batch holds, in batch order.

    ``rows`` are ``(corpus_index, row)`` pairs aligned with dim 0 of every batch
    tensor; ``context_length`` is the shared context window of the batch;
    ``epoch`` is the stream epoch the batch was drawn from (0-based; leftovers are
    dropped at the boundary) and ``batch_index`` the global 0-based ordinal of the
    batch in the stream.
    """

    rows: tuple[Row, ...]
    context_length: int
    epoch: int
    batch_index: int


def _geometry(corpus: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """``(N, K, M, J, T)`` of a schema-4 corpus; the array checks ``corpus_batch`` repeats."""
    diagnostics = corpus.get("diagnostics")
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("schema_version") not in SUPPORTED_CORPUS_SCHEMAS
    ):
        raise ValueError(
            f"CorpusStream consumes corpora with diagnostics schema_version "
            f"{SUPPORTED_CORPUS_SCHEMAS}"
        )
    treatment = np.shape(corpus["treatment_raw"])
    covariates = np.shape(corpus["covariates"])
    latent = np.shape(corpus["latent_active_mask"])
    if len(treatment) != 3 or len(covariates) != 3 or len(latent) != 2:
        raise ValueError("treatments/covariates must have shape [N,T,D], latent mask [N,J]")
    n, t, k = treatment
    if covariates[:2] != (n, t) or latent[0] != n:
        raise ValueError("corpus arrays must share world and time dimensions")
    return int(n), int(k), int(covariates[2]), int(latent[1]), int(t)


def _context_lengths(corpus: Mapping[str, Any]) -> np.ndarray:
    """Per-world context length ``support_mask.sum(axis=1)`` as ``int64 [N]``."""
    support = np.asarray(corpus["support_mask"])
    if support.ndim != 2:
        raise ValueError("support_mask must have shape [N,T]")
    return support.sum(axis=1).astype(np.int64)


def _build_batch(
    corpora: Sequence[Mapping[str, Any]],
    rows: Sequence[Row],
    device: str | torch.device,
) -> dict[str, Tensor]:
    """Encode ``rows`` (one shared context length) with dim 0 in the given order.

    Rows are grouped by corpus for ``corpus_batch`` and written back into their
    original positions, so multi-corpus batches cost one copy and keep the order.
    """
    by_corpus: dict[int, list[int]] = {}
    positions: dict[int, list[int]] = {}
    for position, (corpus_index, row) in enumerate(rows):
        by_corpus.setdefault(corpus_index, []).append(row)
        positions.setdefault(corpus_index, []).append(position)
    pieces = [
        (corpus_batch(corpora[corpus_index], by_corpus[corpus_index], device=device), where)
        for corpus_index, where in positions.items()
    ]
    if len(pieces) == 1:
        return pieces[0][0]
    merged: dict[str, Tensor] = {}
    for key, reference in pieces[0][0].items():
        out = reference.new_empty((len(rows), *reference.shape[1:]))
        for piece, where in pieces:
            out[torch.as_tensor(where, dtype=torch.long, device=out.device)] = piece[key]
        merged[key] = out
    return merged


def batches_for_rows(
    corpora: Sequence[Mapping[str, Any]],
    rows: Sequence[Row],
    *,
    device: str | torch.device = "cpu",
) -> list[tuple[dict[str, Tensor], list[Row]]]:
    """Encode arbitrary ``(corpus_index, row)`` pairs as context-length-homogeneous batches.

    Rows are grouped by context length (batches are returned in increasing context
    length); inside a batch the input order is kept, and the returned row list is
    aligned with dim 0 of that batch. Every input row appears exactly once across the
    returned batches. Used by evaluation, which selects rows by other means than the
    training stream. Raises ``ValueError`` on an empty selection or an index outside
    the corpora.
    """
    if not rows:
        raise ValueError("rows must select at least one world")
    lengths: dict[int, np.ndarray] = {}
    groups: dict[int, list[Row]] = {}
    for corpus_index, row in rows:
        corpus_index, row = int(corpus_index), int(row)
        if not 0 <= corpus_index < len(corpora):
            raise ValueError(f"corpus index {corpus_index} is outside [0, {len(corpora)})")
        if corpus_index not in lengths:
            lengths[corpus_index] = _context_lengths(corpora[corpus_index])
        if not 0 <= row < len(lengths[corpus_index]):
            raise ValueError(
                f"world index {row} is outside [0, {len(lengths[corpus_index])}) "
                f"of corpus {corpus_index}"
            )
        groups.setdefault(int(lengths[corpus_index][row]), []).append((corpus_index, row))
    return [(_build_batch(corpora, group, device), group) for _, group in sorted(groups.items())]


class CorpusStream:
    """Serve context-length-homogeneous training batches from loaded corpora.

    ``corpora`` are schema-4 dicts that must share ``(K, M, J, T)``, exposed as
    ``geometry``; ``split`` selects ``is_val == 0`` (``"train"``), ``is_val == 1``
    (``"holdout"``) or every world.
    The stream is infinite over epochs: each epoch permutes the split index with the
    seeded RNG, buckets it by context length, and :meth:`next_batch` pops exactly
    ``batch_size`` rows from a bucket chosen with probability proportional to the rows
    remaining in the buckets that can still fill a batch. When no bucket can, the next
    call increments ``epoch`` and reshuffles; the leftover rows of the old epoch are
    dropped. Construction fails when no context length has ``batch_size`` rows.
    Not thread-safe: exactly one consumer (see :class:`PrefetchingStream`).
    """

    def __init__(
        self,
        corpora: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        seed: int,
        split: str = "train",
        device: str | torch.device = "cpu",
    ):
        if not corpora:
            raise ValueError("CorpusStream needs at least one corpus")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._corpora = list(corpora)
        self.batch_size = batch_size
        self.seed = seed
        self.split = split
        self.device = device
        self.rng = np.random.default_rng(seed)

        index: list[np.ndarray] = []
        index_lengths: list[np.ndarray] = []
        self._is_val: list[np.ndarray] = []
        for corpus_index, corpus in enumerate(self._corpora):
            n, *geometry = _geometry(corpus)
            if corpus_index == 0:
                self.geometry = tuple(geometry)
            elif tuple(geometry) != self.geometry:
                raise ValueError(
                    f"corpus {corpus_index} has geometry (K, M, J, T)={tuple(geometry)}; "
                    f"corpus 0 has {self.geometry}. A stream needs one geometry."
                )
            is_val = np.asarray(corpus["is_val"]).astype(bool)
            lengths = _context_lengths(corpus)
            if is_val.shape != (n,) or lengths.shape != (n,):
                raise ValueError(f"corpus {corpus_index}: is_val and support_mask need {n} worlds")
            self._is_val.append(is_val)
            if split == "train":
                rows = np.flatnonzero(~is_val)
            elif split == "holdout":
                rows = np.flatnonzero(is_val)
            else:
                rows = np.arange(n)
            index.append(np.stack((np.full(len(rows), corpus_index), rows), axis=1))
            index_lengths.append(lengths[rows])
        self._index = np.concatenate(index).astype(np.int64)
        self._index_lengths = np.concatenate(index_lengths)

        keys, counts = np.unique(self._index_lengths, return_counts=True)
        self._bucket_sizes = {int(key): int(count) for key, count in zip(keys, counts)}
        self._rows_per_epoch = int(sum(count // batch_size * batch_size for count in counts))
        if self._rows_per_epoch == 0:
            raise ValueError(
                f"no context length can fill a batch of {batch_size}: {len(self._index)} "
                f"{split!r} rows across buckets {self._bucket_sizes} (context_length: rows). "
                f"Reduce batch_size or add worlds per context length."
            )

        self.epoch = 0
        self.consumed_rows = 0
        self.batches = 0
        self._first_seen: dict[Row, None] = {}
        self._buckets: dict[int, np.ndarray] = {}
        self._cursor: dict[int, int] = {}
        self._start_epoch()

    @property
    def n_rows(self) -> int:
        """Rows in this split across all corpora (the epoch index size)."""
        return len(self._index)

    @property
    def rows_per_epoch(self) -> int:
        """Rows actually served per epoch: per bucket, the largest multiple of batch_size."""
        return self._rows_per_epoch

    @property
    def batches_per_epoch(self) -> int:
        """Batches served per epoch before a reshuffle becomes necessary."""
        return self._rows_per_epoch // self.batch_size

    def _start_epoch(self) -> None:
        """Permute rows and build FIFO context-length buckets.

        Bucket order follows the first matching row in the permutation so
        weighted selection remains reproducible.
        """
        order = self.rng.permutation(len(self._index))
        lengths = self._index_lengths[order]
        keys = dict.fromkeys(lengths.tolist())
        self._buckets = {int(key): order[lengths == key] for key in keys}
        self._cursor = dict.fromkeys(self._buckets, 0)

    def _emittable(self) -> list[int]:
        return [
            key
            for key, bucket in self._buckets.items()
            if len(bucket) - self._cursor[key] >= self.batch_size
        ]

    def next_batch(self) -> tuple[dict[str, Tensor], BatchProvenance]:
        """Pop one whole batch; reshuffle into a new epoch first when no bucket can fill."""
        emittable = self._emittable()
        if not emittable:
            self.epoch += 1
            self._start_epoch()
            emittable = self._emittable()
        weights = np.array(
            [len(self._buckets[key]) - self._cursor[key] for key in emittable], dtype=np.float64
        )
        key = emittable[int(self.rng.choice(len(emittable), p=weights / weights.sum()))]
        start = self._cursor[key]
        picked = self._buckets[key][start : start + self.batch_size]
        self._cursor[key] = start + self.batch_size
        rows = [(int(corpus_index), int(row)) for corpus_index, row in self._index[picked]]
        batch = _build_batch(self._corpora, rows, self.device)
        provenance = BatchProvenance(
            rows=tuple(rows), context_length=key, epoch=self.epoch, batch_index=self.batches
        )
        self.consumed_rows += len(rows)
        self.batches += 1
        for item in rows:
            self._first_seen.setdefault(item, None)
        return batch, provenance

    @property
    def distinct_consumed_rows(self) -> int:
        """Distinct ``(corpus_index, row)`` pairs consumed so far; what ``seen_rows`` can serve."""
        return len(self._first_seen)

    def seen_rows(self, n: int) -> list[Row]:
        """First ``n`` distinct consumed rows in consumption order (memorization probe).

        Does not change consumption state. Raises ``ValueError`` when fewer than ``n``
        distinct rows have been consumed.
        """
        if n > self.distinct_consumed_rows:
            raise ValueError(
                f"seen_rows({n}) needs {n} distinct consumed rows; "
                f"{self.distinct_consumed_rows} so far"
            )
        return list(self._first_seen)[:n]

    def holdout_rows(self, n: int, *, seed: int) -> list[Row]:
        """``n`` rows with ``is_val == 1`` from every corpus, permuted by an independent RNG.

        Regardless of ``split``; the training RNG and counters are untouched, so the
        same ``seed`` yields the same rows at any point of training. Raises
        ``ValueError`` when fewer than ``n`` held-out rows exist.
        """
        rows = [
            (corpus_index, int(row))
            for corpus_index, is_val in enumerate(self._is_val)
            for row in np.flatnonzero(is_val)
        ]
        if n > len(rows):
            raise ValueError(f"holdout_rows({n}) needs {n} is_val rows; corpora hold {len(rows)}")
        order = np.random.default_rng(seed).permutation(len(rows))
        return [rows[i] for i in order[:n]]

    def state(self) -> dict[str, Any]:
        """JSON-safe consumption bookkeeping for run logs and checkpoints."""
        return {
            "split": self.split,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "n_corpora": len(self._corpora),
            "geometry": dict(zip("KMJT", self.geometry)),
            "n_rows": self.n_rows,
            "rows_per_epoch": self.rows_per_epoch,
            "bucket_sizes": {str(key): size for key, size in self._bucket_sizes.items()},
            "epoch": self.epoch,
            "batches": self.batches,
            "consumed_rows": self.consumed_rows,
        }


class PrefetchingStream:
    """Fill a bounded queue from ``stream.next_batch()`` on one background thread.

    Consumption order equals the un-prefetched order because the single producer is
    the only caller of the wrapped stream. Nothing else is forwarded: read counters
    (``consumed_rows``, ``epoch``, ``state()``, ``seen_rows``, ``holdout_rows``) from
    the wrapped ``CorpusStream`` directly, knowing they run up to ``depth`` batches
    ahead of what the consumer has received; per-batch bookkeeping must use the
    returned provenance. An exception raised by the producer surfaces on the next
    :meth:`next_batch` and every later one. :meth:`stop` ends the producer and joins
    it; :meth:`next_batch` after ``stop`` raises.
    """

    def __init__(self, stream: CorpusStream, *, depth: int = 2):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.stream = stream
        self.last_wait_seconds: float | None = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._produce, name="corpus-prefetch", daemon=True)
        self._thread.start()

    def _put(self, item: Any) -> bool:
        """Enqueue with periodic stop checks; False once stopped."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
            except queue.Full:
                continue
            return True
        return False

    def _produce(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._put(self.stream.next_batch()):
                    return
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
            self._put(exc)

    def next_batch(self) -> tuple[dict[str, Tensor], BatchProvenance]:
        """Next ``(batch, provenance)`` in stream order; records ``last_wait_seconds``."""
        if self._failure is not None:
            raise self._failure
        if self._stop.is_set():
            raise RuntimeError("PrefetchingStream is stopped")
        started = time.perf_counter()
        item = self._queue.get()
        self.last_wait_seconds = time.perf_counter() - started
        if isinstance(item, BaseException):
            self._failure = item
            raise item
        return item

    def stop(self) -> None:
        """Signal the producer and join it; bounded so a stuck producer cannot hang exit."""
        self._stop.set()
        self._thread.join(timeout=2.0)
