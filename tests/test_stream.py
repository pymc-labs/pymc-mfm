"""Generator-free contracts of epoch-based corpus consumption."""

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.data.corpus import corpus_batch
from pymc_mfm.training.stream import CorpusStream, PrefetchingStream, batches_for_rows


def train_rows(corpus, corpus_index=0):
    return {(corpus_index, int(r)) for r in np.flatnonzero(np.asarray(corpus["is_val"]) == 0)}


def context_length(corpus, row):
    return int(np.asarray(corpus["support_mask"][row]).sum())


def draw(stream, n):
    return [stream.next_batch() for _ in range(n)]


@pytest.fixture
def corpus():
    # Train cells 0..5 alternate context length 6/9: two buckets of 12 rows each.
    return make_synthetic_corpus(n_cells=8, draws_per_cell=4, val_cells=2)


def test_every_batch_shares_one_context_length_and_matches_its_provenance(corpus):
    stream = CorpusStream([corpus], batch_size=4, seed=0)
    for batch, provenance in draw(stream, 9):
        lengths = {context_length(corpus, row) for _, row in provenance.rows}
        assert lengths == {provenance.context_length}
        assert batch["ctx_spend_z"].shape == (4, provenance.context_length, 3)
        assert batch["qry_spend_z"].shape[1] == 12 - provenance.context_length
        assert len(provenance.rows) == 4


def test_epoch_consumes_each_train_row_exactly_once_when_buckets_divide(corpus):
    stream = CorpusStream([corpus], batch_size=4, seed=3)
    assert stream.rows_per_epoch == stream.n_rows == 24
    consumed = [row for _, p in draw(stream, 6) for row in p.rows]
    assert len(consumed) == 24
    assert set(consumed) == train_rows(corpus)
    assert stream.epoch == 0 and stream.consumed_rows == 24
    _, provenance = stream.next_batch()
    assert provenance.epoch == stream.epoch == 1
    assert provenance.batch_index == 6


def test_leftover_rows_are_dropped_and_each_row_seen_at_most_once_per_epoch(corpus):
    stream = CorpusStream([corpus], batch_size=5, seed=3)
    assert stream.rows_per_epoch == 20 and stream.batches_per_epoch == 4
    for epoch in range(3):
        provenances = [p for _, p in draw(stream, stream.batches_per_epoch)]
        consumed = [row for p in provenances for row in p.rows]
        assert len(consumed) == len(set(consumed)) == 20
        assert set(consumed) <= train_rows(corpus)
        assert all(p.epoch == epoch for p in provenances)
        assert sorted(p.context_length for p in provenances) == [6, 6, 9, 9]


def test_train_stream_never_serves_holdout_rows_and_holdout_split_serves_only_them(corpus):
    is_val = np.asarray(corpus["is_val"])
    train = CorpusStream([corpus], batch_size=4, seed=1)
    for _, provenance in draw(train, 14):
        assert all(is_val[row] == 0 for _, row in provenance.rows)
    holdout = CorpusStream([corpus], batch_size=4, seed=1, split="holdout")
    assert holdout.n_rows == 8
    for _, provenance in draw(holdout, 4):
        assert all(is_val[row] == 1 for _, row in provenance.rows)


def test_same_seed_reproduces_row_order_and_different_seed_changes_it(corpus):
    def order(seed):
        return [p.rows for _, p in draw(CorpusStream([corpus], batch_size=4, seed=seed), 8)]

    assert order(7) == order(7)
    assert order(7) != order(8)


def test_holdout_rows_use_an_independent_rng_and_leave_training_order_untouched(corpus):
    is_val = np.asarray(corpus["is_val"])
    probed = CorpusStream([corpus], batch_size=4, seed=5)
    plain = CorpusStream([corpus], batch_size=4, seed=5)
    draw(probed, 2)
    draw(plain, 2)
    rows = probed.holdout_rows(5, seed=11)
    assert len(rows) == len(set(rows)) == 5
    assert all(is_val[row] == 1 for _, row in rows)
    assert rows == probed.holdout_rows(5, seed=11)
    assert rows != probed.holdout_rows(5, seed=12)
    assert probed.consumed_rows == plain.consumed_rows == 8
    assert [p.rows for _, p in draw(probed, 6)] == [p.rows for _, p in draw(plain, 6)]
    with pytest.raises(ValueError):
        probed.holdout_rows(9, seed=0)


def test_multi_corpus_batches_keep_popped_order_per_row(corpus):
    other = make_synthetic_corpus(n_cells=8, draws_per_cell=4, val_cells=2, seed=1)
    stream = CorpusStream([corpus, other], batch_size=6, seed=4)
    assert stream.n_rows == 48
    drawn = draw(stream, 8)
    assert any(len({ci for ci, _ in p.rows}) > 1 for _, p in drawn)
    for batch, provenance in drawn:
        for position, (corpus_index, row) in enumerate(provenance.rows):
            single = corpus_batch([corpus, other][corpus_index], [row])
            torch.testing.assert_close(batch["sales_scale"][position], single["sales_scale"][0])
            torch.testing.assert_close(batch["targets_z"][position], single["targets_z"][0])


def test_prefetched_order_equals_direct_order_and_producer_errors_surface(corpus):
    direct = CorpusStream([corpus], batch_size=4, seed=9)
    prefetched = PrefetchingStream(CorpusStream([corpus], batch_size=4, seed=9), depth=2)
    try:
        for _ in range(10):
            batch, provenance = prefetched.next_batch()
            expected_batch, expected = direct.next_batch()
            assert provenance == expected
            torch.testing.assert_close(batch["targets_z"], expected_batch["targets_z"])
        assert prefetched.last_wait_seconds is not None
    finally:
        prefetched.stop()
    with pytest.raises(RuntimeError):
        prefetched.next_batch()

    class Broken:
        def resume_state(self):
            return {}

        def next_batch(self):
            raise ValueError("unusable row")

    failing = PrefetchingStream(Broken(), depth=1)
    try:
        with pytest.raises(ValueError, match="unusable row"):
            failing.next_batch()
        with pytest.raises(ValueError, match="unusable row"):
            failing.next_batch()
    finally:
        failing.stop()


def test_geometry_mismatch_across_corpora_raises(corpus):
    with pytest.raises(ValueError, match="geometry"):
        CorpusStream([corpus, make_synthetic_corpus(k=4)], batch_size=2, seed=0)
    with pytest.raises(ValueError, match="geometry"):
        CorpusStream(
            [corpus, make_synthetic_corpus(n_weeks=16, context_lengths=(6, 9))],
            batch_size=2,
            seed=0,
        )


def test_batch_larger_than_every_bucket_is_rejected_at_construction(corpus):
    with pytest.raises(ValueError, match="batch of 13"):
        CorpusStream([corpus], batch_size=13, seed=0)
    CorpusStream([corpus], batch_size=12, seed=0)


def test_batches_for_rows_groups_by_context_length_and_aligns_rows(corpus):
    other = make_synthetic_corpus(n_cells=8, draws_per_cell=4, val_cells=2, seed=1)
    corpora = [corpus, other]
    rows = [(1, 3), (0, 0), (0, 5), (1, 30), (0, 4), (1, 0)]
    batches = batches_for_rows(corpora, rows)
    assert [context_length(corpus, 0), context_length(corpus, 30)] == [6, 9]
    assert [batch["ctx_spend_z"].shape[1] for batch, _ in batches] == [6, 9]
    assert [group for _, group in batches] == [[(1, 3), (0, 0), (1, 0)], [(0, 5), (1, 30), (0, 4)]]
    for batch, group in batches:
        assert batch["sales_scale"].shape == (len(group),)
        for position, (corpus_index, row) in enumerate(group):
            single = corpus_batch(corpora[corpus_index], [row])
            torch.testing.assert_close(batch["sales_scale"][position], single["sales_scale"][0])
    with pytest.raises(ValueError):
        batches_for_rows(corpora, [])
    with pytest.raises(ValueError):
        batches_for_rows(corpora, [(2, 0)])
