"""StreamingShardDataset streams one shard at a time (bounded memory) and must
produce exactly the same examples as the eager ShardDataset.

The eager ShardDataset loads all shards into RAM (fine for small sets, but ~40 GB
for 100M positions). StreamingShardDataset (IterableDataset) loads one shard at a
time. With shuffle=False it must yield byte-identical examples in the same order
as the eager loader iterated index 0..N-1.
"""
import numpy as np
import torch

from src.data.dataset import (
    ShardDataset,
    StreamingShardDataset,
    make_stream_dataloader,
)

P = 1924


def _write_shard(path, n, seed):
    rng = np.random.default_rng(seed)
    sq = rng.integers(0, 13, size=(n, 64)).astype(np.int8)
    sf = rng.random((n, 18), dtype=np.float32)
    wdl = rng.random((n, 3), dtype=np.float32)
    ml = rng.random(n, dtype=np.float32)
    counts = rng.integers(1, 5, size=n).astype(np.int32)
    total = int(counts.sum())
    legal_indices = rng.integers(0, P, size=total).astype(np.int32)
    legal_probs = rng.random(total, dtype=np.float32)
    np.savez(path, square_tokens=sq, state_features=sf, wdl=wdl, moves_left=ml,
             legal_indices=legal_indices, legal_probs=legal_probs, counts=counts)


def _shards(tmp_path):
    p0, p1 = tmp_path / "train_00.npz", tmp_path / "train_01.npz"
    _write_shard(p0, 7, 0)
    _write_shard(p1, 5, 1)
    return [str(p0), str(p1)]


def test_streaming_matches_eager_when_unshuffled(tmp_path):
    paths = _shards(tmp_path)
    eager = ShardDataset(paths, policy_size=P)
    stream = list(iter(StreamingShardDataset(paths, policy_size=P, shuffle=False)))
    assert len(stream) == len(eager) == 12
    for i in range(len(eager)):
        (esq, esf), (ep, ewdl, eml) = eager[i]
        (ssq, ssf), (sp, swdl, sml) = stream[i]
        assert torch.equal(ssq, esq) and ssq.dtype == torch.int64
        assert torch.equal(ssf, esf)
        assert torch.allclose(sp, ep) and sp.shape == (P,)
        assert torch.equal(swdl, ewdl)
        assert torch.equal(sml, eml) and sml.shape == (1,)


def test_streaming_shuffled_still_covers_all_examples(tmp_path):
    paths = _shards(tmp_path)
    stream = list(iter(StreamingShardDataset(paths, policy_size=P, shuffle=True, seed=3)))
    assert len(stream) == 12  # every example produced exactly once


def test_make_stream_dataloader_batches_cover_all(tmp_path):
    paths = _shards(tmp_path)
    loader = make_stream_dataloader(paths, batch_size=4, policy_size=P,
                                    shuffle=True, num_workers=0)
    seen = 0
    for (sq, sf), (pol, wdl, ml) in loader:
        assert sq.shape[1] == 64 and sq.dtype == torch.int64
        assert sf.shape[1] == 18
        assert pol.shape[1] == P
        assert wdl.shape[1] == 3
        assert ml.shape[1] == 1
        seen += sq.shape[0]
    assert seen == 12
