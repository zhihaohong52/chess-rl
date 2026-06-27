"""PyTorch DataLoader: npz shards -> (inputs, targets) batches per contract.

Contract:
  __getitem__(i) -> ((square_tokens_long[64], state_features_float[18]),
                     (policy_dense_float[P], wdl_float[3], moves_left_float[1]))

  make_dataloader batch shape:
    inputs  : (sq[B,64] long, sf[B,18] float32)
    targets : (policy[B,P] float32, wdl[B,3] float32, ml[B,1] float32)
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class ShardDataset(Dataset):
    """Loads one or more npz shards and exposes individual examples."""

    def __init__(self, shard_paths, policy_size: int):
        self._policy_size = policy_size

        # Accumulate arrays across all shards.
        sq_parts, sf_parts, wdl_parts, ml_parts = [], [], [], []
        idx_parts, prob_parts, counts_parts = [], [], []

        for p in shard_paths:
            d = np.load(p)
            sq_parts.append(d["square_tokens"])        # [n, 64] int8
            sf_parts.append(d["state_features"])       # [n, 18] float32
            wdl_parts.append(d["wdl"])                 # [n, 3]  float32
            ml_parts.append(d["moves_left"])           # [n]     float32
            idx_parts.append(d["legal_indices"])       # [k]     int32
            prob_parts.append(d["legal_probs"])        # [k]     float32
            counts_parts.append(d["counts"])           # [n]     int32

        self._sq = np.concatenate(sq_parts, axis=0)          # [N, 64]
        self._sf = np.concatenate(sf_parts, axis=0)          # [N, 18]
        self._wdl = np.concatenate(wdl_parts, axis=0)        # [N, 3]
        self._ml = np.concatenate(ml_parts, axis=0)          # [N]
        self._counts = np.concatenate(counts_parts, axis=0)  # [N]

        flat_idx = np.concatenate(idx_parts, axis=0)         # [total]
        flat_prob = np.concatenate(prob_parts, axis=0)       # [total]

        # Precompute per-example start offsets into the flat arrays.
        self._offsets = np.empty(len(self._counts) + 1, dtype=np.int64)
        self._offsets[0] = 0
        np.cumsum(self._counts, out=self._offsets[1:])

        self._flat_idx = flat_idx
        self._flat_prob = flat_prob

    def __len__(self) -> int:
        return len(self._sq)

    def __getitem__(self, i: int):
        start = int(self._offsets[i])
        end = int(self._offsets[i + 1])

        sq = torch.from_numpy(self._sq[i].astype(np.int64))    # [64] long
        sf = torch.from_numpy(self._sf[i])                      # [18] float32

        policy = torch.zeros(self._policy_size, dtype=torch.float32)
        if end > start:
            indices = self._flat_idx[start:end].astype(np.int64)
            probs = self._flat_prob[start:end]
            policy[torch.from_numpy(indices)] = torch.from_numpy(probs)

        wdl = torch.from_numpy(self._wdl[i])                    # [3] float32
        ml = torch.tensor([self._ml[i]], dtype=torch.float32)   # [1] float32

        return (sq, sf), (policy, wdl, ml)


class AVShardDataset(Dataset):
    """Action-value shards: each example is one (state, sampled action, win%)."""

    def __init__(self, shard_paths):
        sq_parts, sf_parts, ai_parts, win_parts = [], [], [], []
        for p in shard_paths:
            d = np.load(p)
            sq_parts.append(d["square_tokens"])    # [n, 64] int8
            sf_parts.append(d["state_features"])   # [n, 18] float32
            ai_parts.append(d["action_idx"])       # [n]     int32
            win_parts.append(d["win"])             # [n]     float32
        self._sq = np.concatenate(sq_parts, axis=0)
        self._sf = np.concatenate(sf_parts, axis=0)
        self._ai = np.concatenate(ai_parts, axis=0)
        self._win = np.concatenate(win_parts, axis=0)

    def __len__(self) -> int:
        return len(self._sq)

    def __getitem__(self, i: int):
        sq = torch.from_numpy(self._sq[i].astype(np.int64))   # [64] long
        sf = torch.from_numpy(self._sf[i])                     # [18] float32
        ai = torch.tensor(int(self._ai[i]), dtype=torch.long)  # scalar long
        win = torch.tensor(float(self._win[i]), dtype=torch.float32)  # scalar
        return (sq, sf), (ai, win)


def make_av_dataloader(
    shard_paths,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build an action-value DataLoader over npz shards.

    Returns batches of shape:
      ((sq[B,64] long, sf[B,18] float32), (action_idx[B] long, win[B] float32))
    """
    ds = AVShardDataset(shard_paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def make_dataloader(
    shard_paths,
    batch_size: int,
    policy_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader over npz shards.

    Returns batches of shape:
      ((sq[B,64] long, sf[B,18] float32), (policy[B,P] float32, wdl[B,3] float32, ml[B,1] float32))
    """
    ds = ShardDataset(shard_paths, policy_size=policy_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class StreamingShardDataset(IterableDataset):
    """Stream examples one npz shard at a time to bound memory.

    The eager ShardDataset concatenates every shard into RAM (~40 GB for 100M
    positions). This loads a single shard at a time (~one shard resident),
    yields its examples, then releases it. Per-example output matches
    ShardDataset exactly. When ``shuffle`` is True the shard order and the
    within-shard order are permuted each epoch; across DataLoader workers the
    shards are partitioned by worker id so every example is produced exactly
    once per epoch. Each yielded tensor owns its memory, so batches that span a
    shard boundary stay valid after the previous shard is freed.
    """

    def __init__(self, shard_paths, policy_size: int, shuffle: bool = True,
                 seed: int = 0):
        self._paths = [str(p) for p in shard_paths]
        if not self._paths:
            raise ValueError("StreamingShardDataset: no shard paths provided")
        self._policy_size = policy_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        rng = random.Random(self._seed + self._epoch * 100003 + worker_id)
        self._epoch += 1

        paths = list(self._paths)
        if self._shuffle:
            rng.shuffle(paths)
        paths = paths[worker_id::num_workers]  # disjoint partition per worker

        for path in paths:
            with np.load(path) as d:
                sq = d["square_tokens"]
                sf = d["state_features"]
                wdl = d["wdl"]
                ml = d["moves_left"]
                counts = d["counts"]
                flat_idx = d["legal_indices"]
                flat_prob = d["legal_probs"]
            offsets = np.empty(len(counts) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])

            order = list(range(len(sq)))
            if self._shuffle:
                rng.shuffle(order)
            for i in order:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                # astype(copy) so each tensor owns memory independent of `d`.
                sq_t = torch.from_numpy(sq[i].astype(np.int64))
                sf_t = torch.from_numpy(sf[i].astype(np.float32))
                policy = torch.zeros(self._policy_size, dtype=torch.float32)
                if end > start:
                    idx = torch.from_numpy(flat_idx[start:end].astype(np.int64))
                    prob = torch.from_numpy(flat_prob[start:end].astype(np.float32))
                    policy[idx] = prob
                wdl_t = torch.from_numpy(wdl[i].astype(np.float32))
                ml_t = torch.tensor([ml[i]], dtype=torch.float32)
                yield (sq_t, sf_t), (policy, wdl_t, ml_t)


def make_stream_dataloader(
    shard_paths,
    batch_size: int,
    policy_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """Memory-bounded DataLoader for large pre-encoded sets (streams shards).

    Same batch contract as make_dataloader. Shuffling is handled inside the
    dataset (IterableDataset forbids DataLoader shuffle=), so it is NOT passed
    to DataLoader.
    """
    ds = StreamingShardDataset(shard_paths, policy_size=policy_size,
                               shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
