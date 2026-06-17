"""PyTorch DataLoader: npz shards -> (inputs, targets) batches per contract.

Contract:
  __getitem__(i) -> ((square_tokens_long[64], state_features_float[18]),
                     (policy_dense_float[P], wdl_float[3], moves_left_float[1]))

  make_dataloader batch shape:
    inputs  : (sq[B,64] long, sf[B,18] float32)
    targets : (policy[B,P] float32, wdl[B,3] float32, ml[B,1] float32)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
