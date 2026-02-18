from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class InteractionSample:
    user_id: int
    sequence: List[int]
    pos_item: int
    neg_item: int


class SyntheticSequentialDataset(Dataset):
    """Synthetic dataset for quickly validating the training/evaluation pipeline."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        max_seq_len: int,
        num_samples: int,
        seed: int,
    ) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> InteractionSample:
        user_id = self.rng.randrange(self.num_users)
        seq_len = self.rng.randint(3, self.max_seq_len)
        sequence = [self.rng.randrange(1, self.num_items) for _ in range(seq_len)]
        pos_item = sequence[-1]
        neg_item = self.rng.randrange(1, self.num_items)
        while neg_item == pos_item:
            neg_item = self.rng.randrange(1, self.num_items)
        return InteractionSample(user_id=user_id, sequence=sequence, pos_item=pos_item, neg_item=neg_item)


def collate_interactions(
    batch: List[InteractionSample],
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    user_ids = torch.tensor([x.user_id for x in batch], dtype=torch.long)
    pos_items = torch.tensor([x.pos_item for x in batch], dtype=torch.long)
    neg_items = torch.tensor([x.neg_item for x in batch], dtype=torch.long)

    padded = []
    for sample in batch:
        seq = sample.sequence[-max_seq_len:]
        pad_len = max_seq_len - len(seq)
        padded.append([0] * pad_len + seq)
    sequences = torch.tensor(padded, dtype=torch.long)
    return user_ids, sequences, pos_items, neg_items
