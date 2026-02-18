from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SeqTrainSample:
    user_id: int
    sequence: List[int]
    pos_item: int
    neg_item: int


@dataclass
class EvalUserSample:
    user_id: int
    sequence: List[int]
    target_item: int


class ML1MSequenceDataset(Dataset):
    """Leave-one-out sequence dataset for MovieLens-1M.

    Per user split:
    - train history: all except last 2 interactions
    - val target: penultimate interaction
    - test target: last interaction
    """

    def __init__(
        self,
        ratings_path: str,
        max_seq_len: int,
        seed: int,
        min_user_interactions: int = 5,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)

        # internal ids start from 1, 0 reserved for padding
        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}

        raw_by_user: Dict[int, List[Tuple[int, int]]] = {}
        with open(ratings_path, "r", encoding="latin-1") as f:
            for line in f:
                user_raw, item_raw, _, ts = line.strip().split("::")
                u = int(user_raw)
                i = int(item_raw)
                t = int(ts)
                raw_by_user.setdefault(u, []).append((t, i))

        self.user_train_items: Dict[int, List[int]] = {}
        self.user_seen_items: Dict[int, set[int]] = {}
        self.val_users: List[EvalUserSample] = []
        self.test_users: List[EvalUserSample] = []

        next_user_idx = 1
        next_item_idx = 1

        for u_raw, events in raw_by_user.items():
            if len(events) < min_user_interactions:
                continue

            events.sort(key=lambda x: x[0])
            mapped_items: List[int] = []
            for _, item_raw in events:
                if item_raw not in self.item2idx:
                    self.item2idx[item_raw] = next_item_idx
                    next_item_idx += 1
                mapped_items.append(self.item2idx[item_raw])

            if u_raw not in self.user2idx:
                self.user2idx[u_raw] = next_user_idx
                next_user_idx += 1
            u = self.user2idx[u_raw]

            train_items = mapped_items[:-2]
            val_item = mapped_items[-2]
            test_item = mapped_items[-1]

            if len(train_items) < 2:
                continue

            self.user_train_items[u] = train_items
            self.user_seen_items[u] = set(mapped_items)

            self.val_users.append(
                EvalUserSample(user_id=u, sequence=train_items[-self.max_seq_len :], target_item=val_item)
            )
            self.test_users.append(
                EvalUserSample(user_id=u, sequence=(train_items + [val_item])[-self.max_seq_len :], target_item=test_item)
            )

        self.num_users = next_user_idx
        self.num_items = next_item_idx
        self._train_samples: List[SeqTrainSample] = self._build_train_samples()

    def _build_train_samples(self) -> List[SeqTrainSample]:
        samples: List[SeqTrainSample] = []
        all_items = list(range(1, self.num_items))
        for u, items in self.user_train_items.items():
            seen = set(items)
            for idx in range(1, len(items)):
                seq = items[:idx]
                pos_item = items[idx]
                neg_item = all_items[self.rng.randrange(len(all_items))]
                while neg_item in seen:
                    neg_item = all_items[self.rng.randrange(len(all_items))]
                samples.append(
                    SeqTrainSample(
                        user_id=u,
                        sequence=seq[-self.max_seq_len :],
                        pos_item=pos_item,
                        neg_item=neg_item,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self._train_samples)

    def __getitem__(self, index: int) -> SeqTrainSample:
        return self._train_samples[index]

    def sample_eval_candidates(self, user_id: int, target_item: int, num_negatives: int, rng: random.Random) -> List[int]:
        seen = self.user_seen_items[user_id]
        negatives: List[int] = []
        while len(negatives) < num_negatives:
            cand = rng.randrange(1, self.num_items)
            if cand == target_item or cand in seen:
                continue
            negatives.append(cand)
        return [target_item] + negatives


def collate_seq_train(
    batch: Sequence[SeqTrainSample],
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


def infer_ml1m_num_items(ratings_path: str) -> int:
    item_ids = set()
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            _u, i, _r, _t = line.strip().split("::")
            item_ids.add(int(i))
    # +1 for padding index 0
    return len(item_ids) + 1
