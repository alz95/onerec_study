from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from onerec.data.synthetic import SyntheticSequentialDataset, collate_interactions
from onerec.metrics.ranking import summarize_metrics
from onerec.models.sequential import SimpleSequentialRec


@dataclass
class TrainConfig:
    seed: int
    num_items: int
    max_seq_len: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_train_samples: int
    num_val_users: int
    negatives_per_user: int
    topk: List[int]


class Trainer:
    def __init__(self, model: SimpleSequentialRec, cfg: TrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(self, num_users: int) -> None:
        dataset = SyntheticSequentialDataset(
            num_users=num_users,
            num_items=self.cfg.num_items,
            max_seq_len=self.cfg.max_seq_len,
            num_samples=self.cfg.num_train_samples,
            seed=self.cfg.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=partial(collate_interactions, max_seq_len=self.cfg.max_seq_len),
        )

        for epoch in range(1, self.cfg.epochs + 1):
            loss = self._train_epoch(loader)
            metrics = self.evaluate(num_users=num_users)
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"epoch={epoch} loss={loss:.4f} {metrics_str}")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for _, sequences, pos_items, neg_items in loader:
            sequences = sequences.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            pos_scores, neg_scores = self.model(sequences, pos_items, neg_items)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            batches += 1
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate(self, num_users: int) -> Dict[str, float]:
        self.model.eval()

        positives = torch.randint(1, self.cfg.num_items, (self.cfg.num_val_users,), device=self.device)
        candidate_negatives = torch.randint(
            1,
            self.cfg.num_items,
            (self.cfg.num_val_users, self.cfg.negatives_per_user),
            device=self.device,
        )
        candidates = torch.cat([positives.unsqueeze(1), candidate_negatives], dim=1)

        sequences = torch.randint(
            1,
            self.cfg.num_items,
            (self.cfg.num_val_users, self.cfg.max_seq_len),
            device=self.device,
        )

        user_repr = self.model.encode_sequence(sequences)
        item_emb = self.model.item_embedding(candidates)
        scores = (user_repr.unsqueeze(1) * item_emb).sum(dim=-1)
        ranking_indices = torch.argsort(scores, dim=1, descending=True)
        ranked_candidates = torch.gather(candidates, dim=1, index=ranking_indices)

        del num_users
        return summarize_metrics(ranked_candidates, positives, self.cfg.topk)
