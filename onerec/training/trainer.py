from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from onerec.data.ml1m import ML1MSequenceDataset, collate_seq_train
from onerec.data.synthetic import SyntheticSequentialDataset, collate_interactions
from onerec.metrics.ranking import summarize_metrics
from onerec.models.openonerec_like import OpenOneRecLikeModel, dpo_pair_loss, sequence_cross_entropy


@dataclass
class TrainConfig:
    seed: int
    dataset_type: str
    objective: str
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
    ratings_path: str | None = None
    eval_mode: str = "sampled"
    filter_seen_in_eval: bool = True
    dpo_beta: float = 0.1
    bos_token: int = 1


class Trainer:
    def __init__(self, model: OpenOneRecLikeModel, cfg: TrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(self, num_users: int) -> None:
        if self.cfg.dataset_type == "ml1m":
            self._fit_ml1m()
            return
        self._fit_synthetic(num_users=num_users)

    def _fit_synthetic(self, num_users: int) -> None:
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
            metrics = self._evaluate_synthetic()
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"epoch={epoch} loss={loss:.4f} {metrics_str}")

    def _fit_ml1m(self) -> None:
        if not self.cfg.ratings_path:
            raise ValueError("train.ratings_path must be provided when dataset_type=ml1m")

        dataset = ML1MSequenceDataset(
            ratings_path=self.cfg.ratings_path,
            max_seq_len=self.cfg.max_seq_len,
            seed=self.cfg.seed,
        )

        print(
            f"Loaded ML-1M split users={len(dataset.user_train_items)} items={dataset.num_items - 1} "
            f"train_samples={len(dataset)} objective={self.cfg.objective} eval_mode={self.cfg.eval_mode}"
        )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=partial(collate_seq_train, max_seq_len=self.cfg.max_seq_len),
        )

        for epoch in range(1, self.cfg.epochs + 1):
            loss = self._train_epoch(loader)
            val_metrics = self._evaluate_ml1m(dataset, split="val")
            test_metrics = self._evaluate_ml1m(dataset, split="test")
            val_str = " ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
            test_str = " ".join(f"test_{k}={v:.4f}" for k, v in test_metrics.items())
            print(f"epoch={epoch} loss={loss:.4f} {val_str} {test_str}")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for _, sequences, pos_items, neg_items in loader:
            sequences = sequences.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            decoder_input = torch.full((sequences.size(0), 1), self.cfg.bos_token, dtype=torch.long, device=self.device)
            logits = self.model.forward_teacher_forcing(sequences, decoder_input)
            next_logits = logits[:, -1, :]

            if self.cfg.objective == "dpo":
                pos_logp = torch.log_softmax(next_logits, dim=-1).gather(1, pos_items.unsqueeze(1)).squeeze(1)
                neg_logp = torch.log_softmax(next_logits, dim=-1).gather(1, neg_items.unsqueeze(1)).squeeze(1)
                loss = dpo_pair_loss(pos_logp, neg_logp, beta=self.cfg.dpo_beta)
            elif self.cfg.objective == "bpr":
                pos_scores = next_logits.gather(1, pos_items.unsqueeze(1)).squeeze(1)
                neg_scores = next_logits.gather(1, neg_items.unsqueeze(1)).squeeze(1)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            else:  # generative_ce
                target = pos_items.unsqueeze(1)
                loss = sequence_cross_entropy(logits, target, ignore_index=0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            batches += 1
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def _evaluate_synthetic(self) -> Dict[str, float]:
        self.model.eval()

        positives = torch.randint(1, self.cfg.num_items, (self.cfg.num_val_users,), device=self.device)
        candidate_negatives = torch.randint(
            1,
            self.cfg.num_items,
            (self.cfg.num_val_users, self.cfg.negatives_per_user),
            device=self.device,
        )
        candidates = torch.cat([positives.unsqueeze(1), candidate_negatives], dim=1)
        sequences = torch.randint(1, self.cfg.num_items, (self.cfg.num_val_users, self.cfg.max_seq_len), device=self.device)

        ranked_candidates = self._rank_candidates(sequences, candidates)
        return summarize_metrics(ranked_candidates, positives, self.cfg.topk)

    @torch.no_grad()
    def _evaluate_ml1m(self, dataset: ML1MSequenceDataset, split: str) -> Dict[str, float]:
        self.model.eval()

        eval_users = dataset.val_users if split == "val" else dataset.test_users
        rng = random.Random(self.cfg.seed + (0 if split == "val" else 7))

        all_rankings: List[torch.Tensor] = []
        all_positives: List[torch.Tensor] = []

        for start in range(0, len(eval_users), self.cfg.batch_size):
            chunk = eval_users[start : start + self.cfg.batch_size]
            padded_sequences = []
            positives = []
            user_ids = []
            for sample in chunk:
                seq = sample.sequence[-self.cfg.max_seq_len :]
                pad_len = self.cfg.max_seq_len - len(seq)
                padded_sequences.append([0] * pad_len + seq)
                positives.append(sample.target_item)
                user_ids.append(sample.user_id)

            seq_tensor = torch.tensor(padded_sequences, dtype=torch.long, device=self.device)
            pos_tensor = torch.tensor(positives, dtype=torch.long, device=self.device)

            if self.cfg.eval_mode == "all":
                candidates = torch.arange(1, dataset.num_items, dtype=torch.long, device=self.device)
                candidates = candidates.unsqueeze(0).expand(seq_tensor.size(0), -1)

                logits = self.model.next_item_logits(seq_tensor, bos_token=self.cfg.bos_token)
                scores = logits[:, 1:dataset.num_items].clone()
                if self.cfg.filter_seen_in_eval:
                    for row, uid in enumerate(user_ids):
                        seen = dataset.user_seen_items[uid]
                        seen_tensor = torch.tensor(list(seen), dtype=torch.long, device=self.device)
                        scores[row, seen_tensor - 1] = -1e9
                        scores[row, positives[row] - 1] = logits[row, positives[row]]
                ranking_indices = torch.argsort(scores, dim=1, descending=True)
                ranked_candidates = torch.gather(candidates, dim=1, index=ranking_indices)
            else:
                candidates_list = []
                for uid, tgt in zip(user_ids, positives):
                    candidates_list.append(
                        dataset.sample_eval_candidates(
                            user_id=uid,
                            target_item=tgt,
                            num_negatives=self.cfg.negatives_per_user,
                            rng=rng,
                        )
                    )
                candidate_tensor = torch.tensor(candidates_list, dtype=torch.long, device=self.device)
                ranked_candidates = self._rank_candidates(seq_tensor, candidate_tensor)

            all_rankings.append(ranked_candidates.cpu())
            all_positives.append(pos_tensor.cpu())

        rankings = torch.cat(all_rankings, dim=0)
        positives = torch.cat(all_positives, dim=0)
        return summarize_metrics(rankings, positives, self.cfg.topk)

    def _rank_candidates(self, sequences: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        logits = self.model.next_item_logits(sequences, bos_token=self.cfg.bos_token)
        scores = logits.gather(1, candidates)
        ranking_indices = torch.argsort(scores, dim=1, descending=True)
        return torch.gather(candidates, dim=1, index=ranking_indices)
