from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional
from pathlib import Path
import random
import time
import os
import json
from collections import deque

import torch
from torch.utils.data import DataLoader

from onerec.data.synthetic import SyntheticSequentialDataset, collate_interactions
from onerec.data.ml1m import load_ml1m, ML1MSequentialDataset, ML1MAllPositionsDataset, collate_ml1m
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
    data_source: str = "synthetic"
    ml1m_raw_dir: Optional[str] = None
    rating_threshold: int = 0
    sample_mode: str = "sampled"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    log_interval: int = 50
    profile: bool = False
    sma_window: int = 50
    eval_mode: str = "sampled"  # "sampled" or "all"


class Trainer:
    def __init__(self, model: SimpleSequentialRec, cfg: TrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self._ml1m = None
        os.makedirs("runs", exist_ok=True)
        self._log_file = Path("runs") / "metrics.jsonl"
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(self, num_users: int) -> None:
        if self.cfg.data_source == "synthetic":
            dataset = SyntheticSequentialDataset(
                num_users=num_users,
                num_items=self.cfg.num_items,
                max_seq_len=self.cfg.max_seq_len,
                num_samples=self.cfg.num_train_samples,
                seed=self.cfg.seed,
            )
            dl_kwargs = dict(
                batch_size=self.cfg.batch_size,
                shuffle=True,
                collate_fn=partial(collate_interactions, max_seq_len=self.cfg.max_seq_len),
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            )
            if self.cfg.num_workers > 0 and self.cfg.prefetch_factor is not None:
                dl_kwargs["prefetch_factor"] = self.cfg.prefetch_factor
            loader = DataLoader(dataset, **dl_kwargs)
        elif self.cfg.data_source == "ml1m":
            assert self.cfg.ml1m_raw_dir is not None
            data = load_ml1m(Path(self.cfg.ml1m_raw_dir), rating_threshold=self.cfg.rating_threshold, min_seq_len=3)
            self._ml1m = data
            if self.cfg.sample_mode == "full":
                dataset = ML1MAllPositionsDataset(
                    user_sequences=data.user_sequences,
                    user_item_sets=data.user_item_sets,
                    num_items=data.num_items,
                    max_seq_len=self.cfg.max_seq_len,
                    seed=self.cfg.seed,
                )
            else:
                dataset = ML1MSequentialDataset(
                    user_sequences=data.user_sequences,
                    user_item_sets=data.user_item_sets,
                    num_items=data.num_items,
                    max_seq_len=self.cfg.max_seq_len,
                    num_samples=self.cfg.num_train_samples,
                    seed=self.cfg.seed,
                )
            dl_kwargs = dict(
                batch_size=self.cfg.batch_size,
                shuffle=True,
                collate_fn=partial(collate_ml1m, max_seq_len=self.cfg.max_seq_len),
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            )
            if self.cfg.num_workers > 0 and self.cfg.prefetch_factor is not None:
                dl_kwargs["prefetch_factor"] = self.cfg.prefetch_factor
            loader = DataLoader(dataset, **dl_kwargs)
        else:
            raise ValueError("unknown data source")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            loss = self._train_epoch(loader)
            t1 = time.perf_counter()
            metrics = self.evaluate(num_users=num_users)
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"epoch={epoch} loss={loss:.4f} time={t1 - t0:.2f}s {metrics_str}")
            rec = {
                "epoch": epoch,
                "loss": float(loss),
                "time_s": float(t1 - t0),
                "metrics": metrics,
                "device": self.device.type,
                "batch_size": self.cfg.batch_size,
                "data_source": self.cfg.data_source,
                "sample_mode": self.cfg.sample_mode,
            }
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        sma_iter: deque = deque(maxlen=self.cfg.sma_window)
        sma_comp: deque = deque(maxlen=self.cfg.sma_window)
        sma_data: deque = deque(maxlen=self.cfg.sma_window)
        device_type = self.device.type
        for it, (_, sequences, pos_items, neg_items) in enumerate(loader, start=1):
            t0 = time.perf_counter()
            sequences = sequences.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            t1 = time.perf_counter()

            if device_type == "cuda":
                torch.cuda.synchronize()
            elif device_type == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
            comp_start = time.perf_counter()
            pos_scores, neg_scores = self.model(sequences, pos_items, neg_items)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if device_type == "cuda":
                torch.cuda.synchronize()
            elif device_type == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
            comp_end = time.perf_counter()
            t2 = time.perf_counter()
            total_loss += float(loss.item())
            batches += 1
            iter_time = t2 - t0
            comp_time = comp_end - comp_start
            data_time = t1 - t0
            sma_iter.append(iter_time)
            sma_comp.append(comp_time)
            sma_data.append(data_time)
            if self.cfg.profile and it % max(self.cfg.log_interval, 1) == 0:
                util = (sum(sma_comp) / max(sum(sma_iter), 1e-8)) if sma_iter else 0.0
                ips = (self.cfg.batch_size * len(sma_iter)) / max(sum(sma_iter), 1e-8)
                print(f"iter={it} sma_iter={sum(sma_iter)/len(sma_iter):.4f}s sma_comp={sum(sma_comp)/len(sma_comp):.4f}s sma_data={sum(sma_data)/len(sma_data):.4f}s util={util:.2f} ips={ips:.2f}")
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate(self, num_users: int) -> Dict[str, float]:
        self.model.eval()
        if self.cfg.data_source == "synthetic":
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
        elif self.cfg.data_source == "ml1m":
            assert self._ml1m is not None
            n = min(self.cfg.num_val_users, len(self._ml1m.test_prefixes))
            prefixes = self._ml1m.test_prefixes[:n]
            positives = torch.tensor(self._ml1m.test_pos[:n], device=self.device, dtype=torch.long)
            padded = []
            for seq in prefixes:
                seq = seq[-self.cfg.max_seq_len :]
                pad_len = self.cfg.max_seq_len - len(seq)
                padded.append([0] * pad_len + seq)
            sequences = torch.tensor(padded, device=self.device, dtype=torch.long)
            if self.cfg.eval_mode == "sampled":
                negs = []
                for u in range(n):
                    row = []
                    user_set = self._ml1m.user_item_sets[u]
                    for _ in range(self.cfg.negatives_per_user):
                        x = random.randrange(1, self._ml1m.num_items)
                        tries = 0
                        while (x in user_set or x == int(positives[u].item())) and tries < 50:
                            x = random.randrange(1, self._ml1m.num_items)
                            tries += 1
                        row.append(x)
                    negs.append(row)
                candidate_negatives = torch.tensor(negs, device=self.device, dtype=torch.long)
                candidates = torch.cat([positives.unsqueeze(1), candidate_negatives], dim=1)
            else:
                # All-item ranking with seen-item filtering
                all_items = torch.arange(1, self._ml1m.num_items, device=self.device, dtype=torch.long)
                user_repr = self.model.encode_sequence(sequences)
                item_emb = self.model.item_embedding(all_items)  # [num_items-1, dim]
                scores = user_repr @ item_emb.T  # [n, num_items-1]
                # Mask seen items
                mask = torch.zeros_like(scores, dtype=torch.bool)
                for u in range(n):
                    seen = self._ml1m.user_item_sets[u]
                    if seen:
                        idx = torch.tensor([i - 1 for i in seen if i > 0 and i < self._ml1m.num_items], device=self.device)
                        mask[u, idx] = True
                scores = scores.masked_fill(mask, float("-inf"))
                topk_vals, topk_idx = torch.topk(scores, k=max(self.cfg.topk), dim=1)
                ranked_candidates = (topk_idx + 1)  # back to item ids
                # Summarize metrics with provided positives
                return summarize_metrics(ranked_candidates, positives, self.cfg.topk)
        else:
            raise ValueError("unknown data source")

        user_repr = self.model.encode_sequence(sequences)
        item_emb = self.model.item_embedding(candidates)
        scores = (user_repr.unsqueeze(1) * item_emb).sum(dim=-1)
        ranking_indices = torch.argsort(scores, dim=1, descending=True)
        ranked_candidates = torch.gather(candidates, dim=1, index=ranking_indices)
        del num_users
        return summarize_metrics(ranked_candidates, positives, self.cfg.topk)
