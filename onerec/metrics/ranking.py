from __future__ import annotations

import math
from typing import Dict, Iterable

import torch


def hitrate_at_k(rankings: torch.Tensor, positives: torch.Tensor, k: int) -> float:
    topk = rankings[:, :k]
    hits = (topk == positives.unsqueeze(1)).any(dim=1).float()
    return float(hits.mean().item())


def ndcg_at_k(rankings: torch.Tensor, positives: torch.Tensor, k: int) -> float:
    topk = rankings[:, :k]
    gains = (topk == positives.unsqueeze(1)).float()
    discounts = torch.tensor([1.0 / math.log2(i + 2) for i in range(k)], device=rankings.device)
    dcg = (gains * discounts).sum(dim=1)
    return float(dcg.mean().item())


def summarize_metrics(rankings: torch.Tensor, positives: torch.Tensor, topk: Iterable[int]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for k in topk:
        hr = hitrate_at_k(rankings, positives, k)
        results[f"HR@{k}"] = hr
        results[f"Recall@{k}"] = hr
        results[f"NDCG@{k}"] = ndcg_at_k(rankings, positives, k)
    return results
