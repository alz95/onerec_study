from __future__ import annotations

import torch
from torch import nn


class SimpleSequentialRec(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, hidden_dim: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, embedding_dim)

    def encode_sequence(self, sequences: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(sequences)
        _, hidden = self.encoder(emb)
        return self.proj(hidden[-1])

    def score_items(self, user_repr: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        return (user_repr * item_emb).sum(dim=-1)

    def forward(self, sequences: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_repr = self.encode_sequence(sequences)
        pos_scores = self.score_items(user_repr, pos_items)
        neg_scores = self.score_items(user_repr, neg_items)
        return pos_scores, neg_scores
