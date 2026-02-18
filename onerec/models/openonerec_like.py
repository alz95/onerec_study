from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class SparseMoEFeedForward(nn.Module):
    """Lightweight top-1 sparse MoE FFN block.

    This is a compact approximation to the sparse expert routing idea used in
    OpenOneRec-like architectures.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, dropout: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        router_logits = self.router(x)
        top_expert = torch.argmax(router_logits, dim=-1)  # [B, T]

        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = top_expert == expert_id
            if not mask.any():
                continue
            selected = x[mask]
            transformed = expert(selected)
            out[mask] = transformed
        return self.dropout(out)


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int, num_experts: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.moe_ffn = SparseMoEFeedForward(hidden_dim, ffn_dim, num_experts=num_experts, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None) -> torch.Tensor:
        a1, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)
        tgt = self.norm1(tgt + self.dropout(a1))

        a2, _ = self.cross_attn(tgt, memory, memory, need_weights=False)
        tgt = self.norm2(tgt + self.dropout(a2))

        ffn = self.moe_ffn(tgt)
        tgt = self.norm3(tgt + ffn)
        return tgt


@dataclass
class OpenOneRecLikeConfig:
    num_items: int
    max_seq_len: int
    embedding_dim: int
    hidden_dim: int
    n_heads: int
    num_decoder_layers: int
    ffn_dim: int
    num_experts: int
    dropout: float


class OpenOneRecLikeModel(nn.Module):
    """Encoder-decoder, session-wise autoregressive recommender.

    This is a compact, reproducible implementation that aligns with the high-level
    OpenOneRec methodology: encoder-decoder generation + sparse MoE in decoder.
    """

    def __init__(self, cfg: OpenOneRecLikeConfig, padding_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.padding_idx = padding_idx

        self.item_embedding = nn.Embedding(cfg.num_items, cfg.embedding_dim, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len + 2, cfg.embedding_dim)
        self.in_proj = nn.Linear(cfg.embedding_dim, cfg.hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim=cfg.hidden_dim,
                    n_heads=cfg.n_heads,
                    ffn_dim=cfg.ffn_dim,
                    num_experts=cfg.num_experts,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_decoder_layers)
            ]
        )
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.num_items)

    def _embed_with_pos(self, token_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = token_ids.shape
        pos = torch.arange(seqlen, device=token_ids.device).unsqueeze(0).expand(bsz, seqlen)
        emb = self.item_embedding(token_ids) + self.pos_embedding(pos)
        return self.in_proj(emb)

    def encode_history(self, history_tokens: torch.Tensor) -> torch.Tensor:
        src = self._embed_with_pos(history_tokens)
        src_key_padding_mask = history_tokens.eq(self.padding_idx)
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def decode(self, decoder_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt = self._embed_with_pos(decoder_tokens)
        t = decoder_tokens.size(1)
        causal_mask = torch.triu(torch.ones(t, t, device=decoder_tokens.device), diagonal=1).bool()
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask=causal_mask)
        return self.out_proj(tgt)

    def forward_teacher_forcing(self, history_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.encode_history(history_tokens)
        return self.decode(decoder_tokens, memory)

    def next_item_logits(self, history_tokens: torch.Tensor, bos_token: int) -> torch.Tensor:
        decoder_tokens = torch.full(
            (history_tokens.size(0), 1),
            bos_token,
            dtype=torch.long,
            device=history_tokens.device,
        )
        logits = self.forward_teacher_forcing(history_tokens, decoder_tokens)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate_session(self, history_tokens: torch.Tensor, bos_token: int, gen_len: int) -> List[torch.Tensor]:
        generated = torch.full(
            (history_tokens.size(0), 1),
            bos_token,
            dtype=torch.long,
            device=history_tokens.device,
        )
        steps: List[torch.Tensor] = []
        for _ in range(gen_len):
            logits = self.forward_teacher_forcing(history_tokens, generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            steps.append(next_token.squeeze(1))
        return steps


def sequence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:
    # logits: [B, T, V], labels: [B, T]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)


def dpo_pair_loss(
    preferred_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return -F.logsigmoid(beta * (preferred_logp - rejected_logp)).mean()
