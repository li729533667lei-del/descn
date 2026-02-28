from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn


@dataclass
class FeatureConfig:
    num_dense_cont: int
    num_seq_cont: int
    dense_cat_vocab_sizes: List[int]
    seq_cat_vocab_sizes: List[int]
    seq_len: int


class CrossLayer(nn.Module):
    """Classic DCN cross layer: x_{l+1}=x_0 * (w^T x_l + b) + x_l."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        xlw = self.weight(xl)  # [B, 1]
        return x0 * xlw + self.bias + xl


class DESCN(nn.Module):
    """Deep Entire Space Cross Networks with 4 heads (1 control + 3 treatments)."""

    def __init__(
        self,
        feature_cfg: FeatureConfig,
        emb_dim: int = 16,
        cont_seq_proj_dim: int = 16,
        deep_hidden: List[int] | None = None,
        cross_layers: int = 3,
    ) -> None:
        super().__init__()
        if deep_hidden is None:
            deep_hidden = [128, 64]

        self.feature_cfg = feature_cfg

        # Dense categorical embeddings
        self.dense_cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, emb_dim) for vocab_size in feature_cfg.dense_cat_vocab_sizes]
        )

        # Sequence categorical embeddings
        self.seq_cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, emb_dim) for vocab_size in feature_cfg.seq_cat_vocab_sizes]
        )

        self.cont_seq_proj = nn.Linear(feature_cfg.num_seq_cont, cont_seq_proj_dim)

        input_dim = (
            feature_cfg.num_dense_cont
            + cont_seq_proj_dim
            + emb_dim * len(feature_cfg.dense_cat_vocab_sizes)
            + emb_dim * len(feature_cfg.seq_cat_vocab_sizes)
        )

        self.cross = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])

        deep_layers: List[nn.Module] = []
        prev = input_dim
        for h in deep_hidden:
            deep_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        self.deep_net = nn.Sequential(*deep_layers)

        final_dim = input_dim + prev
        self.control_head = nn.Linear(final_dim, 1)
        self.treatment_heads = nn.ModuleList([nn.Linear(final_dim, 1) for _ in range(3)])

    def encode_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        dense_cont = batch["dense_cont"].float()
        parts.append(dense_cont)

        seq_cont = batch["seq_cont"].float()  # [B, L, C]
        seq_cont_pooled = seq_cont.mean(dim=1)
        parts.append(self.cont_seq_proj(seq_cont_pooled))

        dense_cat = batch["dense_cat"].long()  # [B, N]
        dense_embs = [emb(dense_cat[:, i]) for i, emb in enumerate(self.dense_cat_embeddings)]
        if dense_embs:
            parts.append(torch.cat(dense_embs, dim=-1))

        seq_cat = batch["seq_cat"].long()  # [B, L, N]
        seq_embs = []
        for i, emb in enumerate(self.seq_cat_embeddings):
            e = emb(seq_cat[:, :, i])  # [B, L, D]
            e = e.mean(dim=1)
            seq_embs.append(e)
        if seq_embs:
            parts.append(torch.cat(seq_embs, dim=-1))

        return torch.cat(parts, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x0 = self.encode_features(batch)

        xl = x0
        for layer in self.cross:
            xl = layer(x0, xl)

        xd = self.deep_net(x0)
        x = torch.cat([xl, xd], dim=-1)

        out = {
            "control": torch.sigmoid(self.control_head(x)),
            "treatment_1": torch.sigmoid(self.treatment_heads[0](x)),
            "treatment_2": torch.sigmoid(self.treatment_heads[1](x)),
            "treatment_3": torch.sigmoid(self.treatment_heads[2](x)),
        }
        return out
