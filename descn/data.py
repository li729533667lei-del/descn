from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from .model import FeatureConfig


@dataclass
class DemoBatch:
    features: Dict[str, torch.Tensor]
    labels: Dict[str, torch.Tensor]


class SyntheticDESCNDataset(Dataset):
    def __init__(self, size: int, feature_cfg: FeatureConfig, seed: int = 42):
        super().__init__()
        self.size = size
        self.feature_cfg = feature_cfg

        g = torch.Generator().manual_seed(seed)

        dense_cont = torch.randn(size, feature_cfg.num_dense_cont, generator=g)
        seq_cont = torch.randn(size, feature_cfg.seq_len, feature_cfg.num_seq_cont, generator=g)

        dense_cat = torch.cat(
            [
                torch.randint(0, vocab, (size, 1), generator=g)
                for vocab in feature_cfg.dense_cat_vocab_sizes
            ],
            dim=1,
        )

        seq_cat = torch.cat(
            [
                torch.randint(0, vocab, (size, feature_cfg.seq_len, 1), generator=g)
                for vocab in feature_cfg.seq_cat_vocab_sizes
            ],
            dim=2,
        )

        base = (
            0.6 * dense_cont[:, 0]
            - 0.4 * dense_cont[:, 1]
            + 0.3 * seq_cont.mean(dim=(1, 2))
            + 0.05 * dense_cat[:, 0].float()
            - 0.03 * dense_cat[:, 1].float()
        )

        noise = 0.3 * torch.randn(size, generator=g)
        control_logit = base + noise
        t1_logit = base + 0.5 + 0.15 * dense_cat[:, 0].float() + noise
        t2_logit = base - 0.2 + 0.25 * dense_cont[:, 2] + noise
        t3_logit = base + 0.1 - 0.2 * seq_cont[:, :, 0].mean(dim=1) + noise

        self.features = {
            "dense_cont": dense_cont,
            "seq_cont": seq_cont,
            "dense_cat": dense_cat,
            "seq_cat": seq_cat,
        }
        self.labels = {
            "control": torch.bernoulli(torch.sigmoid(control_logit)).unsqueeze(1),
            "treatment_1": torch.bernoulli(torch.sigmoid(t1_logit)).unsqueeze(1),
            "treatment_2": torch.bernoulli(torch.sigmoid(t2_logit)).unsqueeze(1),
            "treatment_3": torch.bernoulli(torch.sigmoid(t3_logit)).unsqueeze(1),
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        feat = {k: v[idx] for k, v in self.features.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        return feat, label
