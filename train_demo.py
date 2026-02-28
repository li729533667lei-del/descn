from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from descn.data import SyntheticDESCNDataset
from descn.model import DESCN, FeatureConfig


def collate_fn(batch):
    features = {k: torch.stack([item[0][k] for item in batch], dim=0) for k in batch[0][0].keys()}
    labels = {k: torch.stack([item[1][k] for item in batch], dim=0) for k in batch[0][1].keys()}
    return features, labels


def run_demo() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_cfg = FeatureConfig(
        num_dense_cont=6,
        num_seq_cont=4,
        dense_cat_vocab_sizes=[20, 10, 30],
        seq_cat_vocab_sizes=[50, 15],
        seq_len=12,
    )

    train_ds = SyntheticDESCNDataset(size=2048, feature_cfg=feature_cfg, seed=42)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)

    model = DESCN(feature_cfg=feature_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for features, labels in train_loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            preds = model(features)
            loss = sum(criterion(preds[k], labels[k]) for k in preds.keys())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        sample_features, sample_labels = next(iter(train_loader))
        sample_features = {k: v.to(device) for k, v in sample_features.items()}
        outputs = model(sample_features)

    print("\nSample prediction (first 5 rows):")
    for head, value in outputs.items():
        print(f"{head}: {value[:5].squeeze(-1).cpu().numpy()}")


if __name__ == "__main__":
    run_demo()
