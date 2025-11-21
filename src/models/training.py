from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from .cnn import PatchCNN
from .dataset import create_dataloader, DatasetConfig, CLASSES


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 32
    device: str = "cpu"
    num_samples: int = 256
    patch_size: int = 32


def train_quick_classifier(config: TrainingConfig) -> PatchCNN:
    dataset_cfg = DatasetConfig(patch_size=config.patch_size, num_samples=config.num_samples)
    dataloader = create_dataloader(dataset_cfg, batch_size=config.batch_size, shuffle=True)
    model = PatchCNN(num_classes=len(CLASSES)).to(config.device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    model.train()
    for _ in range(config.epochs):
        for batch, labels in dataloader:
            batch = batch.to(config.device)
            labels = labels.to(config.device)
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run_inference(model: PatchCNN, patches: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(patches.to(device))
        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
    return {"probs": probs.cpu(), "preds": preds.cpu(), "confidences": confidences.cpu()}

