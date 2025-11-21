from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def normalize_patch(patch: torch.Tensor) -> torch.Tensor:
    min_val = patch.amin(dim=(-2, -1), keepdim=True)
    max_val = patch.amax(dim=(-2, -1), keepdim=True)
    return (patch - min_val) / (max_val - min_val + 1e-8)

