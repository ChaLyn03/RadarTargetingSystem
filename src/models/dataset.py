from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASSES = ["car", "pedestrian", "drone", "clutter"]


@dataclass
class DatasetConfig:
    patch_size: int = 32
    num_samples: int = 256
    snr_db: float = 5.0
    seed: int = 0


def _gaussian_blob(size: int, center: Tuple[float, float], sigma: float) -> np.ndarray:
    x = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, x)
    cx, cy = center
    return np.exp(-(((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2)))


class SyntheticRDPatchDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.samples: List[Tuple[np.ndarray, int]] = []
        for _ in range(config.num_samples):
            label_idx = self.rng.integers(0, len(CLASSES))
            patch = self._generate_patch(label_idx)
            self.samples.append((patch, label_idx))

    def _generate_patch(self, label_idx: int) -> np.ndarray:
        size = self.config.patch_size
        noise = self.rng.normal(scale=10 ** (-self.config.snr_db / 20), size=(size, size))
        patch = noise
        center = (self.rng.uniform(-0.3, 0.3), self.rng.uniform(-0.3, 0.3))

        if CLASSES[label_idx] == "car":
            patch += 3.0 * _gaussian_blob(size, center, sigma=0.12)
        elif CLASSES[label_idx] == "pedestrian":
            patch += 1.8 * _gaussian_blob(size, center, sigma=0.18)
            patch += 0.6 * _gaussian_blob(size, (center[0] + 0.15, center[1]), sigma=0.1)
        elif CLASSES[label_idx] == "drone":
            patch += 1.5 * _gaussian_blob(size, center, sigma=0.1)
            patch += 0.8 * _gaussian_blob(size, (center[0] - 0.12, center[1] + 0.12), sigma=0.08)
        else:  # clutter
            patch += self.rng.normal(scale=0.4, size=(size, size))
        patch = patch.astype(np.float32)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        return patch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        patch, label_idx = self.samples[idx]
        tensor = torch.from_numpy(patch)[None, :, :]
        return tensor, label_idx


def create_dataloader(config: DatasetConfig, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    dataset = SyntheticRDPatchDataset(config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

