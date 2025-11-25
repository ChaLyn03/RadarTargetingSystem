from __future__ import annotations

import numpy as np


def global_threshold(power_map: np.ndarray, k_sigma: float = 3.0) -> np.ndarray:
    mean = power_map.mean()
    std = power_map.std()
    threshold = mean + k_sigma * std
    return power_map > threshold
