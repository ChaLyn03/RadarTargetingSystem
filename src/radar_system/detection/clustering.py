from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.ndimage import label, center_of_mass


@dataclass
class Detection:
    range_idx: float
    doppler_idx: float
    strength: float


@dataclass
class DetectionSummary:
    detections: List[Detection]
    mask: np.ndarray


def cluster_detections(power_map: np.ndarray, detection_mask: np.ndarray) -> DetectionSummary:
    labeled, num_features = label(detection_mask)
    detections: List[Detection] = []
    for label_id in range(1, num_features + 1):
        component_mask = labeled == label_id
        if not component_mask.any():
            continue
        strength = power_map[component_mask].sum()
        r_idx, d_idx = center_of_mass(power_map, labels=labeled, index=label_id)
        detections.append(Detection(range_idx=float(r_idx), doppler_idx=float(d_idx), strength=float(strength)))
    return DetectionSummary(detections=detections, mask=labeled)
