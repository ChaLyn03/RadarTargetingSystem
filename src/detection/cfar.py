from __future__ import annotations

import numpy as np


def ca_cfar(power_map: np.ndarray, guard_cells: tuple[int, int] = (1, 1), training_cells: tuple[int, int] = (4, 4), rate: float = 1.4) -> np.ndarray:
    n_rows, n_cols = power_map.shape
    g_row, g_col = guard_cells
    t_row, t_col = training_cells
    margin_row = g_row + t_row
    margin_col = g_col + t_col
    detections = np.zeros_like(power_map, dtype=bool)

    for r in range(margin_row, n_rows - margin_row):
        for c in range(margin_col, n_cols - margin_col):
            r_start = r - margin_row
            r_end = r + margin_row + 1
            c_start = c - margin_col
            c_end = c + margin_col + 1
            window = power_map[r_start:r_end, c_start:c_end]

            guard_r_start = t_row
            guard_r_end = t_row + 2 * g_row + 1
            guard_c_start = t_col
            guard_c_end = t_col + 2 * g_col + 1

            mask = np.ones_like(window, dtype=bool)
            mask[guard_r_start:guard_r_end, guard_c_start:guard_c_end] = False
            training_cells_values = window[mask]
            noise_level = training_cells_values.mean()
            threshold = noise_level * rate
            detections[r, c] = power_map[r, c] > threshold

    return detections

