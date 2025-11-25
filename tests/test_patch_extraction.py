import numpy as np

from radar_system.pipeline.pipeline import _extract_patch


def test_extract_patch_centers_edge_detection():
    power_map = np.zeros((5, 5), dtype=np.float32)
    power_map[0, 0] = 10.0

    patch = _extract_patch(power_map, center=(0, 0), size=5)

    center_idx = 5 // 2
    assert patch[center_idx, center_idx] == 10.0
    # No energy should be lost or duplicated in the patch
    assert np.isclose(patch.sum(), power_map.sum())
