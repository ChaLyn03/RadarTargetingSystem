from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann, hamming


WINDOWS = {
    "hann": hann,
    "hamming": hamming,
}


def apply_window(iq: np.ndarray, window_type: str = "hann", axis: int = -1) -> np.ndarray:
    if window_type not in WINDOWS:
        raise ValueError(f"Unsupported window: {window_type}")
    window = WINDOWS[window_type](iq.shape[axis], sym=False)
    reshaped = [1] * iq.ndim
    reshaped[axis] = -1
    window = window.reshape(reshaped)
    return iq * window

