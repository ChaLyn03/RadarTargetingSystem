from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .windowing import apply_window


def compute_range_fft(iq: np.ndarray, window_type: str | None = "hann") -> np.ndarray:
    data = apply_window(iq, window_type=window_type, axis=-1) if window_type else iq
    return np.fft.fft(data, axis=-1)


def compute_range_doppler_map(iq: np.ndarray, window_type: str | None = "hann") -> np.ndarray:
    range_fft = compute_range_fft(iq, window_type=window_type)
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
    # Return with range on axis 0 and Doppler on axis 1 for consistency elsewhere
    return doppler_fft.T


def compute_spectrogram(iq: np.ndarray, sample_rate: float, nperseg: int = 128, noverlap: int = 64):
    _, _, Zxx = stft(iq, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    return Zxx
