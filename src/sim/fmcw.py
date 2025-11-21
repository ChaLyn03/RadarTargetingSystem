from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np

C = 299_792_458.0


def _db_to_linear(power_db: float) -> float:
    return 10 ** (power_db / 10)


@dataclass
class Target:
    range_m: float
    velocity_mps: float
    rcs: float = 1.0
    micro_doppler_hz: float = 0.0


@dataclass
class SimulationConfig:
    sample_rate: float = 2e6
    sweep_bandwidth: float = 150e6
    sweep_time: float = 1e-3
    carrier_freq: float = 77e9
    n_chirps: int = 32
    n_samples: int = 512
    noise_power_db: float = -20.0
    random_seed: int | None = 0
    wobble_scale: float = 0.25

    def slope(self) -> float:
        return self.sweep_bandwidth / self.sweep_time

    def wavelength(self) -> float:
        return C / self.carrier_freq


@dataclass
class SimulationResult:
    iq: np.ndarray
    ranges_m: np.ndarray
    doppler_hz: np.ndarray
    targets: List[Target] = field(default_factory=list)


def simulate_frame(config: SimulationConfig, targets: Iterable[Target]) -> SimulationResult:
    rng = np.random.default_rng(config.random_seed)
    time_fast = np.arange(config.n_samples, dtype=np.float64) / config.sample_rate
    slope = config.slope()
    wavelength = config.wavelength()

    iq = np.zeros((config.n_chirps, config.n_samples), dtype=np.complex128)

    for chirp_idx in range(config.n_chirps):
        for target in targets:
            beat_freq = 2 * slope * target.range_m / C
            doppler_freq = 2 * target.velocity_mps / wavelength
            wobble = target.micro_doppler_hz * math.sin(2 * math.pi * config.wobble_scale * chirp_idx)
            phase_fast = 2 * math.pi * (beat_freq * time_fast)
            phase_slow = 2 * math.pi * (doppler_freq + wobble) * config.sweep_time * chirp_idx
            amplitude = math.sqrt(target.rcs) / max(1.0, target.range_m)
            iq[chirp_idx, :] += amplitude * np.exp(1j * (phase_fast + phase_slow))

    noise_power = _db_to_linear(config.noise_power_db)
    noise = (rng.normal(scale=math.sqrt(noise_power / 2), size=iq.shape) +
             1j * rng.normal(scale=math.sqrt(noise_power / 2), size=iq.shape))
    iq += noise

    ranges_m = np.linspace(0, C * config.sweep_time / (2 * config.sweep_bandwidth), config.n_samples, endpoint=False)
    doppler_hz = np.fft.fftfreq(config.n_chirps, d=config.sweep_time)

    return SimulationResult(iq=iq, ranges_m=ranges_m, doppler_hz=doppler_hz, targets=list(targets))


def generate_random_targets(num_targets: int, config: SimulationConfig, rng: np.random.Generator | None = None) -> List[Target]:
    rng = rng or np.random.default_rng()
    max_range = C * config.sweep_time / (2 * config.sweep_bandwidth)
    targets: List[Target] = []
    for _ in range(num_targets):
        range_m = rng.uniform(0.1 * max_range, 0.8 * max_range)
        velocity_mps = rng.uniform(-30.0, 30.0)
        rcs = rng.uniform(0.5, 5.0)
        micro_doppler_hz = rng.uniform(0.0, 40.0)
        targets.append(Target(range_m=range_m, velocity_mps=velocity_mps, rcs=rcs, micro_doppler_hz=micro_doppler_hz))
    return targets

