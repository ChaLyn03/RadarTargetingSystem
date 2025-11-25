import numpy as np

from radar_system.dsp.range_doppler import compute_range_doppler_map
from radar_system.sim.fmcw_simulator import SimulationConfig, Target, simulate_frame


def test_rd_map_axes_align_with_simulation_config():
    sim_cfg = SimulationConfig(n_chirps=6, n_samples=32)
    target = Target(range_m=30.0, velocity_mps=5.0)
    sim = simulate_frame(sim_cfg, [target])

    rd_map = compute_range_doppler_map(sim.iq)

    # axis 0 is range (n_samples), axis 1 is Doppler (n_chirps)
    assert rd_map.shape == (sim_cfg.n_samples, sim_cfg.n_chirps)


def test_doppler_axis_is_shifted_and_monotonic():
    sim_cfg = SimulationConfig(n_chirps=8, n_samples=16)
    sim = simulate_frame(sim_cfg, [])

    # fftshifted doppler_hz should be sorted ascending around zero
    diffs = np.diff(sim.doppler_hz)
    assert np.all(diffs > 0)
