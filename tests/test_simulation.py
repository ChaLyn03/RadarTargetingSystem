from radar_system.sim.fmcw_simulator import SimulationConfig, Target, simulate_frame


def test_simulate_frame_shape():
    cfg = SimulationConfig(n_chirps=8, n_samples=128)
    t = Target(range_m=50.0, velocity_mps=5.0)
    res = simulate_frame(cfg, [t])
    assert res.iq.shape == (8, 128)
    assert len(res.ranges_m) == 128
    assert len(res.doppler_hz) == 8
