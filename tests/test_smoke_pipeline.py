"""Smoke test for end-to-end pipeline using config loader.

This test ensures the config loader works and the pipeline can be executed
in a minimal configuration without raising exceptions.
"""
from radar_system.config.loader import load_configs
from radar_system.sim.fmcw_simulator import Target, generate_random_targets
from radar_system.pipeline.pipeline import run_pipeline


def test_pipeline_smoke_runs():
    sim_cfg, pipe_cfg, train_cfg = load_configs("config")
    # make it small and deterministic for CI
    sim_cfg.n_chirps = 8
    sim_cfg.n_samples = 128
    targets = generate_random_targets(1, sim_cfg)
    out = run_pipeline(targets, pipe_cfg, classifier=None, device=train_cfg.device)
    assert out.sim.iq.shape == (sim_cfg.n_chirps, sim_cfg.n_samples)
    # power_map should have same doppler/range dims
    assert out.power_map.shape[0] == sim_cfg.n_samples or out.power_map.ndim == 2
