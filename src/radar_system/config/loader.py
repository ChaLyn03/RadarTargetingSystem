"""Simple YAML config loader for RadarTargetingSystem.

This module provides helpers to load the YAML files in `config/` and
convert them into the project's dataclasses (SimulationConfig, PipelineConfig,
and TrainingConfig). It is intentionally small and dependency-light.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

from radar_system.sim.fmcw_simulator import SimulationConfig
from radar_system.pipeline.pipeline import PipelineConfig
from radar_system.models.training.train import TrainingConfig


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all(config_dir: str = "config") -> dict[str, dict]:
    """Load all known YAML configs from `config_dir`.

    Returns a mapping: {"radar": dict, "pipeline": dict, "ml": dict}
    """
    radar = _load_yaml(os.path.join(config_dir, "radar.yaml"))
    pipeline = _load_yaml(os.path.join(config_dir, "pipeline.yaml"))
    ml = _load_yaml(os.path.join(config_dir, "ml.yaml"))
    return {"radar": radar, "pipeline": pipeline, "ml": ml}


def simulation_config_from_dict(d: dict | None) -> SimulationConfig:
    d = (d or {}).copy()
    # YAML uses nested key 'simulation'
    if "simulation" in d:
        d = d["simulation"]
    # map YAML keys directly to dataclass fields where possible
    return SimulationConfig(**d)


def pipeline_config_from_dict(d: dict | None, sim_config: SimulationConfig) -> PipelineConfig:
    d = (d or {}).copy()
    if "pipeline" in d:
        d = d["pipeline"]
    # convert list guard/training cells to tuple if present
    guard = tuple(d.get("guard_cells", (1, 1)))
    training = tuple(d.get("training_cells", (4, 4)))
    return PipelineConfig(
        sim_config=sim_config,
        window_type=d.get("window_type", "hann"),
        use_cfar=bool(d.get("use_cfar", True)),
        threshold_k=float(d.get("threshold_k", 3.0)),
        guard_cells=guard,
        training_cells=training,
        cfar_rate=float(d.get("cfar_rate", 1.4)),
        patch_size=int(d.get("patch_size", 32)),
    )


def training_config_from_dict(d: dict | None) -> TrainingConfig:
    d = (d or {}).copy()
    if "training" in d:
        d = d["training"]
    return TrainingConfig(
        lr=float(d.get("lr", 1e-3)),
        epochs=int(d.get("epochs", 3)),
        batch_size=int(d.get("batch_size", 32)),
        device=str(d.get("device", "cpu")),
        num_samples=int(d.get("num_samples", 256)),
        patch_size=int(d.get("patch_size", 32)),
    )


def load_configs(config_dir: str = "config") -> tuple[SimulationConfig, PipelineConfig, TrainingConfig]:
    allc = load_all(config_dir)
    sim_cfg = simulation_config_from_dict(allc.get("radar"))
    pipe_cfg = pipeline_config_from_dict(allc.get("pipeline"), sim_cfg)
    train_cfg = training_config_from_dict(allc.get("ml"))
    return sim_cfg, pipe_cfg, train_cfg
