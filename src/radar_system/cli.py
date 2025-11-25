"""Command-line interface for RadarTargetingSystem.

Provides basic headless operations:
 - run: simulate + pipeline (no UI)
 - train: train quick classifier on synthetic patches
 - demo: run a single-run simulation and print summary

Usage examples:
    python -m radar_system.cli run --config-dir=config --targets 2
    python -m radar_system.cli train --config-dir=config
"""
from __future__ import annotations

import argparse
import pprint
from typing import List

from radar_system.config.loader import load_configs
from radar_system.sim.fmcw_simulator import Target, generate_random_targets
from radar_system.pipeline.pipeline import run_pipeline
from radar_system.models.training.train import train_quick_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="radar_system")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run simulation + pipeline (headless)")
    run.add_argument("--config-dir", default="config")
    run.add_argument("--targets", type=int, default=2)

    train = sub.add_parser("train", help="Train a quick classifier using synthetic patches")
    train.add_argument("--config-dir", default="config")

    demo = sub.add_parser("demo", help="Run a single small run and print a short summary")
    demo.add_argument("--config-dir", default="config")
    demo.add_argument("--targets", type=int, default=1)

    return p.parse_args()


def run_headless(sim_cfg, pipe_cfg, train_cfg, n_targets: int = 2):
    # create random targets and run pipeline
    targets = generate_random_targets(n_targets, sim_cfg)
    out = run_pipeline(targets, pipe_cfg, classifier=None, device=train_cfg.device)
    # print a short summary
    print("Simulation shapes: IQ=", out.sim.iq.shape)
    print("Detections:")
    for d in out.detections:
        print(f"  range_idx={d.range_idx:.2f}, doppler_idx={d.doppler_idx:.2f}, strength={d.strength:.2f}")
    return out


def main():
    args = parse_args()
    sim_cfg, pipe_cfg, train_cfg = load_configs(args.config_dir)

    if args.cmd == "run":
        run_headless(sim_cfg, pipe_cfg, train_cfg, n_targets=args.targets)
    elif args.cmd == "train":
        print("Training quick classifier with config:")
        pprint.pprint(train_cfg.__dict__)
        clf = train_quick_classifier(train_cfg)
        print("Trained classifier:", clf)
    elif args.cmd == "demo":
        out = run_headless(sim_cfg, pipe_cfg, train_cfg, n_targets=args.targets)
        print("Demo complete. Power range:", out.power_map.min(), out.power_map.max())


if __name__ == "__main__":
    main()
