from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..detection.cfar import ca_cfar
from ..detection.clustering import cluster_detections, Detection
from ..detection.threshold import global_threshold
from ..dsp.range_doppler import compute_range_doppler_map, compute_range_fft
from ..models.cnn import normalize_patch, PatchCNN
from ..models.dataset import CLASSES
from ..models.training import run_inference
from ..sim.fmcw import (
    SimulationConfig,
    SimulationResult,
    Target,
    generate_random_targets,
    simulate_frame,
)


@dataclass
class PipelineConfig:
    sim_config: SimulationConfig
    window_type: str = "hann"
    use_cfar: bool = True
    threshold_k: float = 3.0
    guard_cells: tuple[int, int] = (1, 1)
    training_cells: tuple[int, int] = (4, 4)
    cfar_rate: float = 1.4
    patch_size: int = 32


@dataclass
class PipelineOutput:
    sim: SimulationResult
    range_fft: np.ndarray
    rd_map: np.ndarray
    power_map: np.ndarray
    detection_mask: np.ndarray
    detections: List[Detection]
    classification: Optional[List[Dict]] = None


def _extract_patch(power_map: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    half = size // 2
    r, c = center
    r_start = max(0, r - half)
    r_end = min(power_map.shape[0], r + half)
    c_start = max(0, c - half)
    c_end = min(power_map.shape[1], c + half)
    patch = np.zeros((size, size), dtype=np.float32)
    patch_slice = power_map[r_start:r_end, c_start:c_end]
    patch[: patch_slice.shape[0], : patch_slice.shape[1]] = patch_slice
    return patch


def run_pipeline(targets: List[Target], config: PipelineConfig, classifier: PatchCNN | None = None, device: str = "cpu") -> PipelineOutput:
    sim = simulate_frame(config.sim_config, targets)
    range_fft = compute_range_fft(sim.iq, window_type=config.window_type)
    rd_map = compute_range_doppler_map(sim.iq, window_type=config.window_type)
    power_map = np.abs(rd_map) ** 2

    if config.use_cfar:
        detection_mask = ca_cfar(power_map, guard_cells=config.guard_cells, training_cells=config.training_cells, rate=config.cfar_rate)
    else:
        detection_mask = global_threshold(power_map, k_sigma=config.threshold_k)

    summary = cluster_detections(power_map, detection_mask)
    classification: Optional[List[Dict]] = None

    if classifier and summary.detections:
        patches = []
        for det in summary.detections:
            patch = _extract_patch(np.log1p(power_map), (int(det.range_idx), int(det.doppler_idx)), size=config.patch_size)
            patches.append(patch)
        patch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1)
        patch_tensor = normalize_patch(patch_tensor)
        results = run_inference(classifier, patch_tensor, device=device)
        classification = []
        for det, prob, pred, conf in zip(summary.detections, results["probs"], results["preds"], results["confidences"]):
            classification.append(
                {
                    "detection": det,
                    "label": CLASSES[int(pred)],
                    "probabilities": prob.tolist(),
                    "confidence": float(conf),
                }
            )

    return PipelineOutput(
        sim=sim,
        range_fft=range_fft,
        rd_map=rd_map,
        power_map=power_map,
        detection_mask=detection_mask,
        detections=summary.detections,
        classification=classification,
    )


def _format_detection(det: Detection, sim: SimulationResult) -> str:
    return (
        f"range={sim.ranges_m[int(det.range_idx)]:.2f} m, "
        f"doppler={sim.doppler_hz[int(det.doppler_idx)]:.1f} Hz, "
        f"snr={det.snr_db:.1f} dB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-off radar pipeline simulation")
    parser.add_argument("--targets", type=int, default=2, help="number of synthetic targets")
    parser.add_argument("--noise-db", type=float, default=-20.0, help="noise power in dB")
    parser.add_argument("--chirps", type=int, default=32, help="chirps per frame")
    parser.add_argument("--samples", type=int, default=512, help="samples per chirp")
    parser.add_argument("--no-cfar", action="store_true", help="disable CA-CFAR and use global thresholding")
    parser.add_argument("--threshold-k", type=float, default=3.0, help="k·sigma for global thresholding")
    parser.add_argument("--guard", type=int, default=1, help="guard cells for CFAR")
    parser.add_argument("--train", type=int, default=4, help="training cells for CFAR")
    args = parser.parse_args()

    sim_config = SimulationConfig(
        n_chirps=args.chirps,
        n_samples=args.samples,
        noise_power_db=args.noise_db,
    )
    targets = generate_random_targets(args.targets, sim_config)
    pipeline_config = PipelineConfig(
        sim_config=sim_config,
        use_cfar=not args.no_cfar,
        threshold_k=args.threshold_k,
        guard_cells=(args.guard, args.guard),
        training_cells=(args.train, args.train),
    )

    output = run_pipeline(targets, pipeline_config)
    print("Simulation complete\n-------------------")
    print(f"Targets simulated: {len(targets)}")
    print(f"Detections found: {len(output.detections)}\n")

    if output.detections:
        print("Detections (range, Doppler, SNR):")
        for det in output.detections:
            print(f"- {_format_detection(det, output.sim)}")
    else:
        print("No detections above threshold")


if __name__ == "__main__":
    main()

