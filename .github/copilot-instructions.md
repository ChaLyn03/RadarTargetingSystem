# RadarTargetingSystem AI Agent Instructions

## Project Overview
**RadarTargetingSystem** is an end-to-end FMCW radar demonstrator with signal simulation, DSP processing, target detection, and CNN-based classification. The application runs as a Streamlit interactive dashboard.

### How to Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The dashboard opens at `http://localhost:8501` and allows real-time configuration of radar simulation parameters.

## Architecture & Data Flow

### Core Processing Pipeline
The five-stage pipeline (in `src/ui/pipeline.py`) processes radar frames:

1. **Simulate** (`src/sim/fmcw.py`): Generate FMCW IQ samples
   - Input: `SimulationConfig` (n_chirps, n_samples, noise_db) + list of `Target` objects
   - Output: `SimulationResult` with `iq` matrix (n_chirps, n_samples) and calibration data (ranges, doppler frequencies)
   - Key: Computes beat frequencies and Doppler shifts using physics equations

2. **Process** (`src/dsp/range_doppler.py`): Transform IQ to range–Doppler map
   - Apply Hann windowing per chirp, then range FFT → Doppler FFT
   - Outputs log-power `power_map` (range × Doppler bins)
   - Alternatives: `compute_spectrogram()` for time-frequency analysis

3. **Detect** (`src/detection/`): Identify signal peaks
   - Choose CA-CFAR (`cfar.py`) or global threshold (`threshold.py`)
   - Returns binary `detection_mask`
   - Cluster connected components (`clustering.py`) → list of `Detection` objects with centroids

4. **Classify** (`src/models/training.py` + `cnn.py`): Tag detections
   - Extract RD patches around each detection, normalize with `normalize_patch()`
   - Run CNN inference on PyTorch `PatchCNN` model
   - Classes: "car", "pedestrian", "drone", "clutter" (defined in `src/models/dataset.py`)

5. **Visualize** (`app.py`): Display results in Streamlit UI
   - IQ waveforms (Plotly), range profiles, RD heatmaps with detection overlays, classification tables

### Key Data Structures
- **`SimulationConfig`** (dataclass): Sample rate, chirp parameters, noise level, random seed
- **`Target`** (dataclass): range_m, velocity_mps, rcs, micro_doppler_hz
- **`Detection`** (dataclass): range_idx, doppler_idx, strength (float bin coordinates, not physical units)
- **`PipelineConfig`**: Selects window type, CFAR params (guard/training cells, rate), patch size
- **`PipelineOutput`**: Bundles simulation result, FFT outputs, masks, detections, classifications

## Project-Specific Patterns

### Configuration via Dataclasses
Every stage has a config dataclass (e.g., `SimulationConfig`, `PipelineConfig`, `TrainingConfig`). **Always instantiate configs as defaults are sensible**—modify only fields needed for the use case. Example:
```python
cfg = SimulationConfig(n_chirps=16, noise_power_db=-25)  # Other fields use defaults
```

### Streamlit Session State
Classification model persists in `st.session_state["classifier"]`. The UI trains on synthetic patches from `SyntheticRDPatchDataset`. **No file I/O for models**—state is in-memory.

### Synthetic Data Generation
`SyntheticRDPatchDataset` procedurally generates class-specific RD patch signatures (Gaussian blobs for car/pedestrian/drone, random noise for clutter) instead of using external datasets. Seed-controlled for reproducibility.

### Window Application Convention
In `src/dsp/windowing.py`, the `apply_window()` function reshapes window arrays to broadcast correctly along any axis. Always use `window_type="hann"` by default unless experimenting with alternatives.

### Index vs. Physical Units
`Detection` objects store `range_idx`, `doppler_idx` as **bin indices** (floats from center-of-mass). Convert to physical units using:
```python
physical_range = sim_result.ranges_m[int(det.range_idx)]
physical_doppler = sim_result.doppler_hz[int(det.doppler_idx)]
```

## Integration Points

### FMCW Physics
- **Beat frequency formula**: $f_b = \frac{2KR}{c}$ where $K$ is sweep slope, $R$ is range, $c$ is speed of light
- **Doppler**: $f_d = \frac{2v}{\lambda}$ where $v$ is velocity, $\lambda$ is wavelength at carrier freq
- See `src/sim/fmcw.py` lines computing `beat_freq` and `doppler_freq` in the chirp loop

### DSP Conventions
- FFT outputs are **not normalized**; use magnitude `|X|` or power `|X|²`
- Doppler FFT uses `fftshift()` for zero-centered bins (see `compute_range_doppler_map()`)
- Log-power maps use `10 * log10(|X|² + 1e-9)` to avoid log(0)

### CA-CFAR Implementation
Guard and training cell rings are applied **symmetrically around each cell**. Margins are small (typically 1×1 guard, 4×4 training) because demo maps are tiny. Scale these for larger grids.

### CNN Architecture
`PatchCNN` uses 3 conv blocks (8→16→32 channels) + AdaptiveAvgPool → FC classifier. Input patches are **1-channel** (single RD map intensity). Normalization: min-max per-patch via `normalize_patch()`.

## Critical Developer Workflows

### Adding a New Detection Algorithm
1. Create module in `src/detection/` (e.g., `src/detection/new_method.py`)
2. Signature: `function(power_map: np.ndarray, **params) -> np.ndarray` (returns bool mask)
3. Add option to `PipelineConfig` and `run_pipeline()` conditional logic

### Extending the Simulation
- Add fields to `SimulationConfig` (e.g., frequency hop, PRF variations)
- Modify chirp loop in `simulate_frame()` to use new parameters
- Test with a quick script: `from src.sim.fmcw import *; cfg = SimulationConfig(...); result = simulate_frame(cfg, [Target(...)])`

### Training a Custom Classifier
1. Create dataset class inheriting from `torch.utils.data.Dataset` (see `SyntheticRDPatchDataset`)
2. Pass to `create_dataloader()`
3. Call `train_quick_classifier(TrainingConfig(device=device, epochs=E, num_samples=S))`
4. Modify `CLASSES` in `src/models/dataset.py` if adding new target types

### Debugging Detection Issues
Print intermediate results after each stage:
```python
output = run_pipeline(targets, cfg, device=device)
print(f"Detections found: {len(output.detections)}")
print(f"Power range: {output.power_map.min():.3f} to {output.power_map.max():.3f}")
print(f"Mask coverage: {output.detection_mask.sum()} cells")
```

## File Reference Map

| File/Module | Purpose | Key Classes/Functions |
|---|---|---|
| `app.py` | Streamlit dashboard | `run_dashboard()` (main entry) |
| `src/sim/fmcw.py` | FMCW signal generation | `SimulationConfig`, `Target`, `simulate_frame()` |
| `src/dsp/range_doppler.py` | FFT-based processing | `compute_range_doppler_map()`, `compute_range_fft()` |
| `src/detection/cfar.py` | CA-CFAR detector | `ca_cfar()` |
| `src/detection/threshold.py` | Simple threshold detector | `global_threshold()` |
| `src/detection/clustering.py` | Connected-component analysis | `cluster_detections()`, `Detection` |
| `src/models/cnn.py` | PyTorch CNN classifier | `PatchCNN`, `normalize_patch()` |
| `src/models/training.py` | Model training loop | `train_quick_classifier()`, `run_inference()` |
| `src/models/dataset.py` | Synthetic dataset | `SyntheticRDPatchDataset`, `CLASSES` |
| `src/ui/pipeline.py` | End-to-end orchestration | `PipelineConfig`, `run_pipeline()` |

## Dependencies
- `numpy`, `scipy`: Signal processing, windowing
- `torch`: PyTorch for CNN and inference
- `streamlit`: Interactive dashboard UI
- `plotly`: Interactive visualizations
- `pandas`: Table formatting in UI

---

**Last Updated:** November 26, 2025  
**Codebase Version:** Main branch
