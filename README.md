# Radar Target Viewer & Classifier

This project is an end-to-end FMCW radar demonstrator that simulates radar returns, processes them with DSP, detects targets, and classifies detections with a lightweight CNN.

The code has been reorganized into a proper Python package under `src/radar_system`.

Top-level quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Run the Streamlit app from the package directory
streamlit run src/radar_system/app.py
```

Project layout (new)

```
.
├─ config/                   # YAML configurations (radar, ml, pipeline)
├─ src/
│  └─ radar_system/          # Python package
│     ├─ app.py              # Streamlit dashboard (entrypoint)
│     ├─ sim/                # FMCW simulator (fmcw_simulator.py)
│     ├─ dsp/                # Windowing and FFT utilities
│     ├─ detection/          # threshold, cfar, clustering
│     ├─ pipeline/           # orchestration (formerly ui)
│     └─ models/             # models/architecture, datasets, training
├─ tests/                    # minimal tests
├─ requirements.txt
└─ pyproject.toml
```

Pipeline flow

```mermaid
flowchart LR
  A[Simulate\n(fmcw_simulator.simulate_frame)] --> B[Process\n(range FFT → Doppler FFT)]
  B --> C[Detect\n(ca_cfar / threshold)]
  C --> D[Cluster\n(cluster_detections → Detection)]
  D --> E[Classify\n(extract patch → PatchCNN)]
  E --> F[Visualize\n(Streamlit app)]
```

1. Simulate: `radar_system.sim.fmcw_simulator.simulate_frame()` produces IQ frames and calibration arrays.
2. Process: `radar_system.dsp.range_doppler` computes range FFT and Doppler FFT to build an RD map.
3. Detect: `radar_system.detection` offers `ca_cfar()` and a global threshold detector; connected components are clustered into `Detection` objects.
4. Classify: RD patches around detections are normalized and passed to `PatchCNN` (training helpers in `radar_system.models.training`).
5. Visualize: `src/radar_system/app.py` is the Streamlit UI which wires the pipeline together.

Notes

- The FMCW simulator file was renamed to `fmcw_simulator.py` to clarify its role.
- Configuration values were externalized to `config/*.yaml`.
- Model code was split into `models/architecture`, `models/datasets`, and `models/training` for clarity.
- Use `make` shortcuts (see `Makefile`) for common tasks: `make run`, `make test`, `make lint`.


