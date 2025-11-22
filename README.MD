# Radar Target Viewer & Classifier

An end-to-end FMCW radar demonstrator that simulates radar returns, runs a DSP pipeline (windowing, range/Doppler FFTs, CFAR detection), and classifies detections with a lightweight CNN. A Streamlit dashboard ties it together for interactive exploration.

## Signal model
For a linear FMCW chirp with carrier frequency $f_0$, sweep slope $K = B/T_{\text{sweep}}$, and sweep duration $T_{\text{sweep}}$, the transmitted baseband signal is

\[
\begin{aligned}
s_{\text{tx}}(t) &= \exp\left(j2\pi\left(f_0 t + \tfrac{K}{2} t^2\right)\right) \\
s_{\text{rx}}(t) &= \sum_{i=1}^{N_\text{targets}} A_i\, \exp\left(j2\pi\left((f_0 + K(t-\tau_i))(t-\tau_i) + f_{d,i} t\right)\right) + w(t)
\end{aligned}
\]

where $\tau_i = 2R_i/c$ is the round-trip delay for target range $R_i$, $f_{d,i} = 2v_i/\lambda$ is the Doppler shift for radial velocity $v_i$, $A_i$ encodes RCS and path loss, and $w(t)$ is white Gaussian noise. After dechirping and sampling, the baseband beat frequency can be approximated as
\[
 f_{b,i} \approx \frac{2 K R_i}{c} + \frac{2 v_i f_0}{c}.
\]

This repository simulates frames of shape `(N_chirps, N_samples)` using this structure, adds optional micro-Doppler wobble, and feeds the result into the DSP stack.

## Repository layout
- `src/sim/`: Radar signal simulation utilities.
- `src/dsp/`: Windowing, FFTs, and range–Doppler map generation.
- `src/detection/`: Thresholding, CA-CFAR, and clustering.
- `src/models/`: CNN classifier and synthetic patch dataset utilities.
- `src/ui/`: Pipeline orchestration for the Streamlit app.
- `app.py`: Streamlit dashboard entrypoint.
- `requirements.txt`: Python dependencies.
- `notebooks/`, `docs/`: Space for experiments and architecture notes.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Workflow
1. **Simulate**: Generate FMCW frames with configurable targets, SNR, and clutter.
2. **Process**: Apply Hann windowing, range FFT per chirp, and Doppler FFT across chirps to form a range–Doppler (RD) map. An STFT-based spectrogram is also available for single-channel views.
3. **Detect**: Use global thresholding or 2D CA-CFAR to identify detections. Connected-component clustering merges neighboring cells and yields range/velocity centroids.
4. **Classify**: Extract RD patches around detections, normalize them, and classify with a small CNN trained on synthetic signatures (car/pedestrian/drone/clutter).
5. **Visualize**: The Streamlit UI shows raw IQ, range profiles, RD heatmaps with detection overlays, and classification tables. Controls let you toggle CFAR, adjust noise/SNR, and run a quick CNN training pass on synthetic data.

## Notes for demos
- Default parameters keep runtime low; increase `n_chirps`, `n_samples`, or training epochs in the UI for richer outputs.
- The CNN trains on procedurally generated RD patches to stay self-contained; accuracy improves with more epochs and patch count.
- The CFAR guard/train cells are small to suit demo-sized RD maps—tune them for larger grids.

