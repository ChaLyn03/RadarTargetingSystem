import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from radar_system.detection.clustering import Detection
from radar_system.models import CLASSES
from radar_system.models.training.train import TrainingConfig, train_quick_classifier
from radar_system.sim.fmcw_simulator import SimulationConfig, Target, generate_random_targets
from radar_system.pipeline.pipeline import PipelineConfig, run_pipeline
from radar_system.config.loader import load_configs

st.set_page_config(page_title="Radar Target Viewer & Classifier", layout="wide")


def get_classifier_state():
    if "classifier" not in st.session_state:
        st.session_state.classifier = None
    return st.session_state.classifier


def train_classifier_ui(device: str, training_cfg: TrainingConfig):
    st.markdown("### CNN training")
    epochs = st.slider("Epochs", 1, 10, int(training_cfg.epochs))
    samples = st.slider("Synthetic samples", 64, 1024, int(training_cfg.num_samples), step=64)
    patch_options = [16, 32, 48]
    default_idx = patch_options.index(training_cfg.patch_size) if training_cfg.patch_size in patch_options else 1
    patch_size = st.selectbox("Patch size", patch_options, index=default_idx)
    if st.button("Train quick CNN"):
        with st.spinner("Training CNN on synthetic RD patches..."):
            cfg = TrainingConfig(device=device, epochs=epochs, num_samples=samples, patch_size=patch_size)
            clf = train_quick_classifier(cfg)
        st.session_state.classifier = clf
        st.success("Training complete")


def plot_range_profile(range_bins: np.ndarray, profile: np.ndarray):
    fig = px.line(x=range_bins, y=profile, labels={"x": "Range (m)", "y": "Amplitude"})
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_rd_map(power_map: np.ndarray, ranges: np.ndarray, doppler: np.ndarray, detections: list[Detection], labels: list[str] | None = None):
    log_power = 10 * np.log10(np.abs(power_map) + 1e-9)
    fig = px.imshow(
        log_power,
        x=doppler,
        y=ranges,
        color_continuous_scale="Viridis",
        origin="lower",
        aspect="auto",
        labels={"x": "Doppler (Hz)", "y": "Range (m)", "color": "dB"},
    )
    if detections:
        ys = [ranges[int(det.range_idx)] for det in detections]
        xs = [doppler[int(det.doppler_idx)] for det in detections]
        text = labels if labels else None
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if labels else "markers",
                marker=dict(color="red", size=8, symbol="x"),
                text=text,
                textposition="top center",
                name="Detections",
            )
        )
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_iq(iq: np.ndarray, sample_rate: float):
    samples = np.arange(iq.shape[-1]) / sample_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=iq.real, mode="lines", name="I"))
    fig.add_trace(go.Scatter(x=samples, y=iq.imag, mode="lines", name="Q"))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig, use_container_width=True)


def run_dashboard():
    # Load configs from YAML files in ./config
    sim_cfg, pipeline_cfg, training_cfg = load_configs("config")

    st.title("Radar Target Viewer & Classifier (config-driven)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sidebar: allow overrides for a few key simulation values
    st.sidebar.markdown("## Simulation controls (configurable)")
    n_targets = st.sidebar.slider("Targets", 1, 4, 2)
    noise_db = st.sidebar.slider("Noise power (dB)", -60.0, 0.0, float(sim_cfg.noise_power_db), step=1.0)
    n_chirps = st.sidebar.slider("Chirps", 4, 128, int(sim_cfg.n_chirps), step=4)
    n_samples = st.sidebar.slider("Samples per chirp", 64, 2048, int(sim_cfg.n_samples), step=64)

    # Pipeline settings
    use_cfar = st.sidebar.checkbox("Use CA-CFAR", value=bool(pipeline_cfg.use_cfar))
    guard = st.sidebar.slider("Guard cells", 0, 8, int(pipeline_cfg.guard_cells[0]))
    training_cells = st.sidebar.slider("Training cells", 1, 12, int(pipeline_cfg.training_cells[0]))
    threshold_k = st.sidebar.slider("Global threshold k·σ", 1.0, 8.0, float(pipeline_cfg.threshold_k), step=0.5)

    # Training UI
    train_classifier_ui(device, training_cfg)

    # Build simulation & pipeline configs (apply possible overrides)
    sim_cfg.n_chirps = int(n_chirps)
    sim_cfg.n_samples = int(n_samples)
    sim_cfg.noise_power_db = float(noise_db)

    pipeline_cfg.use_cfar = bool(use_cfar)
    pipeline_cfg.guard_cells = (int(guard), int(guard))
    pipeline_cfg.training_cells = (int(training_cells), int(training_cells))
    pipeline_cfg.threshold_k = float(threshold_k)

    targets = generate_random_targets(n_targets, sim_cfg)

    if st.button("Simulate & run pipeline"):
        classifier = get_classifier_state()
        with st.spinner("Running radar pipeline..."):
            output = run_pipeline(targets, pipeline_cfg, classifier=classifier, device=device)

        st.subheader("Raw IQ (first chirp)")
        plot_iq(output.sim.iq[0], sim_cfg.sample_rate)

        st.subheader("Range profile (mean across chirps)")
        range_profile = np.mean(np.abs(output.range_fft), axis=0)
        plot_range_profile(output.sim.ranges_m, range_profile)

        st.subheader("Range–Doppler map")
        labels = None
        if output.classification:
            labels = [f"{c['label']} ({c['confidence']:.2f})" for c in output.classification]
        plot_rd_map(output.power_map, output.sim.ranges_m, output.sim.doppler_hz, output.detections, labels)

        if output.classification:
            st.subheader("Classification summary")
            rows = []
            for det, cls in zip(output.detections, output.classification):
                rows.append(
                    {
                        "Range (m)": output.sim.ranges_m[int(det.range_idx)],
                        "Doppler (Hz)": output.sim.doppler_hz[int(det.doppler_idx)],
                        "Class": cls["label"],
                        "Confidence": cls["confidence"],
                    }
                )
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("Train the CNN to see classification labels over detections.")


if __name__ == "__main__":
    run_dashboard()
