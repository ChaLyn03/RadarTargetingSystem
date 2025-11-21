import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.detection.clustering import Detection
from src.models.dataset import CLASSES
from src.models.training import TrainingConfig, train_quick_classifier
from src.sim.fmcw import SimulationConfig, Target, generate_random_targets
from src.ui.pipeline import PipelineConfig, run_pipeline

st.set_page_config(page_title="Radar Target Viewer & Classifier", layout="wide")


def get_classifier_state():
    if "classifier" not in st.session_state:
        st.session_state.classifier = None
    return st.session_state.classifier


def train_classifier_ui(device: str):
    st.markdown("### CNN training")
    epochs = st.slider("Epochs", 1, 5, 2)
    samples = st.slider("Synthetic samples", 64, 512, 256, step=64)
    patch_size = st.selectbox("Patch size", [16, 32, 48], index=1)
    if st.button("Train quick CNN"):
        with st.spinner("Training CNN on synthetic RD patches..."):
            clf = train_quick_classifier(TrainingConfig(device=device, epochs=epochs, num_samples=samples, patch_size=patch_size))
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
    st.title("Radar Target Viewer & Classifier")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown("## Simulation controls")
    n_targets = st.sidebar.slider("Targets", 1, 4, 2)
    noise_db = st.sidebar.slider("Noise power (dB)", -40.0, 0.0, -20.0, step=1.0)
    n_chirps = st.sidebar.slider("Chirps", 8, 64, 32, step=8)
    n_samples = st.sidebar.slider("Samples per chirp", 128, 1024, 512, step=64)
    use_cfar = st.sidebar.checkbox("Use CA-CFAR", value=True)
    guard = st.sidebar.slider("Guard cells", 0, 4, 1)
    training = st.sidebar.slider("Training cells", 1, 6, 4)
    threshold_k = st.sidebar.slider("Global threshold k·σ", 1.0, 6.0, 3.0, step=0.5)

    train_classifier_ui(device)

    targets = generate_random_targets(n_targets, SimulationConfig(n_chirps=n_chirps, n_samples=n_samples, noise_power_db=noise_db))

    if st.button("Simulate & run pipeline"):
        sim_config = SimulationConfig(n_chirps=n_chirps, n_samples=n_samples, noise_power_db=noise_db)
        pipeline_config = PipelineConfig(
            sim_config=sim_config,
            use_cfar=use_cfar,
            threshold_k=threshold_k,
            guard_cells=(guard, guard),
            training_cells=(training, training),
        )
        classifier = get_classifier_state()
        with st.spinner("Running radar pipeline..."):
            output = run_pipeline(targets, pipeline_config, classifier=classifier, device=device)

        st.subheader("Raw IQ (first chirp)")
        plot_iq(output.sim.iq[0], sim_config.sample_rate)

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

