"""Plotly interactive visualizations for astronomical light curves, cutouts, and metrics."""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.schema import AstronomicalEvent, Hypothesis


BAND_COLORS = {
    "g": "#2ca02c",     # Green
    "r": "#d62728",     # Red
    "i": "#ff7f0e",     # Orange/Infrared
    "V": "#9467bd",     # Purple
    "NIR": "#8c564b"    # Brown/NIR
}


def plot_multiband_lightcurve(event: AstronomicalEvent, max_time: Optional[float] = None) -> go.Figure:
    """Render interactive multi-band light curve with photometric error bars."""
    fig = go.Figure()
    lc = event.lightcurve

    if lc and len(lc.observations) > 0:
        obs_list = lc.observations
        if max_time is not None:
            # Replay mode: filter observations up to max_time
            obs_list = [o for o in obs_list if (o.time - obs_list[0].time) <= max_time]

        bands = sorted(list(set(o.band for o in obs_list)))
        t0 = obs_list[0].time if obs_list else 0.0

        for b in bands:
            b_obs = [o for o in obs_list if o.band == b]
            times = [o.time - t0 for o in b_obs]
            fluxes = [o.flux for o in b_obs]
            errs = [o.flux_err for o in b_obs]

            color = BAND_COLORS.get(b, "#1f77b4")
            fig.add_trace(go.Scatter(
                x=times,
                y=fluxes,
                mode="markers+lines",
                name=f"{b}-band",
                marker=dict(size=8, color=color),
                line=dict(color=color, width=1.5),
                error_y=dict(type="data", array=errs, visible=True, color=color, thickness=1.2)
            ))

    fig.update_layout(
        title=f"Multiband Photometry: {event.object_id}",
        xaxis_title="Time Since Discovery (Days)",
        yaxis_title="Calibrated Flux (counts / microJy)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cutout_channels(event: AstronomicalEvent) -> go.Figure:
    """Render 3-channel astronomical postage stamps: Science, Template, Difference."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Science (New)", "Template (Reference)", "Difference (Subtraction)"])

    if event.image is not None and isinstance(event.image.data, np.ndarray):
        img = event.image.data
        for i, col in enumerate([1, 2, 3]):
            channel_data = img[i] if i < img.shape[0] else img[0]
            fig.add_trace(
                go.Heatmap(z=channel_data, colorscale="Viridis", showscale=False),
                row=1, col=col
            )
    else:
        for col in [1, 2, 3]:
            dummy = np.zeros((64, 64))
            fig.add_trace(go.Heatmap(z=dummy, colorscale="Greys", showscale=False), row=1, col=col)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white"
    )
    return fig


def plot_anomaly_signals(detector_scores: Dict[str, float], threshold: float = 0.65) -> go.Figure:
    """Bar chart decomposing the 3 signals of the anomaly ensemble."""
    signals = ["Autoencoder Recon", "Mahalanobis Dist", "Energy OOD", "Ensemble Score"]
    values = [
        detector_scores.get("autoencoder_norm", 0.0),
        detector_scores.get("mahalanobis_norm", 0.0),
        detector_scores.get("energy_norm", 0.0),
        detector_scores.get("ensemble_score", 0.0)
    ]

    colors = ["#3498db", "#9b59b6", "#e67e22", "#e74c3c" if values[3] >= threshold else "#2ecc71"]

    fig = go.Figure(go.Bar(
        x=signals,
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="auto"
    ))

    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="#c0392b",
                  annotation_text=f"Anomaly Threshold ({threshold})", annotation_position="top right")

    fig.update_layout(
        title="3-Signal Anomaly Ensemble Breakdown",
        yaxis=dict(range=[0.0, 1.05], title="Normalized Score [0, 1]"),
        template="plotly_white",
        height=280,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    return fig


def plot_hypotheses_distribution(hypotheses: List[Hypothesis]) -> go.Figure:
    """Horizontal bar chart showing calibrated probability distribution across hypotheses."""
    if not hypotheses:
        fig = go.Figure()
        fig.update_layout(title="No Active Hypotheses (Normal In-Distribution Event)")
        return fig

    names = [h.name for h in reversed(hypotheses)]
    probs = [h.probability * 100.0 for h in reversed(hypotheses)]

    fig = go.Figure(go.Bar(
        x=probs,
        y=names,
        orientation="h",
        marker_color="#2980b9",
        text=[f"{p:.1f}%" for p in probs],
        textposition="auto"
    ))

    fig.update_layout(
        title="AI Investigator: Ranked Hypothesis Probabilities",
        xaxis=dict(range=[0, 100], title="Calibrated Posterior Probability (%)"),
        template="plotly_white",
        height=250,
        margin=dict(l=150, r=40, t=40, b=40)
    )
    return fig
