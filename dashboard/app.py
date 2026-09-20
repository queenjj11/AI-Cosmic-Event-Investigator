from typing import Set
"""Streamlit Web Dashboard for the AI Cosmic Event Investigator (ACEI)."""

import os
import sys
import numpy as np
import streamlit as st

# Add project root to python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.schema import AstronomicalEvent, CutoutImage
from src.data.lightcurve_loader import LightCurveLoader
from src.data.image_loader import ImageLoader
from src.pipeline.acei_pipeline import ACEIPipeline
from dashboard.plots import (
    plot_multiband_lightcurve,
    plot_cutout_channels,
    plot_anomaly_signals,
    plot_hypotheses_distribution
)
from dashboard.components import (
    render_alert_banner,
    render_hypothesis_card,
    render_recommendation_card
)

# Page layout
st.set_page_config(
    page_title="AI Cosmic Event Investigator (ACEI)",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_initialized_pipeline():
    """Cache the initialized pipeline instance for responsive dashboard performance."""
    pipeline = ACEIPipeline(anomaly_threshold=0.65)
    pipeline.initialize_system(
        papers_dir=os.path.join(ROOT_DIR, "knowledge_base/papers"),
        catalogs_dir=os.path.join(ROOT_DIR, "knowledge_base/catalogs")
    )
    return pipeline


def load_sample_event(event_type: str) -> AstronomicalEvent:
    """Load curated astronomical events for interactive exploration."""
    is_anomaly = event_type in ["LRN", "SLSN", "TDE"]

    lc = LightCurveLoader.generate_synthetic_lightcurve(event_type=event_type, is_anomaly=is_anomaly)
    img_data = ImageLoader.generate_synthetic_cutout(is_anomaly=is_anomaly, event_type=event_type)

    coords_map = {
        "LRN": (210.1234, -12.3456),
        "SLSN": (165.7890, 42.1122),
        "TDE": (148.5678, 15.9876),
        "SN_Ia": (184.2345, 29.8765),
        "SN_II": (195.4321, -5.6789),
        "Stellar_Flare": (88.1234, 12.4567),
        "Variable_Star": (270.9876, -28.5432)
    }
    ra, dec = coords_map.get(event_type, (180.0, 0.0))

    return AstronomicalEvent(
        object_id=f"ZTF24_{event_type}_sample",
        ra=ra, dec=dec,
        lightcurve=lc,
        image=CutoutImage(data=img_data),
        true_label=event_type,
        is_anomaly=is_anomaly,
        metadata={"survey": "ZTF Public Alert Stream", "field": 712}
    )


# --- MAIN APPLICATION ---

st.title("🌌 AI Cosmic Event Investigator (ACEI)")
st.caption("Multimodal Anomaly Detection & Autonomous Scientific Reasoning for Astronomical Transients")

pipeline = get_initialized_pipeline()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔭 Target Event Selection")

event_choices = {
    "LRN — Luminous Red Nova (Held-out Merger Anomaly)": "LRN",
    "SLSN — Superluminous Supernova (Held-out Magnetar Anomaly)": "SLSN",
    "TDE — Tidal Disruption Event (Nuclear Black Hole Anomaly)": "TDE",
    "SN Ia — Type Ia Supernova (Normal Known Class)": "SN_Ia",
    "SN II — Core-Collapse Supernova (Normal Known Class)": "SN_II",
    "Stellar Flare — M-Dwarf Magnetic Reconnection (Normal Known Class)": "Stellar_Flare",
    "Variable Star — Periodic Pulsator (Normal Known Class)": "Variable_Star"
}

selected_label = st.sidebar.selectbox("Select Target Transient:", list(event_choices.keys()))
selected_type = event_choices[selected_label]
event = load_sample_event(selected_type)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Triage & Replay Settings")

threshold = st.sidebar.slider("Anomaly Detection Threshold:", min_value=0.40, max_value=0.90, value=0.65, step=0.05)
pipeline.anomaly_ensemble.threshold = threshold

enable_replay = st.sidebar.checkbox("Enable Temporal Replay Mode", value=False)
max_days = None
if enable_replay:
    max_days = st.sidebar.slider("Replay Light Curve Up To Day:", min_value=2.0, max_value=45.0, value=15.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.info("""
**About ACEI:**
- **Observe:** Heterogeneous ZTF/ASAS-SN alerts
- **Detect:** 3-Signal calibrated anomaly ensemble
- **Hypothesize:** RAG over astrophysics literature
- **Recommend:** Active observation scheduling
""")

# --- RUN PIPELINE ---
result = pipeline.investigate_event(event)

# --- TOP BANNER ---
render_alert_banner(result.is_anomaly, result.anomaly_score, threshold)

# Metadata Cards
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Object ID", event.object_id)
col_m2.metric("Coordinates (RA, Dec)", f"{event.ra:.2f}°, {event.dec:.2f}°")
col_m3.metric("Observations Count", len(event.lightcurve.observations))
col_m4.metric("Pipeline Latency", f"{result.execution_time_seconds:.2f}s")

st.markdown("---")

# --- MIDDLE SECTION: MULTIMODAL DATA & ANOMALY SIGNALS ---
col_left, col_right = st.columns([1.2, 1.0])

with col_left:
    st.subheader("📈 Multiband Light Curve (Photometry)")
    st.plotly_chart(plot_multiband_lightcurve(event, max_time=max_days), use_container_width=True)

    st.subheader("🖼️ Astronomical Postage Stamps (Cutouts)")
    st.plotly_chart(plot_cutout_channels(event), use_container_width=True)

with col_right:
    st.subheader("🎯 3-Signal Anomaly Ensemble Breakdown")
    st.plotly_chart(plot_anomaly_signals(result.detector_scores, threshold=threshold), use_container_width=True)

    st.subheader("🏷️ Closed-Set Classifier Baseline")
    if result.classifier_prediction:
        st.write(f"**Predicted Known Class:** `{result.classifier_prediction}` "
                 f"(Confidence: **{result.classifier_confidence * 100:.1f}%**)")
        st.progress(min(1.0, max(0.0, result.classifier_confidence)))

        with st.expander("View Full Class Probability Vector"):
            for cls_name, p in result.class_probabilities.items():
                st.write(f"- **{cls_name}**: {p * 100:.2f}%")

# --- BOTTOM SECTION: SCIENTIFIC INVESTIGATOR & RECOMMENDER ---
if result.is_anomaly:
    st.markdown("---")
    st.header("🧠 AI Investigator: Literature-Grounded Hypotheses")
    st.caption("Hypotheses generated by RAG reasoning over peer-reviewed astrophysics papers and taxonomy catalogs.")

    col_h1, col_h2 = st.columns([1.0, 1.2])

    with col_h1:
        st.plotly_chart(plot_hypotheses_distribution(result.hypotheses), use_container_width=True)

    with col_h2:
        for hyp in result.hypotheses:
            render_hypothesis_card(hyp)

    st.markdown("---")
    st.header("🔭 Bayesian Active Observation Recommender")
    st.caption("Optimal follow-up observation maximizing expected Shannon Information Gain (entropy reduction) under cost constraints.")

    if result.recommended_action:
        render_recommendation_card(result.recommended_action)

        with st.expander("Compare All Candidate Follow-Up Actions"):
            all_actions = pipeline.recommender.rank_all_actions(event, result.hypotheses)
            st.table(all_actions)
else:
    st.markdown("---")
    st.info("ℹ️ **No anomaly detected.** Event matches known transient profiles. Automated alert stream monitoring active.")
