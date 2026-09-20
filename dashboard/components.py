"""Streamlit UI components for rendering hypotheses, evidence, and recommendations."""

import streamlit as st
from typing import Any, Dict, List
from src.data.schema import Hypothesis, RecommendedObservation


def render_alert_banner(is_anomaly: bool, anomaly_score: float, threshold: float = 0.65):
    """Render top alert status banner."""
    if is_anomaly:
        st.error(
            f"🚨 **ASTRONOMICAL ANOMALY DETECTED** | Ensemble Score: **{anomaly_score:.4f}** "
            f"(Threshold: {threshold}) — Out-Of-Distribution Event Flagged for Scientific Investigation"
        )
    else:
        st.success(
            f"✅ **IN-DISTRIBUTION TRANSIENT** | Ensemble Anomaly Score: **{anomaly_score:.4f}** "
            f"(Threshold: {threshold}) — Matches Standard Astrophysical Classification Models"
        )


def render_hypothesis_card(hyp: Hypothesis):
    """Render a clean, scientific card for a single hypothesis."""
    badge_color = "#2ecc71" if hyp.rank == 1 else "#3498db"
    st.markdown(f"""
    <div style="border: 1px solid #e0e0e0; border-left: 5px solid {badge_color}; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; background-color: #fafafa;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #2c3e50;">#{hyp.rank}: {hyp.name}</h4>
            <span style="background-color: {badge_color}; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.9em;">
                {hyp.probability * 100:.1f}% Calibrated Prob
            </span>
        </div>
        <p style="margin-top: 8px; margin-bottom: 6px; color: #34495e;"><b>Astrophysical Rationale:</b> {hyp.justification}</p>
        <p style="margin-top: 0; margin-bottom: 6px; color: #7f8c8d; font-size: 0.9em;"><b>Distinguishing Diagnostic Test:</b> {hyp.distinguishing_criteria}</p>
        <div style="font-size: 0.82em; color: #95a5a6;">
            <b>Cited Evidence Sources:</b> <code>{'</code>, <code>'.join(hyp.evidence_sources)}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(rec: RecommendedObservation):
    """Render the active observation recommendation decision card."""
    st.markdown(f"""
    <div style="border: 2px solid #f39c12; border-radius: 8px; padding: 16px 20px; background-color: #fef9e7; margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: #d35400;">⭐ Recommended Next Follow-Up Observation</h3>
            <span style="background-color: #e67e22; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;">
                Cost: {rec.cost:.1f} hrs
            </span>
        </div>
        <h4 style="margin-top: 10px; margin-bottom: 6px; color: #2c3e50;">{rec.name}</h4>
        <p style="margin: 4px 0; color: #444;"><b>Observation Band:</b> <code>{rec.band}</code> | <b>Timing:</b> Within {rec.delay_hours:.0f} hours</p>
        <p style="margin: 4px 0; color: #27ae60;"><b>Expected Shannon Information Gain:</b> <b>{rec.expected_information_gain:.4f} nats</b> (Entropy Reduction)</p>
        <p style="margin-top: 8px; margin-bottom: 0; color: #555; font-size: 0.95em;"><b>Scientific Decision Rationale:</b> {rec.rationale}</p>
    </div>
    """, unsafe_allow_html=True)
