"""Deterministic, serializable schemas for the ACEI Real-ZTF Investigator pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
import json
import os


@dataclass
class CharacterizedValue:
    """A single scientific quantity with explicit provenance and measurement status."""
    value: Optional[Union[float, int, str]]
    status: str  # "MEASURED", "DERIVED", "PROXY", "UNAVAILABLE"
    unit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "unit": self.unit,
            "notes": self.notes
        }


@dataclass
class EventCharacterization:
    """Non-learned, scientifically interpretable observational features."""
    num_observations: CharacterizedValue
    num_filters: CharacterizedValue
    time_baseline_days: CharacterizedValue
    window_duration_days: CharacterizedValue
    peak_normalized_flux: CharacterizedValue
    minimum_normalized_flux: CharacterizedValue
    median_normalized_flux: CharacterizedValue
    flux_std: CharacterizedValue
    median_flux_err: CharacterizedValue
    variability_amplitude: CharacterizedValue
    rise_time_proxy_days: CharacterizedValue
    decline_time_proxy_days: CharacterizedValue
    rise_decline_asymmetry: CharacterizedValue
    cadence_median_days: CharacterizedValue
    cadence_min_days: CharacterizedValue
    cadence_max_days: CharacterizedValue
    per_band_counts: Dict[str, int]
    missing_bands: List[str]
    valid_token_count: CharacterizedValue
    padding_fraction: CharacterizedValue
    classification_disclaimer: str = "No astrophysical classification claimed from non-learned features alone."

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, CharacterizedValue):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d


@dataclass
class RepresentationSummary:
    """Light-curve representation extracted by the frozen production encoder."""
    object_id: str
    embedding_dimension: int
    embedding_norm: float
    embedding: List[float]
    valid_token_count: int
    padding_fraction: float
    available_filters: List[str]
    time_span_days: float
    preprocessing_status: str
    warnings: List[str] = field(default_factory=list)
    encoder_model: str = "Production LightCurveEncoder (128-D Transformer)"
    checkpoint_source: str = "models/checkpoints/acei_multimodal_production.pt"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyAssessment:
    """Conservative scientific anomaly assessment acknowledging domain boundaries."""
    representation_available: bool
    anomaly_score_status: str  # "NOT_VALIDATED_FOR_REAL_ZTF", "VALIDATED_SYNTHETIC", "TRANSFER_DIAGNOSTIC"
    anomaly_score: Optional[float] = None
    anomaly_flag: Optional[bool] = None
    detector_source: str = "ACEI Multimodal Anomaly Ensemble"
    uncertainty_or_status: str = "NOT_VALIDATED_FOR_REAL_ZTF"
    limitations: List[str] = field(default_factory=list)
    domain_gap_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisItem:
    """A scientific hypothesis grounded in observational data and literature evidence."""
    rank: int
    name: str
    supporting_evidence: List[str]
    contradictory_evidence: List[str]
    confidence_or_status: str
    probability: Optional[float] = None
    provenance_or_source: str = "ACEI Domain Grounded Reasoning Engine"
    justification: str = ""
    distinguishing_criteria: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    """Astrophysical literature passage or catalog excerpt supporting reasoning."""
    doc_id: str
    title: str
    relevance_score: float
    snippet: str
    source_uri: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationRecommendation:
    """Prioritized telescope follow-up action to resolve hypothesis ambiguity."""
    priority: int
    action: str
    scientific_rationale: str
    information_gap: str
    status: str = "RECOMMENDED"
    target_band: Optional[str] = None
    urgency: str = "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationResult:
    """Complete, end-to-end scientific investigation output for an astronomical object."""
    object_id: str
    metadata: Dict[str, Any]
    event_characterization: EventCharacterization
    representation_summary: RepresentationSummary
    anomaly_assessment: AnomalyAssessment
    candidate_hypotheses: List[HypothesisItem]
    evidence: List[EvidenceItem]
    confidence_summary: Dict[str, Any]
    recommended_observations: List[ObservationRecommendation]
    limitations: List[str]
    provenance: Dict[str, Any]
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "metadata": self.metadata,
            "event_characterization": self.event_characterization.to_dict(),
            "representation_summary": self.representation_summary.to_dict(),
            "anomaly_assessment": self.anomaly_assessment.to_dict(),
            "candidate_hypotheses": [h.to_dict() for h in self.candidate_hypotheses],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence_summary": self.confidence_summary,
            "recommended_observations": [r.to_dict() for r in self.recommended_observations],
            "limitations": self.limitations,
            "provenance": self.provenance,
            "execution_time_seconds": round(self.execution_time_seconds, 4)
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def to_markdown(self) -> str:
        """Render a readable scientific investigation report."""
        lines = [
            f"# ACEI Investigation Report: {self.object_id}",
            "",
            "## 1. Object Metadata & Provenance",
            f"- **Target Object**: `{self.object_id}`",
            f"- **Benchmark Class**: `{self.metadata.get('claimed_type', self.metadata.get('class', 'Unknown'))}`",
            f"- **Coordinates**: RA = {self.metadata.get('ra', 'N/A')} deg, Dec = {self.metadata.get('dec', 'N/A')} deg",
            f"- **Survey**: {self.provenance.get('source_survey', 'ZTF Public Data Release / IRSA')}",
            f"- **Total Raw Observations**: {self.provenance.get('raw_observation_count', 'N/A')}",
            f"- **Pipeline Execution Time**: {self.execution_time_seconds:.3f} s",
            "",
            "## 2. Cleaned Light-Curve Representation",
            f"- **Encoder**: {self.representation_summary.encoder_model}",
            f"- **Embedding Dimension**: {self.representation_summary.embedding_dimension}",
            f"- **Embedding L2-Norm**: {self.representation_summary.embedding_norm:.4f}",
            f"- **Valid Sequence Tokens**: {self.representation_summary.valid_token_count} / 50",
            f"- **Padding Fraction**: {self.representation_summary.padding_fraction * 100:.1f}%",
            f"- **Active Passbands**: {', '.join(self.representation_summary.available_filters) if self.representation_summary.available_filters else 'None'}",
            f"- **Window Time Span**: {self.representation_summary.time_span_days:.2f} days",
            f"- **Preprocessing Status**: `{self.representation_summary.preprocessing_status}`",
            "",
            "## 3. Non-Learned Event Characterization",
            f"> *Note: {self.event_characterization.classification_disclaimer}*",
            "",
            "| Feature | Value | Status | Unit | Notes |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        # Table rows for characterization
        for feat_name, char_val in self.event_characterization.to_dict().items():
            if isinstance(char_val, dict) and "status" in char_val:
                val = char_val.get("value")
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                unit = char_val.get("unit") or "-"
                notes = char_val.get("notes") or ""
                lines.append(f"| `{feat_name}` | {val_str} | **{char_val['status']}** | {unit} | {notes} |")

        lines.extend([
            "",
            f"- **Per-Band Observation Counts**: {self.event_characterization.per_band_counts}",
            f"- **Missing Photometric Bands**: {self.event_characterization.missing_bands}",
            "",
            "## 4. Scientific Anomaly Assessment",
            f"- **Evaluation Status**: `{self.anomaly_assessment.anomaly_score_status}`",
            f"- **Anomaly Score**: {self.anomaly_assessment.anomaly_score if self.anomaly_assessment.anomaly_score is not None else 'N/A (Unvalidated)'}",
            f"- **Anomaly Trigger Flag**: {self.anomaly_assessment.anomaly_flag if self.anomaly_assessment.anomaly_flag is not None else 'N/A'}",
            f"- **Detector Architecture**: {self.anomaly_assessment.detector_source}",
            f"- **Domain Status**: {self.anomaly_assessment.uncertainty_or_status}",
            "",
            "> [!WARNING]",
            f"> **Scientific Boundary**: {self.anomaly_assessment.domain_gap_notes}",
            "",
            "## 5. Candidate Transient Hypotheses",
            ""
        ])

        if not self.candidate_hypotheses:
            lines.append("*No hypotheses generated.*")
        else:
            for h in self.candidate_hypotheses:
                prob_str = f" (Calibrated Prob: {h.probability:.3f})" if h.probability is not None else ""
                lines.extend([
                    f"### Rank {h.rank}: {h.name}{prob_str}",
                    f"- **Status**: `{h.confidence_or_status}`",
                    f"- **Justification**: {h.justification}",
                    f"- **Supporting Evidence**: {', '.join(h.supporting_evidence) if h.supporting_evidence else 'None cited'}",
                    f"- **Contradictory / Missing Evidence**: {', '.join(h.contradictory_evidence) if h.contradictory_evidence else 'None noted'}",
                    f"- **Distinguishing Criteria**: {h.distinguishing_criteria}",
                    ""
                ])

        lines.extend([
            "## 6. Literature & Knowledge Base Evidence",
            ""
        ])

        if not self.evidence:
            lines.append("*No direct literature passages retrieved.*")
        else:
            for ev in self.evidence:
                lines.extend([
                    f"#### [{ev.doc_id}] {ev.title} *(Relevance: {ev.relevance_score:.3f})*",
                    f"> {ev.snippet}",
                    ""
                ])

        lines.extend([
            "## 7. Recommended Follow-Up Observations",
            ""
        ])

        if not self.recommended_observations:
            lines.append("*No follow-up observations recommended.*")
        else:
            for rec in self.recommended_observations:
                lines.extend([
                    f"- **Priority {rec.priority} ({rec.urgency})**: {rec.action}",
                    f"  - *Information Gap*: {rec.information_gap}",
                    f"  - *Scientific Rationale*: {rec.scientific_rationale}",
                    f"  - *Target Band*: `{rec.target_band}`"
                ])

        lines.extend([
            "",
            "## 8. Pipeline Limitations & Methodological Constraints",
            ""
        ])

        for lim in self.limitations:
            lines.append(f"- {lim}")

        lines.append("")
        return "\n".join(lines)

    def save_markdown(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
