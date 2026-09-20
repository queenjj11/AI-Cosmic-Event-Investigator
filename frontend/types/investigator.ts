export interface CandidateListItem {
  candidate_id: string;
  ztf_designation: string;
  class: string;
  dataset_role: string;
  ra: number;
  dec: number;
  raw_observations: number;
  clean_observations: number;
  valid_tokens: number;
  baseline_days: number;
  available_filters: string[];
  has_investigation_report: boolean;
}

export interface CharacterizedValue {
  value: number | string | null;
  status: "MEASURED" | "DERIVED" | "PROXY" | "UNAVAILABLE";
  unit?: string | null;
  notes?: string | null;
}

export interface RepresentationSummary {
  object_id: string;
  embedding_dimension: number;
  embedding_norm: number;
  embedding: number[];
  valid_token_count: number;
  padding_fraction: number;
  available_filters: string[];
  time_span_days: number;
  preprocessing_status: string;
  warnings: string[];
  encoder_model: string;
  checkpoint_source: string;
}

export interface AnomalyAssessment {
  representation_available: boolean;
  anomaly_score_status: "NOT_VALIDATED_FOR_REAL_ZTF" | "VALIDATED_SYNTHETIC" | "TRANSFER_DIAGNOSTIC" | string;
  anomaly_score: number | null;
  anomaly_flag: boolean | null;
  detector_source: string;
  uncertainty_or_status: string;
  limitations: string[];
  domain_gap_notes: string;
}

export interface HypothesisItem {
  rank: number;
  name: string;
  supporting_evidence: string[];
  contradictory_evidence: string[];
  confidence_or_status: string;
  probability: number | null;
  provenance_or_source: string;
  justification: string;
  distinguishing_criteria: string;
}

export interface EvidenceItem {
  doc_id: string;
  title: string;
  relevance_score: number;
  snippet: string;
  source_uri: string;
}

export interface ObservationRecommendation {
  priority: number;
  action: string;
  scientific_rationale: string;
  information_gap: string;
  status: string;
  target_band?: string | null;
  urgency: string;
}

export interface InvestigationResult {
  object_id: string;
  metadata: {
    candidate_id?: string;
    claimed_type?: string;
    class?: string;
    ra?: number;
    dec?: number;
    redshift?: number | null;
    ztf_object_id?: string;
    [key: string]: any;
  };
  event_characterization: {
    num_observations: CharacterizedValue;
    num_filters: CharacterizedValue;
    time_baseline_days: CharacterizedValue;
    window_duration_days: CharacterizedValue;
    peak_normalized_flux: CharacterizedValue;
    minimum_normalized_flux: CharacterizedValue;
    median_normalized_flux: CharacterizedValue;
    flux_std: CharacterizedValue;
    median_flux_err: CharacterizedValue;
    variability_amplitude: CharacterizedValue;
    rise_time_proxy_days: CharacterizedValue;
    decline_time_proxy_days: CharacterizedValue;
    rise_decline_asymmetry: CharacterizedValue;
    cadence_median_days: CharacterizedValue;
    cadence_min_days: CharacterizedValue;
    cadence_max_days: CharacterizedValue;
    per_band_counts: Record<string, number>;
    missing_bands: string[];
    valid_token_count: CharacterizedValue;
    padding_fraction: CharacterizedValue;
    classification_disclaimer: string;
    [key: string]: any;
  };
  representation_summary: RepresentationSummary;
  anomaly_assessment: AnomalyAssessment;
  candidate_hypotheses: HypothesisItem[];
  evidence: EvidenceItem[];
  confidence_summary: Record<string, string>;
  recommended_observations: ObservationRecommendation[];
  limitations: string[];
  provenance: {
    source_survey: string;
    candidate_id: string;
    raw_observation_count: number;
    passed_observation_count: number;
    resolved_filepath?: string;
    checkpoint_sha256: string;
    production_encoder_dim: number;
    investigator_pipeline_version: string;
    [key: string]: any;
  };
  execution_time_seconds: number;
}

export interface LightCurvePoint {
  rel_time: number;
  norm_flux: number;
  norm_flux_err: number;
  band: string;
}

export interface SystemHealth {
  status: string;
  service: string;
  model: string;
  device: string;
  checkpoint_sha256: string;
  frozen_benchmark_sha256: string;
  primary_benchmark_objects: number;
  real_ztf_data_used_in_training: boolean;
  anomaly_status_default: string;
  utc_time: string;
}
