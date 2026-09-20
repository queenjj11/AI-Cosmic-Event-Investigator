"""Command-line interface to run the Real-ZTF light-curve-first investigator on an astronomical candidate."""

import argparse
import os
import sys
import json

from src.investigator.investigation_service import RealZTFInvestigationService


def parse_args():
    parser = argparse.ArgumentParser(
        description="ACEI Real-ZTF Light-Curve-First Scientific Investigation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Target candidate ID (e.g. CAND_SNIa_002, CAND_SLSN_001, CAND_TDE_001)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional explicit path to raw photometric CSV (defaults to data/real_ztf_benchmark/raw/<candidate>/raw_irsa.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/investigations",
        help="Directory to save JSON and Markdown investigation reports"
    )
    parser.add_argument(
        "--format",
        choices=["both", "json", "markdown"],
        default="both",
        help="Output report formats to persist"
    )
    return parser.parse_args()


def print_terminal_summary(result):
    """Render a clean, formatted terminal investigation summary."""
    meta = result.metadata
    char = result.event_characterization
    rep = result.representation_summary
    anom = result.anomaly_assessment

    print("=" * 72)
    print(f"🌌 ACEI SCIENTIFIC INVESTIGATION: {result.object_id}")
    print("=" * 72)
    print(f"Target Object      : {result.object_id}")
    print(f"Benchmark Class    : {meta.get('claimed_type', meta.get('class', 'Unknown'))}")
    print(f"Coordinates        : RA={meta.get('ra', 0.0):.5f}°, Dec={meta.get('dec', 0.0):.5f}°")
    print(f"Raw Observations   : {result.provenance.get('raw_observation_count', 'N/A')}")
    print(f"Clean Observations : {char.num_observations.value} ({char.num_filters.value} filters: {list(char.per_band_counts.keys())})")
    print(f"Time Baseline      : {char.time_baseline_days.value} days (Outburst window: {char.window_duration_days.value} days)")
    print(f"Execution Time     : {result.execution_time_seconds:.3f} s")
    print("-" * 72)
    print(f"REPRESENTATION (Production LightCurveEncoder):")
    print(f"  Embedding Shape  : ({rep.embedding_dimension},) L2-Norm = {rep.embedding_norm:.4f}")
    print(f"  Valid Tokens     : {rep.valid_token_count} / 50 (Padding: {rep.padding_fraction*100:.1f}%)")
    print(f"  Preprocessing    : {rep.preprocessing_status}")
    print("-" * 72)
    print(f"SCIENTIFIC ANOMALY ASSESSMENT:")
    print(f"  Evaluation Status: {anom.anomaly_score_status}")
    print(f"  Detector Source  : {anom.detector_source}")
    print(f"  Scientific Note  : {anom.domain_gap_notes[:110]}...")
    print("-" * 72)
    print("CANDIDATE TRANSIENT HYPOTHESES:")
    for h in result.candidate_hypotheses:
        prob_str = f" [P = {h.probability:.2f}]" if h.probability is not None else ""
        print(f"  #{h.rank}: {h.name}{prob_str} ({h.confidence_or_status})")
        print(f"      Justification: {h.justification}")
    print("-" * 72)
    print("RECOMMENDED OBSERVATIONS:")
    for r in result.recommended_observations[:3]:
        print(f"  Priority {r.priority} ({r.urgency}): {r.action}")
        print(f"      Rationale: {r.scientific_rationale}")
    print("=" * 72)


def main():
    args = parse_args()

    service = RealZTFInvestigationService()

    try:
        result = service.investigate_candidate(
            candidate_id=args.candidate,
            raw_csv_path=args.file
        )
    except Exception as e:
        print(f"[ERROR] Failed to investigate candidate '{args.candidate}': {e}", file=sys.stderr)
        sys.exit(1)

    # Print terminal output
    print_terminal_summary(result)

    # Save output artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.candidate}_investigation.json")
    md_path = os.path.join(args.output_dir, f"{args.candidate}_investigation.md")

    if args.format in ["both", "json"]:
        result.save_json(json_path)
        print(f"Saved machine-readable JSON: {json_path}")

    if args.format in ["both", "markdown"]:
        result.save_markdown(md_path)
        print(f"Saved human-readable report: {md_path}")


if __name__ == "__main__":
    main()
