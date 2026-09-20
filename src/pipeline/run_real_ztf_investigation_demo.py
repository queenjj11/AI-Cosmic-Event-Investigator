"""Demonstration pipeline executing end-to-end scientific investigations on 5 real ZTF candidates.

IMPORTANT CONSTRAINTS:
- Demonstration only; no benchmark metric optimization.
- No model parameter updates.
- No threshold tuning.
- Preserves all frozen datasets and checkpoints.
"""

import os
import sys
import time
from typing import List, Dict, Any

from src.investigator.investigation_service import RealZTFInvestigationService


DEMO_CANDIDATES = [
    "CAND_SNIa_002",
    "CAND_SLSN_001",
    "CAND_TDE_001",
    "CAND_CV_001",
    "CAND_FieldStar_001"
]


def run_investigation_demonstration(output_dir: str = "reports/investigations") -> List[Dict[str, Any]]:
    """Execute investigation on the 5 representative demonstration candidates."""
    print("=" * 78)
    print("🌌 ACEI REAL-ZTF INVESTIGATION DEMONSTRATION")
    print(f"Targeting {len(DEMO_CANDIDATES)} diverse astronomical classes across the frozen benchmark")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)
    service = RealZTFInvestigationService()

    summaries = []
    total_start = time.time()

    for idx, candidate_id in enumerate(DEMO_CANDIDATES, 1):
        print(f"\n[{idx}/{len(DEMO_CANDIDATES)}] Investigating: {candidate_id} ...")
        result = service.investigate_candidate(candidate_id=candidate_id)

        # Save JSON and Markdown
        json_path = os.path.join(output_dir, f"{candidate_id}_investigation.json")
        md_path = os.path.join(output_dir, f"{candidate_id}_investigation.md")
        result.save_json(json_path)
        result.save_markdown(md_path)

        char = result.event_characterization
        rep = result.representation_summary
        anom = result.anomaly_assessment
        top_hyp = result.candidate_hypotheses[0] if result.candidate_hypotheses else None

        summary = {
            "candidate_id": candidate_id,
            "claimed_type": result.metadata.get("claimed_type", "Unknown"),
            "raw_obs": result.provenance.get("raw_observation_count"),
            "clean_obs": char.num_observations.value,
            "valid_tokens": rep.valid_token_count,
            "embedding_norm": round(rep.embedding_norm, 4),
            "anomaly_status": anom.anomaly_score_status,
            "top_hypothesis": top_hyp.name if top_hyp else "None",
            "top_probability": top_hyp.probability if top_hyp else None,
            "recommendations_count": len(result.recommended_observations),
            "exec_time_sec": round(result.execution_time_seconds, 4)
        }
        summaries.append(summary)

        print(f"    Class           : {summary['claimed_type']}")
        print(f"    Photometry      : {summary['clean_obs']} clean obs ({summary['valid_tokens']} tokens, norm={summary['embedding_norm']:.4f})")
        print(f"    Anomaly Status  : {summary['anomaly_status']}")
        print(f"    Top Hypothesis  : {summary['top_hypothesis']} (P={summary['top_probability']})")
        print(f"    Recommendations : {summary['recommendations_count']} actions formulated")
        print(f"    Saved Artifacts : {json_path}")
        print(f"                      {md_path}")

    total_time = time.time() - total_start
    print("\n" + "=" * 78)
    print(f"✅ DEMONSTRATION COMPLETE: {len(summaries)} investigations generated in {total_time:.2f} s")
    print("=" * 78)

    # Print summary table
    print(f"{'Candidate':<20} | {'Class':<18} | {'Tokens':<6} | {'Norm':<8} | {'Anomaly Status':<26} | {'Top Hypothesis'}")
    print("-" * 110)
    for s in summaries:
        print(f"{s['candidate_id']:<20} | {s['claimed_type']:<18} | {s['valid_tokens']:<6} | {s['embedding_norm']:<8.4f} | {s['anomaly_status']:<26} | {s['top_hypothesis']}")
    print("-" * 110)

    return summaries


if __name__ == "__main__":
    run_investigation_demonstration()
