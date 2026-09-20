"""Explanation groundedness and scientific reasoning metrics."""

from typing import Any, Dict, List, Set
import numpy as np
from src.data.schema import Hypothesis


def compute_explanation_metrics(generated_hypotheses_list: List[List[Hypothesis]],
                                ground_truth_classes: List[str],
                                valid_source_ids: Set[str]) -> Dict[str, float]:
    """
    Evaluates groundedness and accuracy of RAG-generated scientific hypotheses:
    - groundedness_ratio: % of hypotheses citing verifiable retrieved document sources.
    - top1_hypothesis_accuracy: % of events where rank-1 hypothesis matches ground truth.
    - mrr: Mean Reciprocal Rank of ground truth class in the hypothesis candidate list.
    """
    total_hypotheses = 0
    grounded_hypotheses = 0

    top1_correct = 0
    reciprocal_ranks = []

    for hyps, true_cls in zip(generated_hypotheses_list, ground_truth_classes):
        if not hyps:
            reciprocal_ranks.append(0.0)
            continue

        # Check Groundedness: does each hypothesis cite valid sources?
        for h in hyps:
            total_hypotheses += 1
            has_valid_citation = any(s in valid_source_ids for s in h.evidence_sources) if valid_source_ids else bool(h.evidence_sources)
            if has_valid_citation:
                grounded_hypotheses += 1

        # Check Top-1 Accuracy: does rank 1 match ground truth?
        clean_true = true_cls.lower().replace("_", " ")
        top_name = hyps[0].name.lower()
        if clean_true in top_name or any(w in top_name for w in clean_true.split()):
            top1_correct += 1

        # Check Mean Reciprocal Rank
        found_rank = 0
        for h in hyps:
            h_name = h.name.lower()
            if clean_true in h_name or any(w in h_name for w in clean_true.split()):
                found_rank = h.rank
                break
        reciprocal_ranks.append(1.0 / found_rank if found_rank > 0 else 0.0)

    n_events = len(ground_truth_classes)
    return {
        "groundedness_ratio": float(grounded_hypotheses / total_hypotheses) if total_hypotheses > 0 else 1.0,
        "top1_accuracy": float(top1_correct / n_events) if n_events > 0 else 0.0,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    }
