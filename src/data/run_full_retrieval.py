"""Full Real-ZTF Photometric Retrieval & Quality Classification Pipeline.

Retrieves, associates, pre-processes, and classifies the full 180-candidate active benchmark
population from NASA/IPAC IRSA public light curves without running any neural models.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import csv
import io
import json
import time
import concurrent.futures
import numpy as np
import pandas as pd

from src.data.ztf_object_loader import ZTFObjectLoader, AssociatedZTFEvent
from src.data.real_ztf_preprocessing import RealZTFPreprocessor, PreprocessedZTFEvent

CHECKPOINT_PATH = "models/checkpoints/acei_multimodal_production.pt"
EXPECTED_CHECKPOINT_SHA256 = "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"


def get_active_candidates(registry_path: str = "data/real_ztf_benchmark/candidate_registry.csv") -> pd.DataFrame:
    """
    Load all active, verified candidates from the registry.
    Strictly excludes rejected candidates, unconfirmed BTS objects, and quarantined pilot objects.
    """
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Candidate registry not found: {registry_path}")

    df = pd.read_csv(registry_path)
    active = df[df["provenance_status"] == "VERIFIED"].copy()
    active = active[~active["candidate_id"].str.startswith("REJ_")].copy()

    # Safety: ensure quarantined historical pilot objects cannot be included
    quarantined_names = {
        "SN 2019np", "ZTF19aacgslb",
        "SN 2020jfo", "ZTF20aaynrrh",
        "AT 2018cow", "ZTF18abukavn",
        "SN 2018zd", "ZTF18aarkpda",
        "ZTF_J195200.60+295217.4", "Field686_Star"
    }
    active = active[~active["ztf_designation"].isin(quarantined_names)].copy()

    # Order deterministically by candidate_id
    active = active.sort_values("candidate_id").reset_index(drop=True)
    return active


def fetch_or_load_candidate_raw(candidate_id: str,
                                ra: float,
                                dec: float,
                                loader: ZTFObjectLoader,
                                raw_base_dir: str = "data/real_ztf_benchmark/raw",
                                timeout: int = 45,
                                max_retries: int = 2) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """
    Load raw observation records from local disk if cached, or query IRSA with retry behavior.
    Preserves raw data at data/real_ztf_benchmark/raw/<candidate_id>/raw_irsa.csv.
    """
    cand_dir = os.path.join(raw_base_dir, candidate_id)
    os.makedirs(cand_dir, exist_ok=True)

    raw_csv_path = os.path.join(cand_dir, "raw_irsa.csv")
    err_log_path = os.path.join(cand_dir, "error.log")
    meta_json_path = os.path.join(cand_dir, "raw_metadata.json")

    # If already cached and non-empty, load
    if os.path.exists(raw_csv_path) and os.path.getsize(raw_csv_path) > 0:
        with open(raw_csv_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
        if lines and "oid" in lines[0]:
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            return list(reader), raw_csv_path, None
        else:
            return [], raw_csv_path, "Cached CSV has no valid 'oid' column"

    # Query IRSA with retry
    raw_text = ""
    status_code = 0
    err_msg = None

    for attempt in range(max_retries + 1):
        raw_text, status_code, err_msg = loader.query_irsa_raw(ra, dec, timeout=timeout)
        if status_code == 200 and raw_text and not err_msg:
            break
        if attempt < max_retries:
            time.sleep(2.0 * (attempt + 1))

    # Record retrieval metadata
    meta = {
        "candidate_id": candidate_id,
        "query_ra": ra,
        "query_dec": dec,
        "query_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "http_status": status_code,
        "error": err_msg,
        "endpoint": f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra:.6f}+{dec:.6f}+0.000500&FORMAT=csv"
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if err_msg or status_code != 200:
        error_detail = f"HTTP {status_code}: {err_msg or 'Unknown retrieval error'}"
        with open(err_log_path, "w", encoding="utf-8") as f:
            f.write(f"Candidate: {candidate_id}\nRA: {ra}, Dec: {dec}\nError: {error_detail}\n")
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write(raw_text if raw_text else "")
        return [], raw_csv_path, error_detail

    # Write raw response
    with open(raw_csv_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    lines = [l for l in raw_text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines or "oid" not in lines[0]:
        with open(err_log_path, "w", encoding="utf-8") as f:
            f.write(f"Candidate: {candidate_id}\nRA: {ra}, Dec: {dec}\nStatus: NO_SOURCE_FOUND\nRows: {len(lines)}\n")
        return [], raw_csv_path, "No source records returned by IRSA"

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    return list(reader), raw_csv_path, None


def evaluate_single_candidate(row: pd.Series,
                              raw_base_dir: str = "data/real_ztf_benchmark/raw") -> Dict[str, Any]:
    """
    Thread-safe processor: retrieves raw photometry, associates celestial sources,
    and applies production preprocessing without running any neural models.
    """
    cand_id = str(row["candidate_id"])
    target_name = str(row["ztf_designation"])
    astro_class = str(row["astrophysical_class"])
    dataset_role = str(row["candidate_role"])
    ra = float(row["ra"])
    dec = float(row["dec"])
    class_auth = str(row.get("class_authority", ""))
    class_ref = str(row.get("class_reference", ""))

    loader = ZTFObjectLoader(
        max_search_radius_arcsec=1.5,
        min_ambiguity_gap_arcsec=0.3,
        min_source_resolution_arcsec=0.15
    )
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)

    raw_records, raw_path, fetch_err = fetch_or_load_candidate_raw(
        candidate_id=cand_id,
        ra=ra,
        dec=dec,
        loader=loader,
        raw_base_dir=raw_base_dir
    )

    raw_obs_count = len(raw_records)

    event: AssociatedZTFEvent = loader.associate_sources(
        target_ra=ra,
        target_dec=dec,
        records=raw_records,
        object_id=cand_id,
        classification=astro_class,
        classification_source=class_auth,
        source_url=class_ref
    )

    matched_oids_dict = {b: m.oid for b, m in event.matched_sources.items()}
    matched_oids_str = json.dumps(matched_oids_dict) if matched_oids_dict else ""

    if event.matched_sources:
        crossmatch_sep = round(max(m.angular_separation_arcsec for m in event.matched_sources.values()), 4)
    else:
        crossmatch_sep = 0.0

    # Determine retrieval status
    if event.metadata.retrieval_status == "SUCCESS":
        retrieval_status = "SUCCESS"
    elif event.metadata.retrieval_status == "AMBIGUOUS":
        retrieval_status = "AMBIGUOUS_ASSOCIATION"
    elif raw_obs_count == 0:
        retrieval_status = "NO_DATA"
    else:
        retrieval_status = "RETRIEVAL_FAILED"

    # Preprocessing
    if retrieval_status == "SUCCESS" and len(event.raw_records) > 0:
        try:
            prep_event: PreprocessedZTFEvent = preprocessor.process_records(event.raw_records, object_id=cand_id)
            clean_obs_count = prep_event.filter_report.passed_observations
            window_obs_count = prep_event.window_report.obs_in_window
            valid_tok_count = prep_event.valid_token_count
            pad_fraction = round((50.0 - valid_tok_count) / 50.0, 4)
            avail_filters = ",".join(sorted(event.available_filters))
            miss_filters = ",".join(sorted(event.missing_filters))
            part_cov = bool(event.partial_filter_coverage)

            # Baseline MJD calculation
            clean_mjds = []
            for r in event.raw_records:
                try:
                    cflags = int(r.get("catflags", 0))
                    if (cflags & 32768) == 0 and (cflags & 15) == 0:
                        clean_mjds.append(float(r["mjd"]))
                except (ValueError, TypeError):
                    continue

            baseline_days = round(float(np.ptp(clean_mjds)), 4) if len(clean_mjds) >= 2 else 0.0

            # Coverage categorization
            preprocessing_status = "SUCCESS"
            if valid_tok_count < 5:
                coverage_status = "INSUFFICIENT_OBSERVATIONS"
                exclusion_reason = f"INSUFFICIENT_OBSERVATIONS: only {valid_tok_count} valid tokens (<5 required)"
            elif valid_tok_count >= 20 and pad_fraction <= 0.60 and len(event.available_filters) >= 2:
                coverage_status = "COVERAGE_SUFFICIENT"
                exclusion_reason = "NONE"
            else:
                coverage_status = "COVERAGE_LIMITED"
                reasons = []
                if valid_tok_count < 20:
                    reasons.append(f"valid_tokens={valid_tok_count}<20")
                if pad_fraction > 0.60:
                    reasons.append(f"padding_fraction={pad_fraction:.2f}>0.60")
                if len(event.available_filters) < 2:
                    reasons.append(f"available_filters={avail_filters}<2")
                exclusion_reason = f"COVERAGE_LIMITED: {', '.join(reasons)}"

        except Exception as prep_err:
            preprocessing_status = "FAILED"
            coverage_status = "PREPROCESSING_FAILED"
            clean_obs_count = 0
            window_obs_count = 0
            valid_tok_count = 0
            pad_fraction = 1.0
            avail_filters = ",".join(sorted(event.available_filters))
            miss_filters = ",".join(sorted(event.missing_filters))
            part_cov = bool(event.partial_filter_coverage)
            baseline_days = 0.0
            exclusion_reason = f"PREPROCESSING_FAILED: {prep_err}"
    else:
        preprocessing_status = "SKIPPED"
        clean_obs_count = 0
        window_obs_count = 0
        valid_tok_count = 0
        pad_fraction = 1.0
        avail_filters = ""
        miss_filters = "zg,zr,zi"
        part_cov = True
        baseline_days = 0.0

        if retrieval_status == "AMBIGUOUS_ASSOCIATION":
            coverage_status = "AMBIGUOUS_ASSOCIATION"
            exclusion_reason = f"AMBIGUOUS_ASSOCIATION: {event.metadata.failure_reason}"
        elif retrieval_status == "NO_DATA":
            coverage_status = "RETRIEVAL_FAILED"
            exclusion_reason = f"NO_DATA: {fetch_err or 'Zero observations in IRSA cone'}"
        else:
            coverage_status = "RETRIEVAL_FAILED"
            exclusion_reason = f"RETRIEVAL_FAILED: {event.metadata.failure_reason or fetch_err}"

    benchmark_eligible = (coverage_status == "COVERAGE_SUFFICIENT")

    return {
        "candidate_id": cand_id,
        "ztf_designation": target_name,
        "astrophysical_class": astro_class,
        "dataset_role": dataset_role,
        "retrieval_status": retrieval_status,
        "matched_oids": matched_oids_str,
        "crossmatch_separation_arcsec": crossmatch_sep,
        "raw_observation_count": raw_obs_count,
        "clean_observation_count": clean_obs_count,
        "window_observation_count": window_obs_count,
        "valid_token_count": valid_tok_count,
        "padding_fraction": pad_fraction,
        "available_filters": avail_filters,
        "missing_filters": miss_filters,
        "partial_filter_coverage": part_cov,
        "baseline_days": baseline_days,
        "preprocessing_status": preprocessing_status,
        "coverage_status": coverage_status,
        "exclusion_reason": exclusion_reason,
        "benchmark_eligible": benchmark_eligible
    }


def run_full_retrieval(registry_path: str = "data/real_ztf_benchmark/candidate_registry.csv",
                       results_csv: str = "data/real_ztf_benchmark/full_retrieval_results.csv",
                       eligibility_csv: str = "data/real_ztf_benchmark/benchmark_eligibility.csv",
                       raw_base_dir: str = "data/real_ztf_benchmark/raw",
                       max_workers: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full photometric retrieval across all 180 active candidates.
    Outputs full_retrieval_results.csv and benchmark_eligibility.csv.
    """
    print("==================================================")
    print("STARTING FULL REAL-ZTF PHOTOMETRIC RETRIEVAL")
    print("==================================================")

    active_df = get_active_candidates(registry_path)
    total_candidates = len(active_df)
    print(f"Loaded {total_candidates} active candidates across {active_df['astrophysical_class'].nunique()} classes.")
    print(f"Executing retrieval pool with {max_workers} worker threads...")

    results = []
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cid = {
            executor.submit(evaluate_single_candidate, row, raw_base_dir): row["candidate_id"]
            for _, row in active_df.iterrows()
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_cid):
            completed += 1
            cid = future_to_cid[future]
            try:
                res = future.result()
                results.append(res)
                print(f"[{completed:03d}/{total_candidates:03d}] {cid:18s} ({res['ztf_designation']:15s}, {res['astrophysical_class']:12s}) -> "
                      f"Retr: {res['retrieval_status']:9s} | Cov: {res['coverage_status']:25s} | "
                      f"Tokens: {res['valid_token_count']:2d} (Pad: {res['padding_fraction']:.2f}) | "
                      f"Eligible: {res['benchmark_eligible']}")
            except Exception as exc:
                print(f"[{completed:03d}/{total_candidates:03d}] ERROR processing {cid}: {exc}")

    total_time = time.time() - t_start
    res_df = pd.DataFrame(results)

    # Sort deterministically by candidate_id order from registry
    res_df = res_df.set_index("candidate_id").loc[active_df["candidate_id"]].reset_index()

    # Create full_retrieval_results.csv
    full_columns = [
        "candidate_id", "ztf_designation", "astrophysical_class", "dataset_role",
        "retrieval_status", "matched_oids", "crossmatch_separation_arcsec",
        "raw_observation_count", "clean_observation_count", "window_observation_count",
        "valid_token_count", "padding_fraction", "available_filters", "missing_filters",
        "partial_filter_coverage", "baseline_days", "preprocessing_status",
        "coverage_status", "exclusion_reason"
    ]
    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    res_df[full_columns].to_csv(results_csv, index=False)

    # Create benchmark_eligibility.csv
    eligibility_df = pd.DataFrame({
        "candidate_id": res_df["candidate_id"],
        "ztf_designation": res_df["ztf_designation"],
        "astrophysical_class": res_df["astrophysical_class"],
        "dataset_role": res_df["dataset_role"],
        "coverage_status": res_df["coverage_status"],
        "benchmark_eligible": res_df["benchmark_eligible"],
        "eligibility_reason": res_df.apply(
            lambda r: "ELIGIBLE: Satisfies provenance, unambiguous coordinate match (<=1.5\"), >=20 tokens, <=60% padding, >=2 filters"
            if r["benchmark_eligible"] else r["exclusion_reason"],
            axis=1
        )
    })
    eligibility_df.to_csv(eligibility_csv, index=False)

    print("==================================================")
    print(f"FULL RETRIEVAL COMPLETE in {total_time:.1f}s ({total_time / 60.0:.2f} min).")
    print(f"Results saved to: {results_csv}")
    print(f"Eligibility saved to: {eligibility_csv}")
    print(f"Eligible objects for final benchmark: {res_df['benchmark_eligible'].sum()} / {len(res_df)}")
    print("==================================================")

    return res_df, eligibility_df


if __name__ == "__main__":
    run_full_retrieval()
