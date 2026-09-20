"""Small-Batch Photometric Retrieval Gate runner for Real-ZTF Evaluation Benchmark.

Retrieves and pre-processes real ZTF photometry for a stratified 35-candidate batch
across 7 collectible astronomical populations without running any neural models.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import csv
import io
import time
import numpy as np
import pandas as pd

from src.data.ztf_object_loader import ZTFObjectLoader, AssociatedZTFEvent
from src.data.real_ztf_preprocessing import RealZTFPreprocessor, PreprocessedZTFEvent


CLASS_PREFIX_MAP = {
    "SNIa": "SN_Ia",
    "SNII": "SN_II",
    "SLSN": "SLSN",
    "TDE": "TDE",
    "VarStar": "Variable_Star",
    "CV": "Cataclysmic_Variable",
    "FieldStar": "Unclassified_Field_Star"
}


def select_pilot_batch(registry_path: str = "data/real_ztf_benchmark/candidate_registry.csv",
                       n_per_class: int = 5) -> pd.DataFrame:
    """
    Deterministically select exactly 35 active candidates (5 per population across 7 classes).
    Strictly excludes quarantined transfer-pilot objects and unconfirmed candidates.
    """
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    df = pd.read_csv(registry_path)

    # Exclude non-verified candidates
    active_df = df[df["provenance_status"] == "VERIFIED"].copy()

    # Safety check: exclude any candidate starting with REJ_
    active_df = active_df[~active_df["candidate_id"].str.startswith("REJ_")].copy()

    # Quarantined exploratory pilot objects
    quarantined_names = {
        "SN 2019np", "ZTF19aacgslb",
        "SN 2020jfo", "ZTF20aaynrrh",
        "AT 2018cow", "ZTF18abukavn",
        "SN 2018zd", "ZTF18aarkpda",
        "ZTF_J195200.60+295217.4", "Field686_Star"
    }
    active_df = active_df[~active_df["ztf_designation"].isin(quarantined_names)].copy()

    selected_batches = []
    for prefix, standard_class in CLASS_PREFIX_MAP.items():
        sub = active_df[active_df["candidate_id"].str.startswith(f"CAND_{prefix}_")].copy()
        sub = sub.sort_values("candidate_id").head(n_per_class)
        if len(sub) < n_per_class:
            raise ValueError(f"Insufficient active candidates for {prefix}: found {len(sub)}, expected {n_per_class}")
        sub["provisional_class"] = standard_class
        selected_batches.append(sub)

    pilot_df = pd.concat(selected_batches, ignore_index=True)
    if len(pilot_df) != len(CLASS_PREFIX_MAP) * n_per_class:
        raise ValueError(f"Expected {len(CLASS_PREFIX_MAP) * n_per_class} candidates, got {len(pilot_df)}")

    return pilot_df


def fetch_or_load_raw(candidate_id: str,
                      ra: float,
                      dec: float,
                      loader: ZTFObjectLoader,
                      raw_base_dir: str = "data/real_ztf_benchmark/raw",
                      timeout: int = 45) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """
    Load raw observation records from local cache if available, or query IRSA.
    Saves raw response text to data/real_ztf_benchmark/raw/<candidate_id>/raw_irsa.csv.
    Saves error.log on retrieval failure.
    """
    cand_dir = os.path.join(raw_base_dir, candidate_id)
    os.makedirs(cand_dir, exist_ok=True)

    raw_csv_path = os.path.join(cand_dir, "raw_irsa.csv")
    err_log_path = os.path.join(cand_dir, "error.log")

    if os.path.exists(raw_csv_path) and os.path.getsize(raw_csv_path) > 0:
        with open(raw_csv_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
        if lines and "oid" in lines[0]:
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            return list(reader), raw_csv_path, None
        else:
            return [], raw_csv_path, "No valid data columns in cached CSV"

    # Query IRSA
    raw_text, status_code, err_msg = loader.query_irsa_raw(ra, dec, timeout=timeout)

    if err_msg or status_code != 200:
        error_detail = f"HTTP status {status_code}: {err_msg or 'Unknown error'}"
        with open(err_log_path, "w", encoding="utf-8") as f:
            f.write(f"Candidate: {candidate_id}\nRA: {ra}, Dec: {dec}\nError: {error_detail}\n")
        # Save empty/failed CSV as indicator
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write(raw_text if raw_text else "")
        return [], raw_csv_path, error_detail

    # Save raw CSV
    with open(raw_csv_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    lines = [l for l in raw_text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines or "oid" not in lines[0]:
        with open(err_log_path, "w", encoding="utf-8") as f:
            f.write(f"Candidate: {candidate_id}\nRA: {ra}, Dec: {dec}\nStatus: NO_SOURCE_FOUND\nRaw rows: {len(lines)}\n")
        return [], raw_csv_path, "No source records returned by IRSA"

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    return list(reader), raw_csv_path, None


def evaluate_candidate_photometry(row: pd.Series,
                                  loader: ZTFObjectLoader,
                                  preprocessor: RealZTFPreprocessor,
                                  raw_base_dir: str = "data/real_ztf_benchmark/raw") -> Dict[str, Any]:
    """
    Process a single candidate through retrieval and preprocessing.
    Computes all observational, coverage, and filter statistics.
    """
    cand_id = row["candidate_id"]
    target_name = row["ztf_designation"]
    provisional_class = row["provisional_class"]
    ra = float(row["ra"])
    dec = float(row["dec"])
    notes = str(row.get("notes", ""))

    raw_records, raw_path, fetch_err = fetch_or_load_raw(
        candidate_id=cand_id,
        ra=ra,
        dec=dec,
        loader=loader,
        raw_base_dir=raw_base_dir
    )

    n_raw_points = len(raw_records)

    # Perform coordinate association
    event: AssociatedZTFEvent = loader.associate_sources(
        target_ra=ra,
        target_dec=dec,
        records=raw_records,
        object_id=cand_id,
        classification=provisional_class,
        classification_source=str(row.get("class_authority", "")),
        source_url=str(row.get("class_reference", ""))
    )

    if event.matched_sources:
        crossmatch_sep = round(max(m.angular_separation_arcsec for m in event.matched_sources.values()), 4)
    else:
        crossmatch_sep = 0.0

    # Determine retrieval status
    if event.metadata.retrieval_status == "SUCCESS":
        retrieval_status = "SUCCESS"
    elif event.metadata.retrieval_status == "AMBIGUOUS":
        retrieval_status = "AMBIGUOUS_ASSOCIATION"
    elif n_raw_points == 0:
        retrieval_status = "NO_DATA"
    else:
        retrieval_status = "RETRIEVAL_FAILED"

    # Preprocessing
    if retrieval_status == "SUCCESS" and len(event.raw_records) > 0:
        try:
            prep_event: PreprocessedZTFEvent = preprocessor.process_records(event.raw_records, object_id=cand_id)
            n_clean_points = prep_event.filter_report.passed_observations
            n_valid_tokens = prep_event.valid_token_count
            padding_fraction = round((50.0 - n_valid_tokens) / 50.0, 4)
            available_filters = ",".join(sorted(event.available_filters))
            missing_filters = ",".join(sorted(event.missing_filters))
            partial_filter_coverage = bool(event.partial_filter_coverage)

            # Compute clean MJD metrics
            clean_mjds = []
            for r in event.raw_records:
                try:
                    cflags = int(r.get("catflags", 0))
                    # Standard quality flags check
                    if (cflags & 32768) == 0 and (cflags & 15) == 0:
                        clean_mjds.append(float(r["mjd"]))
                except (ValueError, TypeError):
                    continue

            if len(clean_mjds) >= 2:
                baseline_days = round(float(np.ptp(clean_mjds)), 4)
                diffs = np.diff(np.sort(clean_mjds))
                diffs_pos = diffs[diffs > 0]
                cadence_median_days = round(float(np.median(diffs_pos)), 4) if len(diffs_pos) > 0 else 0.0
            else:
                baseline_days = 0.0
                cadence_median_days = 0.0

            peak_snr = round(float(prep_event.window_report.candidate_peak_snr), 2)

            # Classify coverage status
            if n_valid_tokens < 5:
                coverage_status = "INSUFFICIENT_OBSERVATIONS"
            elif n_valid_tokens >= 20 and padding_fraction <= 0.60 and len(event.available_filters) >= 2:
                coverage_status = "COVERAGE_SUFFICIENT"
            else:
                coverage_status = "COVERAGE_LIMITED"

        except Exception as prep_ex:
            coverage_status = "PREPROCESSING_FAILED"
            n_clean_points = 0
            n_valid_tokens = 0
            padding_fraction = 1.0
            available_filters = ",".join(sorted(event.available_filters))
            missing_filters = ",".join(sorted(event.missing_filters))
            partial_filter_coverage = bool(event.partial_filter_coverage)
            baseline_days = 0.0
            cadence_median_days = 0.0
            peak_snr = 0.0
            notes = f"Preprocessing error: {prep_ex}. {notes}".strip()
    else:
        if retrieval_status == "AMBIGUOUS_ASSOCIATION":
            coverage_status = "AMBIGUOUS_ASSOCIATION"
            fail_msg = "Ambiguous coordinate association; competing sources violate gap heuristic"
        elif retrieval_status == "NO_DATA":
            coverage_status = "RETRIEVAL_FAILED"
            fail_msg = fetch_err or "No observations returned from IRSA cone search"
        else:
            coverage_status = "RETRIEVAL_FAILED"
            fail_msg = event.metadata.failure_reason or fetch_err or "Coordinate association failed"

        n_clean_points = 0
        n_valid_tokens = 0
        padding_fraction = 1.0
        available_filters = ""
        missing_filters = "zg,zr,zi"
        partial_filter_coverage = True
        baseline_days = 0.0
        cadence_median_days = 0.0
        peak_snr = 0.0
        notes = f"{fail_msg}. {notes}".strip()

    return {
        "candidate_id": cand_id,
        "target_name": target_name,
        "provisional_class": provisional_class,
        "ra": ra,
        "dec": dec,
        "crossmatch_separation_arcsec": crossmatch_sep,
        "retrieval_status": retrieval_status,
        "coverage_status": coverage_status,
        "n_raw_points": n_raw_points,
        "n_clean_points": n_clean_points,
        "n_valid_tokens": n_valid_tokens,
        "padding_fraction": padding_fraction,
        "available_filters": available_filters,
        "missing_filters": missing_filters,
        "partial_filter_coverage": partial_filter_coverage,
        "baseline_days": baseline_days,
        "cadence_median_days": cadence_median_days,
        "peak_snr": peak_snr,
        "raw_data_path": raw_path,
        "notes": notes
    }

def _process_candidate_worker(row: pd.Series, raw_base_dir: str) -> Dict[str, Any]:
    """Thread-safe worker function for candidate retrieval and preprocessing."""
    loader = ZTFObjectLoader(
        max_search_radius_arcsec=1.5,
        min_ambiguity_gap_arcsec=0.3,
        min_source_resolution_arcsec=0.15
    )
    preprocessor = RealZTFPreprocessor(max_sequence_length=50)
    return evaluate_candidate_photometry(row, loader=loader, preprocessor=preprocessor, raw_base_dir=raw_base_dir)


def run_photometric_pilot(registry_path: str = "data/real_ztf_benchmark/candidate_registry.csv",
                          output_csv: str = "data/real_ztf_benchmark/photometric_pilot.csv",
                          raw_base_dir: str = "data/real_ztf_benchmark/raw",
                          max_workers: int = 3) -> pd.DataFrame:
    """
    Run small-batch photometric retrieval gate for 35 candidates concurrently.
    Outputs results to CSV with deterministic candidate ordering.
    """
    import concurrent.futures

    print("==================================================")
    print("EXECUTING REAL-ZTF PHOTOMETRIC RETRIEVAL PILOT")
    print("==================================================")

    pilot_df = select_pilot_batch(registry_path=registry_path, n_per_class=5)
    print(f"Selected {len(pilot_df)} candidates across {len(CLASS_PREFIX_MAP)} classes (5 per class).")
    print(f"Running concurrent retrieval with {max_workers} worker threads...")

    results = []
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cid = {
            executor.submit(_process_candidate_worker, row, raw_base_dir): row["candidate_id"]
            for _, row in pilot_df.iterrows()
        }

        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_cid):
            completed_count += 1
            cid = future_to_cid[future]
            try:
                res = future.result()
                results.append(res)
                print(f"[{completed_count:02d}/35] Finished {cid} ({res['target_name']}, {res['provisional_class']}) -> "
                      f"Status: {res['retrieval_status']} | Coverage: {res['coverage_status']} | "
                      f"Tokens: {res['n_valid_tokens']} (Pad: {res['padding_fraction']:.2f}) | "
                      f"Filters: {res['available_filters']}")
            except Exception as exc:
                print(f"[{completed_count:02d}/35] Candidate {cid} generated an exception: {exc}")

    total_time = time.time() - t_start
    res_df = pd.DataFrame(results)

    # Sort deterministically according to pilot_df candidate order
    res_df = res_df.set_index("candidate_id").loc[pilot_df["candidate_id"]].reset_index()

    # Ensure output dir exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print("==================================================")
    print(f"PILOT COMPLETE: Processed {len(res_df)} candidates in {total_time:.1f}s.")
    print(f"Saved results to: {output_csv}")
    print("==================================================")

    return res_df


if __name__ == "__main__":
    run_photometric_pilot()

