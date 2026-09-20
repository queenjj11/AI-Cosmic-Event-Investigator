"""Scientifically verified coordinate-based ZTF object ingestion and provenance loader.

Associates multi-band observations based strictly on celestial coordinates (RA/Dec)
within a rigorous angular separation tolerance (<= 1.5 arcseconds).

CRITICAL SCIENTIFIC INTEGRITY RULE:
Never manipulate or increment OID digits to associate filters.
In ZTF Data Releases, per-filter coadd source counters are generated independently;
altering OID digits links celestial sources separated by arcminutes.
All multi-band associations MUST be performed via coordinate cross-matching.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import os
import csv
import io
import json
import math
import time
import urllib.request
import numpy as np


def calculate_angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate the exact great-circle angular separation between two celestial positions
    using the Haversine formula.

    Parameters:
        ra1, dec1: Right Ascension and Declination of position 1 (decimal degrees, J2000).
        ra2, dec2: Right Ascension and Declination of position 2 (decimal degrees, J2000).

    Returns:
        Angular separation in arcseconds (float).
    """
    ra1_rad, dec1_rad = math.radians(ra1), math.radians(dec1)
    ra2_rad, dec2_rad = math.radians(ra2), math.radians(dec2)

    d_ra = ra2_rad - ra1_rad
    d_dec = dec2_rad - dec1_rad

    a = math.sin(d_dec / 2.0) ** 2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(d_ra / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return math.degrees(c) * 3600.0


@dataclass
class ZTFSourceMatch:
    """Provenance record for a single matched ZTF catalog source in a specific passband."""
    oid: str
    filtercode: str
    angular_separation_arcsec: float
    mean_ra: float
    mean_dec: float
    num_observations: int
    mjd_min: float
    mjd_max: float


@dataclass
class ZTFObjectMetadata:
    """Object-level astronomical metadata, authoritative provenance, and retrieval status."""
    object_id: str
    target_ra: float
    target_dec: float
    classification: Optional[str] = None
    classification_source: Optional[str] = None
    source_url: Optional[str] = None
    query_timestamp: str = ""
    data_source: str = "NASA/IPAC IRSA ZTF Data Release"
    retrieval_status: str = "PENDING"  # SUCCESS, AMBIGUOUS, FAILED
    retrieval_failed: bool = False
    failure_reason: Optional[str] = None
    available_filters: List[str] = field(default_factory=list)
    missing_filters: List[str] = field(default_factory=list)
    partial_filter_coverage: bool = False


@dataclass
class AssociatedZTFEvent:
    """Complete container holding coordinate-associated ZTF observations and provenance."""
    metadata: ZTFObjectMetadata
    matched_sources: Dict[str, ZTFSourceMatch]  # Key: filtercode
    raw_records: List[Dict[str, Any]]           # Observation rows retaining original columns
    total_raw_observations: int = 0
    available_filters: List[str] = field(default_factory=list)
    missing_filters: List[str] = field(default_factory=list)
    partial_filter_coverage: bool = False
    time_span_days: float = 0.0
    mjd_min: float = 0.0
    mjd_max: float = 0.0


class ZTFObjectLoader:
    """
    Ingests and associates real ZTF lightcurves using verified celestial coordinates.
    
    Adheres strictly to coordinate cross-matching:
    - Angular separation constraint: separation <= max_search_radius_arcsec (default: 1.5'')
    - Safe rejection of ambiguous associations using an engineering heuristic gap threshold
    - Explicit support and tracking of partial filter coverage (e.g. zg+zr, zr-only, zg-only)
    - Complete preservation of photometric uncertainties, catalog flags, and celestial provenance
    """

    def __init__(self,
                 max_search_radius_arcsec: float = 1.5,
                 min_ambiguity_gap_arcsec: float = 0.3,
                 min_source_resolution_arcsec: float = 0.15,
                 standard_filters: Tuple[str, ...] = ("zg", "zr", "zi")):
        """
        Parameters:
            max_search_radius_arcsec: Maximum allowable great-circle angular separation (arcsec).
            min_ambiguity_gap_arcsec: Minimum separation difference required between competing distinct sources.
                                      NOTE: This is an ENGINEERING HEURISTIC, not an astrophysical truth.
                                      If distinct candidate sources in the same filter are separated by less than
                                      this gap, the loader safely rejects the object as AMBIGUOUS.
            min_source_resolution_arcsec: Spatial resolution threshold below which candidate OIDs in different
                                          ZTF survey fields (e.g., primary vs secondary grid) are identified as
                                          the same physical star rather than competing blends.
            standard_filters: Standard full-coverage filter set (default: zg, zr, zi).
        """
        self.max_search_radius_arcsec = max_search_radius_arcsec
        self.min_ambiguity_gap_arcsec = min_ambiguity_gap_arcsec
        self.min_source_resolution_arcsec = min_source_resolution_arcsec
        self.standard_filters = list(standard_filters)

    def query_irsa_raw(self, ra: float, dec: float, radius_arcsec: Optional[float] = None,
                       timeout: int = 45) -> Tuple[str, int, Optional[str]]:
        """
        Query NASA/IPAC IRSA lightcurve API and return raw response text, HTTP status, and error if any.
        """
        rad_arcsec = radius_arcsec if radius_arcsec is not None else self.max_search_radius_arcsec
        radius_deg = rad_arcsec / 3600.0
        # Ensure minimum radius of 0.0005 deg (~1.8 arcsec) so IRSA grid captures edge candidates
        search_deg = max(0.0005, radius_deg)

        url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra:.6f}+{dec:.6f}+{search_deg:.6f}&FORMAT=csv"
        headers = {"User-Agent": "ACEI-Verified-Pilot/1.0"}

        try:
            import requests
            resp = requests.get(url, headers=headers, timeout=timeout)
            return resp.text, resp.status_code, None
        except Exception as e_req:
            # Fallback to urllib.request
            try:
                req = urllib.request.Request(url, headers=headers)
                ssl_ctx = None
                try:
                    import certifi
                    import ssl
                    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                except Exception:
                    ssl_ctx = None
                with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
                    return resp.read().decode("utf-8"), 200, None
            except Exception as e_url:
                return "", 0, f"HTTP request failed: requests error ({e_req}), urllib error ({e_url})"

    def query_irsa_cone(self, ra: float, dec: float, radius_arcsec: Optional[float] = None,
                        timeout: int = 45) -> List[Dict[str, Any]]:
        """
        Query NASA/IPAC IRSA lightcurve API for observations within a circular cone.

        Parameters:
            ra, dec: Target coordinates in decimal degrees (J2000).
            radius_arcsec: Search cone radius in arcseconds. If None, uses max_search_radius_arcsec.
            timeout: HTTP request timeout in seconds.

        Returns:
            List of raw observation dictionaries parsed from IRSA CSV output.
        """
        text, status, err = self.query_irsa_raw(ra, dec, radius_arcsec=radius_arcsec, timeout=timeout)
        if not text or status != 200:
            return []

        lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
        if not lines or "oid" not in lines[0]:
            return []

        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        return list(reader)

    def associate_sources(self,
                          target_ra: float,
                          target_dec: float,
                          records: List[Dict[str, Any]],
                          object_id: str = "UNKNOWN",
                          classification: Optional[str] = None,
                          classification_source: Optional[str] = None,
                          source_url: Optional[str] = None) -> AssociatedZTFEvent:
        """
        Filter and associate raw observation records by celestial coordinate matching.

        Parameters:
            target_ra, target_dec: Authoritative object coordinates.
            records: Raw observation records from IRSA query or local file.
            object_id: Identifier of the astronomical object.
            classification: Authoritative astrophysical classification.
            classification_source: Provenance source of the classification.
            source_url: Reference URL or publication citation.

        Returns:
            AssociatedZTFEvent with matched sources, filtered observations, and metadata.
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata = ZTFObjectMetadata(
            object_id=object_id,
            target_ra=target_ra,
            target_dec=target_dec,
            classification=classification,
            classification_source=classification_source,
            source_url=source_url,
            query_timestamp=timestamp
        )

        if not records:
            metadata.retrieval_status = "FAILED"
            metadata.retrieval_failed = True
            metadata.failure_reason = "No observations returned from data source."
            metadata.missing_filters = list(self.standard_filters)
            return AssociatedZTFEvent(
                metadata=metadata,
                matched_sources={},
                raw_records=[],
                missing_filters=list(self.standard_filters)
            )

        # 1. Compute angular separation for every observation record
        valid_records: List[Tuple[Dict[str, Any], float]] = []
        for r in records:
            r_ra_val = r.get("ra")
            r_dec_val = r.get("dec")
            if r_ra_val is None or r_dec_val is None:
                continue
            try:
                obs_ra = float(r_ra_val)
                obs_dec = float(r_dec_val)
            except (ValueError, TypeError):
                continue

            sep = calculate_angular_separation_arcsec(target_ra, target_dec, obs_ra, obs_dec)
            if sep <= self.max_search_radius_arcsec:
                valid_records.append((r, sep))

        if not valid_records:
            metadata.retrieval_status = "FAILED"
            metadata.retrieval_failed = True
            metadata.failure_reason = (
                f"Zero observations within maximum search radius ({self.max_search_radius_arcsec} arcsec). "
                f"All {len(records)} candidate rows were spatially mismatched."
            )
            metadata.missing_filters = list(self.standard_filters)
            return AssociatedZTFEvent(
                metadata=metadata,
                matched_sources={},
                raw_records=[],
                missing_filters=list(self.standard_filters)
            )

        # 2. Group candidate observations by (filtercode, oid)
        grouped_candidates: Dict[str, Dict[str, List[Tuple[Dict[str, Any], float]]]] = {}
        for r, sep in valid_records:
            band = str(r.get("filtercode", "")).strip().lower()
            oid = str(r.get("oid", "")).strip()
            if not band or not oid:
                continue
            if band not in grouped_candidates:
                grouped_candidates[band] = {}
            if oid not in grouped_candidates[band]:
                grouped_candidates[band][oid] = []
            grouped_candidates[band][oid].append((r, sep))

        # 3. Assess candidate OIDs per passband and evaluate ambiguity
        matched_sources: Dict[str, ZTFSourceMatch] = {}
        selected_oids: Set[str] = set()

        for band, oid_dict in grouped_candidates.items():
            candidate_stats = []
            for oid, obs_list in oid_dict.items():
                seps = [s for _, s in obs_list]
                mean_sep = float(np.mean(seps))
                ras = [float(rec["ra"]) for rec, _ in obs_list]
                decs = [float(rec["dec"]) for rec, _ in obs_list]
                fields = [int(rec["field"]) for rec, _ in obs_list if "field" in rec]
                dominant_field = fields[0] if fields else 0
                candidate_stats.append({
                    "oid": oid,
                    "mean_sep": mean_sep,
                    "mean_ra": float(np.mean(ras)),
                    "mean_dec": float(np.mean(decs)),
                    "field": dominant_field,
                    "count": len(obs_list),
                    "records": [rec for rec, _ in obs_list]
                })

            # Sort candidate OIDs by mean angular separation ascending
            candidate_stats.sort(key=lambda c: c["mean_sep"])

            if len(candidate_stats) == 1:
                best = candidate_stats[0]
            else:
                # Multiple candidate OIDs exist in this passband.
                # Check whether they represent physically distinct celestial sources
                # or the SAME physical star observed in overlapping ZTF survey fields.
                closest = candidate_stats[0]
                second_closest = candidate_stats[1]

                mutual_sep = calculate_angular_separation_arcsec(
                    closest["mean_ra"], closest["mean_dec"],
                    second_closest["mean_ra"], second_closest["mean_dec"]
                )

                if mutual_sep >= self.min_source_resolution_arcsec:
                    # Physically distinct celestial sources within search cone!
                    gap = second_closest["mean_sep"] - closest["mean_sep"]

                    # Ambiguity Evaluation:
                    # If the separation difference is below the engineering heuristic threshold,
                    # association is ambiguous. Safe rejection is preferred.
                    if gap < self.min_ambiguity_gap_arcsec:
                        metadata.retrieval_status = "AMBIGUOUS"
                        metadata.retrieval_failed = True
                        metadata.failure_reason = (
                            f"Ambiguous distinct candidate sources in passband '{band}': OID {closest['oid']} "
                            f"({closest['mean_sep']:.3f}'') vs OID {second_closest['oid']} "
                            f"({second_closest['mean_sep']:.3f}''). Mutual separation={mutual_sep:.3f}''. "
                            f"Separation gap ({gap:.3f}'') < engineering heuristic threshold ({self.min_ambiguity_gap_arcsec}'')."
                        )
                        return AssociatedZTFEvent(
                            metadata=metadata,
                            matched_sources={},
                            raw_records=[],
                            available_filters=[],
                            missing_filters=list(self.standard_filters)
                        )
                    else:
                        best = closest
                else:
                    # Mutual separation < resolution limit: identical celestial source in overlapping survey fields.
                    # Select the primary survey field (<1000) or highest observation count.
                    primary_candidates = [c for c in candidate_stats if c["field"] < 1000]
                    if primary_candidates:
                        best = max(primary_candidates, key=lambda c: c["count"])
                    else:
                        best = max(candidate_stats, key=lambda c: c["count"])

            selected_oids.add(best["oid"])

            mjds = [float(rec["mjd"]) for rec in best["records"] if rec.get("mjd")]
            mjd_min = float(np.min(mjds)) if mjds else 0.0
            mjd_max = float(np.max(mjds)) if mjds else 0.0

            matched_sources[band] = ZTFSourceMatch(
                oid=best["oid"],
                filtercode=band,
                angular_separation_arcsec=best["mean_sep"],
                mean_ra=best["mean_ra"],
                mean_dec=best["mean_dec"],
                num_observations=best["count"],
                mjd_min=mjd_min,
                mjd_max=mjd_max
            )

        # 4. Filter records strictly to selected unambiguous OIDs
        final_records: List[Dict[str, Any]] = []
        for r, _ in valid_records:
            if str(r.get("oid", "")).strip() in selected_oids:
                # Retain all original columns exactly as retrieved
                final_records.append(dict(r))

        # Sort chronologically by MJD
        final_records.sort(key=lambda rec: float(rec.get("mjd", 0.0)))

        # 5. Compute filter coverage and time span metrics
        available_filters = sorted(list(matched_sources.keys()))
        missing_filters = [f for f in self.standard_filters if f not in available_filters]
        partial_coverage = len(missing_filters) > 0 and len(available_filters) > 0

        all_mjds = [float(rec["mjd"]) for rec in final_records if rec.get("mjd")]
        time_span = float(np.max(all_mjds) - np.min(all_mjds)) if all_mjds else 0.0
        mjd_min = float(np.min(all_mjds)) if all_mjds else 0.0
        mjd_max = float(np.max(all_mjds)) if all_mjds else 0.0

        metadata.retrieval_status = "SUCCESS"
        metadata.retrieval_failed = False
        metadata.available_filters = available_filters
        metadata.missing_filters = missing_filters
        metadata.partial_filter_coverage = partial_coverage

        return AssociatedZTFEvent(
            metadata=metadata,
            matched_sources=matched_sources,
            raw_records=final_records,
            total_raw_observations=len(final_records),
            available_filters=available_filters,
            missing_filters=missing_filters,
            partial_filter_coverage=partial_coverage,
            time_span_days=time_span,
            mjd_min=mjd_min,
            mjd_max=mjd_max
        )

    def load_or_fetch(self,
                      object_id: str,
                      ra: float,
                      dec: float,
                      classification: Optional[str] = None,
                      classification_source: Optional[str] = None,
                      source_url: Optional[str] = None,
                      cache_dir: str = "data/verified_ztf_pilot") -> AssociatedZTFEvent:
        """
        Load observations from local cached CSV if available, or query IRSA and cache locally.
        Then executes coordinate association and returns the AssociatedZTFEvent.
        """
        raw_dir = os.path.join(cache_dir, "raw")
        assoc_dir = os.path.join(cache_dir, "associated")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(assoc_dir, exist_ok=True)

        raw_csv_path = os.path.join(raw_dir, f"{object_id}.csv")

        if os.path.exists(raw_csv_path):
            with open(raw_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        else:
            records = self.query_irsa_cone(ra, dec, timeout=45)
            if records:
                fieldnames = list(records[0].keys())
                with open(raw_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(records)

        event = self.associate_sources(
            target_ra=ra,
            target_dec=dec,
            records=records,
            object_id=object_id,
            classification=classification,
            classification_source=classification_source,
            source_url=source_url
        )

        # Save associated records and metadata for provenance audit
        if event.metadata.retrieval_status == "SUCCESS":
            assoc_csv_path = os.path.join(assoc_dir, f"{object_id}_associated.csv")
            if event.raw_records:
                fieldnames = list(event.raw_records[0].keys())
                with open(assoc_csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(event.raw_records)

            meta_json_path = os.path.join(assoc_dir, f"{object_id}_meta.json")
            meta_dict = {
                "object_id": event.metadata.object_id,
                "target_ra": event.metadata.target_ra,
                "target_dec": event.metadata.target_dec,
                "classification": event.metadata.classification,
                "classification_source": event.metadata.classification_source,
                "source_url": event.metadata.source_url,
                "query_timestamp": event.metadata.query_timestamp,
                "data_source": event.metadata.data_source,
                "retrieval_status": event.metadata.retrieval_status,
                "available_filters": event.available_filters,
                "missing_filters": event.missing_filters,
                "partial_filter_coverage": event.partial_filter_coverage,
                "total_raw_observations": event.total_raw_observations,
                "time_span_days": event.time_span_days,
                "matched_sources": {
                    band: {
                        "oid": match.oid,
                        "angular_separation_arcsec": match.angular_separation_arcsec,
                        "mean_ra": match.mean_ra,
                        "mean_dec": match.mean_dec,
                        "num_observations": match.num_observations,
                        "mjd_min": match.mjd_min,
                        "mjd_max": match.mjd_max
                    } for band, match in event.matched_sources.items()
                }
            }
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)

        return event
