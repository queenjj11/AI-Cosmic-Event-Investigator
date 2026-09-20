"""Isolated preprocessing pipeline for real ZTF light curves.

Transforms raw NASA/IPAC IRSA public light curves or alert packets into the
input tensor contract expected by the existing ACEI LightCurveEncoder:
- Tensor shape: (max_length, 4) with [rel_time, flux, flux_err, band_idx]
- Boolean mask: (max_length,) where True = valid observation, False = padded
- Preserves negative difference fluxes where appropriate without arbitrary clamping
- Performs quality filtering (catflags, uncertainties, missing values)
- Extracts candidate outburst windows ([t_peak - 20d, t_peak + 60d])
- Enforces multi-band mapping (zg->0, zr->1, zi->2)
- Performs dynamic flux normalization to match the [0.0, 5.0] encoder domain
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import csv
import numpy as np
import torch


# Official ZTF filter code to internal ACEI index mapping
# Verified identical to Preprocessor.band_to_idx {"g": 0, "r": 1, "i": 2}
ZTF_BAND_MAP = {
    "zg": 0, "g": 0, "1": 0, 1: 0,
    "zr": 1, "r": 1, "2": 1, 2: 1,
    "zi": 2, "i": 2, "3": 2, 3: 2
}

# Bit 15 = 32768: Cloud cover, Moon pollution, or poor atmospheric transparency
BAD_CATFLAGS_CLOUD_MOON = 32768
# Bits 0-3 = 15: Source near image edge (1), saturated (2), bad pixel (4), blended (8)
BAD_CATFLAGS_SEVERE_ARTIFACTS = 15


@dataclass
class QualityFilterReport:
    """Detailed audit metrics for quality filtering on a light curve."""
    total_raw_observations: int = 0
    passed_observations: int = 0
    removed_missing_or_nan: int = 0
    removed_invalid_time: int = 0
    removed_unsupported_band: int = 0
    removed_cloud_catflags: int = 0
    removed_severe_catflags: int = 0
    removed_bad_uncertainty: int = 0
    removed_unphysical_mag: int = 0
    drb_filtered: int = 0
    drb_available: bool = False


@dataclass
class WindowReport:
    """Audit metrics for candidate transient windowing."""
    original_mjd_min: float = 0.0
    original_mjd_max: float = 0.0
    candidate_peak_mjd: float = 0.0
    candidate_peak_snr: float = 0.0
    window_start_mjd: float = 0.0
    window_end_mjd: float = 0.0
    obs_in_window: int = 0
    has_sufficient_pre_peak: bool = True
    has_sufficient_post_peak: bool = True
    window_warning: Optional[str] = None


@dataclass
class NormalizationReport:
    """Audit metrics for photometric flux transformation and normalization."""
    raw_mag_min: float = 0.0
    raw_mag_max: float = 0.0
    raw_flux_min: float = 0.0
    raw_flux_max: float = 0.0
    norm_flux_min: float = 0.0
    norm_flux_max: float = 0.0
    norm_flux_mean: float = 0.0
    norm_flux_std: float = 0.0
    norm_error_min: float = 0.0
    norm_error_max: float = 0.0
    normalization_reference: float = 1.0
    fraction_negative_raw: float = 0.0
    fraction_negative_norm: float = 0.0
    method: str = "peak_scaled"


@dataclass
class PreprocessedZTFEvent:
    """Container holding preprocessed tensors and complete diagnostic audit logs."""
    object_id: str
    feature_tensor: torch.Tensor  # Shape: (max_length, 4)
    mask_tensor: torch.Tensor     # Shape: (max_length,)
    valid_token_count: int
    filter_report: QualityFilterReport
    window_report: WindowReport
    norm_report: NormalizationReport
    band_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


class ZTFQualityFilter:
    """Filters photometric observations based on ZTF catalog flags, errors, and validity."""

    def __init__(self,
                 filter_clouds: bool = True,
                 filter_artifacts: bool = True,
                 max_mag_err: float = 1.5,
                 min_mag: float = 8.0,
                 max_mag: float = 25.0,
                 min_drb: float = 0.70):
        self.filter_clouds = filter_clouds
        self.filter_artifacts = filter_artifacts
        self.max_mag_err = max_mag_err
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.min_drb = min_drb

    def filter_records(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], QualityFilterReport]:
        """Filter raw ZTF records and generate an observation-level audit report."""
        report = QualityFilterReport(total_raw_observations=len(records))
        filtered: List[Dict[str, Any]] = []

        # Check if DRB is available in these records
        has_drb = any("drb" in r or "rb" in r for r in records)
        report.drb_available = has_drb

        for r in records:
            # 1. Missing or NaN check
            mjd_raw = r.get("mjd") or r.get("jd") or r.get("hmjd")
            mag_raw = r.get("mag") or r.get("magpsf")
            err_raw = r.get("magerr") or r.get("sigmapsf")
            band_raw = r.get("filtercode") or r.get("fid") or r.get("filter")

            if mjd_raw is None or mag_raw is None:
                report.removed_missing_or_nan += 1
                continue

            try:
                mjd = float(mjd_raw)
                mag = float(mag_raw)
                err = float(err_raw) if err_raw is not None else 0.1
            except (ValueError, TypeError):
                report.removed_missing_or_nan += 1
                continue

            if not np.isfinite(mjd) or not np.isfinite(mag) or not np.isfinite(err):
                report.removed_missing_or_nan += 1
                continue

            # 2. Invalid timestamp
            if mjd <= 0:
                report.removed_invalid_time += 1
                continue

            # 3. Unsupported band
            band_str = str(band_raw).strip().lower()
            if band_str not in ZTF_BAND_MAP and band_raw not in ZTF_BAND_MAP:
                report.removed_unsupported_band += 1
                continue

            # 4. Catflags bitmask check
            catflags_val = r.get("catflags")
            if catflags_val is not None:
                try:
                    cflags = int(catflags_val)
                    if self.filter_clouds and (cflags & BAD_CATFLAGS_CLOUD_MOON) != 0:
                        report.removed_cloud_catflags += 1
                        continue
                    if self.filter_artifacts and (cflags & BAD_CATFLAGS_SEVERE_ARTIFACTS) != 0:
                        report.removed_severe_catflags += 1
                        continue
                except (ValueError, TypeError):
                    pass

            # 5. Photometric uncertainty check
            if err <= 0.0 or err > self.max_mag_err:
                report.removed_bad_uncertainty += 1
                continue

            # 6. Unphysical magnitude check
            if mag < self.min_mag or mag > self.max_mag:
                report.removed_unphysical_mag += 1
                continue

            # 7. Real-Bogus check (if available in alert stream)
            if has_drb:
                drb_val = r.get("drb", r.get("rb"))
                if drb_val is not None:
                    try:
                        if float(drb_val) < self.min_drb:
                            report.drb_filtered += 1
                            continue
                    except (ValueError, TypeError):
                        pass

            # Standardized clean record
            clean_rec = dict(r)
            clean_rec["clean_mjd"] = mjd
            clean_rec["clean_mag"] = mag
            clean_rec["clean_magerr"] = err
            clean_rec["clean_band"] = "g" if "g" in band_str or band_raw == 1 else ("r" if "r" in band_str or band_raw == 2 else "i")
            filtered.append(clean_rec)

        report.passed_observations = len(filtered)
        return filtered, report


class PhotometryConverter:
    """Converts photometric measurements into linear flux representations."""

    @staticmethod
    def mag_to_flux(mag: float, magerr: float, zero_point: float = 27.5) -> Tuple[float, float]:
        """
        Convert apparent magnitude and magnitude uncertainty to linear flux units.
        F = 10^(-0.4 * (mag - zero_point))
        sigma_F = (ln(10) / 2.5) * F * sigma_mag
        """
        flux = float(10.0 ** (-0.4 * (mag - zero_point)))
        # Error propagation
        flux_err = float((np.log(10.0) / 2.5) * flux * max(0.005, magerr))
        return flux, flux_err

    @staticmethod
    def process_record_photometry(r: Dict[str, Any], zero_point: float = 27.5) -> Tuple[float, float, bool]:
        """
        Extract flux and flux_err from record.
        Returns: (flux, flux_err, is_negative)
        If difference flux is explicitly present (e.g. forced photometry), preserves negative values.
        """
        if "forcediffimflux" in r and r["forcediffimflux"] is not None:
            raw_f = float(r["forcediffimflux"])
            raw_err = float(r.get("forcediffimfluxunc", abs(raw_f) * 0.1 + 1e-4))
            is_neg = raw_f < 0.0
            return raw_f, raw_err, is_neg

        # Otherwise convert from clean magnitude
        mag = float(r["clean_mag"])
        err = float(r["clean_magerr"])
        flux, flux_err = PhotometryConverter.mag_to_flux(mag, err, zero_point=zero_point)
        return flux, flux_err, False


class TransientWindowSelector:
    """Selects an astrophysically motivated observation window around candidate outbursts."""

    def __init__(self,
                 pre_peak_days: float = 20.0,
                 post_peak_days: float = 60.0,
                 min_snr_threshold: float = 3.0):
        self.pre_peak_days = pre_peak_days
        self.post_peak_days = post_peak_days
        self.min_snr_threshold = min_snr_threshold

    def select_window(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], WindowReport]:
        """Identify candidate peak and extract window [t_peak - pre_peak, t_peak + post_peak]."""
        if not records:
            return [], WindowReport()

        mjds = np.array([r["clean_mjd"] for r in records])
        fluxes = np.array([r["clean_flux"] for r in records])
        flux_errs = np.array([r["clean_flux_err"] for r in records])

        snrs = fluxes / np.maximum(flux_errs, 1e-5)
        peak_idx = int(np.argmax(fluxes))
        peak_mjd = float(mjds[peak_idx])
        peak_snr = float(snrs[peak_idx])

        t_start = peak_mjd - self.pre_peak_days
        t_end = peak_mjd + self.post_peak_days

        in_window_mask = (mjds >= t_start) & (mjds <= t_end)
        selected = [r for i, r in enumerate(records) if in_window_mask[i]]

        # Coverage diagnostics
        has_pre = bool(np.any(mjds < peak_mjd - 2.0))
        has_post = bool(np.any(mjds > peak_mjd + 5.0))

        warning = None
        if peak_snr < self.min_snr_threshold:
            warning = f"Peak SNR ({peak_snr:.2f}) below threshold ({self.min_snr_threshold}); possible non-transient or steady variable"
        elif not has_pre:
            warning = "Insufficient pre-peak coverage; transient onset may be unobserved"
        elif not has_post:
            warning = "Insufficient post-peak coverage; transient decay may be unobserved"

        report = WindowReport(
            original_mjd_min=float(np.min(mjds)),
            original_mjd_max=float(np.max(mjds)),
            candidate_peak_mjd=peak_mjd,
            candidate_peak_snr=peak_snr,
            window_start_mjd=t_start,
            window_end_mjd=t_end,
            obs_in_window=len(selected),
            has_sufficient_pre_peak=has_pre,
            has_sufficient_post_peak=has_post,
            window_warning=warning
        )

        return selected, report


class FluxNormalizer:
    """Normalizes flux values into the standard neural network domain [0.0, 5.0]."""

    @staticmethod
    def normalize_peak_scaled(records: List[Dict[str, Any]],
                              target_peak_scale: float = 1.5) -> Tuple[List[Dict[str, Any]], NormalizationReport]:
        """
        Normalize flux by dividing by the peak flux and scaling to target_peak_scale.
        Preserves relative photometric uncertainties: sigma_norm = (sigma_F / F_peak) * target_peak_scale.
        """
        if not records:
            return [], NormalizationReport()

        fluxes = np.array([r["clean_flux"] for r in records])
        errs = np.array([r["clean_flux_err"] for r in records])
        mags = np.array([r["clean_mag"] for r in records])

        f_peak = float(np.max(fluxes)) if np.max(fluxes) > 0 else 1.0
        neg_count_raw = int(np.sum(fluxes < 0))

        norm_fluxes = (fluxes / f_peak) * target_peak_scale
        norm_errs = (errs / f_peak) * target_peak_scale
        neg_count_norm = int(np.sum(norm_fluxes < 0))

        normalized_records = []
        for i, r in enumerate(records):
            rec = dict(r)
            rec["norm_flux"] = float(norm_fluxes[i])
            rec["norm_flux_err"] = float(norm_errs[i])
            normalized_records.append(rec)

        report = NormalizationReport(
            raw_mag_min=float(np.min(mags)),
            raw_mag_max=float(np.max(mags)),
            raw_flux_min=float(np.min(fluxes)),
            raw_flux_max=float(np.max(fluxes)),
            norm_flux_min=float(np.min(norm_fluxes)),
            norm_flux_max=float(np.max(norm_fluxes)),
            norm_flux_mean=float(np.mean(norm_fluxes)),
            norm_flux_std=float(np.std(norm_fluxes)),
            norm_error_min=float(np.min(norm_errs)),
            norm_error_max=float(np.max(norm_errs)),
            normalization_reference=f_peak,
            fraction_negative_raw=float(neg_count_raw / len(records)),
            fraction_negative_norm=float(neg_count_norm / len(records)),
            method="peak_scaled"
        )

        return normalized_records, report


class TokenSequenceConstructor:
    """Assembles a standard (max_length, 4) tensor and boolean mask for LightCurveEncoder."""

    def __init__(self, max_length: int = 50):
        self.max_length = max_length

    def construct_tensor(self, records: List[Dict[str, Any]],
                         t_zero: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Sort records chronologically, apply intelligent subsampling/binning if > max_length,
        pad with zeros if < max_length, and return (features, mask, band_counts).
        """
        if not records:
            return torch.zeros((self.max_length, 4), dtype=torch.float32), torch.zeros((self.max_length,), dtype=torch.bool), {}

        # 1. Chronological sort
        sorted_records = sorted(records, key=lambda r: r["clean_mjd"])
        t0 = t_zero if t_zero is not None else sorted_records[0]["clean_mjd"]

        # 2. Multi-band same-night binning if observation count exceeds max_length
        if len(sorted_records) > self.max_length:
            binned = self._bin_same_night(sorted_records)
            if len(binned) > self.max_length:
                # Quantile uniform subsampling to preserve evolution across all phases
                indices = np.round(np.linspace(0, len(binned) - 1, self.max_length)).astype(int)
                final_records = [binned[i] for i in indices]
            else:
                final_records = binned
        else:
            final_records = sorted_records

        # 3. Assemble tensor and boolean mask
        features = np.zeros((self.max_length, 4), dtype=np.float32)
        mask = np.zeros((self.max_length,), dtype=bool)
        band_counts = {"g": 0, "r": 0, "i": 0}

        obs_count = min(len(final_records), self.max_length)
        for i in range(obs_count):
            r = final_records[i]
            rel_t = max(0.0, float(r["clean_mjd"] - t0))
            flux = float(r["norm_flux"])
            err = float(r["norm_flux_err"])
            b_str = r["clean_band"]
            b_idx = float(ZTF_BAND_MAP.get(b_str, 0))

            features[i] = [rel_t, flux, err, b_idx]
            mask[i] = True
            band_counts[b_str] = band_counts.get(b_str, 0) + 1

        feature_tensor = torch.from_numpy(features).float()
        mask_tensor = torch.from_numpy(mask).bool()

        return feature_tensor, mask_tensor, band_counts

    @staticmethod
    def _bin_same_night(records: List[Dict[str, Any]], dt_days: float = 0.5) -> List[Dict[str, Any]]:
        """Group observations in the same band within dt_days and compute inverse-variance weighted mean."""
        binned = []
        # Group by band
        by_band = {}
        for r in records:
            b = r["clean_band"]
            by_band.setdefault(b, []).append(r)

        for b, b_records in by_band.items():
            b_sorted = sorted(b_records, key=lambda x: x["clean_mjd"])
            current_bin = [b_sorted[0]]
            for r in b_sorted[1:]:
                if r["clean_mjd"] - current_bin[0]["clean_mjd"] <= dt_days:
                    current_bin.append(r)
                else:
                    binned.append(TokenSequenceConstructor._aggregate_bin(current_bin))
                    current_bin = [r]
            if current_bin:
                binned.append(TokenSequenceConstructor._aggregate_bin(current_bin))

        # Re-sort combined binned observations chronologically
        binned.sort(key=lambda x: x["clean_mjd"])
        return binned

    @staticmethod
    def _aggregate_bin(bin_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute inverse-variance weighted average for a single temporal bin."""
        if len(bin_records) == 1:
            return bin_records[0]

        fluxes = np.array([r["norm_flux"] for r in bin_records])
        errs = np.array([r["norm_flux_err"] for r in bin_records])
        mjds = np.array([r["clean_mjd"] for r in bin_records])
        weights = 1.0 / np.maximum(errs**2, 1e-6)

        weighted_flux = float(np.sum(fluxes * weights) / np.sum(weights))
        weighted_err = float(1.0 / np.sqrt(np.sum(weights)))
        weighted_mjd = float(np.mean(mjds))

        agg = dict(bin_records[0])
        agg["clean_mjd"] = weighted_mjd
        agg["norm_flux"] = weighted_flux
        agg["norm_flux_err"] = weighted_err
        return agg


class RealZTFPreprocessor:
    """Complete end-to-end preprocessor for real ZTF lightcurve observations."""

    def __init__(self,
                 max_sequence_length: int = 50,
                 pre_peak_days: float = 20.0,
                 post_peak_days: float = 60.0):
        self.quality_filter = ZTFQualityFilter()
        self.window_selector = TransientWindowSelector(pre_peak_days=pre_peak_days, post_peak_days=post_peak_days)
        self.sequence_constructor = TokenSequenceConstructor(max_length=max_sequence_length)

    def process_csv_file(self, csv_filepath: str, object_id: str) -> PreprocessedZTFEvent:
        """Process a real ZTF CSV file from IRSA or alert stream."""
        with open(csv_filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        return self.process_records(records, object_id=object_id)

    def process_records(self, records: List[Dict[str, Any]], object_id: str) -> PreprocessedZTFEvent:
        """Execute full preprocessing pipeline on raw observation dictionaries."""
        warnings: List[str] = []

        # 1. Quality filtering
        clean_records, filter_report = self.quality_filter.filter_records(records)
        if len(clean_records) == 0:
            warnings.append("Zero observations passed quality filtering")
            dummy_f = torch.zeros((self.sequence_constructor.max_length, 4), dtype=torch.float32)
            dummy_m = torch.zeros((self.sequence_constructor.max_length,), dtype=torch.bool)
            return PreprocessedZTFEvent(
                object_id=object_id,
                feature_tensor=dummy_f,
                mask_tensor=dummy_m,
                valid_token_count=0,
                filter_report=filter_report,
                window_report=WindowReport(),
                norm_report=NormalizationReport(),
                band_counts={},
                warnings=warnings
            )

        # 2. Photometric conversion
        for r in clean_records:
            flux, flux_err, is_neg = PhotometryConverter.process_record_photometry(r)
            r["clean_flux"] = flux
            r["clean_flux_err"] = flux_err
            r["is_negative_flux"] = is_neg

        # 3. Transient window selection
        window_records, window_report = self.window_selector.select_window(clean_records)
        if window_report.window_warning:
            warnings.append(window_report.window_warning)

        # Fallback to full sequence if window has fewer than 5 points
        if len(window_records) < 5:
            warnings.append(f"Window contained only {len(window_records)} points; falling back to clean sequence")
            window_records = clean_records
            t_zero = float(np.min([r["clean_mjd"] for r in clean_records]))
        else:
            t_zero = window_report.window_start_mjd

        # 4. Normalization
        norm_records, norm_report = FluxNormalizer.normalize_peak_scaled(window_records)

        # 5. Token sequence construction
        feature_tensor, mask_tensor, band_counts = self.sequence_constructor.construct_tensor(
            norm_records, t_zero=t_zero
        )
        valid_tokens = int(torch.sum(mask_tensor).item())

        return PreprocessedZTFEvent(
            object_id=object_id,
            feature_tensor=feature_tensor,
            mask_tensor=mask_tensor,
            valid_token_count=valid_tokens,
            filter_report=filter_report,
            window_report=window_report,
            norm_report=norm_report,
            band_counts=band_counts,
            warnings=warnings
        )
