"""Non-learned astrophysical event characterizer for real ZTF light curves."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.data.real_ztf_preprocessing import RealZTFPreprocessor, PreprocessedZTFEvent
from src.investigator.schemas import EventCharacterization, CharacterizedValue


class EventCharacterizer:
    """
    Computes scientifically interpretable, non-learned features from real ZTF light curves.
    Every quantity is explicitly tagged with measurement status:
    MEASURED, DERIVED, PROXY, or UNAVAILABLE.

    Strictly no model training; strictly no classification asserted from these features alone.
    """

    def __init__(self, preprocessor: Optional[RealZTFPreprocessor] = None):
        self.preprocessor = preprocessor or RealZTFPreprocessor(max_sequence_length=50)

    def characterize(self,
                     records: Optional[List[Dict[str, Any]]] = None,
                     object_id: str = "UNKNOWN",
                     preprocessed: Optional[PreprocessedZTFEvent] = None) -> EventCharacterization:
        """
        Derive all physical and statistical characteristics from records or preprocessed event.
        """
        if preprocessed is None:
            if records is None:
                raise ValueError("Must provide either raw observation records or a PreprocessedZTFEvent")
            preprocessed = self.preprocessor.process_records(records, object_id=object_id)

        # Total observations passed through quality filter
        num_obs_val = preprocessed.filter_report.passed_observations
        num_obs = CharacterizedValue(
            value=int(num_obs_val),
            status="MEASURED",
            unit="observations",
            notes="Observations surviving bitwise catflag, cloud, and artifact filtering"
        )

        # Per-band counts and missing bands
        band_counts = dict(preprocessed.band_counts)
        num_filters_val = sum(1 for c in band_counts.values() if c > 0)
        num_filters = CharacterizedValue(
            value=int(num_filters_val),
            status="MEASURED",
            unit="passbands",
            notes="Active optical passbands (zg, zr, zi) with >= 1 observation"
        )

        all_standard_bands = ["g", "r", "i"]
        missing_bands = [b for b in all_standard_bands if band_counts.get(b, 0) == 0]

        # Valid tokens from tensor
        valid_tokens = preprocessed.valid_token_count
        char_valid_tokens = CharacterizedValue(
            value=int(valid_tokens),
            status="DERIVED",
            unit="tokens",
            notes="Sampled and binned observation tokens populated in model sequence (max 50)"
        )

        padding_frac = 1.0 - (valid_tokens / 50.0)
        char_padding = CharacterizedValue(
            value=round(float(padding_frac), 4),
            status="DERIVED",
            unit="fraction",
            notes="Fraction of sequence capacity filled with zero-padding mask"
        )

        # Time baseline across all raw/clean records
        win = preprocessed.window_report
        if win.original_mjd_max > win.original_mjd_min:
            t_baseline = float(win.original_mjd_max - win.original_mjd_min)
            time_baseline = CharacterizedValue(
                value=round(t_baseline, 3),
                status="MEASURED",
                unit="days",
                notes="Span between first and last clean photometric observation"
            )
        else:
            time_baseline = CharacterizedValue(
                value=0.0,
                status="UNAVAILABLE" if num_obs_val == 0 else "MEASURED",
                unit="days",
                notes="Single observation or no data"
            )

        # Window duration
        if win.window_end_mjd > win.window_start_mjd:
            w_duration = float(win.window_end_mjd - win.window_start_mjd)
            window_duration = CharacterizedValue(
                value=round(w_duration, 3),
                status="DERIVED",
                unit="days",
                notes="Duration of candidate transient outburst evaluation window"
            )
        else:
            window_duration = CharacterizedValue(
                value=None,
                status="UNAVAILABLE",
                unit="days",
                notes="Transient outburst window could not be constructed"
            )

        # Extract numerical arrays from valid tokens
        feat_tensor = preprocessed.feature_tensor.cpu().numpy()
        mask_tensor = preprocessed.mask_tensor.cpu().numpy()
        valid_feats = feat_tensor[mask_tensor]  # (N, 4): [rel_t, flux, flux_err, b_idx]

        if len(valid_feats) > 0:
            times = valid_feats[:, 0]
            fluxes = valid_feats[:, 1]
            flux_errs = valid_feats[:, 2]

            peak_f = float(np.max(fluxes))
            min_f = float(np.min(fluxes))
            med_f = float(np.median(fluxes))
            std_f = float(np.std(fluxes))
            med_err = float(np.median(flux_errs))
            amplitude = float(peak_f - min_f)

            peak_norm_flux = CharacterizedValue(
                value=round(peak_f, 4),
                status="MEASURED",
                unit="normalized_flux",
                notes="Maximum normalized flux observed in sequence"
            )
            min_norm_flux = CharacterizedValue(
                value=round(min_f, 4),
                status="MEASURED",
                unit="normalized_flux",
                notes="Minimum normalized flux observed in sequence"
            )
            med_norm_flux = CharacterizedValue(
                value=round(med_f, 4),
                status="DERIVED",
                unit="normalized_flux",
                notes="Median normalized flux of sequence"
            )
            flux_std = CharacterizedValue(
                value=round(std_f, 4),
                status="DERIVED",
                unit="normalized_flux",
                notes="Standard deviation of normalized flux"
            )
            med_flux_err = CharacterizedValue(
                value=round(med_err, 4),
                status="DERIVED",
                unit="normalized_flux",
                notes="Median normalized photometric measurement uncertainty"
            )
            var_amplitude = CharacterizedValue(
                value=round(amplitude, 4),
                status="DERIVED",
                unit="normalized_flux",
                notes="Peak-to-trough normalized flux variation"
            )

            # Cadence statistics
            sorted_times = np.sort(times)
            diffs = np.diff(sorted_times)
            diffs = diffs[diffs > 1e-4]  # Ignore identical epoch bins

            if len(diffs) > 0:
                cad_med = CharacterizedValue(
                    value=round(float(np.median(diffs)), 3),
                    status="DERIVED",
                    unit="days",
                    notes="Median inter-observation interval within window"
                )
                cad_min = CharacterizedValue(
                    value=round(float(np.min(diffs)), 3),
                    status="DERIVED",
                    unit="days",
                    notes="Shortest interval between distinct observation epochs"
                )
                cad_max = CharacterizedValue(
                    value=round(float(np.max(diffs)), 3),
                    status="DERIVED",
                    unit="days",
                    notes="Longest observational gap within window"
                )
            else:
                cad_med = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days", notes="Insufficient epochs")
                cad_min = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days", notes="Insufficient epochs")
                cad_max = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days", notes="Insufficient epochs")

            # Rise / Decline Half-Maximum Proxies
            peak_idx = int(np.argmax(fluxes))
            t_peak = float(times[peak_idx])
            t_start = float(np.min(times))
            t_end = float(np.max(times))

            # Half-max threshold between median/min baseline and peak
            f_baseline = max(min_f, med_f)
            f_half = f_baseline + 0.5 * max(1e-4, peak_f - f_baseline)

            # Rise: time from when flux crosses half-max to peak
            pre_mask = times <= t_peak
            pre_half_times = times[pre_mask & (fluxes >= f_half)]
            if len(pre_half_times) > 0:
                t_rise_onset = float(np.min(pre_half_times))
                rise_days = max(0.1, t_peak - t_rise_onset)
            else:
                rise_days = max(0.1, t_peak - t_start)

            # Decline: time from peak to when flux drops to half-max
            post_mask = times >= t_peak
            post_below_times = times[post_mask & (fluxes <= f_half)]
            if len(post_below_times) > 0:
                t_decline_end = float(np.min(post_below_times))
                decline_days = max(0.1, t_decline_end - t_peak)
            else:
                decline_days = max(0.1, t_end - t_peak)

            rise_proxy = CharacterizedValue(
                value=round(rise_days, 3),
                status="PROXY",
                unit="days",
                notes="Half-maximum rise duration proxy to peak observation"
            )
            decline_proxy = CharacterizedValue(
                value=round(decline_days, 3),
                status="PROXY",
                unit="days",
                notes="Half-maximum decline duration proxy from peak observation"
            )

            if (rise_days + decline_days) > 0:
                asym = (decline_days - rise_days) / (decline_days + rise_days)
                asym_val = CharacterizedValue(
                    value=round(float(asym), 4),
                    status="PROXY",
                    unit="dimensionless",
                    notes="Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline"
                )
            else:
                asym_val = CharacterizedValue(
                    value=0.0,
                    status="PROXY",
                    unit="dimensionless",
                    notes="Symmetric or instantaneous duration"
                )
        else:
            # Fallback when no valid tokens exist
            peak_norm_flux = CharacterizedValue(value=None, status="UNAVAILABLE", unit="normalized_flux")
            min_norm_flux = CharacterizedValue(value=None, status="UNAVAILABLE", unit="normalized_flux")
            med_norm_flux = CharacterizedValue(value=None, status="UNAVAILABLE", unit="normalized_flux")
            flux_std = CharacterizedValue(value=None, status="UNAVAILABLE", unit="normalized_flux")
            med_flux_err = CharacterizedValue(value=None, status="UNAVAILABLE", unit="normalized_flux")
            var_amplitude = CharacterizedValue(value=None, status="UNAVAILABLE", unit="normalized_flux")
            cad_med = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days")
            cad_min = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days")
            cad_max = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days")
            rise_proxy = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days")
            decline_proxy = CharacterizedValue(value=None, status="UNAVAILABLE", unit="days")
            asym_val = CharacterizedValue(value=None, status="UNAVAILABLE", unit="dimensionless")

        return EventCharacterization(
            num_observations=num_obs,
            num_filters=num_filters,
            time_baseline_days=time_baseline,
            window_duration_days=window_duration,
            peak_normalized_flux=peak_norm_flux,
            minimum_normalized_flux=min_norm_flux,
            median_normalized_flux=med_norm_flux,
            flux_std=flux_std,
            median_flux_err=med_flux_err,
            variability_amplitude=var_amplitude,
            rise_time_proxy_days=rise_proxy,
            decline_time_proxy_days=decline_proxy,
            rise_decline_asymmetry=asym_val,
            cadence_median_days=cad_med,
            cadence_min_days=cad_min,
            cadence_max_days=cad_max,
            per_band_counts=band_counts,
            missing_bands=missing_bands,
            valid_token_count=char_valid_tokens,
            padding_fraction=char_padding,
            classification_disclaimer="No astrophysical classification claimed from non-learned features alone."
        )
