"""Astrophysical feature extraction from astronomical light curves."""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats
from src.data.schema import LightCurve


class LightCurveFeatureExtractor:
    """Extracts astrophysical, statistical, and photometric variability features."""

    def __init__(self, bands: Optional[List[str]] = None):
        self.bands = bands or ["g", "r", "i"]

    def extract_features(self, lc: LightCurve) -> Dict[str, float]:
        """Extract a flat feature vector dictionary from a LightCurve."""
        features: Dict[str, float] = {}

        # 1. Global features across all bands
        times, fluxes, errs, bands = lc.to_arrays()
        if len(fluxes) < 2:
            return self._default_features()

        features["global_amplitude"] = float(np.max(fluxes) - np.min(fluxes))
        features["global_mean"] = float(np.mean(fluxes))
        features["global_std"] = float(np.std(fluxes))
        features["global_skewness"] = float(stats.skew(fluxes))
        features["global_kurtosis"] = float(stats.kurtosis(fluxes))
        features["duration_days"] = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        # Stetson K index (robust kurtosis-like measure of variability)
        mean_flux = np.mean(fluxes)
        delta_flux = (fluxes - mean_flux) / (errs + 1e-5)
        k_val = (1.0 / np.sqrt(len(fluxes))) * np.sum(np.abs(delta_flux)) / (np.sqrt(np.sum(delta_flux**2)) + 1e-6)
        features["stetson_k"] = float(k_val)

        # 2. Per-band features
        band_peaks: Dict[str, float] = {}
        for b in self.bands:
            b_times, b_fluxes, b_errs = lc.get_band_data(b)
            if len(b_fluxes) < 2:
                features[f"{b}_amplitude"] = 0.0
                features[f"{b}_max_flux"] = 0.0
                features[f"{b}_rise_rate"] = 0.0
                features[f"{b}_decay_rate"] = 0.0
                continue

            max_idx = int(np.argmax(b_fluxes))
            max_flux = float(b_fluxes[max_idx])
            t_peak = float(b_times[max_idx])
            band_peaks[b] = max_flux

            features[f"{b}_amplitude"] = float(np.max(b_fluxes) - np.min(b_fluxes))
            features[f"{b}_max_flux"] = max_flux

            # Rise rate: (peak_flux - initial_flux) / (t_peak - t_start)
            if max_idx > 0 and (t_peak - b_times[0]) > 0:
                features[f"{b}_rise_rate"] = float((max_flux - b_fluxes[0]) / (t_peak - b_times[0]))
            else:
                features[f"{b}_rise_rate"] = 0.0

            # Decay rate: (peak_flux - last_flux) / (t_end - t_peak)
            if max_idx < len(b_fluxes) - 1 and (b_times[-1] - t_peak) > 0:
                features[f"{b}_decay_rate"] = float((max_flux - b_fluxes[-1]) / (b_times[-1] - t_peak))
            else:
                features[f"{b}_decay_rate"] = 0.0

        # 3. Color indices (g - r) and (r - i)
        if "g" in band_peaks and "r" in band_peaks and band_peaks["r"] > 0:
            features["color_g_over_r"] = float(band_peaks["g"] / band_peaks["r"])
        else:
            features["color_g_over_r"] = 1.0

        if "r" in band_peaks and "i" in band_peaks and band_peaks["i"] > 0:
            features["color_r_over_i"] = float(band_peaks["r"] / band_peaks["i"])
        else:
            features["color_r_over_i"] = 1.0

        # 4. Period estimation via scipy or astropy
        features["period_days"] = 0.0
        features["period_power"] = 0.0
        if len(times) >= 5:
            try:
                from scipy.signal import lombscargle
                freqs = np.linspace(0.01, 2.0, 100)
                pgram = lombscargle(times, fluxes - np.mean(fluxes), freqs * 2 * np.pi, normalize=True)
                best_freq = freqs[np.argmax(pgram)]
                features["period_days"] = float(1.0 / (best_freq + 1e-6))
                features["period_power"] = float(np.max(pgram))
            except Exception:
                pass


        return features

    def _default_features(self) -> Dict[str, float]:
        """Return zeroes for degenerate light curves."""
        keys = [
            "global_amplitude", "global_mean", "global_std", "global_skewness", "global_kurtosis",
            "duration_days", "stetson_k", "color_g_over_r", "color_r_over_i",
            "period_days", "period_power"
        ]
        for b in self.bands:
            keys.extend([f"{b}_amplitude", f"{b}_max_flux", f"{b}_rise_rate", f"{b}_decay_rate"])
        return {k: 0.0 for k in keys}
