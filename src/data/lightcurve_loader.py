"""Light curve time series loader, cleaning, and quality filtering."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from src.data.schema import LightCurve, Observation


class LightCurveLoader:
    """Utilities for loading, filtering, and organizing astronomical light curves."""

    @staticmethod
    def filter_outliers(lc: LightCurve, sigma: float = 4.0) -> LightCurve:
        """Apply sigma-clipping per passband to reject photometric artifacts."""
        cleaned = LightCurve()
        all_bands = set(o.band for o in lc.observations)

        for b in all_bands:
            obs_b = [o for o in lc.observations if o.band == b]
            if len(obs_b) < 3:
                for o in obs_b:
                    cleaned.observations.append(o)
                continue

            fluxes = np.array([o.flux for o in obs_b])
            median = np.median(fluxes)
            mad = np.median(np.abs(fluxes - median))
            std = 1.4826 * mad if mad > 0 else np.std(fluxes) + 1e-6

            for o in obs_b:
                if np.abs(o.flux - median) <= sigma * std:
                    cleaned.observations.append(o)

        cleaned.sort_chronologically()
        return cleaned

    @staticmethod
    def normalize_times_to_discovery(lc: LightCurve) -> LightCurve:
        """Shift times so the first observation is at t = 0.0 days."""
        if len(lc.observations) == 0:
            return lc
        lc.sort_chronologically()
        t0 = lc.observations[0].time
        norm_lc = LightCurve()
        for o in lc.observations:
            norm_lc.add_observation(
                time=o.time - t0,
                band=o.band,
                flux=o.flux,
                flux_err=o.flux_err,
                mag=o.mag,
                mag_err=o.mag_err
            )
        return norm_lc

    @classmethod
    def generate_synthetic_lightcurve(cls, event_type: str = "SN_Ia",
                                      is_anomaly: bool = False,
                                      num_points: int = 30) -> LightCurve:
        """Generate physically-motivated synthetic light curves for various transient classes."""
        lc = LightCurve()
        bands = ["g", "r", "i"]

        # Time range: -10 to +40 days from peak
        times = np.sort(np.random.uniform(-10.0, 45.0, num_points))

        for t in times:
            for b in bands:
                # Probability of being observed in this band at this epoch
                if np.random.rand() < 0.65:
                    flux, flux_err = cls._calculate_model_flux(t, b, event_type, is_anomaly)
                    lc.add_observation(time=float(t), band=b, flux=float(flux), flux_err=float(flux_err))

        lc.sort_chronologically()
        return lc

    @staticmethod
    def _calculate_model_flux(t: float, band: str, event_type: str, is_anomaly: bool) -> Tuple[float, float]:
        """Compute synthetic flux based on empirical parametric models for astronomical transients."""
        # Band wavelength offset (g is bluer, r is mid, i is redder)
        band_shift = {"g": 0.0, "r": 2.5, "i": 5.0}.get(band, 0.0)
        t_eff = t - band_shift

        if event_type == "SN_Ia":
            # Bazin function: A * exp(-(t - t0)/t_fall) / (1 + exp(-(t - t0)/t_rise))
            t_rise, t_fall = 15.0, 25.0
            amp = 1.0 if band == "g" else 1.2
            val = amp * np.exp(-max(0.0, t_eff) / t_fall) / (1.0 + np.exp(-t_eff / t_rise))
        elif event_type == "SN_II":
            # Plateau supernova: long flat phase followed by exponential tail
            if t_eff < 0:
                val = np.exp(t_eff / 8.0)
            elif t_eff < 30:
                val = 0.8 + 0.1 * np.cos(t_eff / 10.0)
            else:
                val = 0.8 * np.exp(-(t_eff - 30.0) / 15.0)
        elif event_type == "Stellar_Flare":
            # Fast rise exponential decay (FRED): rise in minutes/hours, decay over 1-2 days
            if t_eff < 0:
                val = np.exp(t_eff / 1.0)
            else:
                val = np.exp(-t_eff / 3.0)
        elif event_type == "Variable_Star":
            # Periodic sinusoidal modulation
            period = 12.0
            val = 0.5 + 0.4 * np.sin(2 * np.pi * t / period)
        elif event_type == "LRN":
            # Luminous Red Nova (LRN): stellar merger transient.
            # Characteristics:
            # 1. Early blue precursor outburst at t ~ 5d (g-dominant, minimal r/i)
            # 2. Dramatic envelope cooling and dust condensation quenching optical g-band flux
            # 3. Broad red/infrared secondary merger maximum at t ~ 25d (prominent r, extreme i excess)
            if band == "g":
                val = 0.80 * np.exp(-((t_eff - 5.0)**2) / 45.0) + 0.05 * np.exp(-((t_eff - 25.0)**2) / 100.0)
            elif band == "r":
                val = 0.35 * np.exp(-((t_eff - 5.0)**2) / 50.0) + 0.85 * np.exp(-((t_eff - 25.0)**2) / 110.0)
            elif band == "i":
                val = 0.15 * np.exp(-((t_eff - 5.0)**2) / 50.0) + 3.00 * np.exp(-((t_eff - 25.0)**2) / 120.0)
            else:
                val = 0.30 * np.exp(-((t_eff - 5.0)**2) / 50.0) + 0.90 * np.exp(-((t_eff - 25.0)**2) / 110.0)
        elif event_type == "SLSN":
            # Superluminous Supernova: extremely high peak, slow rise (30+ days), slow decline (60+ days)
            t_rise, t_fall = 30.0, 50.0
            amp = 4.5
            val = amp * np.exp(-max(0.0, t_eff) / t_fall) / (1.0 + np.exp(-t_eff / t_rise))
        else:
            val = np.exp(-np.abs(t_eff) / 15.0)

        noise = np.random.normal(0.0, 0.04)
        flux = max(0.01, float(val + noise))
        flux_err = max(0.01, float(0.03 + 0.02 * flux))
        return flux, flux_err
