"""Time series preprocessing, padding, masking, and Gaussian Process interpolation."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from src.data.schema import LightCurve


BAND_TO_INDEX = {"g": 0, "r": 1, "i": 2, "V": 3, "NIR": 4}


class Preprocessor:
    """Prepares irregular light curves into padded tensors or GP-interpolated grids."""

    def __init__(self, max_length: int = 50, bands: Optional[List[str]] = None):
        self.max_length = max_length
        self.bands = bands or ["g", "r", "i"]
        self.band_to_idx = {b: i for i, b in enumerate(self.bands)}

    def lightcurve_to_padded_tensor(self, lc: LightCurve) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a LightCurve into:
        - features: (max_length, 4) containing [time, flux, flux_err, band_idx]
        - mask: (max_length,) boolean mask where True indicates valid observation
        """
        lc.sort_chronologically()
        features = np.zeros((self.max_length, 4), dtype=np.float32)
        mask = np.zeros((self.max_length,), dtype=bool)

        if len(lc.observations) == 0:
            return features, mask

        # Time relative to discovery
        t0 = lc.observations[0].time
        obs_count = min(len(lc.observations), self.max_length)

        for i in range(obs_count):
            obs = lc.observations[i]
            band_idx = self.band_to_idx.get(obs.band, 0)
            rel_time = obs.time - t0
            features[i] = [rel_time, obs.flux, obs.flux_err, float(band_idx)]
            mask[i] = True

        return features, mask

    def gaussian_process_interpolate(self, lc: LightCurve, band: str,
                                    target_grid: Optional[np.ndarray] = None,
                                    length_scale: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1D Gaussian Process regression with RBF kernel to interpolate flux on a uniform time grid.
        Returns: (time_grid, mean_flux, std_err)
        """
        times, fluxes, errs = lc.get_band_data(band)
        if len(times) < 2:
            grid = target_grid if target_grid is not None else np.linspace(0, 50, 50)
            return grid, np.zeros_like(grid), np.ones_like(grid)

        if target_grid is None:
            target_grid = np.linspace(min(times), max(times), 50)

        # RBF Kernel: K(x1, x2) = exp(-||x1 - x2||^2 / (2 * l^2))
        t_obs = times[:, None]
        t_star = target_grid[:, None]

        K_obs = np.exp(-0.5 * ((t_obs - t_obs.T) / length_scale)**2)
        K_obs += np.diag(errs**2 + 1e-4)  # Observation noise covariance

        K_star = np.exp(-0.5 * ((t_star - t_obs.T) / length_scale)**2)
        K_star_star = np.exp(-0.5 * ((t_star - t_star.T) / length_scale)**2)

        try:
            # GP Predictive Mean: K_* * K_obs^{-1} * y
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, fluxes))
            mean = K_star @ alpha

            # GP Predictive Variance
            v = np.linalg.solve(L, K_star.T)
            var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            std = np.sqrt(np.clip(var, 1e-6, None))
            return target_grid, mean, std
        except np.linalg.LinAlgError:
            # Fallback simple 1D linear interpolation if matrix is singular
            mean = np.interp(target_grid, times, fluxes)
            std = np.ones_like(target_grid) * 0.1
            return target_grid, mean, std
