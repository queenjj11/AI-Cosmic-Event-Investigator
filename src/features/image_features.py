"""Astronomical cutout image morphology and photometric feature extraction."""

from typing import Dict
import numpy as np


class ImageFeatureExtractor:
    """Extracts morphological and PSF features from astronomical image cutouts."""

    @staticmethod
    def extract_features(img: np.ndarray) -> Dict[str, float]:
        """
        Extract morphology metrics from cutout image of shape (C, H, W) or (H, W).
        If 3-channel [science, template, difference], primary metrics come from difference image.
        """
        if img.ndim == 3 and img.shape[0] >= 3:
            diff = img[2]  # difference image
            sci = img[0]   # science image
        elif img.ndim == 3:
            diff = img[0]
            sci = img[0]
        else:
            diff = img
            sci = img

        H, W = diff.shape
        cy, cx = H // 2, W // 2

        # 1. Peak flux and background noise
        # Background estimated from borders
        border_pixels = np.concatenate([diff[0, :], diff[-1, :], diff[:, 0], diff[:, -1]])
        bg_mean = float(np.mean(border_pixels))
        bg_std = float(np.std(border_pixels)) + 1e-6

        peak_val = float(np.max(diff[cy-4:cy+5, cx-4:cx+5]))
        snr = float((peak_val - bg_mean) / bg_std)

        # 2. Second-order moments (Ellipticity and FWHM)
        y, x = np.mgrid[0:H, 0:W]
        sub = np.clip(diff - bg_mean, 0, None)
        total_flux = np.sum(sub) + 1e-6

        x_bar = np.sum(x * sub) / total_flux
        y_bar = np.sum(y * sub) / total_flux

        m_xx = np.sum((x - x_bar)**2 * sub) / total_flux
        m_yy = np.sum((y - y_bar)**2 * sub) / total_flux
        m_xy = np.sum((x - x_bar) * (y - y_bar) * sub) / total_flux

        # Moments-based FWHM approximation: FWHM ~ 2.355 * sigma
        sigma_moment = np.sqrt(np.clip(0.5 * (m_xx + m_yy), 0.1, None))
        fwhm = float(2.355 * sigma_moment)

        # Ellipticity: e = sqrt((m_xx - m_yy)^2 + 4 * m_xy^2) / (m_xx + m_yy)
        denominator = m_xx + m_yy + 1e-6
        ellipticity = float(np.sqrt((m_xx - m_yy)**2 + 4 * m_xy**2) / denominator)
        ellipticity = min(1.0, max(0.0, ellipticity))

        # 3. Concentration index (ratio of flux within 3px radius to 10px radius)
        r_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        flux_inner = np.sum(diff[r_dist <= 3.0])
        flux_outer = np.sum(diff[r_dist <= 10.0]) + 1e-6
        concentration = float(np.clip(flux_inner / flux_outer, 0.0, 5.0))

        # 4. Host galaxy offset: distance between science peak and difference peak
        sci_peak = np.unravel_index(np.argmax(sci), sci.shape)
        diff_peak = np.unravel_index(np.argmax(diff), diff.shape)
        offset_pixels = float(np.sqrt((sci_peak[0] - diff_peak[0])**2 + (sci_peak[1] - diff_peak[1])**2))

        return {
            "image_fwhm": fwhm,
            "image_ellipticity": ellipticity,
            "image_concentration": concentration,
            "image_snr": snr,
            "host_transient_offset": offset_pixels
        }
