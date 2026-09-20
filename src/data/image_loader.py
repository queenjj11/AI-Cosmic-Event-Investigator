"""Astronomical image cutout loader, stretch transforms, and normalization."""

import os
from typing import Optional, Tuple, Union
import numpy as np


class ImageLoader:
    """Loads and preprocesses astronomical cutout postage stamps."""

    def __init__(self, target_size: Tuple[int, int] = (64, 64), stretch: str = "asinh"):
        self.target_size = target_size
        self.stretch = stretch

    @staticmethod
    def asinh_stretch(img: np.ndarray, beta: float = 0.05) -> np.ndarray:
        """Apply astronomical arcsinh stretch to compress wide dynamic flux range."""
        valid_mask = np.isfinite(img)
        if not np.any(valid_mask):
            return np.zeros_like(img)
        med = np.median(img[valid_mask])
        std = np.std(img[valid_mask])
        if std > 0:
            norm_img = (img - med) / std
        else:
            norm_img = img - med
        stretched = np.arcsinh(norm_img / (beta + 1e-6))
        # Min-max scale to [0, 1]
        s_min, s_max = np.min(stretched), np.max(stretched)
        if s_max > s_min:
            return (stretched - s_min) / (s_max - s_min)
        return np.zeros_like(img)

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Ensure shape is (C, H, W) with target_size and normalized values."""
        # Clean invalid values
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure channel dimension: if 2D (H, W) -> expand to (1, H, W) -> repeat to 3 channels
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
        elif img.ndim == 3 and img.shape[2] in [1, 3]:  # (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        elif img.shape[0] > 3:
            img = img[:3]

        # Stretch each channel
        channels = []
        for c in range(img.shape[0]):
            ch = self.asinh_stretch(img[c]) if self.stretch == "asinh" else img[c]
            # Simple bilinear/nearest resize if needed
            if ch.shape != self.target_size:
                from scipy.ndimage import zoom
                zoom_factors = (self.target_size[0] / ch.shape[0], self.target_size[1] / ch.shape[1])
                ch = zoom(ch, zoom_factors, order=1)
            channels.append(ch)

        return np.stack(channels, axis=0).astype(np.float32)

    def load_from_npy(self, path: str) -> np.ndarray:
        """Load from a saved numpy file."""
        data = np.load(path)
        return self.preprocess_image(data)

    def load_from_fits(self, path: str) -> np.ndarray:
        """Load primary image data from a FITS file."""
        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                data = hdul[0].data
                return self.preprocess_image(data)
        except Exception:
            # Fallback if astropy is not available or fits corrupted
            return np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float32)

    @classmethod
    def generate_synthetic_cutout(cls, size: Tuple[int, int] = (64, 64),
                                  is_anomaly: bool = False,
                                  event_type: str = "SN_Ia") -> np.ndarray:
        """Generate a realistic 3-channel astronomical cutout: [science, template, difference]."""
        H, W = size
        y, x = np.mgrid[0:H, 0:W]
        cy, cx = H // 2, W // 2

        # Galaxy background in template and science
        host_profile = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (8.0**2)))
        sky_noise = np.random.normal(0.0, 0.05, (H, W))
        template = host_profile * 0.8 + sky_noise

        # Transient point-source PSF at center conditioned on astrophysical class
        if "LRN" in event_type:
            psf_width = 5.0  # Extended dust envelope
            peak_flux = 0.8
        elif "SLSN" in event_type:
            psf_width = 2.5
            peak_flux = 2.5  # Highly luminous point source
        elif "TDE" in event_type:
            psf_width = 2.0
            peak_flux = 1.5
        else:
            psf_width = 2.5
            peak_flux = 1.2
        point_source = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (psf_width**2)))

        science = template + point_source * peak_flux + np.random.normal(0.0, 0.04, (H, W))
        difference = science - template

        channels = np.stack([science, template, difference], axis=0)
        loader = cls(target_size=size)
        return loader.preprocess_image(channels)
