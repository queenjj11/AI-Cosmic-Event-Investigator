"""ZTF (Zwicky Transient Facility) alert packet and light curve loader."""

import os
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.data.schema import AstronomicalEvent, LightCurve, CutoutImage


ZTF_FID_TO_BAND = {
    1: "g",
    2: "r",
    3: "i"
}


class ZTFLoader:
    """Ingests ZTF alert packets, photometry histories, and postage stamps."""

    @staticmethod
    def mag_to_flux(mag: float, mag_err: float, zero_point: float = 27.5) -> tuple[float, float]:
        """Convert apparent magnitude and magnitude error to microJansky flux units."""
        if mag is None or np.isnan(mag) or mag > 90:
            return 0.0, 0.0
        flux = 10.0 ** (-0.4 * (mag - zero_point))
        # Error propagation: sigma_flux = ln(10)/2.5 * flux * sigma_mag
        flux_err = (np.log(10.0) / 2.5) * flux * (mag_err if mag_err and not np.isnan(mag_err) else 0.1)
        return float(flux), float(flux_err)

    @classmethod
    def parse_alert_dict(cls, alert: Dict[str, Any]) -> AstronomicalEvent:
        """Parse a ZTF alert dictionary into an AstronomicalEvent."""
        candidate = alert.get("candidate", {})
        object_id = alert.get("objectId", candidate.get("objectId", "ZTF_UNKNOWN"))
        ra = float(candidate.get("ra", 0.0))
        dec = float(candidate.get("dec", 0.0))

        lc = LightCurve()

        # Parse historical candidate observations (prv_candidates)
        prv_candidates = alert.get("prv_candidates", [])
        all_obs = list(prv_candidates) + [candidate]

        for obs in all_obs:
            jd = obs.get("jd")
            fid = obs.get("fid")
            mag = obs.get("magpsf")
            sigmag = obs.get("sigmapsf")

            if jd is None or fid not in ZTF_FID_TO_BAND:
                continue

            band = ZTF_FID_TO_BAND[fid]
            if mag is not None and not np.isnan(mag):
                flux, flux_err = cls.mag_to_flux(mag, sigmag)
                lc.add_observation(time=float(jd), band=band, flux=flux, flux_err=flux_err,
                                   mag=float(mag), mag_err=float(sigmag) if sigmag else None)

        lc.sort_chronologically()

        # Parse cutout postage stamps if present
        cutout = None
        cutout_science = alert.get("cutoutScience")
        if isinstance(cutout_science, np.ndarray):
            cutout = CutoutImage(data=cutout_science)
        elif "cutouts" in alert and isinstance(alert["cutouts"], np.ndarray):
            cutout = CutoutImage(data=alert["cutouts"])

        metadata = {
            "survey": "ZTF",
            "field": candidate.get("field"),
            "rcid": candidate.get("rcid"),
            "drb": candidate.get("drb", candidate.get("rb")),
            "distnr": candidate.get("distnr"),
        }

        return AstronomicalEvent(
            object_id=object_id,
            ra=ra,
            dec=dec,
            lightcurve=lc,
            image=cutout,
            true_label=alert.get("classification"),
            metadata=metadata
        )

    @classmethod
    def load_from_file(cls, filepath: str) -> AstronomicalEvent:
        """Load ZTF alert from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.parse_alert_dict(data)
