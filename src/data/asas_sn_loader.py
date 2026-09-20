"""ASAS-SN (All-Sky Automated Survey for Supernovae) light curve loader."""

import os
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.data.schema import AstronomicalEvent, LightCurve, CutoutImage


class ASASSNLoader:
    """Ingests ASAS-SN light curves and variable star / supernova catalogs."""

    @staticmethod
    def parse_csv(filepath: str, object_id: Optional[str] = None,
                  ra: float = 0.0, dec: float = 0.0,
                  label: Optional[str] = None) -> AstronomicalEvent:
        """Parse an ASAS-SN light curve CSV file into an AstronomicalEvent."""
        df = pd.read_csv(filepath)

        # Standardize column naming across ASAS-SN variations
        col_map = {
            "hjd": "time", "HJD": "time", "mjd": "time", "MJD": "time",
            "mag": "mag", "MAG": "mag",
            "mag_err": "mag_err", "MAG_ERR": "mag_err", "err": "mag_err",
            "flux": "flux", "flux_err": "flux_err",
            "filter": "band", "Filter": "band", "camera": "band"
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        lc = LightCurve()
        for _, row in df.iterrows():
            t = float(row.get("time", 0.0))
            band = str(row.get("band", "V")).strip().lower()
            if "g" in band:
                band = "g"
            else:
                band = "V"

            mag = row.get("mag")
            mag_err = row.get("mag_err", 0.05)

            if pd.notna(mag) and float(mag) < 90:
                mag_val = float(mag)
                err_val = float(mag_err) if pd.notna(mag_err) else 0.05
                # Convert to relative flux
                flux = 10.0 ** (-0.4 * (mag_val - 20.0))
                flux_err = (np.log(10.0) / 2.5) * flux * err_val
                lc.add_observation(time=t, band=band, flux=flux, flux_err=flux_err,
                                   mag=mag_val, mag_err=err_val)

        lc.sort_chronologically()

        obj_name = object_id or os.path.splitext(os.path.basename(filepath))[0]
        return AstronomicalEvent(
            object_id=obj_name,
            ra=ra,
            dec=dec,
            lightcurve=lc,
            image=None,
            true_label=label,
            metadata={"survey": "ASAS-SN"}
        )
