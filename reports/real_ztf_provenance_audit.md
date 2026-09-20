# ACEI Real ZTF Pilot: Strict Provenance & Scientific Validity Audit

**Audit Date:** September 19, 2026  
**Phase:** Phase 3 — Transition to Real Astronomical Data  
**Scope:** Strict Object-by-Object Provenance & Scientific-Validity Audit of the 10 Pilot Objects  
**Production Integrity:** Zero production code changes; zero model modifications; zero retraining; zero threshold tuning; zero real-data performance claims.  

---

## 1. Executive Summary & Critical Audit Findings

This audit investigated the scientific validity, catalog provenance, coordinate integrity, and preprocessing behavior of the 10 real ZTF objects processed in `reports/real_ztf_pilot.csv` and `data/real_ztf_pilot/`.

```
                        CRITICAL AUDIT VERDICT:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Engineering Compatibility: VERIFIED                                     │
│    - Pipeline parsing, quality cuts, padding, masking, and tensor          │
│      construction operate stably.                                           │
│    - Zero-shot forward pass through LightCurveEncoder yields 100% finite   │
│      128-D embeddings with zero NaNs/Infs.                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Scientific & Provenance Validity: FAILED FOR ALL 10 OBJECTS              │
│    - Multi-Band Spatial Mismatch: Incrementing filter IDs in ZTF catalog    │
│      OIDs (e.g. 6861... vs 6862...) merged stars separated by 8 to 44       │
│      arcminutes on the sky.                                                 │
│    - Zero Authoritative Labels: None of the 10 objects possess verified     │
│      spectroscopic classifications from TNS, BTS, or Simbad.                │
│    - Peak-Window Invalidity: Burst windowing on multi-year static stars     │
│      isolates arbitrary noise peaks rather than physical explosions.        │
│    - Benchmark Readiness: ALL 10 OBJECTS MARKED benchmark_ready = False.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Comprehensive Provenance Audit Table

The table below provides the object-by-object audit covering all 23 mandated dimensions. The raw machine-readable data is recorded in [`reports/real_ztf_provenance_audit.csv`](file:///Users/jiajadhav/Desktop/BTECH%20/btech/3RD%20YR/AI-Cosmic-Event-Investigator/reports/real_ztf_provenance_audit.csv).

| Object ID | RA, Dec (deg) | Filters | Authoritative Class & Source | Pilot Manual Description | Object Role | Peak Window Valid? | Binning / Subsampling / Padding | Benchmark Ready? | Key Scientific Failure / Audit Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ZTF_VAR_01** | 297.8863, +29.8799 | zg, zr, zi | **None** (Unclassified tutorial star) | Variable Star (zg, zr, zi) in Field 686 | Stress Test Only | **False** (Static star) | Bin: Yes / Sub: Yes / Pad: No | **FALSE** | **Severe Spatial Mismatch**: OID `...34440` in zg, zr, zi are 3 distinct stars separated by up to 14.6 arcmin. |
| **ZTF_VAR_02** | 297.8854, +29.9344 | zg, zr, zi | **None** (Unclassified tutorial star) | Variable Candidate (zg, zr, zi) in Field 686 | Stress Test Only | **False** (Static star) | Bin: Yes / Sub: Yes / Pad: No | **FALSE** | **Severe Spatial Mismatch**: OIDs represent distinct stars separated by up to 44.5 arcmin. |
| **ZTF_VAR_03** | 297.9143, +29.9817 | zg, zr | **None** (Unclassified catalog star) | Periodic Variable (zg, zr) in Field 686 | Stress Test Only | **False** (Static star) | Bin: Yes / Sub: No / Pad: Yes | **FALSE** | **Spatial Mismatch**: zg and zr OIDs are distinct stars separated by 14.8 arcmin. |
| **ZTF_SRC_04** | 297.4792, +30.4848 | zg, zr | **None** (Field 686 catalog counter) | Faint Variable Star (zg, zr) in Field 686 | Stress Test Only | **False** (Only 6 window obs) | Bin: No / Sub: No / Pad: Yes | **FALSE** | **Spatial Mismatch**: zg and zr OIDs separated by 8.2 arcmin. Window has only 6 observations. |
| **ZTF_SRC_05** | 297.7222, +30.4844 | zg, zr | **None** (Field 686 catalog counter) | Variable Source (zg, zr) in Field 686 | Stress Test Only | **False** (Static star) | Bin: Yes / Sub: No / Pad: Yes | **FALSE** | **Spatial Mismatch**: zg and zr OIDs separated by 34.1 arcmin. No authoritative classification. |
| **ZTF_FLARE_06**| 297.8300, +30.4828 | zg, zr | **None** (Field 686 catalog counter) | High-Amplitude Variable / Flare (zg, zr) | Stress Test Only | **False** (Window warning) | Bin: No / Sub: No / Pad: Yes | **FALSE** | **Spatial Mismatch**: zg and zr OIDs separated by 24.2 arcmin. Flagged insufficient post-peak coverage. |
| **ZTF_FAINT_07**| 297.4567, +30.4691 | zg, zr | **None** (Faint catalog star) | Faint Detection-Limit Source (zg, zr) | Stress Test Only | **False** (Static noise) | Bin: Yes / Sub: No / Pad: Yes | **FALSE** | **Spatial Mismatch**: zg and zr OIDs separated by 17.0 arcmin. Unclassified catalog detection. |
| **ZTF_BRIGHT_08**| 298.1060, +30.2630 | zg, zr | **None** (Bright field star) | Bright Variable Star (zg, zr) | Stress Test Only | **False** (Static star) | Bin: Yes / Sub: Yes / Pad: No | **FALSE** | **Spatial Mismatch**: zg and zr OIDs separated by 8.5 arcmin. Unclassified bright field star. |
| **ZTF_SPARSE_09**| 298.0025, +29.8715 | zg | **None** (Synthetic engineering slice) | Sparse Real Sample (18 obs) for Padding Test | Stress Test Only | **False** (Fragmented slice) | Bin: Yes / Sub: No / Pad: Yes | **FALSE** | Single-band artificial slice (18 obs) created to stress-test zero-padding. No astrophysical ground truth. |
| **ZTF_BURST_10** | 297.5644, +30.4878 | zr | **None** (Synthetic engineering slice) | Outburst Episode (70 obs) for Windowing Test | Stress Test Only | **False** (Arbitrary noise peak) | Bin: Yes / Sub: No / Pad: Yes | **FALSE** | Single-band slice (70 obs) created to test burst windowing. No physical explosion provenance. |

---

## 3. Detailed Scientific & Methodological Audit

### Finding 1: The ZTF Catalog OID Multi-Band Spatial Mismatch

In NASA/IPAC IRSA static catalog releases (DR), light curves are tagged by an internal 64-bit integer `oid`:
$$\text{oid} = \text{field (3 digits)} + \text{filterid (1 digit)} + \text{ccdid (2 digits)} + \text{qid (1 digit)} + \text{counter (8 digits)}$$

The pilot implementation assumed that incrementing the filter digit from `1` (zg) to `2` (zr) or `3` (zi) while holding the counter fixed would query the same celestial source in other filters:
- E.g., `686103400034440` ($g$) paired with `686203400034440` ($r$) and `686303400034440` ($i$).

**Astronomical Reality Discovered During Audit:**
In ZTF catalog data releases, **each passband coadd is cataloged independently**. Source counters are assigned sequentially within each filter's detection pipeline. Consequently, counter `00034440` in $g$-band and counter `00034440` in $r$-band are **completely distinct celestial objects** located at different coordinates on the detector quadrant!
- For `ZTF_VAR_01`:
  - `686103400034440` (zg): $\alpha = 298.002521^\circ, \delta = +29.871492^\circ$
  - `686203400034440` (zr): $\alpha = 297.719812^\circ, \delta = +29.885357^\circ$ ($\Delta \theta = 14.6\text{ arcminutes}$)
  - `686303400034440` (zi): $\alpha = 297.991481^\circ, \delta = +29.900491^\circ$ ($\Delta \theta = 1.8\text{ arcminutes}$)

**Scientific Consequence:**
Objects `ZTF_VAR_01` through `ZTF_BRIGHT_08` did not combine multi-band photometry of single astrophysical objects; they concatenated the lightcurves of 2 to 3 physically separate stars located arcminutes apart. While this successfully stressed the software mechanics of multi-band token construction, it is **astrophysically invalid**.

Multi-band ZTF catalog lightcurves **must be associated by celestial coordinate cross-matching ($\Delta \theta \le 1.5''$)**, never by manipulating catalog OID digits.

---

### Finding 2: Classification Provenance & Label Separation

This audit strictly distinguishes between three tiers of object descriptions:

1. **Authoritative Astrophysical Classification**:
   - Must originate from a peer-reviewed survey catalog with spectroscopic verification (e.g. ZTF Bright Transient Survey BTS, IAU Transient Name Server TNS, Gaia DR3 Vari-Classifier, Simbad).
   - **Audit Result**: **0 out of 10 objects** possess an authoritative astrophysical classification.
2. **Catalog / Source Description**:
   - Originates from the data archive (e.g. "IRSA Lightcurve API Tutorial Sample Object").
   - `686103400034440` and `686103400067717` were sample IDs taken directly from the IRSA API documentation tutorial. The remaining OIDs were synthetic counter selections from Field 686.
3. **Pilot Manual Description**:
   - Labels such as `"Variable Star (zg, zr, zi) in Field 686"`, `"High-Amplitude Variable / Flare"`, and `"Faint Detection-Limit Source"` were **manually assigned by the pilot script** based on heuristic inspection of magnitude variations. They do NOT represent physical truth.

**Scientific Consequence:**
No object in the pilot can be used for classification evaluation or anomaly benchmarking. Every object must be marked `benchmark_ready = False`.

---

### Finding 3: Validity of Peak-Based Transient Windowing

The readiness audit proposed a transient windowing rule:
$$t \in [t_{\text{peak}} - 20.0\,\text{days}, \ t_{\text{peak}} + 60.0\,\text{days}]$$
where $t_{\text{peak}} = \arg\max(F)$.

**Audit Evaluation:**
- **For Explosive Transients (Supernovae, Fast Transients, Flares)**: Peak-based windowing is physically grounded. It isolates the physical explosion from years of archival non-detections.
- **For Static Stars and Long-Period Variables (All 10 Pilot Objects)**: The pilot objects are static or slowly modulating field stars observed continuously over 6 years (2,743 days). For these sources, the maximum flux point is simply an arbitrary positive noise spike or seasonal atmospheric fluctuation. Selecting an 80-day window around this point extracts an arbitrary sub-sample of observations that has no physical meaning as an "eruption."
- In `ZTF_FLARE_06`, the window selector correctly issued a diagnostic warning: `"Insufficient post-peak coverage; transient decay may be unobserved"`.
- In `ZTF_SRC_04` and `ZTF_FLARE_06`, the window captured only 5–6 observations, forcing the sequence constructor to zero-pad 44–45 tokens.

---

### Finding 4: Preprocessing Heuristics vs. Physical Validation

The user explicitly instructed not to label preprocessing transformations as scientifically validated without proof:

1. **The $\times 1.5$ Normalization Scaling**:
   $$F_{\text{norm}} = \left( \frac{F}{F_{\text{peak}}} \right) \times 1.5$$
   - **Classification**: **Engineering Compatibility Heuristic**.
   - **Rationale**: This scaling was introduced purely to map real flux values into the $[0.0, 1.5]$ numerical range so they align with the pre-existing synthetic weights of `LightCurveEncoder` (where synthetic Type Ia SNe peaked at $\sim 1.0 - 1.2$). It is not an astrophysically calibrated physical flux or distance-corrected absolute luminosity.
2. **The 50-Token Quantile Subsampling**:
   - **Classification**: **Baseline Sequence-Compression Heuristic**.
   - **Rationale**: When an object has $> 50$ binned observations, sampling 50 points along uniform quantiles retains the temporal bounds, but it discards photometric information and does not guarantee optimal phase coverage for periodic variables or rapid rise capture for fast transients.
3. **Negative Flux**:
   - In static catalog magnitudes ($m$), flux is strictly positive ($F = 10^{-0.4(m - zp)}$). No negative values were present in the 10 pilot objects ($0.0\%$).
   - Negative fluxes occur exclusively in difference-imaging forced photometry (`forcediffimflux`), which was not queried in this catalog-based pilot.

---

### Finding 5: Data Leakage & Selection Bias Evaluation

- **Label Leakage**: **None**. Because no ground-truth labels were fed into the encoder or used to guide preprocessing, label leakage was zero.
- **Selection Bias**: **High**. Objects were selected by arbitrary OID sampling in Field 686. They represent an unvetted sample of field stars rather than a scientifically controlled astronomical sample.
- **Cross-Set Contamination**: **None** (isolated pilot directory).

---

## 4. Final Recommendation

The user requested a definitive recommendation between:
1. **EXPAND TO 50 LABELED OBJECTS**
2. **FIX PROVENANCE FIRST**
3. **REVISE PREPROCESSING FIRST**

### Definitive Recommendation: **2. FIX PROVENANCE FIRST**

#### Scientific & Engineering Justification:
- **Why NOT Option 1 (Do NOT Expand Yet)**: Expanding to 50 objects using the current OID-based retrieval would merely replicate the spatial cross-matching failure across 50 objects, merging unrelated stars and polluting the benchmark with unclassified sources.
- **Why Option 2 (FIX PROVENANCE FIRST)**:
  1. We must retrieve real ZTF transients using **verified coordinates and IAU/ZTF identifiers from authoritative catalogs** (ZTF Bright Transient Survey BTS and IAU Transient Name Server TNS for SNe/SLSNe/TDEs, and the ZTF Periodic Variable Star Catalog for known variables).
  2. Multi-band lightcurves must be aggregated using **spatial cone search cross-matching ($\Delta \theta \le 1.5''$)**, ensuring that $g$, $r$, and $i$ measurements belong to the exact same astrophysical coordinate.
  3. Preprocessing code (`src/data/real_ztf_preprocessing.py`) is already functionally solid (passes all tests, handles padding/subsampling/binning/encoder forward passes cleanly). The bottleneck is not the preprocessing mechanics—it is the **data source provenance and cross-matching protocol**.

---

## 5. Test Suite Verification
- Production code: **Zero lines modified**.
- Model architecture & weights: **Zero changes**.
- Tests executed:
  `python -m pytest -v`: **33 passed, 0 failed** in 14.82s.
- All 25 regression tests and 8 new preprocessing tests pass cleanly.
