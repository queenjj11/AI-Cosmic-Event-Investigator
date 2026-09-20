# ACEI Real Astronomical Data Readiness Audit

**Document Version:** 1.0  
**Phase:** Phase 3 — Transition to Real Astronomical Observations  
**Target Survey:** Zwicky Transient Facility (ZTF) via NASA/IPAC IRSA & Alert Streams  
**Status:** Audit & Architecture Design (No Training / No Code Modification)  

---

## 1. Current ACEI Input Contract

The current ACEI pipeline processes astrophysical events through a modular pipeline from raw data structures to deep neural feature extraction and multi-detector anomaly scoring:
$$\text{Raw Lightcurve} \longrightarrow \text{LightCurveLoader} \longrightarrow \text{Preprocessor} \longrightarrow \text{Tensors} \longrightarrow \text{LightcurveEncoder} \longrightarrow \text{CrossAttentionFusion} \longrightarrow \text{Anomaly Ensemble}$$

### Detailed Subsystem Contracts

| Dimension | Pipeline Layer | Internal Specification & Contract |
| :--- | :--- | :--- |
| **A. Expected Columns** | `Observation` / `LightCurve` | `time: float`, `band: str`, `flux: float`, `flux_err: float`, `mag: Optional[float]`, `mag_err: Optional[float]` |
| **B. Expected Units** | `Observation` | `time`: relative days ($t - t_0 \ge 0.0$); `flux`: unitless / relative flux scaled to $[0.0, 5.0]$; `flux_err`: 1-sigma uncertainty on flux ($0.01 - 0.20$); `band`: optical identifier |
| **C. Tensor Shapes** | `AstronomicalDataset` | `lightcurve`: `(B, 50, 4)` containing `[rel_time, flux, flux_err, band_idx]`; `mask`: `(B, 50)` boolean mask (`True` = valid observation, `False` = padding); `image`: `(B, 3, 64, 64)`; `label`: `(B,)` int64 |
| **D. Band Encoding** | `Preprocessor` | `BAND_TO_INDEX = {"g": 0, "r": 1, "i": 2, "V": 3, "NIR": 4}`. Mapped via `nn.Embedding(5, 16)` in `LightCurveEncoder`. |
| **E. Time Representation** | `LightCurveEncoder` | Continuous relative elapsed time $t_{\text{rel}} = t - t_0$. Input to `Time2Vec(output_dim=32)` which computes linear projection $\omega_0 t + \phi_0$ and 31 periodic components $\sin(\omega_i t + \phi_i)$. |
| **F. Flux Representation** | `LightCurveEncoder` | Linear positive flux values strictly within standard network activation range (synthetic transients peak around $1.0 - 4.5$). Clamped to $\ge 0.01$. |
| **G. Error Representation** | `LightCurveEncoder` | Linear 1-sigma Gaussian photometric uncertainty. Joined with flux into a 2D vector `[flux, flux_err]` and projected via `nn.Linear(2, 32)`. |
| **H. Missing Values** | `Preprocessor` | Missing bands or missing epochs are not explicitly interpolated in the default padded tensor representation; missing positions are zero-padded up to sequence length 50. |
| **I. Chronological Sorting** | `LightCurve` | Requires strict monotonic chronological ordering: `lc.sort_chronologically()` using `o.time`. `Preprocessor` enforces `rel_time[i] >= 0.0`. |
| **J. Normalization** | Pipeline Entry | Lightcurve features assume calibrated relative amplitudes where Type Ia SNe have peak $\approx 1.0$, SLSN peak $\approx 4.5$, and quiescent baselines $\approx 0.0$. No dynamic per-object scaler currently exists in `Preprocessor`. |
| **K. Sequence Length** | `Preprocessor` | Fixed `max_length = 50`. Objects with $>50$ observations are truncated blindly at the first 50 observations: `obs_count = min(len(lc.observations), 50)`. |
| **L. Padding / Masking** | `LightCurveEncoder` | Padded tokens have feature vector `[0.0, 0.0, 0.0, 0.0]` with `mask = False`. Transformer uses PyTorch convention `src_key_padding_mask = ~mask` (True = ignored). Pooled via masked average pooling: $\bar{z} = \sum (z_i \cdot m_i) / \sum m_i$. |
| **M. Cadence Assumptions** | Synthetic Generator | Assumes regular random sampling over $-10$ to $+45$ days with $65\%$ probability of observation per passband per visit, yielding dense, multi-band, contemporaneous observations. |
| **N. Ground-Truth Isolation**| `AstronomicalEvent` | `true_label: Optional[str]` and `is_anomaly: bool` exist on the data dataclass but are strictly isolated from model inference (`investigate_event` only consumes `lightcurve` and `image`). |

---

## 2. Real ZTF Data Contract

Real ZTF observations are disseminated through two primary NASA/IPAC IRSA and alert mechanisms:
1. **ZTF Alert Packets (Avro / JSON via Brokers like ALeRCE / Fink / Lasair)**: Real-time difference-imaging alerts.
2. **IRSA ZTF Lightcurve API (`nph_light_curves`)**: Archival calibrated light curves across Data Releases (e.g., DR18–DR24).
3. **ZTF Forced Photometry Service (ZFP)**: Point-spread-function difference flux evaluated at fixed celestial coordinates over the full baseline.

### Primary ZTF Field Definitions & Observational Realities

```
Real ZTF Data Flow:
[CCD Exposure] -> [Difference Image Analysis (ZOGY)] -> [PSF Photometry] -> [Catflags & DRB Scoring] -> [IRSA Archive / Kafka Alert]
```

- **`oid` / `objectId`**: Archival object identifiers are 64-bit integers (e.g. `686103400067717`) per spatial match, whereas alert stream names are alphanumeric strings (e.g. `ZTF20acvppvo`).
- **`mjd` / `jd`**: Modified Julian Date ($MJD = JD - 2400000.5$). Timestamps are absolute astronomical epochs spanning years ($MJD \sim 58178$ to $60400+$), with large seasonal gaps (months when fields are behind the Sun) and daytime gaps.
- **`mag` / `magpsf`**: Calibrated PSF magnitude on the AB system. Available only for statistically significant detections ($S/N \gtrsim 5$). Non-detections have no valid `mag` value, only a 5-sigma limiting magnitude (`diffmaglim`).
- **`magerr` / `sigmapsf`**: 1-sigma uncertainty in magnitude. Explodes as $S/N \to 5$ ($\sigma_m \sim 0.2 - 0.5$ mag).
- **`forcediffimflux` & `forcediffimfluxunc`**: Linear difference flux and 1-sigma uncertainty in counts from forced PSF photometry. Can be negative due to background subtraction noise when the transient has faded or before explosion.
- **`filtercode` / `fid`**: Filter identifier: `fid = 1` or `zg` (ZTF $g$, $\lambda_{\text{eff}} \approx 472$ nm), `fid = 2` or `zr` (ZTF $r$, $\lambda_{\text{eff}} \approx 634$ nm), `fid = 3` or `zi` (ZTF $i$, $\lambda_{\text{eff}} \approx 788$ nm).
- **`catflags`**: 16-bit catalog quality bitmask. `catflags = 0` indicates pristine data. Critical bits:
  - Bit 0 (1): Source near edge of image/bad pixel.
  - Bit 1 (2): Source saturated.
  - Bit 2 (4): Bad pixel in PSF aperture.
  - Bit 3 (8): Source blended with another object.
  - Bit 15 (32768): Cloud cover, moon contamination, or poor atmospheric transparency.
  - Setting `BAD_CATFLAGS_MASK = 32768` filters cloudy epochs. A strict clean mask is `catflags == 0` or mask `65535`.
- **`drb` / `rb`**: Deep-learning Real-Bogus score ($\in [0.0, 1.0]$). Distinguishes genuine astrophysical transients from subtraction artifacts, optical reflections, and cosmic rays. Standard threshold: $\text{drb} \ge 0.70$.

---

## 3. Photometric Representation Analysis: Magnitude vs. Flux vs. Normalized Flux

ACEI must explicitly establish whether to feed magnitude, calibrated physical flux ($\mu\text{Jy}$), or normalized flux into the neural pipeline.

| Criterion | 1. Magnitude ($m$) | 2. Calibrated Physical Flux ($F_{\mu\text{Jy}}$) | 3. Normalized Relative Flux ($F / F_{\text{scale}}$) |
| :--- | :--- | :--- | :--- |
| **Mathematical Formulation** | $m = -2.5 \log_{10}(F / F_0)$ | $F_{\mu\text{Jy}} = 10^{0.4(23.9 - m)}$ or $10^{0.4(23.9 - zp)} F_{\text{counts}}$ | $F_{\text{norm}} = \frac{F - F_{\text{base}}}{\max(F) - F_{\text{base}}}$ or $F / \text{Scale}$ |
| **Negative Values** | **Fails completely**. Cannot take logarithm of negative flux fluctuations common in difference imaging. | **Supported**. Correctly represents zero/negative baseline noise ($0 \pm \sigma$). | **Supported**. Preserves zero and negative noise fluctuations without numerical singularity. |
| **Non-Detections / Limits** | Discards pre-explosion epochs or requires ad-hoc magnitude imputation ($m = 21.0$). | Retains exact non-detection measurements ($F_{\mu\text{Jy}} \approx 0 \pm 3 \mu\text{Jy}$). | Retains exact non-detection measurements ($F_{\text{norm}} \approx 0.0 \pm \sigma_{\text{norm}}$). |
| **Error Distribution** | Severe skewness: $\sigma_m \approx 1.0857 \frac{\sigma_F}{F} \to \infty$ as $F \to 0$. Highly non-Gaussian. | Symmetric Gaussian uncertainty: $\sigma_F$ is constant across flux levels for background-limited observations. | Symmetric Gaussian uncertainty: $\sigma_{\text{norm}} = \sigma_F / \text{Scale}$. |
| **Neural Dynamic Range** | Dynamic range $[14.0, 22.0]$ (inverted: brighter is smaller). | Dynamic range spans 5 orders of magnitude ($1 \mu\text{Jy}$ to $10^5 \mu\text{Jy}$). Blows up `Linear(2, 32)`. | Dynamic range matches $[0.0, 5.0]$ exactly. Perfectly aligns with pretrained weights. |
| **Distance Invariance** | Brightness depends on luminosity distance ($d_L$). SNe at different redshifts look completely different. | Strongly redshift-dependent. Anomaly detector would flag nearby SNe as outliers solely due to apparent brightness. | **Distance-invariant lightcurve morphology**. Focuses anomaly detection on physical evolution and color, not distance. |

### Architectural Decision:
**ACEI MUST OPERATE ON NORMALIZED RELATIVE FLUX.**
Operating on raw magnitudes is scientifically unacceptable because it corrupts error distributions, discards pre-explosion upper limits, and fails on negative difference flux. Operating on raw microJanskys without scaling would saturate the existing neural layers. Normalized flux preserves linear Gaussian photometric error propagation while matching the $[0.0, 5.0]$ numerical regime expected by `LightCurveEncoder`.

---

## 4. Synthetic Assumptions Audit

Every assumption identified where the current codebase relies on synthetic artifacts:

1. **Taxonomic Completeness**: Synthetic generation assumes the universe consists solely of 4 known classes (`SN_Ia`, `SN_II`, `Stellar_Flare`, `Variable_Star`) and 3 anomaly classes (`LRN`, `SLSN`, `TDE`). Real surveys contain AGN flares, quasi-periodic eruptions, cataclysmic variables, asteroid reflections, satellite glints, and instrumental ghosts.
2. **Deterministic Discovery at $t=0$**: Current `Preprocessor` sets $t_0 = \text{observations}[0].\text{time}$ and takes the first 50 observations. For real ZTF light curves, the first 50 observations are often quiescent archival points from 2 years before the supernova exploded! The explosion would be truncated completely.
3. **Contemporaneous Multi-Band Cadence**: Synthetic data samples $g, r, i$ within hours of each other ($65\%$ probability). Real ZTF often observes $g$ and $r$ on alternating nights or separates them by several days, with $i$-band observed only on select high-cadence fields.
4. **Clean Baseline & Gaussian Additive Noise**: Synthetic light curves have zero true background flux and purely additive Gaussian noise ($\sigma = 0.04$). Real difference imaging has systematic residual dipole artifacts, host galaxy subtraction residuals, seeing-dependent PSF variations, and weather noise.
5. **Absence of Negative Flux**: The synthetic generator explicitly clamps flux with `max(0.01, val + noise)`. Real difference imaging produces negative flux when sky fluctuations or template subtraction over-subtracts background.
6. **Synthetic Image-Lightcurve Synchronization**: Synthetic cutouts are generated on-the-fly with simulated PSFs matched to the exact synthetic class label. Real ZTF cutouts are discrete FITS images captured at specific exposure times, each with real seeing conditions, sky background, and host galaxy morphology.
7. **Sequence Length Fixed at 50**: Current transformer expects at most 50 observations. Real transient light curves frequently range from 15 observations (fast transients) to 1,500+ observations (long-lived variables / AGN).

---

## 5. Compatibility Gaps

```
[Real ZTF Catalog / Alert]
  |-- mjd (58000+)  --> [GAP 1: MJD scale vs relative days] 
  |-- magpsf / sigmapsf --> [GAP 2: Mag-to-flux conversion & non-detections]
  |-- 500+ epochs  --> [GAP 3: Blind truncation cuts off transient peak]
  |-- catflags != 0 --> [GAP 4: Artifacts, clouds, moon pollution]
  |-- FITS cutout  --> [GAP 5: Gzip FITS decoding vs synthetic 3-ch numpy]
```

1. **Windowing & Peak Alignment Gap**:
   - *Current Code*: Blindly takes `observations[:50]`.
   - *Real Data Requirement*: An automated burst-detection / transient-windowing algorithm must identify the significant photometric excursion (e.g. using signal-to-noise peak or rolling median) and extract a $[-20\text{d}, +60\text{d}]$ window of $\le 50$ points around the event.
2. **Quality Filtering Gap**:
   - *Current Code*: `ZTFLoader.parse_alert_dict` checks `if mag is not None and not np.isnan(mag)`. It does NOT inspect `catflags`, `drb`, or `fwhm`.
   - *Real Data Requirement*: Must reject individual epochs with bad `catflags` ($\text{mask} = 32768$) and reject bogus candidates with $\text{drb} < 0.70$.
3. **Difference Flux vs Apparent Magnitude**:
   - *Current Code*: Uses formula $F = 10^{-0.4(m - 27.5)}$.
   - *Real Data Requirement*: Must handle difference fluxes, negative fluxes, and compute proper error propagation $\sigma_F = \frac{\ln(10)}{2.5} F \sigma_m$ with a robust noise floor.
4. **Missing Modality Handling (Image Cutouts)**:
   - *Current Code*: If `event.image` is None, `AstronomicalDataset` generates a *synthetic* cutout based on `event.true_label`! In real deployment, `true_label` is unknown at inference time, so generating a synthetic image conditioned on `true_label` would be an impermissible ground-truth leak.
   - *Real Data Requirement*: The system must support lightcurve-only inference or provide a neutral blank/template image cutout when real FITS cutouts are absent.

---

## 6. Required Preprocessing Pipeline

To safely pass real ZTF data into ACEI without architectural changes:

1. **Alert / Table Ingestion**:
   - Parse ZTF IRSA IPAC / CSV tables or alert JSON.
   - Extract `mjd`, `filtercode` / `fid`, `mag` / `forcediffimflux`, `magerr` / `forcediffimfluxunc`, `catflags`.
2. **Quality Filtering Stage**:
   - Discard all epochs where `(catflags & 32768) != 0` (or `catflags != 0`).
   - Discard alerts with $\text{drb} < 0.70$ or seeing $\text{FWHM} > 5.0$ pixels.
   - Discard observations with non-physical error estimates ($\sigma_m \le 0$ or $\sigma_m > 1.5$).
3. **Transient Detection & Temporal Windowing**:
   - Identify candidate outburst epoch $t_{\text{peak}} = \arg\max(S/N)$.
   - Extract observations in physical temporal window $t \in [t_{\text{peak}} - 20\,\text{days}, t_{\text{peak}} + 60\,\text{days}]$.
   - If points $> 50$, perform density-preserving sub-sampling or multi-band binning (e.g. 1 observation per band per night).
4. **Photometric Calibration & Normalization**:
   - Convert magnitudes to linear flux: $F = 10^{-0.4(m - zp)}$.
   - Propagate uncertainties: $\sigma_F = \frac{\ln(10)}{2.5} F \sigma_m$.
   - Scale flux so that median peak flux of known transients maps to $\sim 1.0 - 1.5$ (preserving the $[0.0, 5.0]$ range expected by the transformer).
5. **Relative Time Shifting**:
   - Shift time so the first windowed observation is $t = 0.0$ days: $t_{\text{rel}} = t - t_{\text{window\_start}}$.
6. **Chronological Sort & Tensor Formatting**:
   - Ensure strictly sorted ascending $t_{\text{rel}}$.
   - Format into `(50, 4)` tensor with corresponding boolean mask.

---

## 7. Required Code Changes (Future Phase)

When transitioning to real data execution in subsequent tasks, the following isolated modules will require enhancements:

1. **`src/data/ztf_loader.py`**:
   - Add support for IRSA IPAC / CSV table parsing in addition to alert JSON dictionaries.
   - Implement bitmask filtering on `catflags`.
   - Implement real-bogus filtering on `drb`.
2. **`src/data/preprocessing.py`**:
   - Add `TransientWindowSelector`: detects transient peak and extracts $[-20\text{d}, +60\text{d}]$ window rather than blindly taking the first 50 observations.
   - Add multi-band night-binning when observation count exceeds 50.
   - Add robust flux normalization function.
3. **`src/data/dataset_builder.py`**:
   - Remove fallback synthetic image generation conditioned on `event.true_label`! Replace with a neutral host-galaxy template or zero-filled image when cutouts are unavailable.
4. **`src/data/schema.py`**:
   - Add metadata fields to `Observation` for quality tracking (`catflags`, `drb`, `chi`, `sharp`).

---

## 8. Changes That Are NOT Required

To maintain scientific integrity and prevent scope creep:

1. **NO changes to `LightCurveEncoder` architecture**:
   The existing `Time2Vec(32)` + `Embedding(5, 16)` + `Linear(2, 32)` + 3-layer `TransformerEncoder(128)` natively accepts irregular time series and arbitrary sequence lengths up to 50.
2. **NO changes to `MultimodalTransientModel`**:
   Cross-attention fusion and classification heads remain completely valid.
3. **NO changes to Anomaly Detectors**:
   Autoencoder, Mahalanobis distance, and Energy score detectors are modality-agnostic and operate purely on 256-D latent embeddings and logits.
4. **NO changes to Anomaly Threshold ($\theta = 0.65$)**:
   The calibrated decision boundary is maintained.
5. **NO changes to RAG or Active Observation Recommender**:
   The literature retrieval and Bayesian information-gain recommendation logic are driven entirely by extracted physical features and class probabilities.

---

## 9. Comprehensive Risk Assessment

| Risk Identifier | Failure Mode | Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **R1. Pre-Explosion Truncation** | Real ZTF light curves have 2+ years of non-detections. Taking first 50 points yields zero transient signal. | **Critical** | Implement peak-centered windowing ($t_{\text{peak}} - 20\text{d}$ to $t_{\text{peak}} + 60\text{d}$) before sequence construction. |
| **R2. Activation Saturation** | Passing unnormalized $\mu\text{Jy}$ ($10^4$) into `nn.Linear(2, 32)` saturates GELU and LayerNorm. | **Critical** | Strictly normalize fluxes to $[0.0, 5.0]$ matching training scale. |
| **R3. Ground-Truth Image Leak** | `AstronomicalDataset` generates synthetic images using `event.true_label`. | **High** | Replace fallback image generation with neutral zero-cutout or pure lightcurve forward pass. |
| **R4. Subtraction Artifacts** | Real difference images contain bogus dipoles and bad pixel subtraction spikes flagged as "anomalies". | **High** | Enforce strict quality cuts: $\text{drb} \ge 0.70$, $\text{catflags} == 0$, $\text{fwhm} \le 5.0$ px. |
| **R5. Irregular / Sparse Cadence** | High cadence in $g$ and $r$, zero observations in $i$, or 2-week weather gaps confuse Time2Vec. | **Medium** | Time2Vec natively encodes continuous time; verify attention mask handles sparse band coverage gracefully. |

---

## 10. Recommended Implementation Order

When authorized to begin the real data transition:

1. **Step 1: Schema & Parsing Utilities**:
   Upgrade `ZTFLoader` to parse IRSA IPAC / CSV files with `catflags` and `drb` filtering.
2. **Step 2: Neutral Cutout Decoupling**:
   Decouple `AstronomicalDataset` from `true_label` image generation to ensure zero ground-truth leakage.
3. **Step 3: Small Real-Data Pilot ($N=50$)**:
   Retrieve a small, curated pilot of confirmed ZTF objects to verify ingestion, parsing, windowing, and tensor creation end-to-end.
4. **Step 4: Synthetic-Trained Model Inference on Real Pilot**:
   Run zero-shot inference with the existing production baseline model on the preprocessed real pilot without retraining to measure baseline transferability.
5. **Step 5: Full Real-Data Benchmark Construction**:
   Build the scientifically verified object-level real benchmark with strict spectroscopic label provenance.
