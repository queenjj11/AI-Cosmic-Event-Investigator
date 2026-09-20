# ACEI Investigation Report: CAND_SLSN_001

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_SLSN_001`
- **Benchmark Class**: `SLSN-II`
- **Coordinates**: RA = 143.865875 deg, Dec = 52.632222 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 24
- **Pipeline Execution Time**: 0.003 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3224
- **Valid Sequence Tokens**: 11 / 50
- **Padding Fraction**: 78.0%
- **Active Passbands**: g, r
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 21 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 2 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 380.9970 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.5000 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.7023 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 1.1121 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.2682 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.1059 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 0.7977 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 10.0610 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 27.0020 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | 0.4571 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 4.9940 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 0.0960 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 27.0020 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 11 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.7800 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 5, 'r': 6, 'i': 0}
- **Missing Photometric Bands**: ['i']

## 4. Scientific Anomaly Assessment
- **Evaluation Status**: `NOT_VALIDATED_FOR_REAL_ZTF`
- **Anomaly Score**: N/A (Unvalidated)
- **Anomaly Trigger Flag**: N/A
- **Detector Architecture**: ACEI Multimodal Anomaly Ensemble
- **Domain Status**: NOT_VALIDATED_FOR_REAL_ZTF

> [!WARNING]
> **Scientific Boundary**: The production anomaly detector is NOT validated for real ZTF light curves. Scores are suppressed to prevent scientific overclaiming. Light-curve-first downstream reasoning proceeds using non-learned physical characterization.

## 5. Candidate Transient Hypotheses

### Rank 1: Superluminous Supernova (SLSN) / Magnetar Spin-Down (Calibrated Prob: 0.650)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Broad multi-week light curve (half-max decline ~27.0d) and high amplitude require central engine power.
- **Supporting Evidence**: lrn_astrophysics.md, transient_taxonomy.json_Variable_Star
- **Contradictory / Missing Evidence**: Rest-frame UV color temperature and host galaxy offset unmeasured
- **Distinguishing Criteria**: Target-of-Opportunity UV/optical spectroscopy to identify broad O II absorption multiplets.

### Rank 2: Tidal Disruption Event (TDE) (Calibrated Prob: 0.220)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Luminous blue transient with prolonged decay timescale.
- **Supporting Evidence**: tde_supermassive_blackhole.md
- **Contradictory / Missing Evidence**: Requires nuclear alignment (<0.1 arcsec) with supermassive black hole host
- **Distinguishing Criteria**: Sub-arcsecond astrometric centroiding and soft X-ray detection.

### Rank 3: Interacting Core-Collapse Supernova (Type IIn) (Calibrated Prob: 0.130)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Shock interaction with circumstellar medium extending optical luminosity.
- **Supporting Evidence**: lrn_astrophysics.md
- **Contradictory / Missing Evidence**: Lacks high-resolution spectroscopy confirming dense circumstellar shock interaction
- **Distinguishing Criteria**: Spectroscopic detection of narrow Balmer emission atop electron-scattering wings.

## 6. Literature & Knowledge Base Evidence

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.245)*
> ### Diagnostic Photometric Signatures 1. **Light Curve Morphology:** Frequently exhibit an initial blue precursor peak followed weeks to months later by a broader, red secondary peak. 2. **Color Evolution:** Rapid red...

#### [transient_taxonomy.json_Variable_Star] Catalog: Periodic Variable Star (Cepheid / RR Lyrae) *(Relevance: 0.226)*
> CATALOG REFERENCE: Periodic Variable Star (Cepheid / RR Lyrae) (Variable_Star) Physical Mechanism: Stellar pulsations driven by kappa-mechanism in helium ionization zone Peak Absolute Magnitude Range: [-4.0, 1.0] Rise...

#### [tde_supermassive_blackhole.md] Photometric and Spectroscopic Evolution of Tidal Disruption Events *(Relevance: 0.186)*
> ### Diagnostic Photometric Signatures 1. **Host Nuclear Position:** Coincident with the precise photometric centroid of the host galaxy nucleus (offset $< 0.1$ arcseconds / $< 0.5$ pixels). 2. **Canonical $t^{-5/3}$ F...

## 7. Recommended Follow-Up Observations

- **Priority 1 (NORMAL)**: Obtain calibrated i-band photometry
  - *Information Gap*: No observations in passband 'i'
  - *Scientific Rationale*: Missing i-band measurements prevent color temperature and extinction determinations.
  - *Target Band*: `i`
- **Priority 2 (NORMAL)**: Increase temporal observation cadence to daily sampling
  - *Information Gap*: Sparse temporal cadence during key evolutionary phases
  - *Scientific Rationale*: Current median cadence of 5.0 days undersamples fast light-curve inflections and peak.
  - *Target Band*: `r`
- **Priority 3 (HIGH)**: Target-of-opportunity optical spectroscopy (3500-9000 A)
  - *Information Gap*: Zero spectroscopic observations available; physical expansion velocities and composition unconstrained
  - *Scientific Rationale*: Disambiguate competing scientific hypotheses ('Superluminous Supernova (SLSN) / Magnetar Spin-Down') via spectral line identification and redshift determination.
  - *Target Band*: `optical_spectrum`
- **Priority 4 (LOW)**: Extend late-time photometric monitoring (t > 40 days)
  - *Information Gap*: Post-peak light curve decline rate unconstrained at late epochs
  - *Scientific Rationale*: Constrain radioactive decay tail slope (Co-56 vs circumstellar interaction) or confirm quiescent return.
  - *Target Band*: `r`

## 8. Pipeline Limitations & Methodological Constraints

- Anomaly scores are suppressed: multimodal ensemble was trained on synthetic data with co-temporal difference images.
- Real ZTF input is processed through the Light-Curve-First pathway without image-modality fusion.
- Hypotheses reflect physical heuristics and retrieved literature, not automated ML classification.
- Redshift, host galaxy associations, and absolute luminosities are unconstrained without external catalog queries.
- Production checkpoint remained strictly frozen (no retraining, no fine-tuning, no threshold alteration).
