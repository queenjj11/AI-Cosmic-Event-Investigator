# ACEI Investigation Report: CAND_TDE_001

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_TDE_001`
- **Benchmark Class**: `TDE`
- **Coordinates**: RA = 105.827667 deg, Dec = 23.029083 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 1019
- **Pipeline Execution Time**: 0.010 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3230
- **Valid Sequence Tokens**: 27 / 50
- **Padding Fraction**: 46.0%
- **Active Passbands**: g, r, i
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 960 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 3 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 2746.2740 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.5000 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.6246 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 1.1072 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.2614 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.0547 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 0.8754 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 0.1070 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 3.0190 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | 0.9315 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 1.3990 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 0.0000 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 10.9840 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 27 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.4600 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 8, 'r': 18, 'i': 1}
- **Missing Photometric Bands**: []

## 4. Scientific Anomaly Assessment
- **Evaluation Status**: `NOT_VALIDATED_FOR_REAL_ZTF`
- **Anomaly Score**: N/A (Unvalidated)
- **Anomaly Trigger Flag**: N/A
- **Detector Architecture**: ACEI Multimodal Anomaly Ensemble
- **Domain Status**: NOT_VALIDATED_FOR_REAL_ZTF

> [!WARNING]
> **Scientific Boundary**: The production anomaly detector is NOT validated for real ZTF light curves. Scores are suppressed to prevent scientific overclaiming. Light-curve-first downstream reasoning proceeds using non-learned physical characterization.

## 5. Candidate Transient Hypotheses

### Rank 1: Tidal Disruption Event (TDE) (Calibrated Prob: 0.620)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Light-curve decay slope and prolonged blue emission match stellar disruption accretion fallback.
- **Supporting Evidence**: transient_taxonomy.json_LRN, lrn_astrophysics.md
- **Contradictory / Missing Evidence**: Centroid offset relative to host galaxy nucleus requires confirmation
- **Distinguishing Criteria**: UV and soft X-ray follow-up confirming non-cooling thermal emission.

### Rank 2: Superluminous Supernova (SLSN-I) (Calibrated Prob: 0.230)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Prolonged high-luminosity optical emission.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Typically shows early-time UV cooling and broad optical line absorption
- **Distinguishing Criteria**: Optical spectroscopy to differentiate metal-line absorption from featureless blue continuum.

### Rank 3: Active Galactic Nucleus (AGN) Outburst (Calibrated Prob: 0.150)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Accretion disk instability in a pre-existing active galactic nucleus.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Lacks pre-existing stochastic AGN variability in archival data
- **Distinguishing Criteria**: Multi-year archival light curve check for historical stochastic variability.

## 6. Literature & Knowledge Base Evidence

#### [transient_taxonomy.json_LRN] Catalog: Luminous Red Nova *(Relevance: 0.353)*
> CATALOG REFERENCE: Luminous Red Nova (LRN) Physical Mechanism: Stellar merger in binary system accompanied by common-envelope dynamic ejection Peak Absolute Magnitude Range: [-14.5, -11.5] Rise Time: [25, 60] days Dec...

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.350)*
> ### Diagnostic Photometric Signatures 1. **Light Curve Morphology:** Frequently exhibit an initial blue precursor peak followed weeks to months later by a broader, red secondary peak. 2. **Color Evolution:** Rapid red...

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.226)*
> ### Abstract & Overview Luminous Red Novae (LRNe) are intermediate-luminosity optical transients ($M_V \approx -11$ to $-15$) characterized by slow expansion velocities ($v \approx 200 - 800\text{ km/s}$), prominent r...

## 7. Recommended Follow-Up Observations

- **Priority 1 (HIGH)**: Target-of-opportunity optical spectroscopy (3500-9000 A)
  - *Information Gap*: Zero spectroscopic observations available; physical expansion velocities and composition unconstrained
  - *Scientific Rationale*: Disambiguate competing scientific hypotheses ('Tidal Disruption Event (TDE)') via spectral line identification and redshift determination.
  - *Target Band*: `optical_spectrum`
- **Priority 2 (LOW)**: Extend late-time photometric monitoring (t > 40 days)
  - *Information Gap*: Post-peak light curve decline rate unconstrained at late epochs
  - *Scientific Rationale*: Constrain radioactive decay tail slope (Co-56 vs circumstellar interaction) or confirm quiescent return.
  - *Target Band*: `r`

## 8. Pipeline Limitations & Methodological Constraints

- Anomaly scores are suppressed: multimodal ensemble was trained on synthetic data with co-temporal difference images.
- Real ZTF input is processed through the Light-Curve-First pathway without image-modality fusion.
- Hypotheses reflect physical heuristics and retrieved literature, not automated ML classification.
- Redshift, host galaxy associations, and absolute luminosities are unconstrained without external catalog queries.
- Production checkpoint remained strictly frozen (no retraining, no fine-tuning, no threshold alteration).
