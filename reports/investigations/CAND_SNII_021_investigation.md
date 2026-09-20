# ACEI Investigation Report: CAND_SNII_021

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_SNII_021`
- **Benchmark Class**: `SN IIP`
- **Coordinates**: RA = 240.491375 deg, Dec = 29.414889 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 2205
- **Pipeline Execution Time**: 0.096 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3244
- **Valid Sequence Tokens**: 35 / 50
- **Padding Fraction**: 30.0%
- **Active Passbands**: g, r, i
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 2131 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 3 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 2753.7310 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.5000 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.7915 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 1.2274 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.2017 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.0400 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 0.7085 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 19.8540 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 1.0110 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | -0.9031 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 1.0390 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 0.0000 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 5.9400 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 35 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.3000 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 14, 'r': 12, 'i': 9}
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

### Rank 1: Type Ia Supernova (Thermonuclear) (Calibrated Prob: 0.550)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Observed rise proxy (19.9d) and asymmetric decline conform to canonical Ni-56 powered thermonuclear transients.
- **Supporting Evidence**: lrn_astrophysics.md, transient_taxonomy.json_Variable_Star
- **Contradictory / Missing Evidence**: Exact rest-frame Si II 6355 A absorption unmeasured
- **Distinguishing Criteria**: Phase-matched template fitting to B/V band light curves or optical spectrum at maximum light.

### Rank 2: Core-Collapse Supernova (Type II) (Calibrated Prob: 0.300)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Optical rise and decline compatible with stripped-envelope or plateau core-collapse events.
- **Supporting Evidence**: lrn_astrophysics.md
- **Contradictory / Missing Evidence**: Plateau duration and hydrogen envelope signatures require multi-band color monitoring
- **Distinguishing Criteria**: Low-resolution optical spectroscopy confirming P-Cygni H-alpha absorption/emission.

### Rank 3: Unclassified Transient / Variable (Calibrated Prob: 0.150)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Transient duration broadly consistent with expanding supernova ejecta or long-period variable.
- **Supporting Evidence**: lrn_astrophysics.md
- **Contradictory / Missing Evidence**: Single outburst recorded in evaluation interval
- **Distinguishing Criteria**: Target-of-opportunity optical spectroscopy to identify line features and host redshift.

## 6. Literature & Knowledge Base Evidence

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.245)*
> ### Diagnostic Photometric Signatures 1. **Light Curve Morphology:** Frequently exhibit an initial blue precursor peak followed weeks to months later by a broader, red secondary peak. 2. **Color Evolution:** Rapid red...

#### [transient_taxonomy.json_Variable_Star] Catalog: Periodic Variable Star (Cepheid / RR Lyrae) *(Relevance: 0.226)*
> CATALOG REFERENCE: Periodic Variable Star (Cepheid / RR Lyrae) (Variable_Star) Physical Mechanism: Stellar pulsations driven by kappa-mechanism in helium ionization zone Peak Absolute Magnitude Range: [-4.0, 1.0] Rise...

#### [tde_supermassive_blackhole.md] Photometric and Spectroscopic Evolution of Tidal Disruption Events *(Relevance: 0.186)*
> ### Diagnostic Photometric Signatures 1. **Host Nuclear Position:** Coincident with the precise photometric centroid of the host galaxy nucleus (offset $< 0.1$ arcseconds / $< 0.5$ pixels). 2. **Canonical $t^{-5/3}$ F...

## 7. Recommended Follow-Up Observations

- **Priority 1 (NORMAL)**: Target-of-opportunity optical spectroscopy (3500-9000 A)
  - *Information Gap*: Zero spectroscopic observations available; physical expansion velocities and composition unconstrained
  - *Scientific Rationale*: Disambiguate competing scientific hypotheses ('Type Ia Supernova (Thermonuclear)') via spectral line identification and redshift determination.
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
