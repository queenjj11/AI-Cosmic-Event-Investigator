# ACEI Investigation Report: CAND_SNIa_002

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_SNIa_002`
- **Benchmark Class**: `SN Ia`
- **Coordinates**: RA = 187.281583 deg, Dec = 36.456694 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 1785
- **Pipeline Execution Time**: 0.079 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3190
- **Valid Sequence Tokens**: 34 / 50
- **Padding Fraction**: 32.0%
- **Active Passbands**: g, r, i
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 1628 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 3 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 2650.9530 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.5000 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.2776 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 0.5885 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.3151 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.0117 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 1.2224 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 2.9740 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 0.9660 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | -0.5098 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 0.1430 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 0.0050 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 28.9720 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 34 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.3200 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 16, 'r': 13, 'i': 5}
- **Missing Photometric Bands**: []

## 4. Scientific Anomaly Assessment
- **Evaluation Status**: `REAL_ZTF_EVALUATED_RESEARCH`
- **Anomaly Score**: 0.6161
- **Anomaly Trigger Flag**: False
- **Detector Architecture**: Real-ZTF Light-Curve Anomaly Detector v2 (PCA-Mahalanobis)
- **Domain Status**: REAL_ZTF_EVALUATED_RESEARCH

> [!WARNING]
> **Scientific Boundary**: Evaluated research model operating on real ZTF 128-D light-curve representation space. Image modality mismatch resolved by light-curve-only regularized distance scoring.

## 5. Candidate Transient Hypotheses

### Rank 1: Type Ia Supernova (Thermonuclear) (Calibrated Prob: 0.550)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Observed rise proxy (3.0d) and asymmetric decline conform to canonical Ni-56 powered thermonuclear transients.
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

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.258)*
> ### Diagnostic Photometric Signatures 1. **Light Curve Morphology:** Frequently exhibit an initial blue precursor peak followed weeks to months later by a broader, red secondary peak. 2. **Color Evolution:** Rapid red...

#### [transient_taxonomy.json_Variable_Star] Catalog: Periodic Variable Star (Cepheid / RR Lyrae) *(Relevance: 0.235)*
> CATALOG REFERENCE: Periodic Variable Star (Cepheid / RR Lyrae) (Variable_Star) Physical Mechanism: Stellar pulsations driven by kappa-mechanism in helium ionization zone Peak Absolute Magnitude Range: [-4.0, 1.0] Rise...

#### [transient_taxonomy.json_LRN] Catalog: Luminous Red Nova *(Relevance: 0.229)*
> CATALOG REFERENCE: Luminous Red Nova (LRN) Physical Mechanism: Stellar merger in binary system accompanied by common-envelope dynamic ejection Peak Absolute Magnitude Range: [-14.5, -11.5] Rise Time: [25, 60] days Dec...

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
