# ACEI Investigation Report: CAND_FieldStar_001

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_FieldStar_001`
- **Benchmark Class**: `Unclassified Field Star`
- **Coordinates**: RA = 297.984536 deg, Dec = 29.788558 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 4174
- **Pipeline Execution Time**: 0.035 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3184
- **Valid Sequence Tokens**: 50 / 50
- **Padding Fraction**: 0.0%
- **Active Passbands**: g, r, i
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 3409 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 3 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 2765.6000 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.2018 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.1055 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 0.4503 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.3213 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.0080 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 1.0964 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 8.9450 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 0.1620 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | -0.9645 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 0.9790 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 0.0190 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 6.0170 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 50 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.0000 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 24, 'r': 19, 'i': 7}
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

### Rank 1: Unclassified Field Star / Stellar Variability (Calibrated Prob: 0.580)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Photometric baseline (2765.6d) shows low-amplitude or stellar stochastic variability.
- **Supporting Evidence**: transient_taxonomy.json_Variable_Star
- **Contradictory / Missing Evidence**: Requires multi-epoch baseline to establish definite periodicity or flare cadence
- **Distinguishing Criteria**: Periodogram analysis and Gaia DR3 astrometric parallax / proper motion cross-match.

### Rank 2: Cataclysmic / Eruptive Variable Star (Calibrated Prob: 0.270)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Modest brightening on top of stellar baseline.
- **Supporting Evidence**: lrn_astrophysics.md
- **Contradictory / Missing Evidence**: Outburst amplitude is moderate compared to canonical dwarf novae
- **Distinguishing Criteria**: Optical spectroscopy to determine stellar spectral type and accretion signatures.

### Rank 3: Extragalactic Background Transient (Calibrated Prob: 0.150)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Coincidental alignment with a background extragalactic transient.
- **Supporting Evidence**: lrn_astrophysics.md
- **Contradictory / Missing Evidence**: Coincident stellar point source present in quiescent imaging
- **Distinguishing Criteria**: High-resolution deep imaging to resolve background host galaxy.

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
  - *Scientific Rationale*: Disambiguate competing scientific hypotheses ('Unclassified Field Star / Stellar Variability') via spectral line identification and redshift determination.
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
