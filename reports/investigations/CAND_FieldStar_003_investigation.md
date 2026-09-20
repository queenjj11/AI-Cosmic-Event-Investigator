# ACEI Investigation Report: CAND_FieldStar_003

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_FieldStar_003`
- **Benchmark Class**: `Unclassified Field Star`
- **Coordinates**: RA = 297.981344 deg, Dec = 29.791348 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 1214
- **Pipeline Execution Time**: 0.036 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3211
- **Valid Sequence Tokens**: 6 / 50
- **Padding Fraction**: 88.0%
- **Active Passbands**: r
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 991 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 1 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 2749.6280 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.5000 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.4017 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 0.5618 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.3700 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.0880 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 1.0983 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 0.1000 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 11.0240 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | 0.9820 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 4.9560 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 1.0000 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 11.0240 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 6 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.8800 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 0, 'r': 6, 'i': 0}
- **Missing Photometric Bands**: ['g', 'i']

## 4. Scientific Anomaly Assessment
- **Evaluation Status**: `REAL_ZTF_EVALUATED_RESEARCH`
- **Anomaly Score**: 1.0
- **Anomaly Trigger Flag**: True
- **Detector Architecture**: Real-ZTF Light-Curve Anomaly Detector v2 (PCA-Mahalanobis)
- **Domain Status**: REAL_ZTF_EVALUATED_RESEARCH

> [!WARNING]
> **Scientific Boundary**: Evaluated research model operating on real ZTF 128-D light-curve representation space. Image modality mismatch resolved by light-curve-only regularized distance scoring.

## 5. Candidate Transient Hypotheses

### Rank 1: Unclassified Field Star / Stellar Variability (Calibrated Prob: 0.580)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Photometric baseline (2749.628d) shows low-amplitude or stellar stochastic variability.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Requires multi-epoch baseline to establish definite periodicity or flare cadence
- **Distinguishing Criteria**: Periodogram analysis and Gaia DR3 astrometric parallax / proper motion cross-match.

### Rank 2: Cataclysmic / Eruptive Variable Star (Calibrated Prob: 0.270)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Modest brightening on top of stellar baseline.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Outburst amplitude is moderate compared to canonical dwarf novae
- **Distinguishing Criteria**: Optical spectroscopy to determine stellar spectral type and accretion signatures.

### Rank 3: Extragalactic Background Transient (Calibrated Prob: 0.150)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Coincidental alignment with a background extragalactic transient.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Coincident stellar point source present in quiescent imaging
- **Distinguishing Criteria**: High-resolution deep imaging to resolve background host galaxy.

## 6. Literature & Knowledge Base Evidence

#### [transient_taxonomy.json_LRN] Catalog: Luminous Red Nova *(Relevance: 0.353)*
> CATALOG REFERENCE: Luminous Red Nova (LRN) Physical Mechanism: Stellar merger in binary system accompanied by common-envelope dynamic ejection Peak Absolute Magnitude Range: [-14.5, -11.5] Rise Time: [25, 60] days Dec...

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.350)*
> ### Diagnostic Photometric Signatures 1. **Light Curve Morphology:** Frequently exhibit an initial blue precursor peak followed weeks to months later by a broader, red secondary peak. 2. **Color Evolution:** Rapid red...

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.226)*
> ### Abstract & Overview Luminous Red Novae (LRNe) are intermediate-luminosity optical transients ($M_V \approx -11$ to $-15$) characterized by slow expansion velocities ($v \approx 200 - 800\text{ km/s}$), prominent r...

## 7. Recommended Follow-Up Observations

- **Priority 1 (HIGH)**: Obtain calibrated g-band photometry
  - *Information Gap*: No observations in passband 'g'
  - *Scientific Rationale*: Missing g-band measurements prevent color temperature and extinction determinations.
  - *Target Band*: `g`
- **Priority 2 (NORMAL)**: Increase temporal observation cadence to daily sampling
  - *Information Gap*: Sparse temporal cadence during key evolutionary phases
  - *Scientific Rationale*: Current median cadence of 5.0 days undersamples fast light-curve inflections and peak.
  - *Target Band*: `r`
- **Priority 3 (NORMAL)**: Target-of-opportunity optical spectroscopy (3500-9000 A)
  - *Information Gap*: Zero spectroscopic observations available; physical expansion velocities and composition unconstrained
  - *Scientific Rationale*: Disambiguate competing scientific hypotheses ('Unclassified Field Star / Stellar Variability') via spectral line identification and redshift determination.
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
