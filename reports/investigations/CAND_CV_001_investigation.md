# ACEI Investigation Report: CAND_CV_001

## 1. Object Metadata & Provenance
- **Target Object**: `CAND_CV_001`
- **Benchmark Class**: `Cataclysmic_Variable`
- **Coordinates**: RA = 3.909458 deg, Dec = 26.615694 deg
- **Survey**: ZTF Public Data Release / NASA IPAC IRSA
- **Total Raw Observations**: 2723
- **Pipeline Execution Time**: 0.024 s

## 2. Cleaned Light-Curve Representation
- **Encoder**: Production LightCurveEncoder (128-D Transformer)
- **Embedding Dimension**: 128
- **Embedding L2-Norm**: 11.3226
- **Valid Sequence Tokens**: 50 / 50
- **Padding Fraction**: 0.0%
- **Active Passbands**: g, r
- **Window Time Span**: 80.00 days
- **Preprocessing Status**: `SUCCESS`

## 3. Non-Learned Event Characterization
> *Note: No astrophysical classification claimed from non-learned features alone.*

| Feature | Value | Status | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `num_observations` | 2497 | **MEASURED** | observations | Observations surviving bitwise catflag, cloud, and artifact filtering |
| `num_filters` | 2 | **MEASURED** | passbands | Active optical passbands (zg, zr, zi) with >= 1 observation |
| `time_baseline_days` | 2704.8130 | **MEASURED** | days | Span between first and last clean photometric observation |
| `window_duration_days` | 80.0000 | **DERIVED** | days | Duration of candidate transient outburst evaluation window |
| `peak_normalized_flux` | 1.1980 | **MEASURED** | normalized_flux | Maximum normalized flux observed in sequence |
| `minimum_normalized_flux` | 0.0121 | **MEASURED** | normalized_flux | Minimum normalized flux observed in sequence |
| `median_normalized_flux` | 0.0183 | **DERIVED** | normalized_flux | Median normalized flux of sequence |
| `flux_std` | 0.1731 | **DERIVED** | normalized_flux | Standard deviation of normalized flux |
| `median_flux_err` | 0.0006 | **DERIVED** | normalized_flux | Median normalized photometric measurement uncertainty |
| `variability_amplitude` | 1.1859 | **DERIVED** | normalized_flux | Peak-to-trough normalized flux variation |
| `rise_time_proxy_days` | 0.1000 | **PROXY** | days | Half-maximum rise duration proxy to peak observation |
| `decline_time_proxy_days` | 1.9450 | **PROXY** | days | Half-maximum decline duration proxy from peak observation |
| `rise_decline_asymmetry` | 0.9022 | **PROXY** | dimensionless | Rise/decline asymmetry: >0 means fast rise / slow decline; <0 means slow rise / fast decline |
| `cadence_median_days` | 1.0470 | **DERIVED** | days | Median inter-observation interval within window |
| `cadence_min_days` | 0.9450 | **DERIVED** | days | Shortest interval between distinct observation epochs |
| `cadence_max_days` | 7.9920 | **DERIVED** | days | Longest observational gap within window |
| `valid_token_count` | 50 | **DERIVED** | tokens | Sampled and binned observation tokens populated in model sequence (max 50) |
| `padding_fraction` | 0.0000 | **DERIVED** | fraction | Fraction of sequence capacity filled with zero-padding mask |

- **Per-Band Observation Counts**: {'g': 19, 'r': 31, 'i': 0}
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

### Rank 1: Cataclysmic Variable (CV) / Dwarf Nova Outburst (Calibrated Prob: 0.600)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: High-amplitude repetitive or rapid outburst (rise=0.1d, decline=1.9d) consistent with disk accretion instability.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Lacks spectroscopic confirmation of accretion disk emission lines
- **Distinguishing Criteria**: High-cadence time-series to detect orbital modulation / superhumps and recurrence.

### Rank 2: Stellar Flare / Chromospheric Outburst (Calibrated Prob: 0.250)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Rapid rise phase followed by radiative cooling damping.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: Outburst duration longer than canonical M-dwarf impulsive flare (hours)
- **Distinguishing Criteria**: Quiescent optical spectroscopy to detect M-dwarf molecular bands.

### Rank 3: Fast Optical Transient (FBOT) (Calibrated Prob: 0.150)
- **Status**: `UNVALIDATED_PRIOR`
- **Justification**: Rapid evolutionary timescale matching fast optical transient phase space.
- **Supporting Evidence**: transient_taxonomy.json_LRN
- **Contradictory / Missing Evidence**: High galactic latitude and persistent quiescent star indicate galactic origin
- **Distinguishing Criteria**: Radio follow-up to search for synchrotron relativistic emission.

## 6. Literature & Knowledge Base Evidence

#### [transient_taxonomy.json_LRN] Catalog: Luminous Red Nova *(Relevance: 0.353)*
> CATALOG REFERENCE: Luminous Red Nova (LRN) Physical Mechanism: Stellar merger in binary system accompanied by common-envelope dynamic ejection Peak Absolute Magnitude Range: [-14.5, -11.5] Rise Time: [25, 60] days Dec...

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.350)*
> ### Diagnostic Photometric Signatures 1. **Light Curve Morphology:** Frequently exhibit an initial blue precursor peak followed weeks to months later by a broader, red secondary peak. 2. **Color Evolution:** Rapid red...

#### [lrn_astrophysics.md] Observational Signatures and Astrophysics of Luminous Red Novae (LRNe) *(Relevance: 0.226)*
> ### Abstract & Overview Luminous Red Novae (LRNe) are intermediate-luminosity optical transients ($M_V \approx -11$ to $-15$) characterized by slow expansion velocities ($v \approx 200 - 800\text{ km/s}$), prominent r...

## 7. Recommended Follow-Up Observations

- **Priority 1 (NORMAL)**: Obtain calibrated i-band photometry
  - *Information Gap*: No observations in passband 'i'
  - *Scientific Rationale*: Missing i-band measurements prevent color temperature and extinction determinations.
  - *Target Band*: `i`
- **Priority 2 (NORMAL)**: Target-of-opportunity optical spectroscopy (3500-9000 A)
  - *Information Gap*: Zero spectroscopic observations available; physical expansion velocities and composition unconstrained
  - *Scientific Rationale*: Disambiguate competing scientific hypotheses ('Cataclysmic Variable (CV) / Dwarf Nova Outburst') via spectral line identification and redshift determination.
  - *Target Band*: `optical_spectrum`
- **Priority 3 (LOW)**: Extend late-time photometric monitoring (t > 40 days)
  - *Information Gap*: Post-peak light curve decline rate unconstrained at late epochs
  - *Scientific Rationale*: Constrain radioactive decay tail slope (Co-56 vs circumstellar interaction) or confirm quiescent return.
  - *Target Band*: `r`

## 8. Pipeline Limitations & Methodological Constraints

- Anomaly scores are suppressed: multimodal ensemble was trained on synthetic data with co-temporal difference images.
- Real ZTF input is processed through the Light-Curve-First pathway without image-modality fusion.
- Hypotheses reflect physical heuristics and retrieved literature, not automated ML classification.
- Redshift, host galaxy associations, and absolute luminosities are unconstrained without external catalog queries.
- Production checkpoint remained strictly frozen (no retraining, no fine-tuning, no threshold alteration).
