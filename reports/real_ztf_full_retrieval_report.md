# ACEI Real-ZTF Photometric Retrieval Gate: Full Candidate Population Report

**Document Version:** 1.0.0  
**Phase:** Full Real-ZTF Photometric Retrieval Gate (Dataset Construction Only)  
**Total Candidates Evaluated:** 180 active candidates  
**Model Inference Status:** ZERO model runs, ZERO embedding extractions, ZERO anomaly score evaluations  
**Production Checkpoint Status:** Frozen and verified (`models/checkpoints/acei_multimodal_production.pt`, SHA-256: `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`)  

---

## 1. Executive Summary

The Full Real-ZTF Photometric Retrieval phase was executed across all **180 active, provenance-verified candidate objects** in the ACEI real-ZTF candidate registry (`data/real_ztf_benchmark/candidate_registry.csv`). This operation queried the NASA/IPAC Infrared Science Archive (IRSA) ZTF Data Release light curve API, performed rigorous spatial cross-matching ($\le 1.5''$ with a $0.3''$ ambiguity separation threshold), preserved all raw photometric records to disk, and executed production sequence preprocessing using the frozen `RealZTFPreprocessor`.

### Key Outcomes:
1. **Photometric Retrieval Yield:** 143 out of 180 objects (**79.4%**) yielded successful, unambiguously matched light curve sequences.
2. **Ambiguity Rejections:** 28 objects (**15.6%**) were safely purged due to ambiguous multi-source spatial associations within the $1.5''$ matching cone where competing candidate sources were separated by $< 0.3''$.
3. **Data Non-Detections:** 9 objects (**5.0%**) returned zero photometric observations from the IRSA light curve database.
4. **Usability & Coverage Quality:**
   - **`COVERAGE_SUFFICIENT` ($\ge 20$ valid tokens, $\le 60\%$ padding):** **106 objects** (58.9% overall, 74.1% of successfully retrieved).
   - **`COVERAGE_LIMITED` (5–19 valid tokens):** **31 objects** (17.2% overall, 21.7% of successfully retrieved).
   - **`INSUFFICIENT_OBSERVATIONS` (< 5 valid tokens):** **6 objects** (3.3% overall).
5. **Benchmark Eligibility:** Exactly **106 objects** are provisionally qualified as `benchmark_eligible = True` under the primary gold-standard coverage criteria. If secondary evaluation on `COVERAGE_LIMITED` objects is authorized, an additional 31 objects become available (totaling 137 objects).
6. **Strict Pipeline & Model Isolation:** Absolutely zero model forward passes, feature extractions, or anomaly score computations were conducted. The production multimodal checkpoint remains bitwise identical to its initial synthetic training state.

---

## 2. Retrieval Yield Matrix

Across the 180 candidate queries, the retrieval breakdown is as follows:

| Metric | Count | Fraction of Queried (N=180) | Fraction of Retrieved (N=143) |
| :--- | :---: | :---: | :---: |
| **Total Queried Candidates** | 180 | 100.0% | — |
| **Successful Retrievals (`SUCCESS`)** | 143 | 79.4% | 100.0% |
| **Ambiguous Associations (`AMBIGUOUS_ASSOCIATION`)** | 28 | 15.6% | — |
| **Zero Detections Returned (`NO_DATA`)** | 9 | 5.0% | — |
| **HTTP / Network Failures (`RETRIEVAL_FAILED`)** | 0 | 0.0% | — |
| **Coverage Sufficient (`COVERAGE_SUFFICIENT`)** | 106 | 58.9% | 74.1% |
| **Coverage Limited (`COVERAGE_LIMITED`)** | 31 | 17.2% | 21.7% |
| **Insufficient Observations (`INSUFFICIENT_OBSERVATIONS`)** | 6 | 3.3% | 4.2% |
| **Primary Benchmark Eligible (`COVERAGE_SUFFICIENT`)** | **106** | **58.9%** | **74.1%** |

---

## 3. Class-by-Class Breakdown

The table below presents the empirical yield and quality metrics across all 7 active astrophysical populations:

| Astrophysical Population | Dataset Role | Target / Min | Raw Candidates | Successfully Retrieved | Coverage Sufficient | Coverage Limited | Insufficient Obs (< 5) | Ambiguous Rejections | No Data / Failed | Mean Valid Tokens | Mean Padding | Mean Separation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cataclysmic_Variable** | OOD Anomaly | 15 / 8 | 15 | 15 (100.0%) | 13 (86.7%) | 2 (13.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 42.1 | 15.7% | 0.368'' |
| **SLSN** | OOD Anomaly | 20 / 8 | 20 | 14 (70.0%) | 7 (35.0%) | 6 (30.0%) | 1 (5.0%) | 5 (25.0%) | 1 (5.0%) | 25.1 | 49.9% | 0.443'' |
| **SN_II** | In-Distribution | 40 / 20 | 40 | 26 (65.0%) | 15 (37.5%) | 9 (22.5%) | 2 (5.0%) | 12 (30.0%) | 2 (5.0%) | 29.7 | 40.5% | 0.540'' |
| **SN_Ia** | In-Distribution | 50 / 25 | 50 | 34 (68.0%) | 19 (38.0%) | 12 (24.0%) | 3 (6.0%) | 10 (20.0%) | 6 (12.0%) | 28.5 | 43.0% | 0.505'' |
| **TDE** | OOD Anomaly | 15 / 6 | 15 | 15 (100.0%) | 14 (93.3%) | 1 (6.7%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 39.7 | 20.7% | 0.422'' |
| **Unclassified_Field_Star** | Control | 15 / 8 | 15 | 15 (100.0%) | 14 (93.3%) | 1 (6.7%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 48.7 | 2.7% | 0.470'' |
| **Variable_Star** | In-Distribution | 25 / 15 | 25 | 24 (96.0%) | 24 (96.0%) | 0 (0.0%) | 0 (0.0%) | 1 (4.0%) | 0 (0.0%) | 49.6 | 0.8% | 0.420'' |
| **Total** | — | **180 / 90** | **180** | **143 (79.4%)** | **106 (58.9%)** | **31 (17.2%)** | **6 (3.3%)** | **28 (15.6%)** | **9 (5.0%)** | **35.7** | **28.5%** | **0.466''** |

---

## 4. Empirical Cross-Match Quality Distribution

For the 143 successfully retrieved candidates, spatial cross-matching was conducted using exact Haversine great-circle distances against candidate discovery coordinates:

- **Minimum Separation:** $0.053''$ (`CAND_SNIa_007` / `ZTF18aabssth`)
- **Median Separation:** $0.330''$
- **Mean Separation:** $0.466'' \pm 0.354''$
- **Maximum Separation:** $1.446''$ (`CAND_SNIa_024` / `ZTF18aaxsctv`)
- **Strict Compliance:** 100.0% of accepted matches fall within the scientifically mandated $\le 1.5''$ match radius.
- **Rejection Statistics:** 28 candidates were rejected because the angular difference between the primary matched source and a secondary source within the cone was smaller than the safety margin ($\Delta \theta_{12} < 0.3''$). This strict gating completely prevents blending flux from adjacent unresolved field objects.

---

## 5. Photometric Coverage Distribution

The preprocessed light curves (truncated or padded to $L=50$ sequence steps) exhibit the following empirical distributions across the 143 retrieved objects:

### A. Valid Token Count ($N_{\text{tok}} \in [1, 50]$):
- **Minimum:** 2 tokens (`CAND_SNIa_019`, `CAND_SNIa_020`)
- **First Quartile ($Q_1$):** 21.0 tokens
- **Median:** 44.0 tokens
- **Mean:** $35.7 \pm 14.8$ tokens
- **Third Quartile ($Q_3$):** 50.0 tokens
- **Maximum:** 50 tokens (full sequence capacity reached in 52 objects)

### B. Padding Fraction ($F_{\text{pad}} \in [0.0, 1.0]$):
- **Minimum:** 0.000 (0% padding, 52 objects)
- **First Quartile ($Q_1$):** 0.000
- **Median:** 0.120 (12.0% padding)
- **Mean:** $0.285 \pm 0.296$ (28.5% padding)
- **Third Quartile ($Q_3$):** 0.580 (58.0% padding)
- **Maximum:** 0.960 (96.0% padding)

### C. Baseline Observation Durations:
- **Minimum Baseline:** 20.9 days (`CAND_SNIa_005`)
- **Median Baseline:** 2,712.7 days ($\approx 7.4$ years)
- **Mean Baseline:** $2,298.5 \pm 889.3$ days ($\approx 6.3$ years)
- **Maximum Baseline:** 2,897.8 days ($\approx 7.9$ years)

---

## 6. Filter Representation Analysis

Real-world survey operations produce heterogeneous filter coverage due to weather, seasonal allocations, and detector configurations. The empirical filter breakdown across the 143 retrieved objects is:

| Filter Combination | Object Count | Percentage | Astrophysical Implications |
| :--- | :---: | :---: | :--- |
| **All Three Filters ($zg + zr + zi$)** | **90** | **62.9%** | Full multi-band optical coverage. Maximum color baseline available for transient modeling. |
| **Canonical Two Filters ($zg + zr$)** | **21** | **14.7%** | Canonical ZTF survey filters. Represents the baseline expected by standard transient pipelines. |
| **Infrared Only ($zi$)** | **13** | **9.1%** | Objects observed exclusively during red-passband campaigns or where $zg/zr$ were contaminated/flagged. |
| **Red + Infrared ($zr + zi$)** | **8** | **5.6%** | Missing green band; common in red-transient follow-ups or dusty Galactic plane fields. |
| **Green Only ($zg$)** | **4** | **2.8%** | Very brief coverage during single-filter survey sweeps. |
| **Green + Infrared ($zg + zi$)** | **4** | **2.8%** | Sparse non-standard pairing. |
| **Red Only ($zr$)** | **3** | **2.1%** | Detected only in deepest red filter passes. |

### Confirmation of Amendment 1:
Requiring rigid three-filter coverage ($zg + zr + zi$) would have discarded **37.1% (53 objects)** of valid astronomical events. Allowing partial filter coverage while tracking it explicitly preserved 53 scientifically valuable candidates without violating input tensor shapes (missing filters are naturally encoded via `band_idx`).

---

## 7. Candidate Quality Observations

1. **Persistent & Periodic Sources (`Variable_Star`, `Cataclysmic_Variable`, `Unclassified_Field_Star`):**
   - High retrieval rate ($54/55 = 98.2\%$).
   - Negligible ambiguity rejections ($1/55 = 1.8\%$).
   - High token yield: Mean valid tokens $>42$, mean padding $<16\%$.
   - Baseline durations span up to 7.9 years, providing rich multi-cycle temporal sampling.
2. **Nuclear Flare Transients (`TDE`):**
   - 100% retrieval success ($15/15$).
   - Zero ambiguity rejections, as nuclear sources are well-centered in host galaxy cores.
   - 14/15 (93.3%) meet `COVERAGE_SUFFICIENT` standards.
3. **Explosive Extragalactic Transients (`SN_Ia`, `SN_II`, `SLSN`):**
   - Yielded lower `COVERAGE_SUFFICIENT` rates (35–38%).
   - Driven by two distinct real-world survey mechanisms:
     - **Seasonal Windowing:** Extragalactic supernovae explode and fade on 30–90 day timescales. If discovery occurred near the end of a seasonal observing window, ZTF only obtained 10–20 epochs before solar conjunction.
     - **Host Galaxy Contamination:** In deep coadded catalogs, host galaxy clumps and star-forming regions generate nearby centroid detections, triggering the $0.3''$ ambiguity threshold.

---

## 8. Ambiguity Analysis

A total of **28 candidates (15.6%)** were rejected under the `AMBIGUOUS_ASSOCIATION` rule:
- `SN_II`: 12 rejections (30.0%)
- `SN_Ia`: 10 rejections (20.0%)
- `SLSN`: 5 rejections (25.0%)
- `Variable_Star`: 1 rejection (4.0%)
- `Cataclysmic_Variable`, `TDE`, `Unclassified_Field_Star`: 0 rejections (0.0%)

### Astrophysical Cause:
Extragalactic supernovae and superluminous supernovae occur in external galaxies. In ZTF reference image processing, spiral arms, host galaxy cores, and compact H II regions within $1.5''$ of the transient position are frequently assigned distinct Object IDs (OIDs) in the IPAC catalog. When two cataloged sources lie at $r_1 = 0.5''$ and $r_2 = 0.65''$, their separation difference ($\Delta r = 0.15''$) is below the $0.3''$ confusion threshold. 

Rather than arbitrarily picking the slightly closer source (which may be a static host node rather than the transient), the pipeline safely drops the object. This is a vital scientific safeguard against host galaxy flux contamination.

---

## 9. Failed Retrieval Analysis

Only **9 candidates (5.0%)** returned zero observations (`NO_DATA`):
- `SN_Ia`: 6 candidates (`CAND_SNIa_016`, `CAND_SNIa_017`, `CAND_SNIa_034`, `CAND_SNIa_044`, `CAND_SNIa_046`, `CAND_SNIa_050`)
- `SN_II`: 2 candidates (`CAND_SNII_032`, `CAND_SNII_034`)
- `SLSN`: 1 candidate (`CAND_SLSN_020`)

### Cause:
These events were discovered in early 2018 or at deep negative declinations ($\delta = -16.6^\circ$). In ZTF alert operations, transients detected via difference imaging (subtraction from reference templates) do not always appear as positive detections in the public coadded reference catalog light curve tables if the transient occurred prior to the template coadd generation or was situated in a bad detector quadrant. The IRSA API returns an empty table for these coordinates.

---

## 10. Scientific Caveats & Limitations

1. **Light Curve Truncation to $L=50$:** The ACEI transformer architecture operates on sequences of length $L=50$. For field stars and long-term variables with $>500$ observations, the pipeline downsamples and selects the most informative contiguous window. This preserves transient dynamics but truncates multi-year secular variability.
2. **Partial Filter Ingestion:** 53 retrieved objects lack one or two of the standard $(g, r, i)$ filters. The encoder processes them with pad tokens in the missing bands, which may impact multimodal attention mechanisms that expect simultaneous multi-color measurements.
3. **Difference vs Coadd Photometry:** The IRSA light curves represent aperture photometry on calibrated science images. While high-quality for isolated sources, crowded galactic environments can exhibit slight background offsets relative to forced point-spread-function (PSF) difference photometry.

---

## 11. Provenance Integrity Audit

- **Candidate Registry Verification:** All 180 evaluated candidates strictly originate from `data/real_ztf_benchmark/candidate_registry.csv`.
- **Spectroscopic Authorities:** All supernovae, SLSNe, and TDEs possess verified spectroscopic classifications in IAU TNS, ZTF BTS, or peer-reviewed literature (Fremling et al. 2020, Perley et al. 2020, van Velzen et al. 2021).
- **Periodic & Variable Authorities:** All variable stars and CVs are confirmed in Gaia DR3, Chen et al. (2020), or Szkody et al. (2020).
- **Raw Photometric Immutability:** All 180 raw API responses are persisted under `data/real_ztf_benchmark/raw/<candidate_id>/raw_irsa.csv` with full HTTP query parameters and timestamps in `raw_metadata.json`.

---

## 12. Model & Checkpoint Isolation Verification

- **Production Checkpoint Path:** `models/checkpoints/acei_multimodal_production.pt`
- **File Size:** 4,374,903 bytes
- **SHA-256 Digest:** `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72` (Verified 100% identical to initial checkpoint).
- **Forward Inferences Run:** Exactly **0**.
- **Embeddings Computed:** Exactly **0**.
- **Anomaly Scores Evaluated:** Exactly **0**.
- **Hyperparameter Adjustments:** None.
- **Threshold Adjustments:** None (frozen at 0.65).

---

## 13. Benchmark Readiness Assessment

### Is the dataset ready to be frozen into the final benchmark?

**Verdict: YES, FEASIBILITY IS CONFIRMED.**

The primary benchmark eligibility criterion (`COVERAGE_SUFFICIENT`: $\ge 20$ valid tokens, $\le 60\%$ padding, clean cross-match) is met by **106 high-quality objects**:
- **In-Distribution:** 58 objects (`SN_Ia`: 19, `SN_II`: 15, `Variable_Star`: 24)
- **OOD Anomaly:** 34 objects (`SLSN`: 7, `TDE`: 14, `Cataclysmic_Variable`: 13)
- **Non-Transient Control:** 14 objects (`Unclassified_Field_Star`: 14)

This yields a balanced, highly representative real-world benchmark that is fully grounded in spectroscopic truth. Furthermore, an additional **31 objects** are available under `COVERAGE_LIMITED`, providing an optional expansion tier if a larger sample size is desired.

---

## 14. Next Phase Recommendations

Before proceeding to any model evaluation, the user and project stakeholders should decide on the benchmark freezing configuration:

### Option A (Recommended): Freeze Pure Gold-Standard Benchmark ($N = 106$)
- Lock the 106 `COVERAGE_SUFFICIENT` objects as the primary benchmark.
- Guarantees minimum 20 clean photometric points per sequence with $\le 60\%$ padding.
- Provides a clean scientific baseline with minimal padding noise.

### Option B: Freeze Tiered Benchmark ($N = 137$)
- Include both `COVERAGE_SUFFICIENT` (106 objects) and `COVERAGE_LIMITED` (31 objects).
- Evaluate metrics on the primary gold standard ($N=106$) and report robustness stress-testing on the limited coverage tier ($N=31$).

### Option C: Augmented Supernova Search
- Supplement `SN_Ia` and `SN_II` candidate pools from ZTF BTS to increase their `COVERAGE_SUFFICIENT` counts from 19 and 15 up to 35+ each.

**DO NOT proceed to model inference until the benchmark freezing strategy is explicitly approved.**
