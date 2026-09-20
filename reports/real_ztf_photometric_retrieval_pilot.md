# ACEI Real-ZTF Evaluation Benchmark: Small-Batch Photometric Retrieval Gate Report

**Document Version:** 1.0.0  
**Phase:** Small-Batch Photometric Retrieval Gate (Real-ZTF Pilot)  
**Date:** 2026-09-19  
**Total Candidates Evaluated:** 35 Objects across 7 Classes  

---

## 1. Executive Summary

As part of transitioning the AI Cosmic Event Investigator (ACEI) to a scientifically defensible real-ZTF benchmark, we executed the **Small-Batch Photometric Retrieval Gate**. This gate rigorously probes whether provenance-verified astronomical candidates from the candidate registry (`data/real_ztf_benchmark/candidate_registry.csv`) possess sufficient real-world photometric observations in ZTF Data Releases to support evaluation with the existing frozen production model.

### Key Empirical Findings:
1. **Network & Retrieval Reliability (100% Query Success):** All 35 IRSA cone queries completed successfully without HTTP timeouts, socket resets, or rate-limiting bans.
2. **Coordinate Association Success (88.6% Association Rate):** 31 of 35 objects were unambiguously cross-matched to physical ZTF sources within $\Delta \theta \le 1.5''$ (mean separation $0.483''$).
3. **Ambiguity Safeguard In Action (11.4% Safe Rejections):** Exactly 4 candidates (`CAND_SNII_003`, `CAND_SLSN_003`, `CAND_SLSN_004`, `CAND_VarStar_002`) exhibited multiple distinct spatial sources in the same filter with separation gaps $< 0.3''$. In accordance with Amendment 2, the pipeline safely rejected these objects (`retrieval_status = AMBIGUOUS_ASSOCIATION`), preventing contaminated or blended light curves from entering the benchmark.
4. **Usable Sequence Coverage (80.6% Sufficient Among Retrieved):** Of the 31 associated objects, 25 (80.6%) achieved `COVERAGE_SUFFICIENT` ($\ge 20$ valid tokens, $\le 60\%$ padding, $\ge 2$ filters). 5 objects (16.1%) had `COVERAGE_LIMITED` (5–19 tokens), and only 1 object (3.2%) had `INSUFFICIENT_OBSERVATIONS` (2 tokens).
5. **Validation of Partial Filter Coverage (25.8% Partial):** 8 of the 31 successfully associated objects (25.8%) exhibited partial filter coverage ($zg+zr$, $zg+zi$, $zi+zr$, or $zg$-only). Under historical assumptions requiring all 3 filters ($zg+zr+zi$), these 8 scientifically valid targets would have been discarded. Under Amendment 1, all 8 were successfully ingested without synthetic data fabrication.
6. **Zero Model Leakage & Immutability:** No model forward passes were executed. The production checkpoint (`models/checkpoints/acei_multimodal_production.pt`) remains bitwise identical (SHA-256: `e5c78fdc...bc72`). All 87 repository unit and integration tests pass cleanly.

---

## 2. Batch Selection & Stratification

The 35 candidates were deterministically selected from `data/real_ztf_benchmark/candidate_registry.csv` by taking the first 5 active candidates (`provenance_status == "VERIFIED"`) per collectible class:

| Provisional Class | Candidate IDs | Target Designations | Discovery Source | Role |
| :--- | :--- | :--- | :--- | :--- |
| **SN_Ia** | `CAND_SNIa_001` - `005` | ZTF21acipofv, ZTF20aakyqlh, ZTF19acdgwhq, ZTF18aajpjdi, ZTF18aaqedfj | ZTF BTS / IAU TNS | In-Distribution |
| **SN_II** | `CAND_SNII_001` - `005` | ZTF18aapifti, ZTF18aaqkoyr, ZTF18aasxvsg, ZTF18aavqdyq, ZTF18aawyjjq | ZTF BTS / IAU TNS | In-Distribution |
| **SLSN** | `CAND_SLSN_001` - `005` | ZTF18abxbmqh, ZTF18achdidy, ZTF18acnnevs, ZTF18acsxwdi, ZTF18acyxnyw | ZTF BTS / IAU TNS | OOD Anomaly |
| **TDE** | `CAND_TDE_001` - `005` | ZTF19aabbnzo, ZTF17aaazdba, ZTF19aapreis, ZTF19aarioci, ZTF19abidbya | BTS / van Velzen 2021 | OOD Anomaly |
| **Variable_Star** | `CAND_VarStar_001` - `005`| ZTFJ000000.13+620605.8 to ZTFJ000000.30+711634.1 | Chen 2020 / VizieR | In-Distribution |
| **Cataclysmic_Variable** | `CAND_CV_001` - `005` | ZTF17aaaemzh, ZTF18abdlywu, ZTF18abgopgb, ZTF17aaawpsz, ZTF18acgplgw | Szkody 2020 / VizieR | OOD Anomaly |
| **Unclassified_Field_Star**| `CAND_FieldStar_001` - `005`| Gaia_DR3_2028862462418379264 to ...9639552 | Gaia DR3 / IRSA Field 686 | Control |

Exploratory transfer-pilot objects (`SN 2019np`, `SN 2020jfo`, `AT 2018cow`, `SN 2018zd`, `Field686_Star`) and unconfirmed candidates (`REJ_UNCONFIRMED_001` to `005`) were strictly excluded.

---

## 3. Retrieval Infrastructure & Raw Storage

### Retrieval Mechanics:
- **API Endpoint:** NASA/IPAC IRSA Light Curve Service (`https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves`).
- **Cone Search:** Position circle radius $r = 1.8''$ ($0.0005^\circ$), ensuring complete capture of sources near the $1.5''$ matching boundary.
- **Client Protocol:** Python `requests` session with CA certificate verification (`certifi`), custom User-Agent (`ACEI-Verified-Pilot/1.0`), and 45s socket timeout.
- **Concurrency:** Thread-safe pool of 3 worker threads. Total batch elapsed time: 333.7s (~9.5s per object including IRSA query, disk serialization, coordinate cross-matching, and pre-processing).

### Raw Storage Architecture:
Raw IRSA responses are stored deterministically on disk:
```text
data/real_ztf_benchmark/raw/
├── CAND_SNIa_001/
│   └── raw_irsa.csv
├── CAND_SNIa_002/
│   └── raw_irsa.csv
...
└── CAND_VarStar_002/
    └── raw_irsa.csv
```
Every observation row preserves original catalog attributes: `oid`, `expid`, `hjd`, `mjd`, `mag`, `magerr`, `catflags`, `filtercode`, `ra`, `dec`, `chi`, `sharp`, `limitmag`.

---

## 4. Coordinate Association & Separation Distribution

Spatial cross-matching was conducted using the spherical Haversine formula against authoritative J2000 coordinates:
$$\Delta \theta = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta \delta}{2}\right) + \cos \delta_1 \cos \delta_2 \sin^2\left(\frac{\Delta \alpha}{2}\right)}$$

### Separation Summary Across 31 Successfully Associated Objects:
- **Mean Angular Separation:** $0.4828''$
- **Median Separation:** $0.3418''$
- **Standard Deviation:** $0.3839''$
- **Range:** $[0.0926'', 1.4459'']$
- **100% Compliance:** All 31 associations fall strictly within the maximum allowed boundary ($\le 1.5''$).

### Separation Breakdown by Class:
- **Variable Stars:** Mean $0.121''$ (extremely high precision stellar centroids from VizieR/Gaia)
- **TDEs:** Mean $0.287''$ (well-centered host nuclear positions)
- **Cataclysmic Variables:** Mean $0.305''$
- **SLSN:** Mean $0.308''$
- **Unclassified Field Stars:** Mean $0.449''$
- **SN II:** Mean $0.616''$
- **SN Ia:** Mean $1.179''$ (reflecting minor host-galaxy centroid offsets in early transient alerts)

---

## 5. Photometric Coverage Distribution

The retrieved photometry was passed through `RealZTFPreprocessor` ($ZP=27.5$, $F_{\text{norm}} = 1.5 \times (F / F_{\text{peak}})$, $[-20, +60]$ day outburst window centering, same-night binning, max 50 tokens):

### Observational Property Summary (31 Associated Objects):
| Metric | Mean | Median | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Detections (`n_raw_points`)** | 1,715.5 | 1,785.0 | 1,294.0 | 4 | 4,353 |
| **Clean Detections (`n_clean_points`)** | 1,026.1 | 903.0 | 762.8 | 2 | 2,496 |
| **Valid Tokens (`n_valid_tokens`)** | 35.8 | 44.0 | 15.6 | 2 | 50 |
| **Padding Fraction (`padding_fraction`)** | 0.284 | 0.120 | 0.312 | 0.000 | 0.960 |
| **Baseline Duration (`baseline_days`)** | 2,458.3 | 2,695.8 | 582.2 | 381.0 | 2,764.6 |
| **Median Cadence (`cadence_median_days`)** | 1.018 | 0.999 | 0.380 | 0.042 | 2.105 |
| **Candidate Peak SNR (`peak_snr`)** | 50.65 | 43.62 | 29.75 | 10.16 | 106.32 |

### Coverage Classification Tally:
- **`COVERAGE_SUFFICIENT` ($\ge 20$ tokens, $\le 60\%$ pad, $\ge 2$ filters):** 25 / 35 (71.4% total, 80.6% of retrieved)
- **`COVERAGE_LIMITED` (5–19 tokens or single filter):** 5 / 35 (14.3% total, 16.1% of retrieved)
- **`INSUFFICIENT_OBSERVATIONS` ($< 5$ tokens):** 1 / 35 (2.9% total, 3.2% of retrieved)
- **`AMBIGUOUS_ASSOCIATION` (spatial blend):** 4 / 35 (11.4% total)
- **`RETRIEVAL_FAILED` / `NO_DATA`:** 0 / 35 (0.0%)

---

## 6. Filter Availability & Partial Coverage Analysis

| Filter Combination | Count | Percentage | Exemplar Targets |
| :--- | :---: | :---: | :--- |
| **Full Coverage ($zg + zi + zr$)** | 23 | 74.2% | `CAND_SNIa_004`, `CAND_TDE_001`, `CAND_VarStar_001`, `CAND_CV_001` |
| **Partial: $zg + zr$** | 3 | 9.7% | `CAND_SNIa_003`, `CAND_SLSN_001`, `CAND_SLSN_002` |
| **Partial: $zg + zi$** | 2 | 6.5% | `CAND_SNII_001`, `CAND_SNII_002` |
| **Partial: $zi + zr$** | 2 | 6.5% | `CAND_FieldStar_003`, `CAND_FieldStar_005` |
| **Partial: $zg$ only** | 1 | 3.2% | `CAND_SNIa_001` |

### Key Astrophysical Insight:
In ZTF survey operations, the $i$-band ($zi$) is scheduled far less frequently than $g$ and $r$ filters, while certain sky sectors or observing seasons have gaps in $g$ or $r$. Enforcing strict 3-filter completeness would have caused a catastrophic **25.8% sample loss**. Supporting partial filter coverage with explicit missing-mask tracking is empirically proven to be indispensable for real-survey evaluation.

---

## 7. Population-by-Population Feasibility Analysis

| Population | Evaluated | Retrieval Success | Ambiguous Rejections | Sufficient Coverage | Limited / Insuf. | Feasibility Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cataclysmic_Variable** | 5 | 5 (100%) | 0 (0%) | 4 (80%) | 1 (20%) | **HIGHLY FEASIBLE** |
| **Unclassified_Field_Star**| 5 | 5 (100%) | 0 (0%) | 5 (100%) | 0 (0%) | **HIGHLY FEASIBLE** |
| **TDE** | 5 | 5 (100%) | 0 (0%) | 4 (80%) | 1 (20%) | **HIGHLY FEASIBLE** |
| **Variable_Star** | 5 | 4 (80%) | 1 (20%) | 4 (80%) | 0 (0%) | **HIGHLY FEASIBLE** |
| **SN_II** | 5 | 4 (80%) | 1 (20%) | 3 (60%) | 1 (20%) | **FEASIBLE (Moderate Yield)** |
| **SN_Ia** | 5 | 5 (100%) | 0 (0%) | 3 (60%) | 2 (40%) | **FEASIBLE (Moderate Yield)** |
| **SLSN** | 5 | 3 (60%) | 2 (40%) | 2 (40%) | 1 (20%) | **MARGINAL (High Ambiguity)** |

### Population Analysis Details:
1. **Unclassified Field Stars & Variable Stars (Persistent Objects):**
   - 100% of non-ambiguous objects reached the full sequence capacity (44–50 valid tokens).
   - High baselines (>2,400 days) and dense coverage across all filters.
2. **TDEs & Cataclysmic Variables (High-Amplitude Outbursts):**
   - High sufficient yield (80%). The one limited TDE (`CAND_TDE_002`) had 19 tokens, just 1 token short of the 20-token threshold.
3. **Supernovae (SN Ia & SN II):**
   - 60% sufficient yield. SN Ia transients often have shorter observing windows or occur late in a seasonal campaign (`CAND_SNIa_001` had only 2 detections; `CAND_SNIa_002` had 7 tokens).
4. **Superluminous Supernovae (SLSN):**
   - Lowest yield (40% sufficient, 40% ambiguous). SLSNe frequently occur in distant, faint dwarf host galaxies or dense crowded environments where multiple ZTF catalog coadd sources appear within $1.5''$.

---

## 8. Comparison: Real Photometry vs Preprocessing Assumptions

| Dimension | Synthetic Production Assumption | Real-ZTF Pilot Finding | Alignment / Mitigation |
| :--- | :--- | :--- | :--- |
| **Baseline Length** | Short, isolated transient light curve ($\sim 80$ days) | Multi-year survey baseline ($1,000 - 2,700$ days) | Handled smoothly by `TransientWindowSelector` ($[-20, +60]$ day window). |
| **Noise & Artifacts** | Synthetic Gaussian noise | Atmospheric transparency flags (32768), bad pixels (15) | Handled by `ZTFQualityFilter` (catflags filtering removed $\sim 40\%$ invalid points). |
| **Sequence Length** | Exactly 50 tokens (mostly populated in synthetic) | Real tokens range from 2 to 50; 28.4% mean padding | Handled by boolean attention mask (`mask_tensor`). |
| **Filter Coverage** | Fixed 3-band ($g, r, i$) | 25.8% partial coverage ($zg+zr$, $zg+zi$, etc.) | Handled by Amendment 1 without fabricating bands. |
| **Cadence Uniformity**| Regular temporal sampling | Variable cadence (median $\sim 1.0$ day, but seasonal gaps) | Handled by `Time2Vec` continuous time encoding. |

---

## 9. Failure Mode Analysis

### Detailed Audit of the 4 Ambiguous Objects:
1. **`CAND_SNII_003` (`ZTF18aasxvsg`):**
   - *Passband:* `zr`
   - *Sources:* OID `676216100012911` ($\Delta \theta = 0.454''$) vs OID `1718212300023984` ($\Delta \theta = 0.516''$).
   - *Mutual separation:* $0.171''$.
   - *Separation gap:* $\Delta \text{gap} = 0.063'' < 0.3''$.
   - *Cause:* Blend of two distinct coadd sources in the overlapping survey grid.
2. **`CAND_SLSN_003` (`ZTF18acnnevs`):**
   - *Passband:* `zg`
   - *Sources:* OID `1823103400007095` ($\Delta \theta = 0.427''$) vs OID `789112300013107` ($\Delta \theta = 0.543''$).
   - *Separation gap:* $\Delta \text{gap} = 0.116'' < 0.3''$.
3. **`CAND_SLSN_004` (`ZTF18acsxwdi`):**
   - *Passband:* `zg`
   - *Sources:* OID `602114100024108` ($\Delta \theta = 1.088''$) vs OID `1647106100008246` ($\Delta \theta = 1.096''$).
   - *Separation gap:* $\Delta \text{gap} = 0.008'' < 0.3''$ (near-identical distance from target).
4. **`CAND_VarStar_002` (`ZTFJ000000.14+721413.7`):**
   - *Passband:* `zg`
   - *Sources:* OID `833116100012915` ($\Delta \theta = 0.190''$) vs OID `853113200018291` ($\Delta \theta = 0.242''$).
   - *Separation gap:* $\Delta \text{gap} = 0.052'' < 0.3''$.

### Low Token Count Failure:
- **`CAND_SNIa_001` (`ZTF21acipofv`):** Returned only 4 raw detections from IRSA, of which 2 were clean ($zg$-only). The transient was flagged at the end of a seasonal campaign with negligible ZTF public coverage.

---

## 10. Implications for Full 180-Object Retrieval

Applying the pilot's empirical success rates to the full candidate registry of 180 active candidates:

| Population | Registry Count | Expected Retrieval Success (88.6%) | Expected Ambiguous (11.4%) | Expected Sufficient (71.4%) | Target Quota | Feasibility Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SN_Ia** | 50 | 50 | 0 | $\sim 30$ | 50 (Min 25) | **Meets minimum 25** |
| **SN_II** | 40 | 32 | 8 | $\sim 24$ | 40 (Min 20) | **Meets minimum 20** |
| **SLSN** | 20 | 12 | 8 | $\sim 8$ | 20 (Min 8) | **Borderline (Meets min 8)** |
| **TDE** | 15 | 15 | 0 | $\sim 12$ | 15 (Min 6) | **Meets target 12-15** |
| **Variable_Star** | 25 | 20 | 5 | $\sim 20$ | 25 (Min 15) | **Meets minimum 15** |
| **Cataclysmic_Variable** | 15 | 15 | 0 | $\sim 12$ | 15 (Min 8) | **Meets target 12-15** |
| **Unclassified_Field_Star**| 15 | 15 | 0 | $\sim 15$ | 15 (Min 8) | **Meets target 15** |
| **Total** | **180** | **~159** | **~21** | **~121** | **~180** | **Robust evaluation sample (~120-130 objects)** |

---

## 11. Security, Isolation & Immutability Verification

1. **Production Checkpoint Hash Verification:**
   - Checkpoint path: `models/checkpoints/acei_multimodal_production.pt`
   - Algorithm: SHA-256
   - Expected: `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`
   - Verified: `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72` (**MATCH**)
2. **Zero Model Execution:**
   - No forward passes, embeddings, loss calculations, or gradient updates were performed.
   - Code executed strictly data retrieval, parsing, coordinate cross-matching, and array feature construction.
3. **Exploratory Pilot Quarantine Maintained:**
   - All 5 exploratory pilot objects remain quarantined; none entered the 35-candidate batch.
4. **Automated Test Suite:**
   - 87 / 87 tests passing (`tests/test_real_ztf_photometric_pilot.py` + regression suite).

---

## 12. Recommendations & Next Steps

1. **Approve Full 180-Object Retrieval:** The retrieval infrastructure, caching mechanics, and concurrency controls proved stable and fast (333s for 35 objects). Full 180-object retrieval will complete in $\sim 25$ minutes.
2. **Maintain Ambiguity Threshold (0.3''):** The 4 rejected objects prove that this engineering safeguard successfully purges contaminated coadd sources without human intervention.
3. **Include `COVERAGE_LIMITED` as Diagnostic Stratum:** While `COVERAGE_SUFFICIENT` ($\ge 20$ tokens) forms the primary evaluation split, `COVERAGE_LIMITED` (5–19 tokens) should be retained as an auxiliary test split to benchmark model behavior under sparse observational regimes.
4. **Await User Authorization:** Do NOT proceed to full 180-object retrieval until explicit approval is granted.
