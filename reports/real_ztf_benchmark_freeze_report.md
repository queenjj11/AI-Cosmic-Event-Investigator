# ACEI Real-ZTF Evaluation Benchmark: Freeze Report

**Document Version:** 1.0.0  
**Phase:** Real-ZTF Benchmark Freeze Gate (Dataset Immutability Phase)  
**Freeze Date:** 2026-09-19  
**Model Inference Status:** ZERO model forward passes, ZERO embedding extractions, ZERO anomaly score calculations  
**Production Checkpoint Status:** Frozen and bitwise verified (`models/checkpoints/acei_multimodal_production.pt`, SHA-256: `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`)  

---

## 1. Executive Summary & Purpose of the Freeze

The **Real-ZTF Evaluation Benchmark** is formally frozen into immutable, versioned manifests. This step establishes a permanent, reproducible empirical evaluation foundation for the Antigravity Cosmic Event Investigator (ACEI) multimodal architecture before any real-data model inference is performed.

Freezing the benchmark prior to evaluation enforces core scientific principles:
1. **Zero Post-Hoc Selection Bias:** Objects cannot be retroactively pruned or re-weighted after observing model performance.
2. **Strict Pre-Inference Dataset Locking:** All candidate inclusions, exclusions, and tier assignments were established solely on data quality, spatial cross-matching confidence, and sequence coverage metrics.
3. **Cryptographic Provenance:** Every candidate is bound to its spectroscopic and photometric origins via deterministic SHA-256 hashing, guaranteeing end-to-end verifiability.

---

## 2. Benchmark Architecture: Primary vs. Secondary Tiers

To balance statistical rigor with the realities of astronomical time-domain survey cadences, the dataset is partitioned into two strictly disjoint manifests:

### A. Primary Real-ZTF Benchmark (`frozen_primary_benchmark.csv`, $N = 106$)
- **Purpose:** Primary gold-standard evaluation benchmark for zero-shot representation fidelity, cross-modal attention behavior, and out-of-distribution (OOD) anomaly triage.
- **Criteria:**
  - Unambiguous spatial association ($\Delta \theta \le 1.5''$ with ambiguity gap $\ge 0.3''$).
  - Valid token count $N_{\text{valid}} \ge 20$ (out of $L = 50$).
  - Padding fraction $F_{\text{pad}} \le 0.60$.
  - Authoritative spectroscopic classification (IAU TNS / ZTF BTS / literature) or catalog variability (Gaia DR3 / VizieR).
  - Absolute exclusion of the 5 historical transfer-pilot objects.
  - Zero coordinate collisions with any other benchmark candidate ($\Delta \theta_{\text{mutual}} \ge 1.5''$).

### B. Secondary Robustness & Stress-Test Tier (`frozen_secondary_limited.csv`, $N = 31$)
- **Purpose:** Independent evaluation stratum to quantify model degradation on sparsely sampled transients without diluting the primary benchmark.
- **Criteria:**
  - Same spatial cross-matching ($\Delta \theta \le 1.5''$) and provenance requirements as primary.
  - Limited photometric coverage: $N_{\text{valid}} \in [5, 19]$ tokens ($F_{\text{pad}} > 0.60$).
  - Evaluated separately as a stress-test of Transformer positional encodings under severe padding.

---

## 3. Census & Population Breakdown

Across the 180 queried candidates, 106 qualify for the primary benchmark, 31 are preserved in the secondary tier, and 43 are excluded:

### A. Primary Benchmark Demographic Breakdown ($N = 106$)

| Population Family | Authoritative Subtypes | Dataset Role | Dataset Split | Ground Truth Anomaly ($y$) | Primary Count |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Variable_Star** | Periodic Variable (RR Lyr, Cepheid, EB, Mira) | `test_benchmark` | `test_known` | 0 | 24 |
| **SN_Ia** | Type Ia Supernova (`SN Ia`) | `test_benchmark` | `test_known` | 0 | 19 |
| **SN_II** | Type II Supernova (`SN II`: 9, `SN IIP`: 4, `SN IIb`: 2) | `test_benchmark` | `test_known` | 0 | 15 |
| **TDE** | Tidal Disruption Event (`TDE`) | `test_benchmark` | `test_anomaly` | 1 | 14 |
| **Unclassified_Field_Star** | Non-Variable Field Reference Star | `control` | `control_unclassified` | -1 | 14 |
| **Cataclysmic_Variable** | Dwarf Nova / CV Outburst (`Cataclysmic_Variable`) | `test_benchmark` | `test_anomaly` | 1 | 13 |
| **SLSN** | Superluminous Supernova (`SLSN-I`: 5, `SLSN-II`: 2) | `test_benchmark` | `test_anomaly` | 1 | 7 |
| **Total Primary** | — | — | — | — | **106** |

- **In-Distribution Baseline ($y = 0$):** 58 objects (54.7%)
- **OOD Anomaly ($y = 1$):** 34 objects (32.1%)
- **Non-Transient Control ($y = -1$):** 14 objects (13.2%)

### B. Secondary Limited Tier Breakdown ($N = 31$)

| Population Family | Authoritative Subtypes | Dataset Role | Dataset Split | Ground Truth Anomaly ($y$) | Secondary Count |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **SN_Ia** | Type Ia Supernova (`SN Ia`) | `secondary_stress_test` | `test_known` | 0 | 12 |
| **SN_II** | Type II Supernova (`SN II`: 7, `SN IIP`: 2) | `secondary_stress_test` | `test_known` | 0 | 9 |
| **SLSN** | Superluminous Supernova (`SLSN-II`: 4, `SLSN-I`: 2) | `secondary_stress_test` | `test_anomaly` | 1 | 6 |
| **Cataclysmic_Variable** | Dwarf Nova / CV Outburst | `secondary_stress_test` | `test_anomaly` | 1 | 2 |
| **TDE** | Tidal Disruption Event (`TDE`) | `secondary_stress_test` | `test_anomaly` | 1 | 1 |
| **Unclassified_Field_Star** | Non-Variable Field Reference Star | `control` | `control_unclassified` | -1 | 1 |
| **Total Secondary** | — | — | — | — | **31** |

---

## 4. Exclusion Accounting ($N = 43$)

A total of 43 queried candidates were completely excluded from both primary and secondary benchmarks:

1. **`AMBIGUOUS_ASSOCIATION` (28 Candidates, 15.6% of queried):**
   - Objects where multiple distinct photometric sources were detected within the $1.5''$ matching cone and had a separation gap $< 0.3''$.
   - Prevalent in extragalactic events (`SN_II`: 12, `SN_Ia`: 10, `SLSN`: 5, `Variable_Star`: 1) due to host galaxy star-forming knots and spiral arms. Purging these objects prevents host-flux contamination.
2. **`NO_DATA` (9 Candidates, 5.0% of queried):**
   - Objects that returned zero detections from the IRSA light curve database (`SN_Ia`: 6, `SN_II`: 2, `SLSN`: 1).
   - Caused by early 2018 survey coverage limits or deep negative declinations ($\delta = -16.6^\circ$) where difference alerts occurred on bad detector quadrants without catalog light curves.
3. **`INSUFFICIENT_OBSERVATIONS` (6 Candidates, 3.3% of queried):**
   - Objects with $< 5$ total photometric observations (`SN_Ia`: 3, `SN_II`: 2, `SLSN`: 1).
   - Insufficient temporal support for sequence modeling; purged to maintain meaningful feature representations.

---

## 5. Provenance & Cross-Match Safeguards

1. **Ground-Truth Isolation:**
   - The production multimodal model (`acei_multimodal_production.pt`) was trained strictly on synthetic benchmark data.
   - All real-ZTF labels and coordinates are isolated from training and calibration routines.
2. **Pilot Quarantine:**
   - The 5 historical transfer-pilot objects (`SN_2019np`, `SN_2020jfo`, `AT_2018cow`, `SN_2018zd`, and `Field686_Star`) remain quarantined under `REJ_PILOT_001` through `005` in `candidate_registry.csv` and have zero overlap with the frozen benchmark.
3. **Astrometric Precision:**
   - Across the 106 primary objects, mean cross-match separation is $0.466''$ (max $1.363''$). All objects strictly satisfy the $\Delta \theta \le 1.5''$ matching criterion.
4. **Cryptographic Provenance Binding:**
   - Each manifest record includes a deterministic 64-character SHA-256 `provenance_hash` binding the candidate ID, designation, sky coordinates, astrophysical class, and authority URL.

---

## 6. Checkpoint Isolation & Pre-Inference Confirmation

- **Production Checkpoint:** `models/checkpoints/acei_multimodal_production.pt`
- **File Size:** 4,374,903 bytes
- **SHA-256:** `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`
- **Confirmation:** Exactly **0** model forward passes, **0** embedding extractions, and **0** anomaly evaluations were executed prior to this benchmark freeze. The checkpoint is bitwise identical to its initial synthetic training state.

---

## 7. Cryptographic File Integrity Manifest

| File Path | Description | SHA-256 Digest |
| :--- | :--- | :--- |
| `data/real_ztf_benchmark/candidate_registry.csv` | Active & quarantined candidate catalog (185 rows) | `8c365b996156ce77561ff702343ef01412a7ee3ab306ce4a78e967a1c0f816ad` |
| `data/real_ztf_benchmark/full_retrieval_results.csv` | Full empirical retrieval results (180 rows) | `c5f452221f91f514221ca1c07863f10399833a8d31665ed5a9ad97008c482502` |
| `data/real_ztf_benchmark/benchmark_eligibility.csv` | Eligibility flags and exclusion reasons (180 rows) | `46d17c86592dea0586885f95f6df1a5cded9a30d347a679af10ebb37fc8a14ad` |
| `data/real_ztf_benchmark/frozen_primary_benchmark.csv` | **Primary Real-ZTF Benchmark (106 rows)** | `0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c` |
| `data/real_ztf_benchmark/frozen_secondary_limited.csv` | **Secondary Stress-Test Tier (31 rows)** | `4690dea41c5ecb5b31f8d732905613fbdb52ce8596435b2b22fa7a0ca6bad3f6` |
| `data/real_ztf_benchmark/benchmark_freeze_metadata.json` | Machine-readable audit & freeze metadata | `4604e4615d474cdd1baf023900ff63818baace5b3e94806d3bf6ea1311a5c63a` |
| `models/checkpoints/acei_multimodal_production.pt` | Frozen multimodal production model checkpoint | `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72` |

---

## 8. Verification Suite Status

Automated testing in `tests/test_real_ztf_benchmark_freeze.py` passed with 10/10 checks:
- Primary count = 106
- Secondary count = 31
- Disjoint candidate IDs and ZTF designations
- Valid token count $\ge 20$ and padding $\le 60\%$
- Zero ambiguous or failed records
- Verified provenance citations
- Zero coordinate collisions within $1.5''$
- Zero transfer-pilot contamination
- Production checkpoint SHA-256 match
- Freeze metadata internal consistency

**Full Repository Test Suite:** **104 / 104 passed in 18.28s**.
