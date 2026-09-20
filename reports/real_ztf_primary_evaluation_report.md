# Controlled Zero-Shot Real-ZTF Model Evaluation Report

**Document Date:** September 19, 2026  
**Phase:** Controlled Zero-Shot Real-ZTF Model Evaluation Gate  
**Evaluation Scope:** Primary Frozen Real-ZTF Benchmark ($N = 106$)  
**Target Checkpoint:** `models/checkpoints/acei_multimodal_production.pt` (`SHA-256: e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`)  

---

## 1. Evaluation Objective

The objective of this controlled experiment is to execute a rigorous, zero-shot transfer evaluation of the frozen AI Cosmic Event Investigator (ACEI) production model and anomaly ensemble on the scientifically curated, 106-object Real-ZTF Primary Benchmark. 

Specifically, this evaluation determines:
1. Whether the frozen production `LightCurveEncoder` and multimodal transient architecture can ingest real-world photometric sequences across diverse astrophysical populations without architectural or numerical failure (e.g., zero NaNs, zero Infs, finite bounded embedding norms).
2. The exact quantitative zero-shot anomaly detection performance (AUROC, AUPRC, Average Precision, F1, FPR, TPR) achieved by an anomaly ensemble trained and calibrated *strictly* on synthetic simulations when exposed to real survey data without fine-tuning, threshold tuning, or domain adaptation.
3. The empirical domain-transfer characteristics between idealized synthetic simulations and real-sky Zitter/ZTF observations.

**Strict Scientific Boundary:** This evaluation is designed solely as an empirical measurement of zero-shot transfer compatibility. It is **not** a benchmark designed to claim production readiness or real-world astrophysical validation. In accordance with strict evaluation protocols, **no parameters were updated, no retraining occurred, no threshold tuning was performed, and zero real-ZTF data was used to fit anomaly reference distributions.**

---

## 2. Frozen Dataset Description

The primary benchmark evaluated herein is strictly isolated to the frozen manifest at `data/real_ztf_benchmark/frozen_primary_benchmark.csv`, cryptographically frozen under `data/real_ztf_benchmark/benchmark_freeze_metadata.json`:

* **Total Candidates Evaluated:** $N = 106$
* **Primary Manifest SHA-256:** `0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c`
* **Eligibility Criteria Met:**
  - Positional crossmatch separation $\le 1.5''$ (mean separation $= 0.054''$, max separation $= 0.697''$).
  - Ambiguity safety margin $\ge 0.3''$ to nearest alternative catalog source.
  - Usable valid token count $N_{\text{valid}} \ge 20$.
  - Padding fraction $\le 60\%$.
  - Provenance status: `VERIFIED` via authoritative spectroscopic crossmatches (IAU TNS, ZTF Bright Transient Survey, Gaia DR3, or published literature).
  - Strict isolation: Zero overlap with the 5 historical transfer-pilot objects (`ZTF18abukavn`, `ZTF19abjrhbe`, `ZTF20aaelulu`, `ZTF18aabtxvd`, `Gaia DR3 2028869231302243712`).
  - Zero inclusion of the 31 secondary-tier limited-coverage objects (`data/real_ztf_benchmark/frozen_secondary_limited.csv`).

### Composition Breakdown:
| Population Category | Ground Truth ($y$) | Population Family | Authoritative Class Subtypes | Object Count ($N$) |
| :--- | :---: | :--- | :--- | :---: |
| **In-Distribution Known** | $y = 0$ | Variable_Star | Variable_Star | 24 |
| | | SN_Ia | SN Ia | 19 |
| | | SN_II | SN II (9), SN IIP (4), SN IIb (2) | 15 |
| **Out-of-Distribution Anomaly** | $y = 1$ | TDE | Tidal Disruption Event (TDE) | 14 |
| | | Cataclysmic_Variable | Cataclysmic_Variable | 13 |
| | | SLSN | SLSN-I (5), SLSN-II (2) | 7 |
| **Unclassified Control** | $y = -1$ | Unclassified_Field_Star | Unclassified Field Star | 14 |
| **Total** | | | | **106** |

*Binary Metric Evaluation Subset ($y \in \{0, 1\}$):* Exactly 92 objects (58 in-distribution known, 34 out-of-distribution anomalies). The 14 unclassified field star controls are evaluated separately for score diagnostics.

---

## 3. Checkpoint Provenance

The model checkpoint used for this evaluation was loaded in frozen evaluation mode:

* **Filepath:** `models/checkpoints/acei_multimodal_production.pt`
* **File Size:** 51,048,011 bytes (48.68 MB)
* **Pre-Evaluation SHA-256:** `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`
* **Post-Evaluation SHA-256:** `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72` (Verified Bitwise Invariant)
* **Training Provenance:**
  - Base Model: `MultimodalTransientModel`
  - Total Trainable Parameters: 4,249,732 (LC Encoder: 425,072)
  - Training Seed: 42
  - Training Epochs: 5
  - Optimizer: `AdamW` ($\text{lr} = 0.0005$, $\text{weight\_decay} = 1\times 10^{-4}$)
  - Training Dataset: 100% Synthetic Benchmark (84 train, 16 validation, 50 test)
  - `real_ztf_data_used`: `False` (Stored and verified in checkpoint payload)

---

## 4. Preprocessing Contract

All 106 primary objects were processed via the production `RealZTFPreprocessor` implementing the frozen contract:
* **Sequence Length ($T_{\text{max}}$):** Fixed at 50 tokens.
* **Token Structure:** 4-channel representation `[relative_time_days, flux_scaled, flux_err_scaled, band_index]`.
* **Flux Normalization:** Median absolute deviation (MAD) scaling relative to object baseline.
* **Band Mapping:** Standard integer indices matching production vocabulary (`zg` $\to 0$, `zr` $\to 1$, `zi` $\to 2$).
* **Masking Convention:** Boolean mask where `True` denotes valid observation and `False` denotes padded sequence padding.
* **Cadence & Windowing:** Dynamic 50-observation selection centered around peak brightness.

---

## 5. Embedding Numerical Stability

Light-curve representations were extracted directly from the frozen production `LightCurveEncoder` in evaluation mode (`model.eval()`, `torch.no_grad()`). The generated 128-D embedding representations achieved perfect numerical stability across all 106 objects:

* **Total Embeddings Generated:** 106
* **Latent Dimensionality:** 128
* **NaN Values Encountered:** 0
* **Infinite (Inf) Values Encountered:** 0
* **Minimum $L_2$ Norm:** 11.3130
* **Maximum $L_2$ Norm:** 11.3244
* **Mean $L_2$ Norm:** 11.3211
* **Median $L_2$ Norm:** 11.3215
* **Norm Standard Deviation:** 0.0020
* **Finite Embedding Percentage:** 100.0%

Artifacts generated:
* `data/real_ztf_benchmark/real_ztf_primary_embeddings.csv` (Metadata and norms per object)
* `data/real_ztf_benchmark/real_ztf_primary_embeddings.npy` (Complete $106 \times 128$ float32 tensor matrix)

---

## 6. Anomaly Detector Configuration

The production anomaly ensemble combines three complementary signals:
1. **Multimodal Autoencoder:** Reconstruction error on 256-D fused embeddings ($w_{\text{ae}} = 0.40$).
2. **Mahalanobis Detector:** Minimum Mahalanobis distance to class-conditional centroids in latent space ($w_{\text{mah}} = 0.35$).
3. **Energy-based OOD Detector:** Logit free-energy score from the 4-class classifier head ($w_{\text{energy}} = 0.25$, $T = 1.0$).

### Reference Distribution Provenance:
To guarantee complete freedom from data contamination or target leakage, **all reference distributions were fitted strictly on the synthetic benchmark dataset using seed 42**:
* Autoencoder was trained on the 84 synthetic training events ($N_{\text{epochs}} = 25$).
* Mahalanobis centroids and shared covariance were fitted on the 84 synthetic training embeddings.
* Anomaly ensemble normalization parameters ($\text{median}$, $\text{scale}$) were learned on the 16 held-out synthetic validation events:
  - `autoencoder`: median = 0.021659, scale = 0.005305
  - `mahalanobis`: median = 14.452549, scale = 1.511752
  - `energy`: median = -1.532207, scale = 0.128376
* **Production Decision Threshold:** Stays frozen at $\tau = 0.65$.

---

## 7. Primary Metrics

Binary anomaly classification metrics were computed over the 92 non-control primary objects ($y=0$ vs $y=1$). The 14 unclassified field stars are excluded from the binary ground-truth metrics.

| Metric | Measured Value (Real-ZTF Primary) | Description / Scientific Interpretation |
| :--- | :---: | :--- |
| **AUROC** | **0.3859** | Area Under Receiver Operating Characteristic (continuous score ranking) |
| **AUPRC** | **0.2948** | Area Under Precision-Recall Curve (positive class = OOD anomalies) |
| **Average Precision** | **0.3063** | Weighted mean of precisions achieved at each PR threshold |
| **Accuracy** | **0.4022** (37 / 92) | Overall classification accuracy at frozen threshold $\tau = 0.65$ |
| **Precision** | **0.3091** (17 / 55) | Fraction of flagged candidates that are true OOD anomalies |
| **Recall / TPR** | **0.5000** (17 / 34) | Fraction of true OOD anomalies correctly flagged at $\tau = 0.65$ |
| **Specificity / TNR** | **0.3448** (20 / 58) | Fraction of known in-distribution objects correctly rejected |
| **False Positive Rate (FPR)** | **0.6552** (38 / 58) | Fraction of known objects erroneously flagged as anomalous |
| **F1 Score** | **0.3820** | Harmonic mean of Precision and Recall |
| **True Positives (TP)** | **17** | OOD anomalies correctly flagged |
| **False Positives (FP)** | **38** | Known in-distribution transients erroneously flagged |
| **True Negatives (TN)** | **20** | Known in-distribution transients correctly rejected |
| **False Negatives (FN)** | **17** | OOD anomalies erroneously classified as normal |

---

## 8. Per-Population Results

Detailed score distributions and detection rates across individual astrophysical populations:

| Population Family | Authoritative Subclasses | $N$ | Mean Score | Median Score | Std Dev | Flagged ($\ge 0.65$) | Flagged % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SLSN** | SLSN-I (5), SLSN-II (2) | 7 | 0.7593 | 0.7626 | 0.0398 | 7 | **100.0%** |
| **SN_II** | SN II (9), SN IIP (4), SN IIb (2) | 15 | 0.7615 | 0.7891 | 0.0766 | 13 | **86.7%** |
| **SN_Ia** | SN Ia (19) | 19 | 0.6870 | 0.7012 | 0.0915 | 14 | **73.7%** |
| **TDE** | Tidal Disruption Event (14) | 14 | 0.6530 | 0.6418 | 0.0896 | 7 | **50.0%** |
| **Variable_Star** | Variable Star (24) | 24 | 0.6490 | 0.6324 | 0.1081 | 11 | **45.8%** |
| **Cataclysmic_Var** | Cataclysmic Variable (13) | 13 | 0.5705 | 0.6150 | 0.1354 | 3 | **23.1%** |
| **Controls (Field Stars)** | Unclassified Field Star (14) | 14 | 0.6526 | 0.6882 | 0.1620 | 8 | **57.1%** |

### Per-Subclass Granular Breakdown:
* **SLSN-I** ($N=5$): Mean = 0.7743, Med = 0.7757, Flagged = 5/5 (100.0%)
* **SLSN-II** ($N=2$): Mean = 0.7215, Med = 0.7215, Flagged = 2/2 (100.0%)
* **SN IIP** ($N=4$): Mean = 0.8087, Med = 0.8100, Flagged = 4/4 (100.0%)
* **SN II (standard)** ($N=9$): Mean = 0.7616, Med = 0.7871, Flagged = 8/9 (88.9%)
* **SN IIb** ($N=2$): Mean = 0.6663, Med = 0.6663, Flagged = 1/2 (50.0%)

---

## 9. Confusion Matrix

The $2 \times 2$ contingency table at the frozen threshold of $\tau = 0.65$ (saved to `reports/real_ztf_primary_confusion_matrix.csv`):

| | Actual In-Distribution ($y = 0$) | Actual OOD Anomaly ($y = 1$) | Total Predicted |
| :--- | :---: | :---: | :---: |
| **Predicted Normal ($\text{Score} < 0.65$)** | **TN = 20** (34.5%) | **FN = 17** (50.0%) | 37 |
| **Predicted Anomaly ($\text{Score} \ge 0.65$)** | **FP = 38** (65.5%) | **TP = 17** (50.0%) | 55 |
| **Total Ground Truth** | 58 | 34 | 92 |

*Diagnostic Output Files:*
* `reports/real_ztf_primary_confusion_matrix.csv`
* `reports/real_ztf_primary_roc.csv` (42 ROC points)
* `reports/real_ztf_primary_pr.csv` (93 PR points)

---

## 10. Synthetic-vs-Real Comparison

A direct side-by-side comparison between the established synthetic benchmark baseline (`reports/anomaly_metrics.json`) and the real-ZTF zero-shot primary evaluation:

| Evaluation Dimension | Synthetic Benchmark Baseline | Real-ZTF Primary Benchmark | Absolute Change ($\Delta$) | Scientific Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Test Sample Size ($N$)** | 50 (20 known, 30 anom) | 92 (58 known, 34 anom) | +42 objects | Real benchmark is larger and class-unbalanced |
| **AUROC** | **0.6950** | **0.3859** | **-0.3091** | Severe ranking inversion due to domain gap |
| **AUPRC** | **0.7412** | **0.2948** | **-0.4464** | Baseline prevalence is lower (37% vs 60%) |
| **Average Precision** | **0.7520** | **0.3063** | **-0.4457** | Reflects lower precision across thresholds |
| **Accuracy** | **0.5400** | **0.4022** | **-0.1378** | Elevated false positive rate on real SNe |
| **Precision** | **0.7333** | **0.3091** | **-0.4242** | High FP rate degrades anomaly triage purity |
| **Recall / TPR** | **0.3667** | **0.5000** | **+0.1333** | SLSN detected (100%), but CV missed (77%) |
| **Specificity / TNR** | **0.8000** | **0.3448** | **-0.4552** | In-distribution SNe fail rejection |
| **False Positive Rate (FPR)** | **0.2000** | **0.6552** | **+0.4552** | 65.5% of real known transients flagged |
| **F1 Score** | **0.4889** | **0.3820** | **-0.1069** | Net decline in harmonic performance |

### Scientific Analysis of Differences:
1. **Simulation-to-Reality Domain Shift:** In synthetic training, the model learned joint distributions over synthetic 3-channel cutouts and simulated light curves. Real ZTF data is purely photometric (evaluated with zero-valued image tensors). The absent image modality induces a systematic shift in fused latent representations, elevating Mahalanobis distances for all real transients.
2. **Observational Cadence & Noise Structure:** Real ZTF photometric sequences contain heteroskedastic flux errors, seasonal gaps, and non-uniform cadence not fully captured by the parametric synthetic generator.
3. **Class Overlap:** Cataclysmic Variables exhibit rapid stochastic variability resembling synthetic variable stars, evading anomaly detection ($76.9\%$ false negative rate). Conversely, real Type II and Ia supernovae show complex color evolution that deviates from synthetic templates, triggering false alarms ($65.5\%$ false positive rate).

---

## 11. Failure Case Analysis

### A. Highest-Scoring Known In-Distribution Objects (False Positives)
These known transients produced the highest anomaly scores and represent the primary false alarms:

| Candidate ID | ZTF Designation | Authoritative Class | Ensemble Score | Valid Tokens | Padding Fraction | Available Filters | Primary Driver |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `CAND_SNII_009` | ZTF18abckutn | SN IIP | **0.8260** | 35 | 0.30 | `zg, zi, zr` | Mahalanobis saturation ($d_{\text{mah}} = 1.0$) |
| `CAND_SNIa_003` | ZTF19acdgwhq | SN Ia | **0.8237** | 38 | 0.24 | `zg, zr` | High Mahalanobis ($0.9918$) |
| `CAND_SNII_004` | ZTF18aavqdyq | SN II | **0.8204** | 50 | 0.00 | `zg, zi, zr` | Full sequence, high Mahalanobis ($0.9994$) |
| `CAND_SNII_036` | ZTF18acbwaxk | SN II | **0.8201** | 49 | 0.02 | `zg, zi, zr` | Complex light-curve morphology |
| `CAND_SNII_001` | ZTF18aapifti | SN IIP | **0.8170** | 23 | 0.54 | `zg, zi` | Moderate padding, high Mahalanobis ($1.0$) |

### B. Lowest-Scoring Out-of-Distribution Objects (False Negatives)
These genuine anomalous/rare transients received low scores and evaded detection at $\tau = 0.65$:

| Candidate ID | ZTF Designation | Authoritative Class | Ensemble Score | Valid Tokens | Padding Fraction | Available Filters | Primary Driver |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `CAND_CV_001` | ZTF17aaaemzh | Cataclysmic_Variable | **0.2797** | 50 | 0.00 | `zg, zi, zr` | Low Mahalanobis ($0.473$), maps to variable stars |
| `CAND_CV_003` | ZTF18abgopgb | Cataclysmic_Variable | **0.4501** | 43 | 0.14 | `zg, zi, zr` | Low energy score ($0.285$), moderate Mahalanobis |
| `CAND_CV_002` | ZTF18abdlywu | Cataclysmic_Variable | **0.4544** | 50 | 0.00 | `zg, zi, zr` | Low energy score ($0.259$), moderate Mahalanobis |
| `CAND_CV_010` | ZTF18abscxct | Cataclysmic_Variable | **0.4893** | 50 | 0.00 | `zg, zr` | Periodic flickering resembles normal variables |
| `CAND_CV_013` | ZTF17aaavfwx | Cataclysmic_Variable | **0.5107** | 50 | 0.00 | `zg, zr` | Quiescent baseline dominates sequence |

### C. Highest-Scoring Control Objects (Unclassified Field Stars)
Field stars evaluated to test control stability:

| Candidate ID | ZTF Designation | Authoritative Class | Ensemble Score | Valid Tokens | Padding Fraction | Available Filters |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `CAND_FieldStar_005` | Gaia DR3 2028862462419639552 | Unclassified Field Star | **0.8257** | 36 | 0.28 | `zi, zr` |
| `CAND_FieldStar_003` | Gaia DR3 2028862462419332224 | Unclassified Field Star | **0.8108** | 6 | 0.88 | `zi, zr` |
| `CAND_FieldStar_006` | Gaia DR3 2028862462419639936 | Unclassified Field Star | **0.8038** | 50 | 0.00 | `zi, zr` |
| `CAND_FieldStar_004` | Gaia DR3 2028862462419332608 | Unclassified Field Star | **0.8038** | 50 | 0.00 | `zg, zi, zr` |
| `CAND_FieldStar_007` | Gaia DR3 2028862462419640320 | Unclassified Field Star | **0.8015** | 33 | 0.34 | `zi, zr` |

---

## 12. Scientific Limitations

1. **Successful Technical Ingestion $\ne$ Validated Anomaly Triage:** The fact that the frozen production model successfully completed 106 forward passes with 0 NaNs and finite bounded norms demonstrates software and architectural robustness, **not** astronomical detection validity. The resulting AUROC ($0.3859$) indicates that zero-shot transfer without domain adaptation is insufficient for automated telescope alert triage.
2. **Modality Asymmetry (Light Curve Only vs Multimodal Fusion):** The production `MultimodalTransientModel` was pre-trained to fuse image cutouts and light curves. In the primary ZTF benchmark, only photometric time series were available. The missing visual modality significantly shifted the fused embedding manifold.
3. **Synthetic-to-Real Distribution Mismatch:** Parametric generators for light curves simplify astronomical variability. In particular, Cataclysmic Variables exhibit outburst cycles that the model conflated with normal variable stars, while standard Supernovae exhibit complex filter-dependent rise times that the model flagged as anomalous.
4. **Sample Size and Class Imbalance:** While $N=106$ is sufficient for a controlled feasibility benchmark, it is small relative to full survey operations ($10^6$ alerts/night). Metrics have wider binomial confidence intervals.

---

## 13. Explicit Compliance Statement

It is hereby formally certified that:
1. **ZERO real-ZTF data was used for training any model component.**
2. **ZERO real-ZTF data was used for model calibration.**
3. **ZERO real-ZTF data was used for threshold tuning (threshold remained fixed at $0.65$).**
4. **ZERO real-ZTF data was used to fit anomaly detector reference distributions (Autoencoder, Mahalanobis, Energy).**
5. All reference distributions and calibration parameters were derived strictly from synthetic benchmark partitions using `seed = 42`.
6. The production checkpoint `models/checkpoints/acei_multimodal_production.pt` remained strictly frozen throughout this evaluation, with identical pre- and post-inference SHA-256 digests (`e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`).
