# Scientific Domain Gap Diagnosis: Real-ZTF Primary Benchmark

**Document Date:** September 19, 2026  
**Phase:** Scientific Diagnosis Only (Post-Primary Benchmark Evaluation)  
**Evaluated Artifacts:**  
- Production Checkpoint: `models/checkpoints/acei_multimodal_production.pt` (`SHA-256: e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`)  
- Primary Manifest: `data/real_ztf_benchmark/frozen_primary_benchmark.csv` (`SHA-256: 0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c`)  
- Evaluation Output: `data/real_ztf_benchmark/real_ztf_primary_evaluation.csv`  
- Diagnostic Outputs Directory: `reports/real_ztf_domain_gap/`  

---

## Executive Summary

In the controlled zero-shot evaluation of the frozen ACEI production model on the 106-object Real-ZTF Primary Benchmark, the model achieved an AUROC of **0.3859**, an AUPRC of **0.2948**, a false positive rate (FPR) of **0.6552** on known in-distribution transients, and a recall of **0.5000** on genuine out-of-distribution (OOD) anomalies.

This diagnostic investigation was conducted under strict scientific constraints: **zero model training, zero fine-tuning, zero recalibration, zero threshold adjustments, zero preprocessing modifications, and zero benchmark alterations.**

### Principal Findings:
1. **The Primary Driver is Modality Mismatch (Missing Visual Context):** The production `MultimodalTransientModel` was pre-trained to fuse visual cutouts and light curves via bidirectional cross-attention. When evaluated on real-ZTF photometric data using all-zero image tensors, the cross-attention gate allocates **$51.81\%$** of its weighting to the blank visual branch. This injects a constant, arbitrary out-of-distribution vector into the fused representation, shifting the 256-D fused manifold away from synthetic training centroids by a distance of **$5.5555$** (compared to a synthetic intra-cluster radius of $\approx 6.75$).
2. **Mahalanobis Saturation Drives the High False-Positive Rate:** Because the Mahalanobis detector was fitted on 256-D multimodal embeddings with an underdetermined reference sample ($N=84 < D=256$, Ledoit-Wolf condition number $= 1905.36$), this zero-image centroid shift forces the normalized Mahalanobis anomaly score to saturate at a mean of **$0.9216$** (median **$0.9672$**, with $48.1\%$ of objects scoring $\ge 0.99$). This single component is almost entirely responsible for the $65.52\%$ false positive rate on Type Ia and Type II supernovae.
3. **Severe Light-Curve Distribution Shift Exacerbates Manifold Drift:** Real ZTF light curves span multiple years of survey baseline (mean $2629.7$ days vs synthetic $50.9$ days), exhibit irregular cadence ($0.64$ obs/day vs synthetic $0.98$ obs/day), heteroskedastic flux errors ($\sigma_{\text{err}} = 0.0221$ vs synthetic $0.0060$), and variable band availability. Valid token count correlates negatively with anomaly scores ($r = -0.3742$, $p = 7.77 \times 10^{-5}$), meaning sparser sequences receive artificially higher anomaly scores.
4. **Phenomenological Conflation of Cataclysmic Variables:** Cataclysmic Variables (CVs) achieved an anomaly detection recall of only **$23.08\%$** (mean ensemble score $0.5705$, Autoencoder reconstruction error norm $0.2273$). Without visual morphology (e.g., accretion disk / point source environment), the light curve encoder maps the rapid flickering and quiescent baselines of CVs into the manifold occupied by normal synthetic `Variable_Star` templates.

---

## Hypothesis A: Modality Mismatch

### Background & Mechanism
The production checkpoint contains a `MultimodalTransientModel` with a `CrossAttentionFusion` layer:
$$\mathbf{h}_{\text{img}} = \text{LayerNorm}(\mathbf{h}_{\text{img}} + \text{MultiHeadAttention}(\mathbf{h}_{\text{img}}, \mathbf{h}_{\text{lc}}, \mathbf{h}_{\text{lc}}))$$
$$\mathbf{h}_{\text{lc}} = \text{LayerNorm}(\mathbf{h}_{\text{lc}} + \text{MultiHeadAttention}(\mathbf{h}_{\text{lc}}, \mathbf{h}_{\text{img}}, \mathbf{h}_{\text{img}}))$$
$$\mathbf{z}_{\text{fused}} = \mathbf{W}_{\text{proj}} \left( \mathbf{g} \odot \mathbf{h}_{\text{img}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{lc}} \right)$$

In synthetic training and validation, $\mathbf{h}_{\text{img}}$ is computed from simulated 3-channel visual cutouts. In real-ZTF primary evaluation, where survey cutouts were unavailable, zero-valued image tensors (`torch.zeros(1, 3, 64, 64)`) were supplied.

### Quantitative Measurements:
| Representation Metric | Synthetic Reference (Normal Images, $N=84$) | Synthetic Validation (Zero Images, $N=16$) | Real-ZTF Primary (Zero Images, $N=106$) | Scientific Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Image Embedding $L_2$ Norm** | $11.3210 \pm 0.0011$ | $11.3214$ (fixed) | $11.3214$ (fixed) | Zero-image maps to fixed point in $\mathbb{R}^{128}$ |
| **Distance to Mean Synthetic Visual Centroid** | $0.0000$ | **$4.1821$** | **$4.1821$** | Visual representation displaced by $4.18\sigma$ |
| **Cross-Attention Fusion Gate ($g$)** | $0.5012 \pm 0.0045$ | $0.5182 \pm 0.0009$ | **$0.5181 \pm 0.0011$** | **$51.81\%$** weight assigned to blank image branch |
| **Post-Attention Image Norm ($\|\mathbf{h}_{\text{img}}\|_2$)** | $16.0084$ | $16.0118$ | $16.0117$ | Constant uninformative vector injected into fusion |
| **Post-Attention LC Norm ($\|\mathbf{h}_{\text{lc}}\|_2$)** | $16.0142$ | $16.0160$ | $16.0158$ | Light-curve attends to blank image, corrupting features |
| **256-D Fused Embedding Norm** | $11.8320 \pm 0.3829$ | $11.7533 \pm 0.0662$ | $11.7148 \pm 0.1177$ | Real fused norms variance collapsed by $69\%$ |
| **Fused Centroid Distance from Synthetic** | $0.0000$ | **$4.9221$** | **$5.5555$** | Real fused manifold displaced by $>5.5$ Euclidean units |
| **128-D LC Centroid Distance from Synthetic** | $0.0000$ | $0.0000$ (identical LC) | **$4.8665$** | LC-only manifold displaced by $4.87$ units |

### Direct Proof of Modality Shift Impact on Validation Events:
When the exact same 16 synthetic validation events are scored with their normal synthetic images versus blank zero images:
* **Mean Anomaly Score:** Increases from **$0.1675$** (0% flagged) to **$0.4481$** (25.0% false positive rate).
* **Normalized Mahalanobis Score:** Jumps from **$0.1031$** to **$0.5730$** (a **$+455.8\%$** increase strictly due to the zero image).
* **Normalized Autoencoder Score:** Jumps from **$0.1809$** to **$0.4282$** (a **$+136.7\%$** increase).

**Conclusion:** Hypothesis A is **strongly supported**. Modality mismatch is the single largest contributor to the systematic elevation of anomaly scores.

---

## Hypothesis B: Light-Curve Distribution Shift

To isolate the distribution shift originating in the light curves themselves, we clearly separate actual measured physical properties from engineered preprocessing quantities:

### 1. Actual Measured Survey Quantities:
| Physical Parameter | Synthetic Reference ($N=84$) | Real-ZTF Primary ($N=106$) | Ratio / Shift | Physical Driver |
| :--- | :---: | :---: | :---: | :--- |
| **Full Survey Baseline** | $50.91 \pm 2.50$ days | **$2629.71 \pm 347.48$ days** | **$51.6\times$** | Real ZTF spans 7.2 years vs single synthetic outburst |
| **Available Filters** | Exactly 3 (`g, r, i`) | $2.77 \pm 0.42$ (`zg, zr, zi`) | $37.1\%$ partial | Real objects frequently lack $i$-band or $g$-band coverage |
| **Temporal Sampling Rate** | $0.9822 \pm 0.0521$ obs/day | **$0.6401 \pm 0.1647$ obs/day** | **$-34.8\%$** | Weather, seasonal lunar gaps, and scheduling constraints |
| **Photometric Noise ($\sigma_{\text{err}}$)** | $0.0385 \pm 0.0060$ | **$0.0224 \pm 0.0221$** | $3.7\times$ spread | Real error distribution is heavily heteroskedastic |
| **Maximum Observed Error** | $0.0516$ | **$0.1797$** | **$3.5\times$** | Real data contains low-SNR observations near limiting magnitude |

### 2. Engineered Preprocessing Quantities:
| Engineered Quantity | Synthetic Reference ($N=84$) | Real-ZTF Primary ($N=106$) | Ratio / Shift | Engineering Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Observation Window Span** | $50.91 \pm 2.50$ days | $70.42 \pm 12.77$ days | $+38.3\%$ | Real window selector extracts $[-20, +60]$ days from peak |
| **Valid Token Count ($N_{\text{valid}}$)** | $49.88 \pm 0.62$ (min $45$) | **$44.37 \pm 8.98$ (min $6$)** | Wider variance | Real transients often have $<50$ observations in 80-day window |
| **Padding Fraction ($f_{\text{pad}}$)** | $0.0024 \pm 0.0125$ | **$0.1126 \pm 0.1797$** | **$46.9\times$** | Real benchmark contains objects up to $88\%$ zero-padded |
| **Normalized Flux ($\tilde{F}$)** | $0.4241 \pm 0.3000$ | **$0.7459 \pm 0.4207$** | **$+75.9\%$** | Real peak normalization flattens baseline relative to peak |

**Conclusion:** Hypothesis B is **strongly supported**. The synthetic training data was generated under idealized sampling assumptions (tight 50-day window, homoskedastic errors, $100\%$ complete 3-band coverage, almost zero padding). When exposed to real ZTF sampling, the input feature distribution is substantially displaced.

---

## Hypothesis C: Detector Component Diagnosis

The production anomaly ensemble combines three detectors using frozen weights ($w_{\text{ae}}=0.40, w_{\text{mah}}=0.35, w_{\text{energy}}=0.25$):

$$\text{Ensemble Score} = 0.80 \left( 0.40 \cdot \tilde{S}_{\text{ae}} + 0.35 \cdot \tilde{S}_{\text{mah}} + 0.25 \cdot \tilde{S}_{\text{energy}} \right) + 0.20 \cdot \max(\tilde{S}_{\text{ae}}, \tilde{S}_{\text{mah}}, \tilde{S}_{\text{energy}})$$

### Component Score Breakdown Across Datasets:
| Detector Component | Split / Condition | Raw Score (Mean $\pm$ Std) | Normalized Score $\tilde{S}$ (Mean) | Normalized Score $\tilde{S}$ (Median) | Flags at $\tau \ge 0.65$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Mahalanobis Detector** | Synthetic Val (Normal) | $14.41 \pm 0.08$ | $0.1031$ | $0.0695$ | 0 / 16 (0.0%) |
| | Synthetic Val (Zero-Image) | $17.81 \pm 0.26$ | $0.5730$ | $0.6415$ | 4 / 16 (25.0%) |
| | **Real ZTF Primary** | **$21.96 \pm 0.10$** | **$0.9216$** | **$0.9672$** | **104 / 106 (98.1%)** |
| **Autoencoder** | Synthetic Val (Normal) | $0.0236 \pm 0.005$ | $0.1809$ | $0.0692$ | 0 / 16 (0.0%) |
| | Synthetic Val (Zero-Image) | $0.0301 \pm 0.007$ | $0.4282$ | $0.3508$ | 3 / 16 (18.8%) |
| | **Real ZTF Primary** | **$0.0382 \pm 0.028$** | **$0.6595$** | **$0.6817$** | **59 / 106 (55.7%)** |
| **Energy Detector** | Synthetic Val (Normal) | $-1.499 \pm 0.14$ | $0.1428$ | $0.0714$ | 0 / 16 (0.0%) |
| | Synthetic Val (Zero-Image) | $-1.495 \pm 0.12$ | $0.1364$ | $0.0936$ | 0 / 16 (0.0%) |
| | **Real ZTF Primary** | **$-1.571 \pm 0.12$** | **$0.0846$** | **$0.0310$** | **0 / 106 (0.0%)** |
| **Ensemble** | Synthetic Val (Normal) | — | $0.1675$ | $0.1204$ | 0 / 16 (0.0%) |
| | Synthetic Val (Zero-Image) | — | $0.4481$ | $0.4538$ | 4 / 16 (25.0%) |
| | **Real ZTF Primary** | — | **$0.6704$** | **$0.6898$** | **63 / 106 (59.4%)** |

### Key Diagnostic Takeaway:
* **The Mahalanobis detector is the primary driver of the high false-positive rate.** With a mean normalized score of $0.9216$ and a median of $0.9672$, the Mahalanobis detector flags $98.1\%$ of all real objects as anomalies. Because the ensemble includes a peak-signal response ($0.20 \cdot \max(\tilde{S})$), a near-1.0 Mahalanobis score automatically inflates the ensemble score past $0.65$ even if other detectors are low.
* **The Autoencoder acts as a secondary amplifier**, with a mean normalized score of $0.6595$ on real ZTF.
* **The Energy detector is completely inert against real ZTF transients**, producing very low scores ($\text{mean} = 0.0846$, flagging $0/106$ objects). The classification head's logit distribution did not exhibit high free energy on real data.

---

## Hypothesis D: Population-Specific Failure Analysis

| Population Family | Role | Ground Truth | $N$ | Autoencoder Mean | Mahalanobis Mean | Energy Mean | Ensemble Mean | Flagged % ($\ge 0.65$) | Valid Tokens | Padding % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SN_II** | In-Distribution | $y = 0$ | 15 | **$0.8848$** | **$0.9736$** | $0.0551$ | **$0.7615$** | **$86.67\%$ (13/15)** | $40.8$ | $18.4\%$ |
| **SN_Ia** | In-Distribution | $y = 0$ | 19 | **$0.7248$** | **$0.9296$** | $0.0446$ | **$0.6870$** | **$73.68\%$ (14/19)** | $46.2$ | $7.7\%$ |
| **SLSN** | OOD Anomaly | $y = 1$ | 7 | **$0.8705$** | **$0.9900$** | $0.0275$ | **$0.7593$** | **$100.0\%$ (7/7)** | $37.7$ | $24.6\%$ |
| **TDE** | OOD Anomaly | $y = 1$ | 14 | $0.6494$ | $0.9124$ | $0.0331$ | $0.6530$ | **$50.00\%$ (7/14)** | $43.6$ | $12.9\%$ |
| **Variable_Star** | In-Distribution | $y = 0$ | 24 | $0.6569$ | $0.8992$ | $0.0359$ | $0.6490$ | **$45.83\%$ (11/24)** | $47.1$ | $5.8\%$ |
| **Cataclysmic_Var** | OOD Anomaly | $y = 1$ | 13 | **$0.2273$** | **$0.9031$** | **$0.3215$** | **$0.5705$** | **$23.08\%$ (3/13)** | $46.1$ | $7.8\%$ |
| **Field_Star (Ctrl)** | Control | $y = -1$ | 14 | $0.6395$ | $0.8857$ | $0.1139$ | $0.6526$ | **$57.14\%$ (8/14)** | $43.6$ | $12.9\%$ |

### Astrophysical & Algorithmic Explanations:
1. **Supernovae False Alarms (SN II & SN Ia):** Real supernovae exhibit plateau phases, radioactive nickel-cobalt decline slopes, and multi-filter color changes that differ from idealized parametric templates. Combined with zero-image input, both Autoencoder reconstruction error and Mahalanobis distance explode, resulting in an $86.7\%$ false-positive rate on SN II and $73.7\%$ on SN Ia.
2. **Cataclysmic Variables Evasion (False Negatives):** CVs exhibit low Autoencoder reconstruction errors (mean $\tilde{S}_{\text{ae}} = 0.2273$). Because CVs consist of stochastic quiescent flickering punctuated by rapid dwarf nova outbursts, the LightCurveEncoder projects them near the periodic/stochastic manifold occupied by synthetic `Variable_Star` templates. The model fails to recognize them as anomalous, resulting in a $76.92\%$ false negative rate.
3. **Superluminous Supernovae Success:** SLSNe exhibit extremely high peak luminosities and exceptionally broad light-curve envelopes ($>60$ days). These extreme physical deviations triggered both the Autoencoder ($0.8705$) and Mahalanobis ($0.9900$) detectors, resulting in $100\%$ recall.

---

## Hypothesis E: Padding / Coverage Effect

We investigated whether variations in observational coverage systematically bias the model's anomaly scores.

### Pearson ($r$) and Spearman ($\rho$) Correlation Matrix ($N=106$):
| Coverage / Cadence Variable | Target Anomaly Score | Pearson $r$ | $p$-value ($r$) | Spearman $\rho$ | $p$-value ($\rho$) | Statistically Significant ($\alpha=0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Valid Token Count** | **Ensemble Score** | **$-0.3742$** | **$7.77 \times 10^{-5}$** | **$-0.3694$** | **$9.75 \times 10^{-5}$** | **YES (Highly Significant)** |
| Valid Token Count | Mahalanobis Score | **$-0.3309$** | **$5.31 \times 10^{-4}$** | **$-0.4579$** | **$8.03 \times 10^{-7}$** | **YES (Highly Significant)** |
| Valid Token Count | Autoencoder Score | **$-0.3217$** | **$7.73 \times 10^{-4}$** | **$-0.2929$** | **$2.31 \times 10^{-3}$** | **YES (Highly Significant)** |
| Valid Token Count | Energy Score | $+0.0036$ | $0.9711$ | $+0.0410$ | $0.6762$ | NO ($p > 0.65$) |
| **Padding Fraction** | **Ensemble Score** | **$+0.3742$** | **$7.77 \times 10^{-5}$** | **$+0.3694$** | **$9.75 \times 10^{-5}$** | **YES (Highly Significant)** |
| Padding Fraction | Mahalanobis Score | **$+0.3309$** | **$5.31 \times 10^{-4}$** | **$+0.4579$** | **$8.03 \times 10^{-7}$** | **YES (Highly Significant)** |
| **Available Filter Count** | **Ensemble Score** | **$-0.2831$** | **$3.28 \times 10^{-3}$** | **$-0.3138$** | **$1.05 \times 10^{-3}$** | **YES (Significant)** |
| Available Filter Count | Mahalanobis Score | **$-0.3346$** | **$4.55 \times 10^{-4}$** | **$-0.4833$** | **$1.54 \times 10^{-7}$** | **YES (Highly Significant)** |
| **Full Survey Baseline** | Ensemble Score | $-0.2439$ | $0.0117$ | $-0.2297$ | $0.0179$ | YES ($p < 0.02$) |
| **Window Span (Days)** | Ensemble Score | $-0.2588$ | $0.0074$ | $-0.2297$ | $0.0178$ | YES ($p < 0.02$) |

### Interpretation:
* There is a **statistically significant inverse correlation** ($r = -0.3742, p < 10^{-4}$) between the number of valid observations and the ensemble anomaly score.
* Objects with higher padding fractions and missing filters receive systematically higher Mahalanobis distances ($\rho = +0.4579$ for padding, $\rho = -0.4833$ for filters).
* In the Transformer `LightCurveEncoder`, padded tokens are masked out from self-attention; however, sequences with fewer valid tokens produce embeddings that sit further from the dense core of the synthetic manifold.
* The Energy detector is completely uncorrelated with padding ($r = 0.0036, p = 0.9711$), confirming that the coverage bias operates via latent embedding geometry rather than logit calibration.

---

## Hypothesis F: Synthetic Reference Geometry

We inspected the geometrical properties of the 84 synthetic training embeddings used to fit the Autoencoder and Mahalanobis detector:

| Geometrical Property | Value | Scientific Interpretation |
| :--- | :---: | :--- |
| **Latent Space Dimensionality ($D$)** | **$256$** | Fused multimodal latent dimension |
| **Reference Sample Size ($N$)** | **$84$** | Normal synthetic training objects |
| **Degrees of Freedom** | **$N < D$ ($84 < 256$)** | **Underdetermined, rank-deficient covariance estimation** |
| **Empirical Covariance Rank** | $\le 83$ | At least $173$ null dimensions in empirical covariance |
| **Raw Covariance Condition Number** | **$2.9815 \times 10^{13}$** | Ill-conditioned without regularized shrinkage |
| **Ledoit-Wolf Shrinkage Intensity ($\alpha$)** | **$0.0599$** | Modest shrinkage applied by Ledoit-Wolf estimator |
| **Shrunk Covariance Condition Number** | **$1905.36$** | Ratio of max to min eigenvalue ($\lambda_{\max}=7.394, \lambda_{\min}=0.00388$) |
| **Maximum Eigenvalue of Precision Matrix ($\mathbf{\Sigma}^{-1}$)** | **$257.64$** | **Perturbations along low-variance directions amplified by $257\times$** |
| **Synthetic Reference Centroid $L_2$ Norm** | $9.5505$ | Mean center of synthetic training distribution |
| **Real ZTF Centroid $L_2$ Norm** | $10.8520$ | Mean center of real ZTF primary distribution |
| **Centroid Shift ($\|\mathbf{\mu}_{\text{syn}} - \mathbf{\mu}_{\text{real}}\|_2$)** | **$5.5555$** | Substantial rigid displacement of real data manifold |
| **Min Distance to Any Synthetic Reference Sample** | $2.9134 \pm 0.35$ | Zero real samples overlap with synthetic reference points |

**Conclusion:** Hypothesis F is **strongly supported**. Fitting a 256-dimensional Mahalanobis detector on only 84 synthetic samples created an ill-conditioned metric space where minor out-of-distribution shifts (such as a missing image modality) are magnified by up to $257\times$.

---

## Hypothesis G: Failure Cases

The 15 diagnostic failure cases from `reports/real_ztf_domain_gap/failure_case_summary.csv`:

### 1. Top 5 Known In-Distribution Objects Erroneously Flagged (False Positives)
| Candidate ID | ZTF Designation | Class | Score | AE Score | Mahalanobis | Energy | Tokens | Pad | Filters | Primary Failure Driver |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `CAND_SNII_009` | ZTF18abckutn | SN IIP | **0.8260** | 0.8876 | 0.9999 | 0.1309 | 35 | 0.30 | `zg, zi, zr` | Saturated Mahalanobis + zero-image shift |
| `CAND_SNIa_003` | ZTF19acdgwhq | SN Ia | **0.8237** | 0.7289 | 0.9918 | 0.2305 | 38 | 0.24 | `zg, zr` | Saturated Mahalanobis + missing $i$-band |
| `CAND_SNII_004` | ZTF18aavqdyq | SN II | **0.8204** | 0.8596 | 0.9994 | 0.1065 | 50 | 0.00 | `zg, zi, zr` | Full sequence, high Mahalanobis saturation |
| `CAND_SNII_036` | ZTF18acbwaxk | SN II | **0.8201** | 0.8582 | 0.9997 | 0.1232 | 49 | 0.02 | `zg, zi, zr` | Complex light-curve morphology |
| `CAND_SNII_001` | ZTF18aapifti | SN IIP | **0.8170** | 0.8427 | 1.0000 | 0.0852 | 23 | 0.54 | `zg, zi` | Moderate padding, missing $r$-band |

### 2. Top 5 Out-of-Distribution Anomalies Missed (False Negatives)
| Candidate ID | ZTF Designation | Class | Score | AE Score | Mahalanobis | Energy | Tokens | Pad | Filters | Primary Failure Driver |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `CAND_CV_001` | ZTF17aaaemzh | Cataclysmic_Var | **0.2797** | 0.0391 | 0.4732 | 0.2003 | 50 | 0.00 | `zg, zi, zr` | Maps directly to normal Variable_Star manifold |
| `CAND_CV_003` | ZTF18abgopgb | Cataclysmic_Var | **0.4501** | 0.1554 | 0.7840 | 0.2852 | 43 | 0.14 | `zg, zi, zr` | Low AE reconstruction error |
| `CAND_CV_002` | ZTF18abdlywu | Cataclysmic_Var | **0.4544** | 0.1627 | 0.8001 | 0.2589 | 50 | 0.00 | `zg, zi, zr` | Low AE reconstruction error |
| `CAND_CV_010` | ZTF18abscxct | Cataclysmic_Var | **0.4893** | 0.1770 | 0.8398 | 0.3487 | 50 | 0.00 | `zg, zr` | Stochastic flickering treated as normal variability |
| `CAND_CV_013` | ZTF17aaavfwx | Cataclysmic_Var | **0.5107** | 0.2023 | 0.9089 | 0.2849 | 50 | 0.00 | `zg, zr` | Quiescent baseline dominates 80-day window |

### 3. Top 5 Controls Flagged as Anomalies
| Candidate ID | ZTF Designation | Class | Score | AE Score | Mahalanobis | Energy | Tokens | Pad | Filters | Primary Failure Driver |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `CAND_FieldStar_005` | Gaia DR3 2028862462419639552 | Field Star | **0.8257** | 0.8860 | 0.9999 | 0.1301 | 36 | 0.28 | `zi, zr` | Missing $g$-band, elevated Mahalanobis |
| `CAND_FieldStar_003` | Gaia DR3 2028862462419332224 | Field Star | **0.8108** | 0.8118 | 0.9996 | 0.0549 | 6 | 0.88 | `zi, zr` | Severe padding ($88\%$), sparse coverage |
| `CAND_FieldStar_006` | Gaia DR3 2028862462419639936 | Field Star | **0.8038** | 0.7766 | 0.9999 | 0.0211 | 50 | 0.00 | `zi, zr` | Missing $g$-band |
| `CAND_FieldStar_004` | Gaia DR3 2028862462419332608 | Field Star | **0.8038** | 0.7767 | 0.9999 | 0.0212 | 50 | 0.00 | `zg, zi, zr` | High Mahalanobis saturation |
| `CAND_FieldStar_007` | Gaia DR3 2028862462419640320 | Field Star | **0.8015** | 0.7654 | 0.9998 | 0.0210 | 33 | 0.34 | `zi, zr` | Missing $g$-band, moderate padding |

---

## Hypothesis H: Modality Ablation as Diagnostic Only

To determine whether the fused representation is being dominated by the missing visual modality, we performed a descriptive comparison without altering model weights:

1. **Full Multimodal Forward Pass with Zero Image ($\mathbf{z}_{\text{fused}} \in \mathbb{R}^{256}$):**
   - Centroid norm: **$10.8520$** (vs synthetic reference **$9.5505$**).
   - Centroid shift from synthetic: **$5.5555$**.
   - Variance across dimensions: mean $= 0.0762$ (synthetic was $0.1911$). The zero image suppresses multimodal variance by $60.1\%$.
2. **Light-Curve Encoder Representation ($\mathbf{z}_{\text{lc}} \in \mathbb{R}^{128}$):**
   - Centroid norm: **$10.2288$** (vs synthetic reference **$9.8625$**).
   - Centroid shift from synthetic: **$4.8665$**.
   - Variance across dimensions: mean $= 0.0612$ (synthetic was $0.0743$).
   - The light curve encoder preserves finite, well-bounded representations without visual interference.
3. **Image-Only Zero/Constant Input Representation ($\mathbf{z}_{\text{img}} \in \mathbb{R}^{128}$):**
   - Fixed embedding vector norm: **$11.3214$**.
   - Distance to mean synthetic cutout embedding: **$4.1821$**.
   - Variance across dimensions: **$0.0000$** (completely degenerate zero-variance point).

### Mechanistic Conclusion:
In `CrossAttentionFusion`:
* The attention gate weight $g$ averages **$0.5181$** with a standard deviation of only **$0.0011$**.
* This means that across all 106 real objects, **more than half ($51.81\%$) of the fused representation vector is entirely constituted by the projection of a blank zero image.**
* Furthermore, during the cross-attention step, the light-curve queries attend to the blank image features (`attn_lc`), corrupting the photometric representation prior to classification and anomaly detection.

---

## Verification & Integrity

* **Production Checkpoint SHA-256 (Pre-Diagnosis):** `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`
* **Production Checkpoint SHA-256 (Post-Diagnosis):** `e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72`
* **Primary Benchmark Manifest SHA-256:** `0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c`
* **Parameter Invariance:** Verified bitwise identical (0 parameters modified, zero gradient tensors active).
* **Test Suite Verification:**
  - `tests/test_real_ztf_domain_gap.py`: **8 / 8 passed in 3.00s**
  - Full repository regression suite (`pytest -q`): **127 / 127 passed in 19.85s**
