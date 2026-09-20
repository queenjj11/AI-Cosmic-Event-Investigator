# ACEI Real-ZTF Evaluation Dataset Protocol & Dataset Card

**Protocol Version:** 1.0.0-PROVISIONAL  
**Status:** INFRASTRUCTURE AND PROTOCOL DEFINITION  
**Primary Dataset Manifest:** [`data/real_ztf_benchmark/manifest.csv`](file:///Users/jiajadhav/Desktop/BTECH /btech/3RD YR/AI-Cosmic-Event-Investigator/data/real_ztf_benchmark/manifest.csv)  
**Associated Target Checkpoint:** [`models/checkpoints/acei_multimodal_production.pt`](file:///Users/jiajadhav/Desktop/BTECH /btech/3RD YR/AI-Cosmic-Event-Investigator/models/checkpoints/acei_multimodal_production.pt)  
**Validation Suite:** [`src/data/benchmark_validator.py`](file:///Users/jiajadhav/Desktop/BTECH /btech/3RD YR/AI-Cosmic-Event-Investigator/src/data/benchmark_validator.py)  

---

## 1. Purpose & Objectives

The primary objective of the ACEI Real-ZTF Evaluation Dataset is to establish a rigorous, scientifically defensible evaluation benchmark for machine-learning-driven cosmic event investigation and out-of-distribution (OOD) anomaly detection on real photometric sky survey data from the Zwicky Transient Facility (ZTF).

The dataset is constructed specifically to test:
1. **Within-Distribution Representation Stability:** Whether real astronomical observations belonging to known classes (`SN_Ia`, `SN_II`, `Stellar_Flare`, `Variable_Star`) map into coherent, nominal latent regions without numerical failure.
2. **Out-of-Distribution Anomaly Sensitivity:** Whether genuinely rare, unmodeled, or extreme astrophysical transients (`SLSN`, `TDE`, `FBOT`, `LRN/ILRT`, `SN_Ibn/Icn`, `Cataclysmic_Variable`) trigger elevated anomaly scores in the frozen production ensemble.
3. **Survey Noise & Control Resistance:** Whether non-transient catalog sources (field stars, quiet baselines) are handled properly without excessive false-positive triage alarm.

> [!IMPORTANT]
> **Exploratory Pilot Boundary:**  
> The 5-object verified exploratory pilot (`SN_2019np`, `SN_2020jfo`, `AT_2018cow`, `SN_2018zd`, `ZTF_J195200.60+295217.4`) established **data contract and pipeline compatibility only**. It provided **zero proof of real-world anomaly detection performance or classification accuracy**. The pilot objects are strictly tagged `dataset_role = transfer_pilot` and `dataset_split = transfer_pilot` and are permanently isolated from evaluation metrics.

---

## 2. Dataset Scope & Planning Size

- **Planning Target Size:** Approximately $250$ celestial objects.
- **Scientific Sizing Rule:** The ~250 figure is a **provisional planning target**, not a quota that must be satisfied at any cost. If strict provenance, spectroscopic confirmation, or photometric coverage constraints cannot be satisfied for candidate objects, the benchmark size will scale down accordingly.
- **Strict Quality Rule:** Standards will **never** be relaxed, classes will never be guessed or visually inferred, and unrelated astrophysical populations will never be substituted merely to reach an arbitrary count.

---

## 3. Object Inclusion Criteria

To be admitted into the evaluation benchmark, an astronomical object must satisfy all of the following:

1. **Authoritative Identification:** Must possess documented, published celestial coordinates (RA/Dec, J2000) verified in authoritative astronomical catalogs:
   - IAU Transient Name Server (TNS)
   - ZTF Bright Transient Survey (BTS)
   - Gaia Data Release 3 (Gaia DR3)
   - Published peer-reviewed transient literature (e.g. ApJ, A&A, MNRAS)
2. **Spectroscopic / Catalog Confirmation:** For all in-distribution and anomaly classes, spectroscopic or high-confidence catalog confirmation is strictly required.
3. **Celestial Position Agreement:** Coordinate cross-matching to the ZTF catalog must yield great-circle angular separation $\Delta \theta \le 1.5''$ across all retrieved filters.
4. **Photometric Viability:**
   - Must contain at least one valid ZTF passband ($zg$, $zr$, or $zi$).
   - Must contain $\ge 15$ raw observations spanning the transient outburst or variability baseline.
   - Clean observations after catflags quality filtering must yield $\ge 5$ valid tokens in the transient window.

---

## 4. Object Exclusion Criteria

An object is permanently excluded from the benchmark if any of the following occur:

1. **Ambiguous Source Blending:** Multiple catalog sources within $1.5''$ having angular separation differences $<0.3''$ (engineering ambiguity heuristic).
2. **Missing Authoritative Classification:** Any object claiming a transient class without verifiable spectroscopic or catalog grounding.
3. **Extreme Positional Mismatch:** Best match separation exceeds $1.5''$.
4. **Unphysical Photometry / Severe Corruption:** Entire light curve flagged by catastrophic image artifacts (`catflags` bits 0–3) or missing flux values.
5. **No Transient Coverage:** Objects whose observations do not cover the outburst phase (e.g. only baseline observations years after transient decay).

All excluded candidate objects are retained in the manifest with explicit `exclusion_reason` entries rather than being deleted.

---

## 5. Provenance Requirements & Hierarchy

Every benchmark record must specify:
1. `class_authority`: The authoritative entity or catalog establishing the classification (e.g., `IAU TNS`, `ZTF BTS`, `Gaia DR3`).
2. `class_reference`: The exact permanent URL, DOI, or bibcode establishing the classification.
3. `classification_status`: Categorized strictly as:
   - `spectroscopic`: Confirmed by optical spectroscopy.
   - `catalog_photometric`: Grounded in authoritative multi-epoch variable catalog.
   - `unclassified`: Known field object with unconfirmed astrophysical nature.
   - `control`: Explicit non-transient survey control object.

**Rule:** Classifications must NEVER be inferred from the appearance of the light curve.

---

## 6. Coordinate Matching Policy

Multi-band ZTF observation histories are retrieved and associated strictly via great-circle angular separation computed using the spherical Haversine formula:

$$\Delta \theta = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta \delta}{2}\right) + \cos \delta_1 \cos \delta_2 \sin^2\left(\frac{\Delta \alpha}{2}\right)}$$

- **Cone Search Radius:** $\Delta \theta \le 1.5''$ (selected to match the typical ZTF optical seeing FWHM $\approx 2.0''$ and astrometric RMS $<0.15''$).
- **Multi-Band Preservation:** Each filter passband ($zg$, $zr$, $zi$) is matched independently to the nearest catalog source within the cone.
- **OID Integrity:** ZTF Object IDs (OIDs) are catalog database keys and must **never** be modified or digit-manipulated to link filters.

---

## 7. Photometric Preprocessing Contract

Raw ZTF observation records undergo deterministic preprocessing:
1. **Catalog Quality Bitmask Filtering:**
   - Clouds/Moon contamination: `catflags & 32768 == 0`
   - Saturated, edge, or bad pixel artifacts: `catflags & 15 == 0`
   - Magnitude uncertainties: $\sigma_m \in [0.005, 1.5]$ mag
   - Magnitude validity: $m \in [8.0, 25.0]$ mag
2. **Linear Flux Conversion:**
   $$F = 10^{-0.4(m - 27.5)}, \quad \sigma_F = \frac{\ln(10)}{2.5} F \max(0.005, \sigma_m)$$
   using the official ZTF AB zero-point $ZP = 27.5$.
3. **Peak Scaling Normalization:**
   $$F_{\text{norm}} = 1.5 \times \frac{F}{F_{\text{peak}}}, \quad \sigma_{F,\text{norm}} = 1.5 \times \frac{\sigma_F}{F_{\text{peak}}}$$
   *(Note: The additive zero-point cancels algebraically in the ratio $F / F_{\text{peak}}$).*

---

## 8. Tokenization & Sequence Construction

Processed observations are converted into 50-token PyTorch tensors:
- **Tensor Shape:** $(B, 50, 4)$
- **Column Order:** Strict internal convention:
  $$\mathbf{x}_i = [\text{relative\_time}, \text{norm\_flux}, \text{norm\_flux\_err}, \text{band\_idx}]$$
  where:
  - $\text{relative\_time} = \max(0, t - t_0)$ in days.
  - $\text{band\_idx} \in \{0: zg, 1: zr, 2: zi\}$.
- **Mask Tensor:** $(B, 50)$ boolean mask where `True` indicates a valid observation, and `False` indicates zero-padding.

---

## 9. Maximum Sequence Length & Compression

- Sequence ceiling: $N = 50$ tokens.
- **Compression Heuristic for Dense Sampling ($> 50$ observations):**
  1. Multi-band same-night binning: Observations in the same band within $\Delta t \le 0.5$ days are combined via inverse-variance weighted averaging:
     $$\bar{F} = \frac{\sum w_i F_i}{\sum w_i}, \quad w_i = \frac{1}{\sigma_i^2}, \quad \sigma_{\bar{F}} = \frac{1}{\sqrt{\sum w_i}}$$
  2. If binned observations still exceed 50 tokens, deterministic quantile uniform subsampling preserves temporal morphology across all outburst phases.

---

## 10. Missing-Band Handling

Real ZTF observations frequently lack coverage in one or two passbands (e.g. only $zg+zr$ or $zg+zi$).
- **Policy:** Partial filter coverage is valid. Missing passbands are **never fabricated or imputed**.
- The neural `LightCurveEncoder` handles irregular multi-band sampling naturally via band embeddings and Time2Vec temporal encodings.
- `available_filters`, `missing_filters`, and `partial_filter_coverage` are explicitly recorded in metadata.

---

## 11. Padding & Temporal Sampling Limitation

When an object has fewer than 50 valid observations in its transient window, the sequence is padded with zero tokens.
- **Padding Fraction:** $P_{\text{pad}} = \frac{50 - N_{\text{valid}}}{50}$.
- **Caveat:** Objects with $P_{\text{pad}} \ge 60\%$ (e.g. fewer than 20 valid tokens) demonstrate tensor contract compatibility but provide sparse temporal sampling. The benchmark manifest explicitly logs padding fractions to prevent over-interpreting sparsely sampled objects.

---

## 12. Class-Label Policy

Astronomical classes follow standard IAU/TNS taxonomic conventions:
- **Known / In-Distribution:** `SN_Ia`, `SN_II`, `Stellar_Flare`, `Variable_Star`.
- **Held-Out / OOD Anomaly:** `SLSN`, `TDE`, `FBOT`, `LRN`, `SN_Ibn`, `SN_Icn`, `Cataclysmic_Variable`.
- **Control / Unclassified:** `Unclassified_Field_Star`, `Control_Sparse`.

Ground-truth binary labels for anomaly detection are assigned strictly as:
- In-distribution: `is_anomaly_ground_truth = 0`
- OOD Anomaly: `is_anomaly_ground_truth = 1`
- Unclassified / Control: `is_anomaly_ground_truth = -1` (excluded from binary AUROC/AUPRC evaluation; evaluated separately for false alarm rates).

---

## 13. Object-Level Split Policy

Splits are partitioned strictly at the **OBJECT LEVEL**:
- All multi-band, multi-epoch observations for a given physical object belong exclusively to one split.
- **Celestial Separation Guard:** Distinct candidate object IDs within $1.5''$ of each other are flagged as coordinate collisions and prevented from spanning multiple splits.
- Split Roles:
  - `val_calibration`: Held-out known in-distribution objects used strictly for post-hoc calibration / score normalization without label leakage.
  - `test_known`: In-distribution objects reserved for nominal testing and false-positive measurement.
  - `test_anomaly`: Novel/rare objects reserved for true anomaly detection evaluation.
  - `control_unclassified`: Field stars and control objects for false alarm resistance evaluation.
  - `transfer_pilot`: The 5 initial exploratory targets, permanently isolated.

---

## 14. Calibration Policy

If score normalization or probability calibration (e.g. isotonic regression or Platt scaling) is performed:
- Calibration models must be fit **exclusively on the `val_calibration` split**.
- The `test_known` and `test_anomaly` splits must remain locked and unobserved until final scoring.
- Anomaly ground-truth labels must **never** be used during calibration.

---

## 15. Leakage Controls

To eliminate data leakage:
1. **Zero Real Data in Checkpoint:** The production checkpoint was trained exclusively on synthetic data (`real_ztf_data_used = False`).
2. **No Preprocessing Contamination:** Preprocessing operates on individual light curves independently (no cross-object scaling or dataset-level parameter fitting).
3. **No Threshold Tuning:** The production anomaly threshold is locked at **0.65**.

---

## 16. Intended Evaluation Metrics

When the benchmark is fully populated and approved for scoring, evaluation will measure:
1. **Continuous Ranking Discrimination:**
   - **AUROC** (Area Under the Receiver Operating Characteristic Curve)
   - **AUPRC** (Area Under the Precision-Recall Curve)
2. **Fixed-Threshold Decision Metrics (Threshold = 0.65):**
   - True Positive Rate (Recall / Detection Sensitivity) on anomalous transients
   - False Positive Rate on known transients
   - Precision and F1-Score
3. **Score Distributions:**
   - Median, mean, and IQR anomaly scores partitioned by astrophysical class.

---

## 17. Known Limitations

1. **Cadence Inhomogeneity:** ZTF surveys northern skies with non-uniform weather and seasonal gaps, resulting in variable sampling cadences.
2. **High Padding in Rapid Transients:** Fast-evolving transients (e.g. FBOTs) often yield fewer than 20 observations in a 50-token window.
3. **Photometric Depth:** ZTF difference imaging reaches limiting magnitude $r \approx 20.5$; faint transients near threshold suffer larger photometric uncertainties.

---

## 18. Scientific Boundaries: What This Dataset Can and Cannot Establish

### What This Benchmark CAN Establish:
- Whether the frozen ACEI multimodal architecture ingests real ZTF multi-band light curves with numerical and tensor stability.
- Whether OOD transients exhibit statistically higher anomaly scores than nominal in-distribution supernovae and variable stars.
- Whether false positive triage rates on catalog field stars remain bounded.

### What This Benchmark CANNOT Establish:
- **Complete Astronomical Survey Completeness:** A ~250-object benchmark does not capture all cosmological transient phenomena.
- **Vision Modality Generalization:** This benchmark tests light curve processing; real ZTF cutout image ingestion remains subject to separate image-quality qualification.
- **Model Perfection:** High benchmark scores do not guarantee zero missed discoveries in live alert stream operations.
