# ACEI Real Astronomical Data Transition: Risk & Failure Mode Analysis

**Document Version:** 1.0  
**Phase:** Phase 3 — Transition to Real Astronomical Observations  
**Status:** Risk Assessment & Mitigation Framework  

---

## 1. Executive Summary

Transitioning an astronomical machine learning system from synthetic parameterizations to real observational survey streams introduces non-Gaussian noise, missing data regimes, selection biases, and survey artifacts. This document categorizes all identified risks, their severity, empirical failure signatures, and exact mitigation protocols.

---

## 2. Comprehensive Risk Matrix

| Risk ID | Risk Category | Failure Mode Description | Likelihood | Impact | Severity | Mitigation Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | **Temporal** | **Pre-Explosion Truncation**: ZTF archival lightcurves have years of non-detections. Blindly taking the first 50 epochs captures only quiescent noise, cutting off the explosion entirely. | High | Fatal | **Critical** | Implement SNR-peak detection and temporal windowing ($t_{\text{peak}} - 20\text{d}$ to $t_{\text{peak}} + 60\text{d}$) before sequence construction. |
| **RSK-02** | **Photometric** | **Activation Saturation**: Unnormalized microJansky fluxes ($10^2 - 10^5 \mu\text{Jy}$) saturate `Linear(2, 32)` and LayerNorm, causing gradient explosion or uniform feature collapse. | High | Fatal | **Critical** | Strictly normalize fluxes to $[0.0, 5.0]$ matching the synthetic pretraining dynamic range. |
| **RSK-03** | **Data Leakage**| **Conditioned Image Leakage**: `AstronomicalDataset` generates synthetic cutouts conditioned on `event.true_label` when real images are missing, leaking test labels to the classifier. | High | Fatal | **Critical** | Decouple `AstronomicalDataset` from `true_label`; substitute a neutral zero-tensor image when cutouts are absent. |
| **RSK-04** | **Observational**| **Difference-Image Subtraction Artifacts**: Bad subtraction dipoles, optical glints, and cosmic rays produce high-amplitude single-epoch spikes flagged as false anomalies. | High | Moderate | **High** | Filter individual alerts with Real-Bogus score $\text{drb} \ge 0.70$ and $\text{catflags} == 0$. |
| **RSK-05** | **Astrophysical** | **TDE / SN Ia Lightcurve Degeneracy**: In 1D optical lightcurves, smooth nuclear flares of TDEs closely mimic Type Ia SNe, yielding 0% recall without host-galaxy spatial offset. | High | Moderate | **High** | Document lightcurve-only limitations transparently; prepare host-galaxy nuclear offset ($\Delta r < 0.1''$) as a future multimodal feature. |
| **RSK-06** | **Catalog** | **Catalog Identifier Duplication**: The same physical transient reported under different survey names (e.g. ZTF vs ATLAS vs Gaia) appearing in both train and test splits. | Moderate | Fatal | **Critical** | Perform celestial coordinate cross-matching ($r \le 1.5''$) prior to dataset splitting. |
| **RSK-07** | **Cadence** | **Irregular Sampling & Long Seasonal Gaps**: Fields setting behind the Sun create 3-to-5 month observation blackouts. | High | Low | **Medium** | Time2Vec continuous temporal encoding natively handles non-uniform intervals; windowing restricts sequences to the active eruption phase. |
| **RSK-08** | **Taxonomic** | **Supernova Subtype Mismatches**: Over-luminous 91T-like or sub-luminous 91bg-like Type Ia SNe exhibiting non-standard decay rates flagged as OOD anomalies. | Moderate | Low | **Medium** | Ensure training set includes representative variety of subtype lightcurves during full retraining. |
| **RSK-09** | **Data Quality** | **Missing Uncertainty Estimates**: Occasional corrupted alert records reporting `sigmag <= 0.0` or `NaN`. | Low | Moderate | **Medium** | Enforce input assertion floors: clamp $\sigma_F \ge 0.01 \times F$ to prevent division-by-zero in feature extraction. |

---

## 3. Failure Mode Diagnostic Signatures & Automated Checks

```
Pipeline Ingestion Diagnostics:
1. Window Sanity Check:
   ASSERT max(windowed_fluxes) > 3.0 * median(baseline_fluxes)
   -> Catches RSK-01 (Pre-Explosion Truncation)

2. Scale Sanity Check:
   ASSERT 0.0 <= mean(normalized_flux) <= 5.0
   -> Catches RSK-02 (Activation Saturation)

3. Leakage Guard Sanity Check:
   ASSERT len(set(train_ids).intersection(set(test_ids))) == 0
   ASSERT len(set(train_coords).cross_match(test_coords, 1.5_arcsec)) == 0
   -> Catches RSK-06 (Catalog Identifier Duplication)

4. Image Neutrality Sanity Check:
   ASSERT event.image is None OR event.image.is_neutral OR is_real_fits
   -> Catches RSK-03 (Conditioned Image Leakage)
```

---

## 4. Operational Recommendations for Phase 3

1. **Gate Every Real Ingestion with Preprocessing Assertions**:
   Never feed raw API responses directly into PyTorch datasets. All data must pass through a validated `ZTFPreprocessFilter`.
2. **Preserve Frozen Production Baseline**:
   Run the 50-event real pilot zero-shot on the existing 5-epoch baseline model first. This establishes an empirical transferability baseline before deciding whether real-data fine-tuning is needed.
