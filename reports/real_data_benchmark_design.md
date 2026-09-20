# ACEI Real Astronomical Benchmark Design & Provenance Protocol

**Document Version:** 1.0  
**Phase:** Phase 3 — Transition to Real Astronomical Observations  
**Surveys:** Zwicky Transient Facility (ZTF), Transient Name Server (TNS), ZTF Bright Transient Survey (BTS)  
**Status:** Benchmark Design & Leakage Prevention Protocol  

---

## 1. Executive Summary & Design Principles

The objective of this design is to establish a **scientifically defensible, leakage-free real astronomical benchmark** for quantitative evaluation of ACEI's multimodal classification and out-of-distribution (OOD) anomaly detection capabilities.

### Core Scientific Principles:
1. **Zero Data Leakage**: Enforce strict object-level partitioning using celestial coordinate cross-matching ($\le 1.5''$) and authoritative IAU designations. No observation, temporal slice, or catalog duplicate of an object may span train, validation, or test sets.
2. **Spectroscopic Ground Truth**: Evaluated events must have spectroscopically confirmed classifications from authoritative sources (TNS / ZTF BTS). Machine-learning photometric labels from alert brokers must NEVER be treated as ground-truth evaluation labels.
3. **Genuine Held-Out Anomaly Classes**: In-distribution known classes and held-out anomaly classes must represent distinct astrophysical mechanisms. Anomaly classes (`LRN`, `SLSN`, `TDE`) must remain strictly unseen during any training and calibration phases.
4. **Distance and Cadence Invariance**: Benchmarks must reflect real observational conditions (cadence gaps, weather dropouts, variable signal-to-noise) while evaluating intrinsic physical variability.

---

## 2. Small Real-Data Pilot Dataset Design ($N = 50$)

Before launching large-scale archive queries, a compact **Pilot Dataset** of exactly **50 confirmed astronomical objects** will be assembled to validate the end-to-end data pipeline locally on a standard workstation.

### Pilot Composition

| Class Category | Target Class | Count | Key Physical / Observational Rationale |
| :--- | :--- | :--- | :--- |
| **Known In-Distribution** | Type Ia Supernova (`SN Ia`) | 18 | Canonical thermonuclear standard candle. Validates high S/N peak, rapid rise, and exponential decline. |
| **Known In-Distribution** | Type II Supernova (`SN II`) | 14 | Core-collapse explosion. Validates extended hydrogen plateau (II-P) and slow linear decay (II-L). |
| **Known In-Distribution** | Variable Star (`Variable_Star`) | 10 | Periodic pulsators (RR Lyrae / Cepheids). Validates continuous multi-cycle periodic coverage. |
| **Held-Out Anomaly** | Superluminous Supernova (`SLSN`) | 3 | Magnetar / CSM powered. Validates extreme peak flux and prolonged 60+ day rise. |
| **Held-Out Anomaly** | Tidal Disruption Event (`TDE`) | 3 | Relativistic accretion onto SMBH. Validates nuclear flare with power-law $t^{-5/3}$ decline. |
| **Held-Out Anomaly** | Luminous Red Nova (`LRN`) | 2 | Stellar merger transient. Validates rapid cooling, red color excess, and secondary infrared peak. |
| **Total Pilot Events** | — | **50** | ~3,000–5,000 total photometric epochs. Disk footprint: < 5 MB. Run time: < 3 seconds. |

### Specific Validation Targets for the Pilot:
- Verification of HTTP GET retrieval against NASA/IPAC IRSA API (`nph_light_curves`).
- Filtering of bad quality epochs using `BAD_CATFLAGS_MASK = 32768`.
- Parsing of ZTF alert JSON and IPAC tables into standard `Observation` dataclasses.
- Conversion of apparent magnitudes to linear flux with Gaussian error propagation.
- Robust transient windowing ($t_{\text{peak}} - 20\text{d}$ to $t_{\text{peak}} + 60\text{d}$) to eliminate pre-explosion non-detections.
- Padding and masking into PyTorch `(50, 4)` tensors.
- Forward pass through the production baseline `LightCurveEncoder` to verify shape, activation stability, and embedding generation.

---

## 3. Real Astronomical Taxonomy vs. ACEI Synthetic Classes

The synthetic ACEI prototype utilized a simplified 4-class known taxonomy and 3 anomaly categories. Real astronomical taxonomies are significantly richer and more nuanced.

```
Real Astronomical Taxonomy Mapping:
Synthetic Known:
  SN_Ia ----------> IAU: SN Ia-norm, SN Ia-91T, SN Ia-91bg, SN Ia-CSM
  SN_II ----------> IAU: SN IIP, SN IIL, SN IIn, SN IIb
  Stellar_Flare --> Flare Stars, UV Ceti, Active M-Dwarf Chromospheric Flares
  Variable_Star --> RR Lyrae (ab/c), Classical Cepheids, Delta Scuti, Miras
Synthetic Anomaly:
  LRN ------------> IAU: Luminous Red Novae, Intermediate Luminosity Red Transients (ILRT)
  SLSN -----------> IAU: SLSN-I (hydrogen-poor), SLSN-II (hydrogen-rich)
  TDE ------------> IAU: TDE-H+He, TDE-H, TDE-He (SMBH disruption)
Real Novelty ------> Fast Blue Optical Transients (FBOT / Cow-like), Kilonovae (GW follow-up)
```

### Detailed Taxonomical Mapping & Incompatibilities

| ACEI Class ID | Real IAU / TNS Catalog Labels | Astrophysical Mechanism | Points of Agreement | Incompatibilities / Nuances to Address |
| :--- | :--- | :--- | :--- | :--- |
| **`SN_Ia`** | `SN Ia`, `SN Ia-norm`, `SN Ia-91T-like`, `SN Ia-91bg-like` | White dwarf thermonuclear runaway ($^{56}\text{Ni} \to {}^{56}\text{Co}$) | Characteristic 15–20d rise, rapid blue decay, Si II absorption. | Sub-luminous (91bg) and over-luminous (91T) subtypes exhibit altered decline rates (${\Delta}m_{15}$) that must not be misclassified as anomalies. |
| **`SN_II`** | `SN II`, `SN IIP`, `SN IIL`, `SN IIn`, `SN IIb` | Core-collapse of massive star ($M > 8 M_\odot$) | Prominent Balmer lines; extended plateau phase in IIP. | Type IIn involves circumstellar interaction (CSM), creating narrow lines and prolonged light curves that can mimic SLSN. |
| **`Stellar_Flare`**| `Flare Star`, `UV Ceti-type`, `dMe flare` | Magnetic reconnection in stellar corona | Fast rise in minutes, exponential decay in hours. | Ground-based survey cadence (1 visit/night) poorly samples minute-scale flares. In ZTF, stellar flares often appear as single-epoch spikes. |
| **`Variable_Star`**| `RR Lyrae`, `Cepheid`, `Delta Scuti`, `BY Dra` | Stellar pulsations / rotational modulation | Strictly periodic, repeatable light curve. | In difference imaging (alerts), non-varying host flux is subtracted, so only amplitude residuals appear, altering light curve morphology. |
| **`LRN`** (Anomaly) | `Luminous Red Nova`, `ILRT`, `V838 Mon-like` | Binary stellar merger with dynamic envelope loss | Double-peaked optical/NIR light curve, extreme reddening. | Extremely rare in real sky surveys ($< 1$ per 10,000 transients). Real sample size will be small ($N \sim 5 - 15$). |
| **`SLSN`** (Anomaly) | `SLSN-I`, `SLSN-II` | Millisecond magnetar spin-down or pair-instability | Absolute magnitude $M_V < -21$, prolonged UV excess. | Well-documented in ZTF BTS catalogs ($N > 80$). Clean separation from standard SNe. |
| **`TDE`** (Anomaly) | `TDE`, `TDE-H`, `TDE-He` | SMBH stellar tidal disruption | Coincident with galactic nucleus ($\Delta r < 0.1''$), $t^{-5/3}$ decline. | In 1D light curve shape alone, TDE optical rise and decay closely mimics SN Ia, requiring spatial host galaxy offset as a key discriminator. |

---

## 4. Label Provenance & Data Sources

To ensure scientific integrity, every benchmark event must have documented provenance verifying its celestial identity and physical classification.

### Primary Catalogs & Priority Hierarchy

1. **ZTF Bright Transient Survey (BTS)** (Fremling et al. 2020; Perley et al. 2020):
   - *Description*: Largest magnitude-limited spectroscopic survey of extragalactic transients ($m < 18.5$ mag).
   - *Completeness*: $> 95\%$ spectroscopic completeness for all transients reaching $m < 18.5$.
   - *Provenance Level*: **Gold Standard**. All labels confirmed via Palomar SED Machine (SEDM), Keck, or P200 spectroscopy.
2. **Transient Name Server (TNS - IAU Official Mechanism)**:
   - *Description*: Official IAU clearinghouse for astronomical transients.
   - *Provenance Level*: High. Provides IAU official designations (e.g. `SN 2023ixf`), discovery coordinates, reporter, and linked classification spectra (FITS / ASCII).
3. **ZTF Periodic Variable Star Catalog** (Chen et al. 2020):
   - *Description*: 782,412 periodic variable stars classified with light curves from ZTF DR2.
   - *Provenance Level*: High for periodic variables. Verified period, amplitude, and variability class.
4. **ALeRCE / Fink Community Brokers**:
   - *Role*: Cross-matching and alert stream aggregation only. Broker machine-learning classifications must NOT be used as evaluation ground truth.

### Event Metadata Provenance Contract
Every benchmark event record must store:
```json
{
  "object_id": "ZTF20acvppvo",
  "iau_name": "SN 2020fqv",
  "ra": 190.4908,
  "dec": -13.2081,
  "redshift": 0.0075,
  "spectroscopic_class": "SN IIP",
  "acei_canonical_class": "SN_II",
  "is_anomaly": false,
  "spectral_source": "Keck-I/LRIS",
  "catalog_source": "ZTF Bright Transient Survey (BTS)",
  "reference_bibcode": "2021ApJ...920..127T",
  "discovery_mjd": 59123.4,
  "peak_mag": 15.3,
  "peak_band": "r"
}
```

---

## 5. Comprehensive Data Leakage Safeguards

```
                      Raw Data Ingestion
                              |
       +----------------------+----------------------+
       | Spatial Cross-Match (<= 1.5 arcsec)         |  <-- Guard 1: Coordinate Deduplication
       +----------------------+----------------------+
                              |
       +----------------------+----------------------+
       | IAU / ZTF Object-Level Identifier Hash      |  <-- Guard 2: Cross-Catalog Partitioning
       +----------------------+----------------------+
                              |
       +----------------------+----------------------+
       | Split by Object ID (Train: 70%, Val: 15%, Test: 15%) | <-- Guard 3: Zero Observation Leakage
       +----------------------+----------------------+
                              |
       +----------------------+----------------------+
       | Held-Out Anomalies (LRN, SLSN, TDE -> 100% Test)    | <-- Guard 4: Strict Anomaly Isolation
       +----------------------+----------------------+
                              |
       +----------------------+----------------------+
       | Anomaly Ensemble Calibrated on Val Knowns ONLY      | <-- Guard 5: Zero Calibration Leakage
       +----------------------+----------------------+
                              |
                      Clean Benchmark
```

### Explicit Safeguard Definitions:

1. **Spatial Coordinate Cross-Matching (Guard 1)**:
   - Astronomical sources are often indexed under multiple identifiers across surveys (e.g. `ZTF18abukavn` = `AT 2018cow` = `ATLAS18cow`).
   - *Safeguard*: All candidate objects are cross-matched using a KD-tree spatial cone search with radius $r \le 1.5''$. Any matched entities are unified into a single unique object entry prior to dataset splitting.
2. **Strict Object-Level Partitioning (Guard 2 & 3)**:
   - Under no circumstances may lightcurve points from the same object appear in both training and test sets.
   - *Safeguard*: Splitting is performed strictly by unique object ID via `ObjectLevelSplitter`. All observations of an object remain intact within its assigned partition.
3. **Strict Anomaly Isolation (Guard 4)**:
   - Real anomaly classes (`LRN`, `SLSN`, `TDE`) must be 100% partitioned into the final evaluation test split.
   - *Safeguard*: Anomaly objects never enter the training set, validation set, or calibration set.
4. **Validation-Only Anomaly Detector Calibration (Guard 5)**:
   - ACEI's robust anomaly calibration ($M_{\text{norm}} = \frac{M - \text{median}}{\text{MAD}}$) must be fitted exclusively on the validation set of in-distribution known events.
   - *Safeguard*: Test set embeddings and logits are evaluated strictly zero-shot against frozen calibration parameters.
5. **Temporal & Metadata Leakage Prevention**:
   - `true_label`, `is_anomaly`, spectroscopic redshift, host galaxy classification, and peak magnitude metadata must be stripped from the tensor dictionary fed to `investigate_event`.
   - The neural model and anomaly detector receive only `[rel_time, flux, flux_err, band_idx]`.

---

## 6. Image Modality Integration Roadmap (Future Work)

While the initial real-data transition focuses on the lightcurve branch, the multimodal pipeline is architected to incorporate real astronomical cutouts.

### Requirements for Real Image Integration:
1. **Postage Stamp Sources**: Real difference-imaging cutouts are available as gzip-compressed FITS images in ZTF alert packets:
   - `cutoutScience`: Calibrated science image of the field.
   - `cutoutTemplate`: Deep coadded reference image of the static sky and host galaxy.
   - `cutoutDifference`: Difference image ($D = S - T$) revealing the transient point source.
2. **Image Preprocessing Specifications**:
   - Extraction of the central $64 \times 64$ pixel sub-image (pixel scale $\approx 1.0''/\text{pixel}$, field of view $\approx 1\text{ arcmin}$).
   - Astronomical arcsinh stretch ($\beta = 0.05$) to compress high dynamic range from bright stellar cores to faint host galaxy arms.
   - Stacking into 3-channel tensor `(3, 64, 64)` representing `[science, template, difference]`.
3. **Temporal Correspondence**:
   - In alert streams, each cutout corresponds to a single epoch ($t_{\text{alert}}$).
   - For lightcurve fusion, the image cutout selected should be the observation nearest to the lightcurve peak ($t \approx t_{\text{peak}}$) with high seeing quality ($\text{FWHM} < 2.5''$) and $\text{drb} > 0.70$.
4. **Current Status**:
   - Real image integration is deferred to Phase 4. The initial real pilot and lightcurve benchmark will operate with neutral zero-cutout inputs to evaluate the lightcurve encoder branch in complete isolation.
