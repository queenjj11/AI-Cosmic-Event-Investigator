# Real-ZTF Candidate Availability & Discovery Report

**Document Version:** 1.0.0  
**Phase:** Candidate Discovery & Provenance Verification  
**Registry Path:** [`data/real_ztf_benchmark/candidate_registry.csv`](file:///Users/jiajadhav/Desktop/BTECH /btech/3RD YR/AI-Cosmic-Event-Investigator/data/real_ztf_benchmark/candidate_registry.csv)  
**Total Candidates in Registry:** 190 (180 Active Candidates + 10 Rejected/Isolated Controls)  

---

## 1. Executive Summary

This report documents the results of the **Candidate Discovery Phase** for the ACEI Real-ZTF Evaluation Benchmark.
Following the project's strict scientific protocol:
1. Candidate discovery was restricted to **authoritative astronomical catalogs** (ZTF Bright Transient Survey, VizieR catalogs of Chen et al. 2020 and Szkody et al. 2020, and Gaia DR3).
2. **Zero bulk photometry was downloaded** during this phase.
3. Classifications were accepted **only from documented spectroscopic or multi-epoch catalog citations**; zero classifications were inferred from light curve appearance, colors, or AI models.
4. The 5 historical exploratory transfer-pilot objects are explicitly cataloged in the registry as `candidate_role = "rejected"` to mathematically guarantee they cannot enter the benchmark evaluation set.

---

## 2. Population Candidate Availability Master Table

| Population | Role | Planning Target | Candidates Found | Provenance-Verified | ZTF-Associated | Availability Status | Primary Discovery Catalog |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **SN_Ia** | In-Distribution | 50 | 50 | 50 | 50 | **SUFFICIENT_CANDIDATES** | ZTF BTS (Caltech/TNS) |
| **SN_II** | In-Distribution | 40 | 40 | 40 | 40 | **SUFFICIENT_CANDIDATES** | ZTF BTS (Caltech/TNS) |
| **Variable_Star** | In-Distribution | 25 | 25 | 25 | 25 | **SUFFICIENT_CANDIDATES** | Chen et al. 2020 (ApJS 249, 18) |
| **SLSN** | OOD Anomaly | 20 | 20 | 20 | 20 | **SUFFICIENT_CANDIDATES** | ZTF BTS / Perley et al. 2020 |
| **TDE** | OOD Anomaly | 15 | 15 | 15 | 15 | **SUFFICIENT_CANDIDATES** | ZTF BTS / van Velzen et al. 2021 |
| **Cataclysmic_Variable** | OOD Anomaly | 15 | 15 | 15 | 15 | **SUFFICIENT_CANDIDATES** | Szkody et al. 2020 (AJ 159, 198) |
| **Unclassified_Field_Star** | Control | 15 | 15 | 15 | 15 | **SUFFICIENT_CANDIDATES** | Gaia DR3 (I/355/gaiadr3) |
| **Stellar_Flare** | In-Distribution | 25 | 0 | 0 | 0 | **PROVENANCE_LIMITED** | Survey cadence limited (2–3d) |
| **FBOT** | OOD Anomaly | 10 | 0 | 0 | 0 | **INSUFFICIENT** | Extremely scarce literature sample |
| **LRN / ILRT** | OOD Anomaly | 10 | 0 | 0 | 0 | **INSUFFICIENT** | Scarce literature sample |
| **SN_Ibn / SN_Icn** | OOD Anomaly | 10 | 0 | 0 | 0 | **POTENTIALLY_SUFFICIENT** | 38 confirmed events exist in BTS |
| **Sparse / Edge Control** | Control | 15 | 0 | 0 | 0 | **UNVERIFIED** | Gating criteria pending definition |
| **Rejected / Pilot Controls** | Control / Rejected | — | 10 | 5 | 10 | **REJECTED_CONTROLS** | BTS Unconfirmed (5) + Pilots (5) |

---

## 3. Detailed Population Findings

### A. Fully Populated Provisional Populations (180 Active Candidates)

#### 1. Type Ia Supernovae (`SN_Ia`, 50 Candidates)
- **Candidates Discovered:** 50 spectroscopically confirmed Type Ia supernovae.
- **Source:** ZTF Bright Transient Survey (BTS) Public Explorer.
- **Provenance:** 100% spectroscopically verified with authoritative IAU TNS identifiers, spectroscopic redshifts ($z \in [0.015, 0.090]$), and clean apparent peak magnitudes ($m_{\text{peak}} \in [15.7, 18.5]$).
- **ZTF Association:** 100% coordinate-grounded with exact ZTF alert IDs (e.g. `ZTF21acipofv`, `ZTF18aajpjdi`).
- **Status:** **SUFFICIENT_CANDIDATES**.

#### 2. Type II Supernovae (`SN_II`, 40 Candidates)
- **Candidates Discovered:** 40 spectroscopically confirmed core-collapse supernovae (25 `SN II`, 12 `SN IIP`, 3 `SN IIb`).
- **Source:** ZTF Bright Transient Survey (BTS).
- **Provenance:** 100% spectroscopic confirmation in TNS/BTS with measured redshifts ($z \in [0.012, 0.058]$).
- **ZTF Association:** 100% coordinate-grounded with exact ZTF alert IDs (e.g. `ZTF18aapifti`, `ZTF18aaqkoyr`).
- **Status:** **SUFFICIENT_CANDIDATES**.

#### 3. Superluminous Supernovae (`SLSN`, 20 Candidates)
- **Candidates Discovered:** 20 spectroscopically confirmed SLSNe (10 `SLSN-I`, 10 `SLSN-II`).
- **Source:** ZTF BTS / Perley et al. 2020 SLSN Sample.
- **Provenance:** Spectroscopic classifications confirming extreme peak luminosities ($M < -21$) and characteristic broad light curve evolution.
- **ZTF Association:** 100% coordinate-grounded with exact ZTF alert IDs.
- **Status:** **SUFFICIENT_CANDIDATES**.

#### 4. Tidal Disruption Events (`TDE`, 15 Candidates)
- **Candidates Discovered:** 15 spectroscopically confirmed optical/UV Tidal Disruption Events.
- **Source:** ZTF BTS cross-referenced with van Velzen et al. (2021) and Hammerstein et al. (2023).
- **Provenance:** Spectroscopically confirmed black-hole stellar disruptions with measured redshifts and nuclear galactic coordinates.
- **ZTF Association:** 100% coordinate-grounded with exact ZTF alert IDs.
- **Status:** **SUFFICIENT_CANDIDATES**.

#### 5. Periodic Variable Stars (`Variable_Star`, 25 Candidates)
- **Candidates Discovered:** 25 cataloged periodic variable stars.
- **Source:** The ZTF Catalog of Periodic Variable Stars (Chen et al. 2020, ApJS 249, 18, VizieR `J/ApJS/249/18/table2`).
- **Provenance:** Authoritative catalog classifications including RR Lyrae, Cepheids, Delta Scuti, and W UMa contact binaries (`EW`, `BYDra`, `SR`) with measured periods ($P \in [0.25, 112.0]$ days).
- **ZTF Association:** 100% coordinate-grounded with exact ZTF catalog IDs (e.g. `ZTFJ000000.13+620605.8`).
- **Status:** **SUFFICIENT_CANDIDATES**.

#### 6. Cataclysmic Variables (`Cataclysmic_Variable`, 15 Candidates)
- **Candidates Discovered:** 15 confirmed cataclysmic variables / dwarf novae with observed ZTF outbursts.
- **Source:** Szkody et al. (2020, AJ 159, 198, VizieR `J/AJ/159/198/table1`).
- **Provenance:** Published spectroscopic confirmation and multi-epoch outburst counts ($N_{\text{out}} \ge 1$, up to superoutbursts) during ZTF Year 1.
- **ZTF Association:** 100% coordinate-grounded with exact ZTF identifiers (e.g. `ZTF17aaaemzh`, `ZTF18abdlywu`).
- **Status:** **SUFFICIENT_CANDIDATES**.

#### 7. Unclassified Field Stars (`Unclassified_Field_Star`, 15 Candidates)
- **Candidates Discovered:** 15 non-variable astrometrically standard reference stars in ZTF Field 686.
- **Source:** Gaia Data Release 3 (Gaia Collaboration 2022, VizieR `I/355/gaiadr3`).
- **Provenance:** Gaia DR3 astrometrically verified single stars with `RUWE < 1.15` and `phot_variable_flag = NOT_AVAILABLE`.
- **ZTF Association:** 100% coordinate-grounded within ZTF primary survey field 686.
- **Status:** **SUFFICIENT_CANDIDATES**.

---

### B. Rejected & Control Registry Entries (10 Objects)

To ensure the registry actively tracks rejected and unverified candidates, 10 records are cataloged with `candidate_role = "rejected"`:

1. **Unconfirmed BTS Transients (5 Objects):**
   - `REJ_UNCONFIRMED_001` through `REJ_UNCONFIRMED_005` (e.g. `ZTF20acglhmi / AT2020usb`, `ZTF18acxgqij / AT2017hio`).
   - **Rejection Reason:** "Classification unconfirmed: BTS entry has type='-' with no spectroscopic confirmation."
   - **Scientific Purpose:** Enforces the rule that appearance or alert presence without spectroscopic confirmation results in rejection.
2. **Historical Exploratory Pilot Objects (5 Objects):**
   - `REJ_PILOT_001` (`SN_2019np`), `REJ_PILOT_002` (`SN_2020jfo`), `REJ_PILOT_003` (`AT_2018cow`), `REJ_PILOT_004` (`SN_2018zd`), `REJ_PILOT_005` (`ZTF_J195200.60+295217.4`).
   - **Rejection Reason:** "Excluded from benchmark: isolated exploratory transfer-pilot object."
   - **Scientific Purpose:** Guarantees that the exploratory compatibility pilot targets can never enter the benchmark evaluation set.

---

### C. Unconfirmed Populations Analysis

1. **`Stellar_Flare` (0 / 25 Candidates):**
   - **Status:** `PROVENANCE_LIMITED`.
   - **Diagnosis:** The standard ZTF public survey cadence (2–3 days) severely undersamples impulsive M-dwarf reconnection flares ($t_{\text{rise}} < 1\text{h}$, decay $< 12\text{h}$). Populating this class requires cross-matching ZTF high-cadence partnership fields (e.g. 6-observation/night cadences) against TESS/Kepler flare star catalogs.
2. **`FBOT` (0 / 10 Candidates):**
   - **Status:** `INSUFFICIENT`.
   - **Diagnosis:** Extremely rare class. Fewer than 15 confirmed optical Fast Blue Optical Transients exist across all astronomical literature. Target of 10 in ZTF is unfeasible under strict spectroscopic criteria; the target must scale down to $3 - 5$ verified objects.
3. **`LRN / ILRT` (0 / 10 Candidates):**
   - **Status:** `INSUFFICIENT`.
   - **Diagnosis:** Stellar merger transients and intermediate-luminosity red transients are intrinsically rare ($< 20$ known). Target of 10 must scale down to confirmed literature events (e.g. AT 2019zhd, AT 2020hat).
4. **`SN_Ibn / SN_Icn` (0 / 10 Candidates):**
   - **Status:** `POTENTIALLY_SUFFICIENT`.
   - **Diagnosis:** The ZTF BTS contains 32 confirmed `SN Ibn` and 6 confirmed `SN Icn`. These 38 events can be extracted in the next discovery pass to populate this class.

---

## 4. Summary of Collectible Populations

| Group | Population | Discovered Candidates | Feasibility Status | Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| **In-Distribution** | `SN_Ia` | 50 | SUFFICIENT | Proceed to photometric retrieval gate |
| **In-Distribution** | `SN_II` | 40 | SUFFICIENT | Proceed to photometric retrieval gate |
| **In-Distribution** | `Variable_Star` | 25 | SUFFICIENT | Proceed to photometric retrieval gate |
| **OOD Anomaly** | `SLSN` | 20 | SUFFICIENT | Proceed to photometric retrieval gate |
| **OOD Anomaly** | `TDE` | 15 | SUFFICIENT | Proceed to photometric retrieval gate |
| **OOD Anomaly** | `Cataclysmic_Variable` | 15 | SUFFICIENT | Proceed to photometric retrieval gate |
| **Control** | `Unclassified_Field_Star` | 15 | SUFFICIENT | Proceed to photometric retrieval gate |
| **Subtotal** | **7 Populations** | **180 Candidates** | **SUFFICIENT** | **Ready for photometric coverage qualification** |

---

## 5. Candidate Sequence Boundary

```
[Candidate Discovered (180)]  <--- CURRENT COMPLETED STAGE
            ↓
[Classification Verified (180 / 180 Spectroscopic / Catalog Grounded)]
            ↓
[ZTF Association Verified (100% Coordinate-Grounded)]
            ↓
[Photometry Retrieval (NEXT STAGE - NOT STARTED)]
            ↓
[Quality Filtering & Transient Windowing]
            ↓
[Coverage & Padding Evaluation (N_tokens, P_pad <= threshold)]
            ↓
[Final Benchmark Manifest Inclusion]
```

Candidate discovery establishes catalog availability only. Final benchmark admission is contingent on photometric coverage and padding evaluation in the next phase.
