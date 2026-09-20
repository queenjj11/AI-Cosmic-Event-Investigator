# AI Cosmic Event Investigator (ACEI)
### Multimodal Anomaly Detection & Autonomous Scientific Reasoning for Astronomical Transients

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **AI Cosmic Event Investigator (ACEI)** is a closed-loop scientific reasoning pipeline for time-domain astronomical surveys (e.g., Zwicky Transient Facility, ASAS-SN). Rather than stopping at standard closed-set classification, ACEI:
1. **Detects** out-of-distribution (OOD) cosmic anomalies via a 3-signal calibrated ensemble and the Real-ZTF Light-Curve Anomaly Detector v2.
2. **Explains** detected anomalies by generating ranked, evidence-grounded hypotheses via Retrieval-Augmented Generation (RAG) over astrophysics literature and transient catalogs.
3. **Calibrates** LLM uncertainty into empirical probabilities (isotonic regression, temperature scaling).
4. **Recommends** the optimal follow-up observation by maximizing expected Shannon information gain under observational cost constraints.

---

## Architecture Overview

```
                         Survey Alerts (ZTF, ASAS-SN)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ↓                                                   ↓
   Image Stream (Postage Stamps)                   Light-Curve Stream (Flux vs Time)
           │                                                   │
           ↓                                                   ↓
   ResNet-18 Image Encoder                          Time2Vec + Transformer Encoder
           └─────────────────────────┬─────────────────────────┘
                                     ↓
                         Multimodal Fusion Layer
                     (Cross-Attention / Concat+MLP)
                                     ↓
                         3-Signal Anomaly Ensemble &
                    Real-ZTF Light-Curve Detector v2
                 ┌───────────────────┼───────────────────┐
                 ↓                   ↓                   ↓
       Autoencoder Recon     Mahalanobis Dist      Energy OOD
                 └───────────────────┬───────────────────┘
                                     ↓
                          Calibrated Anomaly Score
                                     │
            ┌────────────────────────┴────────────────────────┐
            ↓ (Normal / In-Distribution)                       ↓ (Anomalous / Research Score)
    Standard Classification                         AI Investigator Agent (RAG)
                                                               │
                                             ┌─────────────────┴─────────────────┐
                                             ↓                                   ↓
                                    Astronomical Literature            Bayesian Hypothesis
                                         Vector Store                        Ranker
                                             └─────────────────┬─────────────────┘
                                                               ↓
                                                    Calibrated Hypotheses
                                                               ↓
                                                Active Observation Recommender
                                            (Max Expected Shannon Information Gain)
                                                               ↓
                                             Interactive Next.js 3D Workstation
```

---

## Quickstart

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/AI-Cosmic-Event-Investigator.git
cd AI-Cosmic-Event-Investigator
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline Demo
Execute the full closed-loop pipeline on sample and held-out astronomical events:
```bash
python -m src.pipeline.acei_pipeline --demo
```

### 3. Launch the Web Dashboards

#### A. Advanced Scientific Web Dashboard (Next.js 14 + FastAPI 3D Workstation)
The primary observatory mission control interface provides multi-band light curves ($g, r, i$), React Three Fiber 3D latent representation subspace models, Bayesian hypothesis evidence chains, and active learning observation queues:

```bash
# Terminal 1: Start FastAPI backend (port 8000)
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Next.js frontend (port 3000)
cd frontend
npm install
npm run dev
```
Navigate to [http://localhost:3000](http://localhost:3000).

#### B. Research / Internal Dashboard (Streamlit)
The original Streamlit interface is preserved intact for research inspection:
```bash
streamlit run dashboard/app.py
```

### 4. Run Test Suite
Run the full automated test suite (147 tests including backend API, frozen benchmark verification, and investigator pipeline):
```bash
pytest tests/ -v
```

---

## Directory Structure

```text
AI-Cosmic-Event-Investigator/
├── config.yaml                     # Central configuration for data, models, anomaly thresholds
├── requirements.txt                # Python package dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules
├── data/                           # Data storage (raw alerts, processed arrays, frozen primary benchmark)
├── models/                         # Checkpoints (acei_multimodal_production.pt, real_ztf_lc_detector_v2.pkl)
├── notebooks/                      # Exploratory and evaluation Jupyter notebooks
├── src/
│   ├── data/                       # Ingestion, schema, loaders, object-level splitters
│   ├── features/                   # Astrophysical light curve & image morphology features
│   ├── baselines/                  # Random Forest, CNN, Bi-LSTM baselines & evaluation harness
│   ├── models/                     # Image encoder, Transformer lightcurve encoder, Multimodal fusion
│   ├── anomaly_detection/          # Autoencoder, Mahalanobis, Energy OOD & Real-ZTF v2 Detector
│   ├── knowledge_base/             # Astro literature chunker, embedder, and vector store
│   ├── investigator/               # Prompt templates, RAG retriever, hypothesis ranker, calibrator
│   ├── recommender/                # Candidate actions, simulator, Bayesian updater, information gain
│   ├── evaluation/                 # AUROC, FDR, ECE, Brier score, bootstrap confidence intervals
│   └── pipeline/                   # Master ACEI pipeline orchestrator
├── frontend/                       # Next.js 14 + React Three Fiber 3D Scientific Workstation
├── dashboard/                      # Streamlit dashboard (app, components, interactive Plotly plots)
├── knowledge_base/                 # Seed astronomy research papers and transient catalogs
├── reports/                        # Logged evaluation reports, metrics, and markdown audits
└── tests/                          # Automated test suite (147 tests)
```

---

## Evaluation Methodology

ACEI is evaluated across five rigorous axes:
1. **Classification Performance**: Precision, Recall, F1, AUROC on known classes.
2. **Anomaly Detection**: AUROC (known vs. held-out rare classes like LRN, SLSN, TDE), False Discovery Rate (FDR), and detection latency.
3. **Real-ZTF Light-Curve Anomaly Detector v2**: Evaluated on 106-object primary benchmark ($\text{AUROC} = 0.6881, \text{AUPRC} = 0.6862$, $\text{FPR} = 5.17\%$, Out-Of-Fold $\text{OOF AUROC} = 0.6795, \text{OOF AUPRC} = 0.6687$).
4. **Hypothesis Grounding**: Groundedness ratio (% claims verifiable from retrieved papers/catalogs), Top-1 hypothesis accuracy.
5. **Calibration Quality**: Expected Calibration Error (ECE) and Brier Score before and after calibration.
6. **Decision Quality**: Average Shannon entropy reduction across candidate follow-up actions compared to random and fixed policies.
7. **Statistical Significance**: All reported metrics include 95% non-parametric bootstrap confidence intervals (1,000 resamples).
