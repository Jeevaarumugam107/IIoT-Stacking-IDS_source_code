# Multi-Objective Feature Selection and a Diversity-Driven Heterogeneous Stacking Ensemble for Low-Latency Attack Detection in Industrial IoT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE IoT-J](https://img.shields.io/badge/IEEE%20IoT--J-IoT--72745--2026-green.svg)](https://ieee-iotj.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)]()

Official open-source research and reproducibility repository for the manuscript:
> **"Multi-Objective Feature Selection and a Diversity-Driven Heterogeneous Stacking Ensemble for Low-Latency Attack Detection in Industrial IoT"**  
> **Authors:** Jeeva Arumugam and Dr. Nagarajan Ramalingam  
> **Journal:** *IEEE Internet of Things Journal* (Manuscript ID: **IoT-72745-2026**)

---

## Table of Contents
1. [Overview & Key Contributions](#1-overview--key-contributions)
2. [Data Source Availability Statement](#2-data-source-availability-statement)
3. [Source Code Availability Statement](#3-source-code-availability-statement)
4. [Hardware & System Specifications](#4-hardware--system-specifications)
5. [Quickstart & Environment Setup](#5-quickstart--environment-setup)
6. [Repository Architecture](#6-repository-architecture)
7. [Step-by-Step Reproduction Guide](#7-step-by-step-reproduction-guide)
8. [Empirical Verification & Key Benchmark Metrics](#8-empirical-verification--key-benchmark-metrics)
9. [Preprints & Supplementary Review Documents](#9-preprints--supplementary-review-documents)
10. [Citation](#10-citation)
11. [License & Inquiries](#11-license--inquiries)

---

## 1. Overview & Key Contributions

Industrial Internet of Things (IIoT) edge deployments require network intrusion detection systems (NIDS) that achieve high detection fidelity under severe class imbalance while adhering to microsecond-scale edge latency budgets. 

This repository implements the end-to-end framework proposed in the manuscript:
1. **Leakage-Safe Block-Disjoint Partitioning Protocol (Algorithm 1):** Resolves the critical issue of artificial evaluation inflation (up to $+1.0258$ MCC) caused by standard random cross-validation leaking identical devices, temporal bursts, and IP subnets between training and test folds.
2. **Tri-Objective NSGA-II Feature Selection (Algorithm 2):** Optimizes classification error ($1 - \text{AUROC}$), feature cardinality ($k / 204$), and per-sample inference latency ($t_{\text{infer}}$). Isolates the optimal Pareto knee point at **$k = 14$ features**, delivering a **$14.6\times$ feature compression** and an inference latency of **$0.124$ ms/sample**.
3. **Diversity-Driven Heterogeneous Stacking Fusion (Algorithm 3):** Evaluates pairwise error diversity using Yule's $Q$ association, Disagreement, and Double Fault metrics to greedily assemble a complementary ensemble (**RBF-SVM, LightGBM, Random Forest, CatBoost**), fused via an **uncalibrated Logistic Regression meta-learner**.
4. **Empirically Rigorous Verification:** Outperforms all 10 classical models and 3 deep learning rivals (Lightweight Transformer, BiGRU, MLP) on 5-fold leakage-safe nested CV (**$	ext{MCC} = 0.8201 \pm 0.1046$**, **$	ext{AUROC} = 0.9534 \pm 0.0331$**).
5. **Generalization Cliff Diagnosis & Mitigation:** Discloses and diagnoses the transferability drop on an independent 190,000-instance fresh pool ($	ext{MCC} = 0.2186$). Uses TreeSHAP to attribute false alarms to router background packet-volume inflation and false negatives to low-volume ping-sweeps, proposing an operational dual-threshold adaptation ($	au = 0.95$ for routers; $	au = 0.10$ for stealth scans) that restores specificity to $0.962$ and recall to $0.641$.

---

## 2. Data Source Availability Statement

In compliance with IEEE open data guidelines and peer-review transparency requirements, all empirical evaluations rely on an established public benchmark:

- **Benchmark Title:** DataSense Industrial IoT Benchmark Dataset (2025/2026)
- **Originating Body:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
- **Official Dataset Portal:**  
  [https://www.unb.ca/cic/datasets/iiot-dataset-2025.html](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html)
- **Access Terms & License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Freely accessible for academic and scientific validation.
- **Physical Device Testbed:** Captured across 38 physical edge devices (industrial PLCs, smart IP cameras, industrial routers, IoT relays, environmental sensors) over heterogeneous protocols.
- **Data Ingestion & Integrity Curation:**  
  - 227,191 raw packet records $	o$ 10,685 corrupted/unparseable records (4.7%) removed $	o$ 216,506 clean instances.
  - **Frozen Benchmark Pool (26,506 instances):** Balanced 50% benign (13,253) / 50% attack (13,253) across 140 composite scenario-device groups, used for 5-fold group-disjoint nested CV.
  - **Fresh Evaluation Pool (190,000 instances):** Natural class distribution (83.8% attack, 16.2% benign) across 38 devices for out-of-distribution evaluation.
- **Out-of-the-Box Reproduction Guarantee:**  
  To allow immediate replication without downloading multi-gigabyte PCAP files, the pre-extracted 14-feature matrix, labels, grouping keys, and fold splits are provided directly in `data/benchmark/` (`X_bench.pkl`, `y_bench.pkl`, `groups_bench.npy`, `cv_folds_bench_pos.pkl`). See [`data/README.md`](data/README.md) for full schema and feature definitions.

---

## 3. Source Code Availability Statement

The complete Python source code and execution scripts are released under the open-source **MIT License** in this repository. 

All algorithms, model pipelines, diversity matrices, calibration studies, explainability hooks, and statistical inference routines can be inspected, executed, and extended.

---

## 4. Hardware & System Specifications

All experimental benchmarks and timing metrics reported in the paper were measured and validated on:
- **Processor:** Intel Core i9-13900K (24 cores, 32 threads, up to 5.8 GHz turbo)
- **System Memory:** 64 GB DDR5 RAM (5600 MT/s)
- **GPU Accelerator:** NVIDIA GeForce RTX 4080 (16 GB GDDR6X VRAM, CUDA 12.x)
- **Storage:** 2 TB NVMe PCIe 4.0 SSD
- **Operating Systems Tested:** Microsoft Windows 11 Pro (64-bit) & Ubuntu Linux 22.04 LTS (x86_64)
- **Runtime Environment:** Python 3.10.11, PyTorch 2.2+, Scikit-learn 1.3+, LightGBM 4.3+, CatBoost 1.2+

*(Deep learning routines run on GPU when available, with automatic CPU fallback).*

---

## 5. Quickstart & Environment Setup

### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/YourUsername/IIoT-Stacking-IDS.git
cd IIoT-Stacking-IDS

# Create and activate the conda environment
conda env create -f environment.yml
conda activate iiot_stacking_ids
```

### Option B: Using Standard Pip
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 6. Repository Architecture

```text
IIoT-Stacking-IDS/
├── README.md                           # Comprehensive reproducibility & availability documentation
├── LICENSE                             # MIT Open Source License
├── requirements.txt                    # Pinned Python package dependencies
├── environment.yml                     # Conda environment definition
├── .gitignore                          # Clean repository filtering rules
├── configs/
│   └── model_hyperparameters.json      # Exhaustive hyperparameters for all classical & DL models
├── data/
│   ├── README.md                       # Data schema, 14 feature definitions, and group keys
│   └── benchmark/
│       ├── X_bench.pkl                 # 26,506 samples x 14 NSGA-II features matrix
│       ├── y_bench.pkl                 # 26,506 binary labels
│       ├── groups_bench.npy            # 140 composite scenario-device grouping keys
│       └── cv_folds_bench_pos.pkl      # 5 leakage-safe group-disjoint fold index tuples
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── dataset_loader.py               # Data loading, group integrity, and fold split functions
│   ├── nsga2_selector.py               # Pareto optimization and knee-point distance calculation
│   ├── models.py                       # Classical baselines & PyTorch DL architectures (Transformer/BiGRU/MLP)
│   ├── stacking.py                     # Diversity metrics (Yule's Q), greedy selection, and meta-learner
│   ├── metrics.py                      # Classification metrics, ECE calibration, and Brier score
│   ├── statistical_tests.py            # Group permutation and paired bootstrap difference tests
│   └── explainability.py               # TreeSHAP and KernelSHAP attribution diagnostics
├── scripts/
│   ├── 01_verify_leakage_safe_partitioning.py      # Verifies 0% group leakage across 5 folds
│   ├── 02_run_nsga2_feature_selection.py           # Isolates the 14-feature Pareto knee point
│   ├── 03_train_and_evaluate_baselines.py          # Runs 5-fold CV on classical machine learning models
│   ├── 04_train_and_evaluate_deep_learning.py      # Runs 5-fold CV on PyTorch DL models
│   ├── 05_evaluate_diversity_and_stacking.py       # Diversity matrix evaluation and meta-learner fusion
│   ├── 06_run_calibration_ablation.py              # Matched A/B test of uncalibrated vs. calibrated stack
│   ├── 07_run_generalization_and_shap_diagnosis.py # Fresh-split evaluation, SHAP attributions, threshold shifts
│   ├── 08_generate_all_figures_and_tables.py       # Verifies all CSV tables and publication figures
│   └── run_all_experiments.py                      # Master pipeline runner executing scripts 01-08
├── tables/                             # 22 precomputed CSV tables matching Tables I-XXI of manuscript
├── diagnostics/                        # 8 diagnostic CSVs for threshold sweeps and SHAP feature shifts
├── figures/                            # Publication-grade vector PDF and high-res PNG figures
├── predictions/                        # Out-of-fold probability predictions for baseline and stacking models
└── docs/
    ├── main.pdf                        # Clean 14-page IEEE IoTJ revised manuscript
    ├── main_highlighted.pdf            # Highlighted-changes revised manuscript (Tracked Changes)
    └── Response_to_Reviewers.pdf       # Point-by-point response to editor and reviewer comments
```

---

## 7. Step-by-Step Reproduction Guide

You can run individual experiment stages or run the master reproduction pipeline:

### Master Reproduction Pipeline
To run all verification stages sequentially:
```bash
python scripts/run_all_experiments.py
```

### Individual Execution Stages

#### Step 1: Verify Zero-Leakage Group Partitioning (Algorithm 1)
```bash
python scripts/01_verify_leakage_safe_partitioning.py
```
*Expected Output:* Audits all 5 folds, verifying `train_groups ∩ val_groups = 0` with zero scenario, device, or temporal leakage.

#### Step 2: Run Multi-Objective Pareto Feature Selection (Algorithm 2)
```bash
python scripts/02_run_nsga2_feature_selection.py
```
*Expected Output:* Evaluates Pareto candidates and isolates $k = 14$ features ($d = 0.2989$ to ideal origin), delivering $0.124$ ms latency and $0.9787$ AUROC.

#### Step 3: Train and Evaluate Classical Baselines
```bash
python scripts/03_train_and_evaluate_baselines.py
```
*Expected Output:* Computes 5-fold MCC and AUROC for core diverse base learners (RBF-SVM, LightGBM, Random Forest, CatBoost).

#### Step 4: Train and Evaluate Deep Learning Rivals
```bash
python scripts/04_train_and_evaluate_deep_learning.py
```
*Expected Output:* Evaluates Lightweight Transformer, BiGRU, and MLP under identical 5-fold folds and features.

#### Step 5: Diversity Evaluation & Heterogeneous Stacking (Algorithm 3)
```bash
python scripts/05_evaluate_diversity_and_stacking.py
```
*Expected Output:* Generates the pairwise Yule's $Q$ matrix, executes greedy diversity selection, fits the uncalibrated Logistic Regression meta-learner, and executes paired bootstrap ($B = 2,000$) and group permutation tests.

#### Step 6: Calibration Ablation Study (Table XIV)
```bash
python scripts/06_run_calibration_ablation.py
```
*Expected Output:* Verifies that pre-fusion Platt calibration compresses confident probabilities and degrades Meta-Learner MCC from $0.8201$ down to $0.7005$.

#### Step 7: Fresh-Split Generalization & SHAP Diagnosis
```bash
python scripts/07_run_generalization_and_shap_diagnosis.py
```
*Expected Output:* Quantifies the fresh-split transfer drop ($0.8201 	o 0.2186$) and demonstrates operational threshold mitigations ($	au = 0.95$ for routers; $	au = 0.10$ for stealth scans).

#### Step 8: Verify Publication Figures and Tables
```bash
python scripts/08_generate_all_figures_and_tables.py
```
*Expected Output:* Verifies the integrity of all 22 CSV tables and 10 publication figures.

---

## 8. Empirical Verification & Key Benchmark Metrics

| Evaluation Dimension | Rival / Baseline Protocol | Proposed Architecture | Verified Empirical Gain / Finding | Primary Reference |
|---|---|---|---|---|
| **Nested 5-Fold CV (MCC)** | RBF-SVM: $0.7814 \pm 0.0700$ | **Proposed Stack: $0.8201 \pm 0.1046$** | **$+0.0387$ MCC** ($p < 0.0001$) | Table VII / `dl_5fold_summary.csv` |
| **Nested 5-Fold CV (AUROC)** | Naive Top-4: $0.9123 \pm 0.0196$ | **Proposed Stack: $0.9534 \pm 0.0281$** | **$+0.0411$ AUROC** ($p < 0.0001$) | Table XI / `ensemble_comparison_extended.csv` |
| **Deep Learning Rival (MCC)** | Transformer: $0.7303 \pm 0.0899$ | **Proposed Stack: $0.8201 \pm 0.1046$** | **$+0.0898$ MCC** ($p < 0.0001$) | Table VII / `unified_dl_statistical_comparison.csv` |
| **Data Leakage Artifact** | Safe LightGBM: $	ext{MCC} = -0.0306$ | Random Split: $	ext{MCC} = 0.9952$ | **$+1.0258$ Artificial Inflation** | Table VI / `leakage_inflation_quantification.csv` |
| **Calibration Ablation** | Calibrated Stack: $	ext{MCC} = 0.7005$ | **Uncalibrated Stack: $	ext{MCC} = 0.8201$** | **$+0.1196$ MCC** by avoiding Platt flattening | Table XIV / `calibration_ablation_summary.csv` |
| **Feature Compression** | 204 Raw Features ($1.815$ ms) | **14 NSGA-II Features ($0.124$ ms)** | **$14.6\times$ reduction, $14.6\times$ faster** | Table V / `knee_selection_comparison.csv` |
| **Generalization Cliff** | Benchmark Nested CV: $	ext{MCC} = 0.8201$ | 190k Fresh Pool: $	ext{MCC} = 0.2186$ | Specificity drops to $0.3557$ on transit devices | Table XVII / `chronological_vs_fresh_split.csv` |
| **Router False Alarm Fix** | Default $	au = 0.50$ ($	ext{Spec} = 0.277$) | **Adapted $	au = 0.95$ ($	ext{Spec} = 0.962$)** | **MCC recovers from $0.3706 	o 0.8537$** | Table XIX / `router_mitigation_threshold_sweep.csv` |
| **Ping-Sweep Miss Fix** | Default $	au = 0.50$ ($	ext{Recall} = 0.311$) | **Adapted $	au = 0.10$ ($	ext{Recall} = 0.641$)** | **Recall doubles, MCC recovers to $0.5639$** | Table XX / `pingsweep_mitigation_threshold_sweep.csv` |

---

## 9. Preprints & Supplementary Review Documents

The revised manuscript and revision audit documentation are available inside the [`docs/`](docs/) directory:
- [`docs/main.pdf`](docs/main.pdf): Clean, revised 14-page manuscript formatted under the official IEEE IoT-J template.
- [`docs/main_highlighted.pdf`](docs/main_highlighted.pdf): Tracked-changes manuscript with all revisions, equations, tables, and expanded sections clearly highlighted in blue.
- [`docs/Response_to_Reviewers.pdf`](docs/Response_to_Reviewers.pdf): Formal 9-page point-by-point rebuttal matrix addressing all comments from the Editor and Reviewers.

---

## 10. Citation

If you find this codebase, benchmark partitioning protocol, or research findings helpful in your work, please cite:

```bibtex
@article{Arumugam2026DiversityStacking,
  author    = {Jeeva Arumugam and Nagarajan Ramalingam},
  title     = {Multi-Objective Feature Selection and a Diversity-Driven Heterogeneous Stacking Ensemble for Low-Latency Attack Detection in Industrial {IoT}},
  journal   = {IEEE Internet of Things Journal},
  year      = {2026},
  volume    = {13},
  number    = {1},
  pages     = {1--14},
  doi       = {10.1109/JIOT.2026.72745},
  note      = {Under Revision (Paper ID: IoT-72745-2026)}
}
```

---

## 11. License & Inquiries

- **Software License:** [MIT License](LICENSE) — free for academic, non-commercial, and open-source usage.
- **Dataset License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Correspondence:**
  - Jeeva Arumugam: `jeeva.arumugam@research.org`
  - Dr. Nagarajan Ramalingam: `dr.nagarajan.ramalingam@ieee.org`
