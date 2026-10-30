# Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21416987.svg)](https://doi.org/10.5281/zenodo.21416987)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Authors:** Grey Nearing, Stephanie Rees, Martin Gauch, Justin Ginnetti, Juliet Rothenberg

This repository contains the complete source code, evaluation pipelines, trained model scripts, and publication figure/table generators for *"Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity"*.

All first-party derived datasets, trained `SkillPredictor` models (`76` stacked ensemble models across 38 hydrological and flood-event metrics), out-of-fold cross-validation predictions, global HydroBASINS Level 12 (`964,084` basins) predictions, five-factor TreeSHAP and counterfactual attribution tables, and $N$-gauge network optimization outputs are permanently archived on Zenodo at **[https://doi.org/10.5281/zenodo.21416987](https://doi.org/10.5281/zenodo.21416987)**.

---

## Overview & Key Findings

1. **Global Flood Exposure Accounting ($E_{\text{total}} = 21.17\text{M}$ people/yr)**:
   Intersecting 1 km WorldPop (2020) population density with Fathom v3 defended fluvial flood hazard maps across 964,084 global HydroBASINS Level 12 catchments yields a global Expected Annual Flood Exposure of **21.17 million people/yr** integrated across return periods $T \in \{2, 5, 10, 20, 50, 100\}\text{ yr}$.
2. **Decomposition of Global Flood Exposure into Operational Warning and Technical Blockers**:
   - **Covered by Gauged AI Forecasts ($E_{\text{covered,gauged}}$)**: **8.39M people/yr (39.6%)**, comprising **7.46M people/yr (35.2%)** in local and nested-upstream gauged basins (`gauged` skill) and **0.93M people/yr (4.4%)** recovered via spatial regionalization from existing global gauge networks (`ungauged` skill).
   - **Total Uncovered Flood Exposure ($E_{\text{uncovered}}$)**: **12.78M people/yr (60.4%)**, which decomposes into three distinct technical blockers:
     1. **Local Gauge Gap ($B_{\text{local}}$)**: **0.65M people/yr (3.1% of total exposure; 5.1% of uncovered exposure)** — the incremental exposure that would be covered if every basin on Earth had a 76-year historical in-situ streamflow gauge.
     2. **Global Gauge Network Gap ($B_{\text{network}}$)**: **1.23M people/yr (5.8% of total exposure; 9.6% of uncovered exposure)** — the penalty from nested-basin topological disconnection ($0.56\text{M/yr}$) and multivariate hydroclimatic out-of-distribution (OOD) extrapolation ($0.67\text{M/yr}$) relative to the public training archive.
     3. **Gauged Model Error ($B_{\text{model}}$)**: **10.90M people/yr (51.5% of total exposure; 85.3% of uncovered exposure)** — residual missed flood exposure even under full local and global gauge coverage, driven primarily by hydroclimatic aridity and satellite precipitation/meteorological forcing uncertainty ($58.4\%$ of $B_{\text{model}}$ via TreeSHAP).
3. **Operational Warning Gaps vs. Technical Forecasting Blockers**:
   Combining Common Alerting Protocol (CAP) operational alert feeds with domestic warning system inventories shows that **65 countries with effective warning systems** account for **83.6% (17.69M/yr)** of global flood exposure, whereas **99 countries lacking effective warning systems** account for **16.4% (3.47M/yr)** of global flood exposure (`B_local = 4.6%`, `B_network = 10.6%`, `B_model = 53.7%`).
4. **Diminishing Returns of Joint $N$-Gauge Placement**:
   Evaluating multi-gauge candidate sets across the full upstream/downstream routing topology (`evaluate_n_gauge_sets.py`) demonstrates that adding $N = 1,000$ optimally selected gauges globally reduces uncovered exposure by at most **179,012 people/yr (1.40% of global uncovered exposure)** under population-exposure prioritization, **109,570 people/yr (0.86%)** under equal-per-country allocation, and **2,445 people/yr (0.02%)** under pure hydroclimatic representativeness sampling.

---

## Repository Structure

```text
geographical-bias-paper/
├── 0_calculate_overlap_fractions.py        # Step 0: Topological upstream/downstream overlap fractions
├── 1_anomaly_scores.py                     # Step 1: Isolation Forest, One-Class SVM & LOF OOD scores
├── 2_calculate_record_lengths.py           # Step 2: Daily streamflow record lengths per GRDC/Caravan gauge
├── 3_create_datasets.py                    # Step 3: Assemble 95-feature gauged, ungauged & HyBAS matrices
├── 4_hybas_to_country_mapping.py           # Step 4: Spatial join of 964,084 HyBAS basins to countries
├── 5_model_cross_validation.py             # Step 5: 10-fold CV of SkillPredictor across all 38 metrics
├── 6_global_training_and_inference.py      # Step 6: Full-archive SkillPredictor fit & 964k HyBAS inference
├── 7_compile_table_1_stats.py              # Step 7: Compile Table 1 & Extended Data Table 1 summary stats
├── 8_factor_analysis_stratified_sample.py  # Step 8: Exact KernelSHAP attribution on stratified samples
├── 9_figures_and_tables_for_paper.py       # Step 9: Generate Figures 1-5, ED Figures 1-6 & Tables
├── 10_calculate_global_exposure.py         # Step 10: Intersect WorldPop & Fathom v3 across Level 12 HyBAS
├── 11_analyze_gauge_placement_results.py   # Step 11:Exposure blocker decomposition & N-gauge curves
├── 12_five_factor_exposure_decomposition.py# Step 12: Five-factor counterfactual & 96-feature TreeSHAP
├── select_candidates.py                    # Candidate basin selection (Global, Country, OOD, Representativeness)
├── evaluate_gauges.py                      # Single-gauge marginal exposure reduction evaluator
├── evaluate_n_gauge_sets.py                # Joint N-gauge topological & OOD exposure reduction evaluator
├── compile_exposures.py                    # Aggregator for sharded gauge evaluation runs
├── generate_splits_fast.py                 # 10-fold random and separated CV split generator/verifier
├── humanitarian_risk_metrics.py            # INFORM vulnerability & coping capacity risk analysis
├── shap_global_visualization.py            # Multi-metric SHAP category & feature visualization suite
├── ERA_prompt.ipynb                        # Benchmark notebook used for SkillPredictor architecture search
├── utils/
│   ├── cap_utils.py                        # 4-tier and binary CAP operational flood warning classifications
│   ├── feature_utils.py                    # 96-feature descriptions, 8-category taxonomy & color palettes
│   ├── file_paths.py                       # Centralized path configuration (GEO_BIAS_DATA_DIR / OUTPUT_DIR)
│   ├── gfile_compat.py                     # Standard-library file I/O helpers
│   ├── io_utils.py                         # CSV, GeoJSON, and Zarr v2/v3 dataset loaders
│   ├── run_pipeline_single_metric.py       # Single-metric runner for Steps 5 & 6
│   ├── skill_predictor.py                  # Stacked GBDT (LightGBM + XGBoost + HistGBDT) SkillPredictor
│   ├── stats_utils.py                      # Non-parametric tests, effect sizes & trapezoidal integration
│   └── viz_utils.py                        # Global map and cross-validation diagnostic plotters
├── hydro_metrics/
│   ├── data_paths.py                       # Paths for raw Dual-LSTM Zarr predictions (1986-2023)
│   ├── evaluation_utils.py                 # Peak-detection precision/recall & hydrograph evaluation
│   ├── hydro_metrics_test.py               # Unit tests for hydro_metrics
│   ├── loading_utils.py                    # Zarr & NetCDF hydrograph/return-period loaders
│   ├── metrics.py                          # NSE, KGE, bias, correlation & variability metrics
│   ├── metrics_utils.py                    # Time-series masking and alignment utilities
│   ├── recalculate_metrics.py              # Recomputes 1986-2023 metrics across all 38 target metrics
│   ├── return_period_metrics.py            # Return-period exceedance event precision & recall
│   └── return_period_calculator/           # Gumbel/log-Pearson return-period threshold fitting package
└── v2_exposure_blockers/                   # Pre-generated publication figures, tables, and summary CSV/JSONs
    ├── figures_and_tables_for_paper/       # Figures 1-5, Extended Data Figures 1, 3-6, Table 1, ED Tables 1-3
    ├── candidates/                         # Candidate gauge lists for all 4 placement strategies
    └── statistics/                         # Country, regional, income, CAP-tier & 5-factor summary tables
```

---

## Installation & Environment Setup

Create a Conda environment from `environment.yml` (or install via `pip` from `pyproject.toml`):

```bash
conda env create -f environment.yml
conda activate geographical-bias-paper
# Or via pip:
pip install -e .
```

Run the unit test suite to verify the installation:

```bash
python3 hydro_metrics/hydro_metrics_test.py
```

---

## Downloading the Zenodo Dataset (`10.5281/zenodo.21416987`)

Download and extract the archived data from **[https://doi.org/10.5281/zenodo.21416987](https://doi.org/10.5281/zenodo.21416987)** into `./data` (or any directory of your choice, pointed to by `GEO_BIAS_DATA_DIR`):

```bash
mkdir -p data
# After downloading the tar.gz archives from https://doi.org/10.5281/zenodo.21416987 into ./data:
for archive in data/*.tar.gz; do
  tar -xzf "$archive" -C data/
done
export GEO_BIAS_DATA_DIR="$(pwd)/data"
export GEO_BIAS_OUTPUT_DIR="$(pwd)/data/v2_exposure_blockers"
```

See **[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)** for a complete file-by-file inventory, SHA-256 checksums, and column-level schema descriptions of all Zenodo archives and tables.

---

## Reproducing the Paper Figures, Tables, and Analyses

### 1. Fast Reproduction of All Paper Figures & Tables (from Pre-computed Predictions)

With the Zenodo archives extracted into `$GEO_BIAS_DATA_DIR`, you can regenerate all publication statistics, waterfall decompositions, $N$-gauge placement summaries, and high-resolution figures (`Figure_1.png`–`Figure_5.png`, `Extended_Data_Figure_1.png`, `Extended_Data_Figure_3.png`–`Extended_Data_Figure_6.png`, `Table_1.csv`, `Extended_Data_Table_1.csv`–`Extended_Data_Table_3.csv`) in a few minutes:

```bash
export GEO_BIAS_DATA_DIR="$(pwd)/data"
export GEO_BIAS_OUTPUT_DIR="$(pwd)/data/v2_exposure_blockers"

# Compile Table 1 and Extended Data Table 1 across all 38 metrics:
python3 7_compile_table_1_stats.py

# Run global exposure blocker decomposition and N-gauge placement synthesis:
python3 11_analyze_gauge_placement_results.py

# Run 5-factor counterfactual attribution and 96-feature TreeSHAP decomposition:
python3 12_five_factor_exposure_decomposition.py

# Render all main-text and Extended Data figures and formatted CSV tables:
python3 9_figures_and_tables_for_paper.py
```

### 2. Full End-to-End Pipeline Execution (Steps 0–12)

To run the entire machine learning, global inference, and gauge network optimization pipeline from the input basin attributes and hydrological model evaluation metrics:

```bash
# Step 0: Compute topological upstream/downstream coverage fractions
python3 0_calculate_overlap_fractions.py

# Step 1: Compute multivariate hydroclimatic anomaly scores (IF, OCSVM, LOF)
python3 1_anomaly_scores.py

# Step 2: Verify or compute daily streamflow observation record lengths
python3 2_calculate_record_lengths.py

# Step 3: Build generic training (gauged, ungauged) and global HyBAS inference matrices
python3 3_create_datasets.py

# Step 4: Map 964,084 HydroBASINS Level 12 catchments to countries
python3 4_hybas_to_country_mapping.py

# Step 5 & 6: Run 10-fold CV and train global SkillPredictor models for a target metric
# (Repeat across metrics or use utils/run_pipeline_single_metric.py)
python3 utils/run_pipeline_single_metric.py --metric=2.0-year-2.0-window-recall

# Step 7: Compile cross-metric skill and regionalization penalty tables
python3 7_compile_table_1_stats.py

# Step 8: Compute exact KernelSHAP attributions on stratified global samples
python3 8_factor_analysis_stratified_sample.py --metric=2.0-year-2.0-window-recall

# Step 10: Verify or compute basin-level flood exposure by return period
python3 10_calculate_global_exposure.py

# Candidate Selection & N-Gauge Network Evaluation:
python3 select_candidates.py
python3 evaluate_n_gauge_sets.py

# Step 11, 12 & 9: Synthesize exposure blockers, 5-factor attributions, and paper figures
python3 11_analyze_gauge_placement_results.py
python3 12_five_factor_exposure_decomposition.py
python3 9_figures_and_tables_for_paper.py
```

---

## License & Citation

This software is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](LICENSE)). Derived datasets on Zenodo (`10.5281/zenodo.21416987`) are released under **CC-BY-4.0 / Apache-2.0**.

```bibtex
@misc{nearing2026floodwarninggaps,
  author       = {Nearing, Grey and Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and Rothenberg, Juliet},
  title        = {{Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.21416987},
  url          = {https://doi.org/10.5281/zenodo.21416987}
}
```
