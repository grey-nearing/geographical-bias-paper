# Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21416987.svg)](https://doi.org/10.5281/zenodo.21416987)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Authors:** Grey Nearing (Google Research; Zurich, Switzerland), Stephanie Rees (Google; London, United Kingdom), Martin Gauch (Google Research; Zurich, Switzerland), Justin Ginnetti (Google Research; Zurich, Switzerland), Juliet Rothenberg (Google Research; Mountain View, California, United States)

This repository contains the complete source code, evaluation pipelines, trained model scripts, and publication figure/table generators for *"Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity"*.

All first-party derived datasets, trained `SkillPredictor` models (`76` Random Forest meta-models across 38 hydrological and flood-event metrics), out-of-fold cross-validation predictions, global HydroBASINS Level 12 (`964,084` sub-basins) predictions, five-factor TreeSHAP and counterfactual attribution tables, and $N$-gauge network optimization outputs are permanently archived on Zenodo at **[https://doi.org/10.5281/zenodo.21416987](https://doi.org/10.5281/zenodo.21416987)**.

---

## Overview & Key Findings

1. **Global Riverine Flood Exposure Accounting (`964,084` HydroBASINS Level 12 sub-basins)**:
   Intersecting 30 m defended Fathom v3 fluvial flood hazard maps with 100 m WorldPop population counts integrated across return periods $T \in \{5, 10, 20, 50, 100\}\text{ yr}$ yields a global Expected Annual Riverine Flood Exposure of **`221,567,096 people/yr`** (`221.57M`: **`160,496,155`** across **`60`** countries with national warning capacity; **`55,980,721`** across **`113`** countries without effective operational warning systems; and **`5,090,220`** across `27,064` unmapped coastal/island sub-basins; `173` mapped countries total, `160` with non-zero riverine flood exposure).
   - **Currently Warned Exposure (`M_ungauged`, $\alpha = 1.0$)**: **`64,037,729 people/yr` (`28.9%` of total)** (`47,245,114` with warning capacity; `15,349,199` without effective systems; `1,443,416` unmapped).
   - **Currently Unwarned Exposure (`M_ungauged`, $\alpha = 1.0$)**: **`157,529,367 people/yr` (`71.1%` of total)** (`113,251,041` [`70.6%`] across the `60` countries with warning capacity; `40,631,522` [`72.6%`] across the `113` countries without effective systems; `3,646,805` unmapped).

2. **Strategic $N$-Gauge Network Expansion (`M_ungauged`, $\alpha = 1.0$, with 20-NN OOD Updates)**:
   - Adding **`N = 1,000` global gauges** reduces unwarned flood exposure by **`+3,780,338 people/yr` (`2.40%` of global unwarned; `68.3%` of the `5.54M` Factor 2 ceiling)** under Exposure-Targeted siting (`Bucket 2A`), and **`+3,629,081 people/yr` (`2.30%`)** under Exposure $\times$ OOD siting (`Bucket 2B`), compared to **`+1,007,822 people/yr` (`0.64%`)** under Unweighted OOD representativeness siting (`Bucket 1` — `3.75x` less). At `N = 100` global gauges, Exposure-Targeted siting (`Bucket 2A`) warns **`+1,601,546 people/yr` (`1.02%`)** (`+1,655,033` for `Bucket 2B`) vs. **`+495,293` (`0.314%`)** for `Bucket 1`.
   - **Coordinated National Expansion (`N = 1, 5, 10, 20` gauges per country across `160` exposed countries)** warns **`+506,660`** (`N = 1`), **`+854,845`** (`N = 5`), **`+1,182,535`** (`N = 10`), and **`+1,621,350 people/yr` (`1.03%` of global unwarned)** at **`N = 20`** (`+1,109,167` in countries with warning capacity; `+512,184` in countries without effective systems).

3. **Five-Factor Predictability Ceilings & TreeSHAP Attribution (`96` catchment features: `88` HydroATLAS physical/anthropogenic + `4` topological coverage + `3` OOD anomaly + `1` record length)**:
   - **Factor 5 Residual Unwarned Ceiling (above `100%` river-network coverage + `90th`-percentile in-distribution similarity)**: **`137,197,608 people/yr` (`87.09%` of global unwarned)** remains unwarned (`98,408,283` with warning capacity; `38,789,325` without effective systems), reflecting meteorological precipitation accuracy limits, 2D overbank inundation hydraulics, and channel rating curves.
   - **Combined Observational Ceiling (`Factors 2 + 3` jointly)**: **`+20,355,848 people/yr` (`12.92%` of global unwarned)** at $\alpha = 1.0$ (`+14,860,964` with warning capacity; `+5,494,884` without effective systems), scaling to **`+16,284,679 people/yr` (`9.6%`)** at high warning compliance ($\alpha = 0.80$) and **`+7,124,547 people/yr` (`3.6%`)** at low compliance ($\alpha = 0.35$).
   - **Factor 3 (Eliminating Out-of-Distribution Representation Bias)**: **`+16,277,249 people/yr` (`10.33%` of global unwarned)** counterfactual ceiling (`+11,919,224` with capacity; `+4,358,024` without effective systems); accounts for **`11,071,508 people/yr` (`34.63%` of the `31.97M` catchment TreeSHAP deficit)** and is the dominant catchment-level blocker in **`111` of `160` exposed countries**.
   - **Factor 2 (Universal `100%` River-Network Coverage)**: **`+5,535,831 people/yr` (`3.51%` of global unwarned)** counterfactual ceiling (`+4,019,704 people/yr` [`2.55%`] independent `M_gauged` confirmation check); accounts for **`1,114,829 people/yr` (`3.49%` of catchment TreeSHAP deficit)**.
   - **Factor 1 (Physical & Anthropogenic Catchment Regimes, `78` features in exposure SHAP / `88` total HydroATLAS features)**: accounts for **`19,780,651 people/yr` (`61.88%` of the `31.97M` catchment TreeSHAP deficit)**, decomposed into:
     - **Floodplains, Wetlands & Lakes (`1c`)**: `12,785,221` TreeSHAP deficit (`+2,883,044` counterfactual benchmark);
     - **Arid/Convective Climate (`1a`)**: `2,812,489` TreeSHAP deficit (`+4,750,214` counterfactual benchmark);
     - **Dam & Irrigation Regulation (`1b`)**: `1,940,683` TreeSHAP deficit (`+1,178,107` counterfactual benchmark);
     - **Topography & Drainage Scale (`1d`)**: `1,170,136` TreeSHAP deficit (`+901,284` counterfactual benchmark);
     - **Subsurface Soil & Karst (`1e`)**: `1,072,121` TreeSHAP deficit (`+718,512` counterfactual benchmark).
   - **Factor 4 (Extending Historical Record Length to `50` Years)**: **`+19,226 people/yr` (`0.01%`)** counterfactual ceiling and **`3,537 people/yr` (`0.01%`)** TreeSHAP deficit, confirming that short (`1–2 yr`) local records suffice for pretrained transfer learning.

---

## Repository Structure

```text
geographical-bias-paper/
├── 0_calculate_overlap_fractions.py        # Step 0: Topological upstream/downstream overlap fractions
├── 1_anomaly_scores.py                     # Step 1: Isolation Forest, One-Class SVM & LOF OOD scores
├── 2_calculate_record_lengths.py           # Step 2: Daily streamflow record lengths per GRDC/Caravan gauge
├── 3_create_datasets.py                    # Step 3: Assemble 92-feature gauged & 96-feature ungauged/HyBAS matrices
├── 4_hybas_to_country_mapping.py           # Step 4: Spatial join of 964,084 HyBAS sub-basins to countries
├── 5_model_cross_validation.py             # Step 5: 10-fold CV of SkillPredictor across all 38 metrics
├── 6_global_training_and_inference.py      # Step 6: Full-archive SkillPredictor fit & 964k HyBAS inference
├── 7_compile_table_1_stats.py              # Step 7: Compile cross-metric skill & regionalization summary stats
├── 8_factor_analysis_stratified_sample.py  # Step 8: Stratified SHAP attribution across metrics
├── 9_figures_and_tables_for_paper.py       # Step 9: Generate Figures 1-5, ED Figures 1, 3-6 & Tables
├── 10_calculate_global_exposure.py         # Step 10: Intersect WorldPop & Fathom v3 across Level 12 HyBAS
├── 11_analyze_gauge_placement_results.py   # Step 11: Exposure blocker decomposition & N-gauge curves
├── 12_five_factor_exposure_decomposition.py# Step 12: Five-factor counterfactual & 96-feature TreeSHAP
├── select_candidates.py                    # Candidate basin selection (Bucket 1, 2A, 2B, Country Top-20)
├── evaluate_gauges.py                      # Single-gauge marginal exposure reduction evaluator
├── evaluate_n_gauge_sets.py                # Joint N-gauge topological & 20-NN OOD exposure reduction evaluator
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
│   ├── skill_predictor.py                  # Random Forest SkillPredictor meta-model & model loader
│   ├── stats_utils.py                      # Non-parametric tests, effect sizes & trapezoidal integration
│   └── viz_utils.py                        # Global map and cross-validation diagnostic plotters
├── hydro_metrics/
│   ├── data_paths.py                       # Paths for raw Dual-LSTM Zarr predictions (1986-2023)
│   ├── evaluation_utils.py                 # Peak-detection precision/recall & hydrograph evaluation
│   ├── hydro_metrics_test.py               # Unit tests for hydro_metrics
│   ├── loading_utils.py                    # Zarr & NetCDF hydrograph/return-period loaders
│   ├── metrics.py                          # NSE, KGE, log-NSE, log-KGE, Alpha/Beta & correlation metrics
│   ├── metrics_utils.py                    # Time-series masking and alignment utilities
│   ├── recalculate_metrics.py              # Recomputes 1986-2023 metrics across all 38 target metrics
│   ├── return_period_metrics.py            # Return-period exceedance event precision & recall
│   └── return_period_calculator/           # Gumbel/log-Pearson return-period threshold fitting package
└── v2_exposure_blockers/                   # Pre-generated publication figures, tables, and summary CSV/JSONs
    ├── figures_and_tables_for_paper/       # Figures 1-5, Extended Data Figures 1, 3-6, Table 1, ED Tables 1-3
    ├── candidates/                         # Candidate gauge lists for all 4 placement strategies
    └── statistics/                         # Country, regional, CAP-tier & 5-factor summary tables
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

# Compile cross-metric skill statistics across all 38 metrics:
python3 7_compile_table_1_stats.py

# Run global exposure blocker decomposition and N-gauge placement synthesis:
python3 11_analyze_gauge_placement_results.py

# Run 5-factor counterfactual attribution and 96-feature TreeSHAP decomposition:
python3 12_five_factor_exposure_decomposition.py

# Render all main-text and Extended Data figures and formatted CSV/HTML tables:
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
python3 utils/run_pipeline_single_metric.py --metric=5.0-year-2.0-window-recall

# Step 7: Compile cross-metric skill and regionalization penalty tables
python3 7_compile_table_1_stats.py

# Step 8: Compute stratified SHAP attributions
python3 8_factor_analysis_stratified_sample.py --metric=5.0-year-2.0-window-recall

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
  version      = {1.0.1},
  doi          = {10.5281/zenodo.21416987},
  url          = {https://doi.org/10.5281/zenodo.21416987}
}
```
