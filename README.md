# Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21416986.svg)](https://doi.org/10.5281/zenodo.21416986)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Authors:** Grey Nearing (Google Research; Zurich, Switzerland), Stephanie Rees (Google; London, United Kingdom), Martin Gauch (Google Research; Zurich, Switzerland), Justin Ginnetti (Google Research; Zurich, Switzerland), Juliet Rothenberg (Google Research; Mountain View, California, United States)

This repository contains the complete source code, evaluation pipelines, trained model scripts, and publication figure/table generators for *"Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity"*.

All first-party derived datasets, trained `SkillPredictor` models (`76` Random Forest meta-models across 38 hydrological and flood-event metrics), out-of-fold cross-validation predictions, global HydroBASINS Level 12 (`964,084` watersheds) predictions across the **`38,069`-gauge open reference network** (`Caravan` + `EStreams` + `GRDC`), five-factor counterfactual and unified in-manifold ("Way B") TreeSHAP attribution tables (`five_factor_way_b_in_manifold_v2.csv`), and $N$-gauge network optimization outputs are permanently archived on Zenodo at **[https://doi.org/10.5281/zenodo.21416986](https://doi.org/10.5281/zenodo.21416986)**.

---

## Overview & Key Findings

1. **Global Riverine Flood Exposure & `38,069`-Gauge Open-Access Reference Network Coverage (`964,084` HydroBASINS Level 12 watersheds)**:
   Intersecting 30 m defended Fathom v3 fluvial flood hazard maps with 100 m WorldPop population counts integrated across return periods $T \in \{5, 10, 20, 50, 100\}\text{ yr}$ yields a global Expected Annual Riverine Flood Exposure of **`221,567,096 people/yr`** (`221.57M`: **`160,496,155`** across **`60`** countries with national warning capacity; **`55,980,721`** across **`113`** countries without effective operational warning systems; and **`5,090,220`** across `27,064` unmapped coastal/island watersheds; `173` mapped countries total, `160` with non-zero riverine flood exposure).
   - **Zero Open-Access Gauge Coverage**: **`637,433` of `964,084` watersheds (`66.1%`), containing `194,516,517 people/yr` (`87.8%` of global riverine flood exposure**: `138,124,432` in countries with warning capacity and `51,518,640` in countries without effective systems), have **`0%` open-access streamflow gauge coverage** (`total_upstream_coverage == 0` and `max_downstream_coverage == 0` outside the `38,069`-gauge open reference network).
   - **Currently Warned Exposure (`M_ungauged`, $\alpha = 1.0$)**: **`64,414,357 people/yr` (`29.1%` of total)** (`47,496,895` with warning capacity; `15,468,445` without effective systems; `1,449,016` unmapped).
   - **Currently Unwarned Exposure (`M_ungauged`, $\alpha = 1.0$)**: **`157,152,740 people/yr` (`70.9%` of total)** (`112,999,260` [`70.4%`] across the `60` countries with warning capacity; `40,512,276` [`72.4%`] across the `113` countries without effective systems; `3,641,204` unmapped).

2. **Observational Counterfactual Ceilings & Factor 5 Residual Ceiling (`M_ungauged`, $\alpha = 1.0$)**:
   - **Factor 5 Residual Unwarned Ceiling (above `100%` river-network coverage + top-quartile in-distribution similarity)**: **`137,181,397 people/yr` (`87.3%` of global unwarned)** remains unwarned (`98,395,813` with warning capacity; `38,785,583` without effective systems), reflecting meteorological precipitation accuracy limits, 2D overbank inundation hydraulics, and channel rating curves.
   - **Combined Observational Ceiling (`Factors 2 + 3` jointly)**: **`+19,996,421 people/yr` (`12.7%` of global unwarned)** at $\alpha = 1.0$ (`+14,622,139` with warning capacity; `+5,374,282` without effective systems), scaling to **`+15,997,137 people/yr` (`9.4%`)** at high warning compliance ($\alpha = 0.80$) and **`+6,998,747 people/yr` (`3.5%`)** at low compliance ($\alpha = 0.35$).
   - **Factor 3 Observational Ceiling (Top-Quartile In-Distribution Similarity)**: **`+15,942,362 people/yr` (`10.1%` of global unwarned)** (`+11,700,777` with capacity; `+4,241,585` without effective systems).
   - **Factor 2 Observational Ceiling (Universal `100%` River-Network Coverage)**: **`+5,453,469 people/yr` (`3.5%` of global unwarned)** (`+3,967,077` with capacity; `+1,486,392` without effective systems), independently confirmed by the temporal-split `M_gauged` predictor check (**`+3,911,650 people/yr`**, `2.5%`).
   - **Factor 4 Observational Ceiling (Extending Record Length to `50` Years)**: **`+20,763 people/yr` (`0.01%`)**, confirming that short (`1–2 yr`) local records suffice for pretrained transfer learning.

3. **Unified In-Manifold ("Way B") TreeSHAP Targeting Across All 10 Factor Groups (`five_factor_way_b_in_manifold_v2.csv`, post-stratified `12,689`-watershed global exposure sample covering `75.7%` direct global exposure; `34.17M people/yr` total feature-level TreeSHAP deficit)**:
   Reporting **(Parameter-Free `100%` SHAP Penalty Elimination $R_{\text{parity}} = \min(1, R + P)$ / Common-Scale $P^* = 0.10$, $\gamma = 25\%$ Nonlinear Missed-Event Reduction)**:
   - **`Factor 3` (Out-of-Distribution Representation Bias)**: **`+11,533,245 / +21,828,952 people/yr`** (`7.3% / 13.9%` of global unwarned; `11,621,851` feature-sum SHAP deficit; `94.9%` of exposed population penalized $>0.5\text{ pp}$).
   - **`Factor 2` (Local River-Network Gauge Coverage)**: **`+1,070,642 / +2,170,675 people/yr`** (`0.7% / 1.4%` of global unwarned; `+5,171,699` at $\gamma = 50\%$, matching the `+5.45M` observational ceiling; `1,089,260` feature-sum SHAP deficit).
   - **`Combined Factor 1` (All Physical & Anthropogenic Catchment Attributes `1a–1g` Net)**: **`+4,973,782 / +8,942,716 people/yr`** (`3.2% / 5.7%` of global unwarned; `21,459,970` feature-sum SHAP deficit), decomposed across the 7 physical and anthropogenic sub-groups:
     - **`1f` (Sparse Canopy, Agricultural & Biome Land Cover)**: **`+5,857,133 / +11,668,564 people/yr`** (`3.7% / 7.4%`; `10,975,946` feature-sum SHAP deficit);
     - **`1a` (Arid & Convective Hydroclimate)**: **`+1,490,251 / +3,124,528 people/yr`** (`0.9% / 2.0%`; `3,330,716` feature-sum SHAP deficit);
     - **`1c` (Floodplain, Wetland & Lake Storage)**: **`+963,825 / +1,963,270 people/yr`** (`0.6% / 1.2%`; `2,388,066` feature-sum SHAP deficit);
     - **`1d` (Steep & Small-Scale Channel Routing)**: **`+657,120 / +1,346,357 people/yr`** (`0.4% / 0.9%`; `1,263,230` feature-sum SHAP deficit);
     - **`1b` (Dam, Reservoir & Irrigation Regulation)**: **`+436,553 / +909,270 people/yr`** (`0.3% / 0.6%`; `825,031` feature-sum SHAP deficit);
     - **`1g` (Urban & Human Footprint)**: **`+419,403 / +906,650 people/yr`** (`0.3% / 0.6%`; `1,457,788` feature-sum SHAP deficit);
     - **`1e` (Subsurface Soil & Karst Hydrogeology)**: **`+336,276 / +699,156 people/yr`** (`0.2% / 0.4%`; `1,219,194` feature-sum SHAP deficit).
   - **`Factor 4` (Historical Record Length)**: **`+2,579 / +4,483 people/yr`** (`0.002% / 0.003%`; `2,579` feature-sum SHAP deficit).

4. **Strategic $N$-Gauge Network Expansion (`M_ungauged`, $\alpha = 1.0$, with 20-NN OOD Updates)**:
   - Adding **`N = 1,000` global gauges** reduces unwarned flood exposure by **`+3,726,000 people/yr` (`2.37%` of global unwarned; `68.3%` of the `5.45M` Factor 2 ceiling)** under Exposure-Targeted siting (`Bucket 2A`), and **`+3,551,975 people/yr` (`2.26%`)** under Exposure $\times$ OOD siting (`Bucket 2B`), compared to **`+996,600 people/yr` (`0.63%`)** under Unweighted OOD representativeness siting (`Bucket 1` — `3.74x` less). At `N = 100` global gauges, Exposure-Targeted siting (`Bucket 2A`) warns **`+1,591,734 people/yr` (`1.01%`)** (`+1,637,585` for `Bucket 2B`) vs. **`+498,444` (`0.317%`)** for `Bucket 1`.
   - **Coordinated National Expansion (`N = 1, 5, 10, 20` gauges per country across `160` exposed countries)** warns **`+503,337`** (`N = 1`), **`+840,103`** (`N = 5`), **`+1,141,400`** (`N = 10`), and **`+1,582,463 people/yr` (`1.01%` of global unwarned)** at **`N = 20`** (`+1,078,295` in countries with warning capacity; `+504,168` in countries without effective systems).

---

## Repository Structure

```text
geographical-bias-paper/
├── 0_calculate_overlap_fractions.py        # Step 0: Topological upstream/downstream overlap fractions
├── 1_anomaly_scores.py                     # Step 1: Isolation Forest, One-Class SVM & LOF OOD scores
├── 2_calculate_record_lengths.py           # Step 2: Daily streamflow record lengths per GRDC/Caravan gauge
├── 3_create_datasets.py                    # Step 3: Assemble 92-feature gauged & 96-feature ungauged/HyBAS matrices
├── 4_hybas_to_country_mapping.py           # Step 4: Spatial join of 964,084 HyBAS watersheds to countries
├── 5_model_cross_validation.py             # Step 5: 10-fold CV of SkillPredictor across all 38 metrics
├── 6_global_training_and_inference.py      # Step 6: Full-archive SkillPredictor fit & 964k HyBAS inference
├── 7_compile_table_1_stats.py              # Step 7: Compile cross-metric skill & regionalization summary stats
├── 8_factor_analysis_stratified_sample.py  # Step 8: Stratified SHAP attribution across metrics
├── 9_figures_and_tables_for_paper.py       # Step 9: Generate Figures 1-5, ED Figures 1, 3-7 & Tables
├── 10_calculate_global_exposure.py         # Step 10: Intersect WorldPop & Fathom v3 across Level 12 HyBAS
├── 11_analyze_gauge_placement_results.py   # Step 11: Exposure blocker decomposition & N-gauge curves
├── 12_five_factor_exposure_decomposition.py# Step 12: Observational ceilings & unified Way B in-manifold TreeSHAP
├── run_reference_network_update.py         # End-to-end 38,069-gauge reference network update orchestrator
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
│   ├── io_utils.py                         # CSV, GeoJSON, Zarr v2/v3 loaders & parallel TreeSHAP helpers
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
    ├── figures_and_tables_for_paper/       # Figures 1-5, Extended Data Figures 1, 3-7, Table 1, ED Tables 1-3
    ├── candidates/                         # Candidate gauge lists for all 4 placement strategies
    └── statistics/                         # Country, CAP-tier, observational ceilings & Way B TreeSHAP tables
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

## Downloading the Zenodo Dataset (`10.5281/zenodo.21416986`)

Download and extract the archived data from **[https://doi.org/10.5281/zenodo.21416986](https://doi.org/10.5281/zenodo.21416986)** into `./data` (or any directory of your choice, pointed to by `GEO_BIAS_DATA_DIR`):

```bash
mkdir -p data
# After downloading the tar.gz archives from https://doi.org/10.5281/zenodo.21416986 into ./data:
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

With the Zenodo archives extracted into `$GEO_BIAS_DATA_DIR`, you can regenerate all publication statistics, waterfall decompositions, $N$-gauge placement summaries, and high-resolution figures (`Figure_1.png`–`Figure_5.png`, `Extended_Data_Figure_1.png`, `Extended_Data_Figure_3.png`–`Extended_Data_Figure_7.png`, `Table_1.csv`, `Extended_Data_Table_1.csv`–`Extended_Data_Table_3.csv`) in a few minutes:

```bash
export GEO_BIAS_DATA_DIR="$(pwd)/data"
export GEO_BIAS_OUTPUT_DIR="$(pwd)/data/v2_exposure_blockers"

# Compile cross-metric skill statistics across all 38 metrics:
python3 7_compile_table_1_stats.py

# Run global exposure blocker decomposition and N-gauge placement synthesis:
python3 11_analyze_gauge_placement_results.py

# Run observational counterfactual ceilings and unified Way B in-manifold TreeSHAP decomposition:
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

# Step 4: Map 964,084 HydroBASINS Level 12 watersheds to countries
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

This software is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](LICENSE)). Derived datasets on Zenodo (`10.5281/zenodo.21416986`) are released under **CC-BY-4.0 / Apache-2.0**.

```bibtex
@misc{nearing2026floodwarninggaps,
  author       = {Nearing, Grey and Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and Rothenberg, Juliet},
  title        = {{Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.3},
  doi          = {10.5281/zenodo.21416986},
  url          = {https://doi.org/10.5281/zenodo.21416986}
}
```
