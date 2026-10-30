# Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21416986.svg)](https://doi.org/10.5281/zenodo.21416986)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Authors:** Grey Nearing (Google Research; Zurich, Switzerland), Stephanie Rees (Google; London, United Kingdom), Martin Gauch (Google Research; Zurich, Switzerland), Justin Ginnetti (Google Research; Zurich, Switzerland), Juliet Rothenberg (Google Research; Mountain View, California, United States)

This repository contains the complete source code, evaluation pipelines, trained model scripts, and publication figure/table generators for *"Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity"*.

All first-party derived datasets, trained `SkillPredictor` models (`76` Random Forest meta-models across 38 hydrological and flood-event metrics), out-of-fold cross-validation predictions, global HydroBASINS Level 12 (`964,084` sub-basins) predictions, five-factor counterfactual and unified in-manifold ("Way B") TreeSHAP attribution tables (`five_factor_way_b_in_manifold_v2.csv`), and $N$-gauge network optimization outputs are permanently archived on Zenodo at **[https://doi.org/10.5281/zenodo.21416986](https://doi.org/10.5281/zenodo.21416986)**.

---

## Overview & Key Findings

1. **Global Riverine Flood Exposure & Open-Access Gauge Coverage (`964,084` HydroBASINS Level 12 sub-basins)**:
   Intersecting 30 m defended Fathom v3 fluvial flood hazard maps with 100 m WorldPop population counts integrated across return periods $T \in \{5, 10, 20, 50, 100\}\text{ yr}$ yields a global Expected Annual Riverine Flood Exposure of **`221,567,096 people/yr`** (`221.57M`: **`160,496,155`** across **`60`** countries with national warning capacity; **`55,980,721`** across **`113`** countries without effective operational warning systems; and **`5,090,220`** across `27,064` unmapped coastal/island sub-basins; `173` mapped countries total, `160` with non-zero riverine flood exposure).
   - **Zero Open-Access Gauge Coverage**: **`709,709` of `964,084` sub-basins (`73.6%`), containing `200,294,602 people/yr` (`90.4%` of global riverine flood exposure**: `141,511,402` in countries with warning capacity and `53,853,166` in countries without effective systems), have **`0%` open-access streamflow gauge coverage** (`total_upstream_coverage == 0` and `max_downstream_coverage == 0` across open Caravan + EStreams networks).
   - **Currently Warned Exposure (`M_ungauged`, $\alpha = 1.0$)**: **`64,037,729 people/yr` (`28.9%` of total)** (`47,245,114` with warning capacity; `15,349,199` without effective systems; `1,443,416` unmapped).
   - **Currently Unwarned Exposure (`M_ungauged`, $\alpha = 1.0$)**: **`157,529,367 people/yr` (`71.1%` of total)** (`113,251,041` [`70.6%`] across the `60` countries with warning capacity; `40,631,522` [`72.6%`] across the `113` countries without effective systems; `3,646,805` unmapped).

2. **Strategic $N$-Gauge Network Expansion (`M_ungauged`, $\alpha = 1.0$, with 20-NN OOD Updates)**:
   - Adding **`N = 1,000` global gauges** reduces unwarned flood exposure by **`+3,780,338 people/yr` (`2.40%` of global unwarned; `68.3%` of the `5.54M` Factor 2 ceiling)** under Exposure-Targeted siting (`Bucket 2A`), and **`+3,629,081 people/yr` (`2.30%`)** under Exposure $\times$ OOD siting (`Bucket 2B`), compared to **`+1,007,822 people/yr` (`0.64%`)** under Unweighted OOD representativeness siting (`Bucket 1` — `3.75x` less). At `N = 100` global gauges, Exposure-Targeted siting (`Bucket 2A`) warns **`+1,601,546 people/yr` (`1.02%`)** (`+1,655,033` for `Bucket 2B`) vs. **`+495,293` (`0.314%`)** for `Bucket 1`.
   - **Coordinated National Expansion (`N = 1, 5, 10, 20` gauges per country across `160` exposed countries)** warns **`+506,660`** (`N = 1`), **`+854,845`** (`N = 5`), **`+1,182,535`** (`N = 10`), and **`+1,621,350 people/yr` (`1.03%` of global unwarned)** at **`N = 20`** (`+1,109,167` in countries with warning capacity; `+512,184` in countries without effective systems).

3. **Observational Counterfactual Ceilings & Factor 5 Residual Ceiling (`M_ungauged`, $\alpha = 1.0$)**:
   - **Factor 5 Residual Unwarned Ceiling (above `100%` river-network coverage + top-quartile in-distribution similarity)**: **`137,197,608 people/yr` (`87.09%` of global unwarned)** remains unwarned (`98,408,283` with warning capacity; `38,789,325` without effective systems), reflecting meteorological precipitation accuracy limits, 2D overbank inundation hydraulics, and channel rating curves.
   - **Combined Observational Ceiling (`Factors 2 + 3` jointly)**: **`+20,355,848 people/yr` (`12.92%` of global unwarned)** at $\alpha = 1.0$ (`+14,860,964` with warning capacity; `+5,494,884` without effective systems), scaling to **`+16,284,679 people/yr` (`9.6%`)** at high warning compliance ($\alpha = 0.80$) and **`+7,124,547 people/yr` (`3.6%`)** at low compliance ($\alpha = 0.35$).
   - **Factor 3 Observational Ceiling (Top-Quartile In-Distribution Similarity)**: **`+16,277,249 people/yr` (`10.33%` of global unwarned)** (`+11,919,224` with capacity; `+4,358,024` without effective systems).
   - **Factor 2 Observational Ceiling (Universal `100%` River-Network Coverage)**: **`+5,535,831 people/yr` (`3.51%` of global unwarned)** (`+4,027,296` with capacity; `+1,508,535` without effective systems), independently confirmed by the temporal-split `M_gauged` predictor check (**`+4,019,704 people/yr`**, `2.55%`).
   - **Factor 4 Observational Ceiling (Extending Record Length to `50` Years)**: **`+19,226 people/yr` (`0.01%`)**, confirming that short (`1–2 yr`) local records suffice for pretrained transfer learning.

4. **Unified In-Manifold ("Way B") TreeSHAP Targeting Across All 10 Factor Groups (`five_factor_way_b_in_manifold_v2.csv`, post-stratified `12,689`-basin global exposure sample covering `75.7%` direct global exposure; `34.33M people/yr` total feature-level TreeSHAP deficit)**:
   Reporting **(Parameter-Free `100%` SHAP Penalty Elimination $R_{\text{parity}} = \min(1, R + P)$ / Common-Scale $P^* = 0.10$, $\gamma = 25\%$ Nonlinear Missed-Event Reduction)**:
   - **`Factor 3` (Out-of-Distribution Representation Bias)**: **`+11,848,086 / +22,431,431 people/yr`** (`7.5% / 14.2%` of global unwarned; `11,907,035` feature-sum SHAP deficit; `96.5%` of exposed population penalized $>0.5\text{ pp}$).
   - **`Factor 2` (Local River-Network Gauge Coverage)**: **`+1,087,001 / +2,209,637 people/yr`** (`0.7% / 1.4%` of global unwarned; `+5,264,336` at $\gamma = 50\%$, matching the `+5.54M` observational ceiling; `1,102,255` feature-sum SHAP deficit).
   - **`Combined Factor 1` (All Physical & Anthropogenic Catchment Attributes `1a–1g` Net)**: **`+4,883,977 / +8,785,869 people/yr`** (`3.1% / 5.6%` of global unwarned; `21,321,973` feature-sum SHAP deficit), decomposed across the 7 physical and anthropogenic sub-groups:
     - **`1f` (Sparse Canopy, Agricultural & Biome Land Cover)**: **`+5,821,935 / +11,626,911 people/yr`** (`3.7% / 7.4%`; `10,958,964` feature-sum SHAP deficit);
     - **`1a` (Arid & Convective Hydroclimate)**: **`+1,463,259 / +3,085,650 people/yr`** (`0.9% / 2.0%`; `3,275,965` feature-sum SHAP deficit);
     - **`1c` (Floodplain, Wetland & Lake Storage)**: **`+922,509 / +1,889,270 people/yr`** (`0.6% / 1.2%`; `2,346,451` feature-sum SHAP deficit);
     - **`1d` (Steep & Small-Scale Channel Routing)**: **`+651,451 / +1,340,702 people/yr`** (`0.4% / 0.9%`; `1,259,602` feature-sum SHAP deficit);
     - **`1b` (Dam, Reservoir & Irrigation Regulation)**: **`+437,897 / +916,212 people/yr`** (`0.3% / 0.6%`; `822,477` feature-sum SHAP deficit);
     - **`1g` (Urban & Human Footprint)**: **`+420,327 / +912,146 people/yr`** (`0.3% / 0.6%`; `1,451,216` feature-sum SHAP deficit);
     - **`1e` (Subsurface Soil & Karst Hydrogeology)**: **`+330,203 / +688,550 people/yr`** (`0.2% / 0.4%`; `1,207,297` feature-sum SHAP deficit).
   - **`Factor 4` (Historical Record Length)**: **`+2,516 / +4,377 people/yr`** (`0.002% / 0.003%`; `2,516` feature-sum SHAP deficit).

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
├── 12_five_factor_exposure_decomposition.py# Step 12: Observational ceilings & unified Way B in-manifold TreeSHAP
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

With the Zenodo archives extracted into `$GEO_BIAS_DATA_DIR`, you can regenerate all publication statistics, waterfall decompositions, $N$-gauge placement summaries, and high-resolution figures (`Figure_1.png`–`Figure_5.png`, `Extended_Data_Figure_1.png`, `Extended_Data_Figure_3.png`–`Extended_Data_Figure_6.png`, `Table_1.csv`, `Extended_Data_Table_1.csv`–`Extended_Data_Table_3.csv`) in a few minutes:

```bash
export GEO_BIAS_DATA_DIR="$(pwd)/data"
export GEO_BIAS_OUTPUT_DIR="$(pwd)/data/v2_exposure_blockers"

# Compile cross-metric skill statistics across all 38 metrics:
python3 7_compile_table_1_stats.py

# Run global exposure blocker decomposition and N-gauge placement synthesis:
python3 11_analyze_gauge_placement_results.py

# Run observational counterfactual ceilings and unified Way B in-manifold TreeSHAP decomposition:
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

This software is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](LICENSE)). Derived datasets on Zenodo (`10.5281/zenodo.21416986`) are released under **CC-BY-4.0 / Apache-2.0**.

```bibtex
@misc{nearing2026floodwarninggaps,
  author       = {Nearing, Grey and Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and Rothenberg, Juliet},
  title        = {{Global Flood Warning Gaps Reflect Forecasting and Observational Blockers Beyond Local Gauge Sparsity}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.2},
  doi          = {10.5281/zenodo.21416986},
  url          = {https://doi.org/10.5281/zenodo.21416986}
}
```
