# What Fraction of the World's Watersheds are Ungauged?

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Authors:**
Grey Nearing (Google Research; Zurich, Switzerland),
Karan Ruparell (European Centre for Medium-Range Weather Forecasts; Reading, United Kingdom & Department of Meteorology, University of Reading; Reading, United Kingdom),
Maria Luisa Taccari (European Centre for Medium-Range Weather Forecasts; Reading, United Kingdom),
Stephanie Rees (Google; London, United Kingdom),
Martin Gauch (Google Research; Zurich, Switzerland),
Justin Ginnetti (Google Research; Zurich, Switzerland),
Oliver Wing (Fathom; Bristol, United Kingdom & School of Geographical Sciences, University of Bristol; Bristol, United Kingdom),
Juliet Rothenberg (Google Research; Mountain View, California, United States),
Frederik Kratzert (Google Research; Vienna, Austria),
Deborah Cohen (Google Research; Tel Aviv, Israel),
Yossi Matias (Google Research; Mountain View, California, United States)

This public repository contains the source code, evaluation pipelines, `SkillPredictor` meta-model training and global inference scripts, and non-proprietary hydrological skill and ungaugedness tables/figures for *"What Fraction of the World's Watersheds are Ungauged?"*.

> **Proprietary Data Licensing Notice (Fathom v3):**
> Commercial Fathom v3 fluvial flood hazard maps and all Fathom-derived flood exposure datasets (`global_basin_exposure_by_return_period.csv`, `basin_exposure_and_blockers_v2.parquet`, `country_exposure_and_blockers_v2.csv`, `Table_S3_lead1.csv`, `anchored_3tiers_exposure_summary_lead1.csv`, `multi_mechanism_tiered_skill_and_exposure_lead1.csv`, and flood exposure figures) are subject to commercial licensing restrictions and are **strictly excluded** from this repository and from external archives. Only open hydrological skill, topological gauge coverage, and HydroATLAS out-of-distribution (OOD) representativeness datasets and figures are included in `science_results/`. Researchers wishing to run the flood exposure intersection scripts must obtain a Fathom v3 license directly from [Fathom](https://www.fathom.global).

---

## Overview of the Methodology & Pipeline

1. **Continuous Topological and Hydroclimatic Ungaugedness Across `964,084` Global Watersheds (`science_tiers_and_figures/`)**:
   - Evaluates continuous river-network gauge coverage ($u_i = (1 - c_{\text{up}}(i))(1 - c_{\text{down}}(i))$) and multivariate HydroATLAS out-of-distribution anomaly scores (combining Isolation Forest, One-Class SVM, and Local Outlier Factor across 88 static catchment attributes) across all `964,084` non-Arctic global HydroBASINS Level 12 watersheds (`961,390` watersheds assigned across `174` countries using a 25 km coastal tolerance, plus `2,694` remote offshore island watersheds).
   - Evaluates three anchored global gauge-accessibility tiers and screens all `121,819` deduplicated physical stations for active operational status (`record_end_year >= 2016` or explicit active portal status: `90,398` active gauges and `31,421` non-active historical gauges):
     1. **Open Community Compilations (`34,577` gauges; `27,828` active)**: Caravan + EStreams standardized open compilations (`91.5%` of global watershed land area effectively ungauged).
     2. **Publicly Accessible Discharge Records (`93,827` gauges; `68,899` active)**: Adding national hydrologic service APIs and bulk public portals (`80.9%` of global watershed land area effectively ungauged).
     3. **All Publicly Knowable Gauges (`121,819` gauges; `90,398` active)**: Including catchment-attribute catalogs, summary flow indices, and restricted metadata inventories (`75.7%` of global watershed land area effectively ungauged).

2. **38-Year (`1986–2023`) Lead-Time-1 Cross-Validation, Local Fine-Tuning (`_ft`), and Data Assimilation (`_da`) (`cv_1995_2025/`)**:
   - Evaluates the open-source Mean Embedding Long Short-Term Memory (`ME-LSTM`, `google-research/flood-forecasting`, 1-day-ahead forecast horizon `lead_time = 1`) over the 38-year (`1986–2023`) evaluation window across `8,811` post-2016 active Caravan gauges (`38` continuous hydrograph and 7-day causal flood-event trajectory precision/recall metrics).
   - Compares three space-time cross-validation designs (`gauged`, `random` ungauged, and hydrologically `separated` ungauged) across four adaptation regimes: base model (`ungauged`), local target fine-tuning (`ungauged_ft`), real-time 30-day variational data assimilation (`ungauged_da`, applied only at active gauges), and combined fine-tuning and data assimilation (`ungauged_ft_da`, where non-active gauges fall back to `ungauged_ft`).

3. **Global `SkillPredictor` Meta-Models & Empirical `KernelSHAP` Attribution (`science_tiers_and_figures/`, `utils/`)**:
   - Trains stacked `SkillPredictor` meta-models mapping the 96 catchment, topological coverage, record-length, and OOD representativeness features to 1-day-ahead hydrological skill across all 38 metrics and 4 adaptation regimes (`Table_S1_4regimes_lead1.csv`), projects skill across all `964,084` global HydroBASINS Level 12 watersheds (`Fig_S4_global_predicted_skill_lead1.png`), and decomposes full-stack skill drivers via exact empirical `KernelSHAP` with signed category summation (`Fig_4_TreeSHAP_lead1.png`, `Fig_S7_TreeSHAP_top35_lead1.png`, `treeshap_category_importances_lead1.csv`).

---

## Repository Structure

```text
geographical-bias-paper/
├── cv_1995_2025/                                     # 1995-2025 Lead-Time-1 Dual-LSTM CV, Fine-Tuning & Data Assimilation
│   ├── prepare_caravan_post2016.py                   # Filter 8,811 Caravan gauges with post-2016 daily observations
│   ├── build_vm_caravan_zarrs.py                     # Build consolidated Zarr stores for fast multi-GPU training
│   ├── build_precomputed_arrays.py                   # Pre-slice 1980-2025 forcing/target tensors for 10-fold CV
│   ├── fast_multimet.py                              # Mixed-precision PyTorch Dual-LSTM trainer & 1-to-7 day predictor
│   ├── run_cv_orchestrator.py                        # Multi-GPU orchestrator for gauged, random & separated 10-fold CV
│   ├── run_ft_da_cv.py                               # Local fine-tuning (_ft) & real-time data assimilation (_da, _ft_da)
│   ├── compute_cv_metrics.py                         # Compute 68 hydrological & flood-event metrics (lead 1 & peak)
│   ├── stitch_and_compute_ft_da_metrics.py           # Aggregate sharded _ft / _da / _ft_da runs & compute 68 metrics
│   ├── train_skill_predictors_lead1.py               # Train 500-tree Random Forest SkillPredictor models & 964k HyBAS preds
│   └── test_and_benchmark_fast_multimet.py           # Numerical equivalence & throughput tests against flood-forecasting
├── science_tiers_and_figures/                        # Anchored Gauge-Accessibility Tiers, KernelSHAP & Science Figures
│   ├── build_verified_active_gauges_121k.py          # Screen all 121,819 deduplicated stations for active status
│   ├── compute_active_masks_for_tiers.py             # Build Option 1 active-only DA masks across gauge tiers
│   ├── build_tier_features.py                        # Topological coverage & OOD feature builder across gauge tiers
│   ├── compute_3_anchored_tiers.py                   # Build the 3 anchored tiers (34,577 / 93,827 / 121,819 gauges)
│   ├── run_stage1_option_b_metrics.py                # Stage 1: 7-day causal Option B return-period precision/recall metrics
│   ├── run_stage3_train_rp_skill_predictors.py       # Stage 3: Gauge-grouped 10-fold CV & global SkillPredictor fitting
│   ├── run_stage5_kernelshap_lead1.py                # Stage 5: Full-stack empirical KernelSHAP (1,000 medoids; Fig 4 & S7)
│   ├── generate_all_updated_figures_and_tables_v2.py # Compile 4-regime Table S1/S3 & render finalized publication figures
│   ├── evaluate_3_anchored_tiers_exposure_v2.py      # Option 1 active-gauge anchored tier exposure evaluation
│   ├── evaluate_multi_mechanism_tiered_exposure_v2.py # Active-gauge multi-mechanism (_ft / _da / _ft_da) evaluation
│   ├── compute_sec3_cv_and_figs_s1_s2.py             # Section 3 CV paired statistics (Wilcoxon/Mann-Whitney) & Figs S2-S3
│   ├── compute_table_s1_and_figs3.py                 # Compile Table S1 across metrics & render global skill map Fig S4
│   ├── compute_treeshap_lead1.py                     # Legacy TreeSHAP attribution script
│   ├── build_figure1_3tiers.py                       # Render Figure 1 (global gauge map across 3 anchored tiers)
│   ├── build_figure4_3panel_anchored.py              # Render 3-panel tiered skill comparison
│   ├── compute_table_s3_and_figs4_s5.py              # Conditional tiered skill & exposure script (requires Fathom inputs)
│   ├── evaluate_3_anchored_tiers_exposure.py         # Legacy anchored tier exposure script (requires Fathom inputs)
│   ├── evaluate_multi_mechanism_tiered_exposure.py   # Legacy multi-mechanism (_ft / _da / _ft_da) evaluation script
│   └── evaluate_tiered_skill_and_exposure.py         # Tiered skill evaluation script
├── science_results/                                  # Non-proprietary hydrological skill & ungaugedness results
│   ├── figures/                                      # Non-Fathom figures (Fig 1, Fig 2, Fig 4, Fig S1-S4, Fig S7)
│   └── statistics/                                   # Non-Fathom tables (Table S1, Table S2, Table S4, CV stats, TreeSHAP)
├── 0_calculate_overlap_fractions.py                  # Core Step 0: Topological upstream/downstream overlap fractions
├── 1_anomaly_scores.py                               # Core Step 1: Isolation Forest, One-Class SVM & LOF OOD scores
├── 2_calculate_record_lengths.py                     # Core Step 2: Daily streamflow record lengths per gauge
├── 3_create_datasets.py                              # Core Step 3: Assemble 92-feature gauged & 96-feature ungauged/HyBAS matrices
├── 4_hybas_to_country_mapping.py                     # Core Step 4: Spatial join of 964,084 HyBAS watersheds to countries
├── 5_model_cross_validation.py                       # Core Step 5: 10-fold CV of SkillPredictor across metrics
├── 6_global_training_and_inference.py                # Core Step 6: Full-archive SkillPredictor fit & 964k HyBAS inference
├── 7_compile_table_1_stats.py                        # Core Step 7: Compile cross-metric skill & regionalization stats
├── 8_factor_analysis_stratified_sample.py            # Core Step 8: Stratified SHAP attribution across metrics
├── 9_figures_and_tables_for_paper.py                 # Core Step 9: Legacy figure/table generator
├── 10_calculate_global_exposure.py                   # Core Step 10: WorldPop x hazard intersection (requires external license)
├── 11_analyze_gauge_placement_results.py             # Core Step 11: Gauge placement curve analyzer
├── 12_five_factor_exposure_decomposition.py          # Core Step 12: Counterfactual & TreeSHAP analyzer
├── utils/                                            # Shared feature definitions, SkillPredictor loader, stats & I/O helpers
└── hydro_metrics/                                    # Hydrological & return-period event precision/recall metric library
```

---

## Installation & Environment Setup

```bash
conda env create -f environment.yml
conda activate geographical-bias-paper
# Or via pip:
pip install -e .
```

Run the unit test suite for `hydro_metrics`:

```bash
python3 hydro_metrics/hydro_metrics_test.py
```

---

## Reproducing the Non-Proprietary Hydrological Skill & Ungaugedness Analyses

```bash
export GEO_BIAS_DATA_DIR="$(pwd)/data"
export GEO_BIAS_OUTPUT_DIR="$(pwd)/science_results"

# 1. Compute topological coverage and OOD features for the 3 anchored gauge tiers:
python3 science_tiers_and_figures/compute_3_anchored_tiers.py

# 2. Compute 1995-2025 Lead-Time-1 cross-validation paired statistics and render Figs S2 & S3:
python3 science_tiers_and_figures/compute_sec3_cv_and_figs_s1_s2.py

# 3. Compile Table S1 across all 68 Lead-Time-1 metrics and render Figure S4:
python3 science_tiers_and_figures/compute_table_s1_and_figs3.py

# 4. Compute exact TreeSHAP feature & category importances and render Figure 4 & Figure S7:
python3 science_tiers_and_figures/compute_treeshap_lead1.py

# 5. Render Figure 1 (global map of the 3 anchored gauge tiers):
python3 science_tiers_and_figures/build_figure1_3tiers.py
```

---

## License & Citation

This software is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](LICENSE)).

```bibtex
@unpublished{nearing2026ungauged,
  author = {Nearing, Grey and Ruparell, Karan and Taccari, Maria Luisa and Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and Wing, Oliver and Rothenberg, Juliet and Kratzert, Frederik and Cohen, Deborah and Matias, Yossi},
  title  = {{What Fraction of the World's Watersheds are Ungauged?}},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```
