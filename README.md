# What Fraction of the World's Watersheds are Ungauged?

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Authors:**
Grey Nearing (Google Research; Zurich, Switzerland),
Karan Ruparell (European Centre for Medium-Range Weather Forecasts; Reading, United Kingdom & Department of Meteorology, University of Reading; Reading, United Kingdom),
Frederik Kratzert (Google Research; Vienna, Austria),
Maria Luisa Taccari (European Centre for Medium-Range Weather Forecasts; Reading, United Kingdom),
Stephanie Rees (Google; London, United Kingdom),
Martin Gauch (Google Research; Zurich, Switzerland),
Justin Ginnetti (Google Research; Zurich, Switzerland),
Oliver Wing (Fathom; Bristol, United Kingdom & School of Geographical Sciences, University of Bristol; Bristol, United Kingdom),
Juliet Rothenberg (Google Research; Mountain View, California, United States),
Christel Prudhomme (European Centre for Medium-Range Weather Forecasts; Reading, United Kingdom),
Florian Pappenberger (European Centre for Medium-Range Weather Forecasts; Reading, United Kingdom),
Deborah Cohen (Google Research; Tel Aviv, Israel),
Yossi Matias (Google Research; Mountain View, California, United States)

This private repository contains the source code, evaluation pipelines, `SkillPredictor` meta-model training and global inference scripts, and non-proprietary hydrological skill and ungaugedness tables/figures for *"What Fraction of the World's Watersheds are Ungauged?"*.

> **Proprietary Data Licensing Notice (Fathom v3):**
> Commercial Fathom v3 fluvial flood hazard maps and all Fathom-derived flood exposure datasets (`global_basin_exposure_by_return_period.csv`, `basin_exposure_and_blockers_v2.parquet`, `country_exposure_and_blockers_v2.csv`, `Table_S3_lead1.csv`, `anchored_3tiers_exposure_summary_lead1.csv`, `multi_mechanism_tiered_skill_and_exposure_lead1.csv`, and flood exposure figures) are subject to commercial licensing restrictions and are **strictly excluded** from this repository and from external archives. Only open hydrological skill, topological gauge coverage, and HydroATLAS out-of-distribution (OOD) representativeness datasets and figures are included in `science_results/`. Researchers wishing to run the flood exposure intersection scripts must obtain a Fathom v3 license directly from [Fathom](https://www.fathom.global).

---

## Overview of the Methodology & Pipeline

1. **Continuous Topological and Hydroclimatic Ungaugedness Across `964,084` Global Watersheds (`science_tiers_and_figures/`)**:
   - Evaluates continuous river-network gauge coverage ($U_{\text{net}} = 1 - \max(C_{\text{up}}, C_{\text{down}})$) and multivariate HydroATLAS out-of-distribution anomaly scores ($U_{\text{clim}}$, combining Isolation Forest, One-Class SVM, and Local Outlier Factor across 88 static catchment attributes) across all `964,084` global HydroBASINS Level 12 watersheds.
   - Evaluates three anchored global gauge-accessibility tiers:
     1. **Open Community Compilations (`34,577` gauges)**: Caravan + EStreams + GRDC open academic compilations (`75.8%` of global watershed land area completely disconnected, $U_{\text{net}} = 1$; `91.4%` with $U_{\text{net}} > 0.5$).
     2. **Publicly Accessible Discharge Records (`93,827` gauges)**: Adding national hydrologic service APIs and bulk public portals (`66.8%` of global land area with $U_{\text{net}} = 1$; `85.2%` with $U_{\text{net}} > 0.5$).
     3. **All Publicly Knowable Gauges (`121,819` gauges)**: Including view-only web portals, stage-only stations, and restricted metadata inventories (`63.5%` of global land area with $U_{\text{net}} = 1$; `81.9%` with $U_{\text{net}} > 0.5$).

2. **30-Year (`1995–2025`) Lead-Time-1 Cross-Validation, Local Fine-Tuning (`_ft`), and Data Assimilation (`_da`) (`cv_1995_2025/`)**:
   - Evaluates the operational Dual-LSTM global flood forecasting architecture (`google-research/flood-forecasting`, 1-day-ahead forecast horizon `lead_time = 1`) over the 30-year (`1995–2025`) evaluation window across `8,811` post-2016 active Caravan gauges (`68` hydrological and flood return-period metrics).
   - Compares six experimental regimens across `8,811` basins:
     - Temporal split (`gauged`: trained `1980–1995`, evaluated `1995–2025`) and real-time autoregressive discharge data assimilation (`gauged_da`);
     - 10-fold random out-of-sample cross-validation (`random`), local per-basin fine-tuning (`random_ft`), real-time data assimilation (`random_da`), and combined local fine-tuning + data assimilation (`random_ft_da`);
     - 10-fold hydroclimatically separated cross-validation (`separated`), local fine-tuning (`separated_ft`), real-time data assimilation (`separated_da`), and combined local fine-tuning + data assimilation (`separated_ft_da`).

3. **Global `SkillPredictor` Meta-Models & TreeSHAP Attribution (`science_tiers_and_figures/`, `utils/`)**:
   - Trains 500-tree Random Forest `SkillPredictor` meta-models mapping the 96 catchment, topological coverage, record-length, and OOD representativeness features to 1-day-ahead hydrological skill across all 68 metrics (`Table_S1_lead1.csv`), projects skill across all `964,084` global HydroBASINS Level 12 watersheds (`Fig_S4_global_predicted_skill_lead1.png`), and decomposes skill drivers via exact TreeSHAP (`Fig_4_TreeSHAP_lead1.png`, `Fig_S7_TreeSHAP_top35_lead1.png`, `treeshap_category_importances_lead1.csv`).

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
├── science_tiers_and_figures/                        # Anchored Gauge-Accessibility Tiers, TreeSHAP & Science Figures
│   ├── build_tier_features.py                        # Topological coverage & OOD feature builder across gauge tiers
│   ├── compute_3_anchored_tiers.py                   # Build the 3 anchored tiers (34,577 / 93,827 / 121,819 gauges)
│   ├── compute_sec3_cv_and_figs_s1_s2.py             # Section 3 CV paired statistics (Wilcoxon/Mann-Whitney) & Figs S2-S3
│   ├── compute_table_s1_and_figs3.py                 # Compile Table S1 across 68 metrics & render global skill map Fig S4
│   ├── compute_treeshap_lead1.py                     # Exact TreeSHAP attribution across 96 features (Fig 4 & Fig S7)
│   ├── build_figure1_3tiers.py                       # Render Figure 1 (global gauge map across 3 anchored tiers)
│   ├── build_figure4_3panel_anchored.py              # Render 3-panel tiered skill comparison
│   ├── compute_table_s3_and_figs4_s5.py              # Conditional tiered skill & exposure script (requires Fathom inputs)
│   ├── evaluate_3_anchored_tiers_exposure.py         # Anchored tier exposure script (requires Fathom inputs)
│   ├── evaluate_multi_mechanism_tiered_exposure.py   # Multi-mechanism (_ft / _da / _ft_da) evaluation script
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
  author = {Nearing, Grey and Ruparell, Karan and Kratzert, Frederik and Taccari, Maria Luisa and Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and Wing, Oliver and Rothenberg, Juliet and Prudhomme, Christel and Pappenberger, Florian and Cohen, Deborah and Matias, Yossi},
  title  = {{What Fraction of the World's Watersheds are Ungauged?}},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```
