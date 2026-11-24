# What Fraction of the World's Watersheds Are Ungauged?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21416986.svg)](https://doi.org/10.5281/zenodo.21416986)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official code repository and public reproduction suite for:

> **What Fraction of the World's Watersheds Are Ungauged?**
> Grey Nearing, Karan Ruparell, Maria Luisa Taccari, Stephanie Rees, Martin Gauch, Justin Ginnetti, Oliver Wing, Paul Bates, Juliet Rothenberg, Frederik Kratzert, Deborah Cohen, and Yossi Matias (2026).

Companion public data archive (**v3.3.1**): **https://doi.org/10.5281/zenodo.21416986**

---

## Overview

This repository reproduces the global hydrological ungaugedness, national flood-alerting gap, machine-learning flood-forecast skill modeling, and factor attribution analyses reported in the manuscript:

1. **Multi-Hydrofabric Global Ungaugedness (`01_hydrofabric_ungaugedness/`)**
   - Benchmarks global streamflow monitoring across **119,169 deduplicated physical stream gauges** (`361,020` archive records across 72 physical-gauge repositories out of 73 audited public repositories; strictly physical in-situ river gauges, excluding simulated runoff and non-river weather/offshore/groundwater/lake stations), nested into *Standardized, Open-License Discharge Datasets* (`32,369` stations redistributing observed streamflow time series in harmonized multi-basin compilations under full-use open licenses), *Gauges with Discharge Records* (`82,596` stations where observed daily or monthly river discharge records exist across open compilations, national portals, and restricted or non-commercial archives, inclusive of the `32,369` standardized open gauges), and *All Publicly Discoverable Gauges* (`119,169` physical river gauges), including observation-type breakdowns and strict $\ge 2022$ daily-discharge operational subsets.
   - Evaluates continuous and binary ungaugedness (Eq. 4, *never passes a gauge*, and 1D flow-volume weighted river discharge) symmetrically across **six fine-scale global hydrofabric configurations** — HydroSHEDS BasinATLAS Level 12, HydroSHEDS Rivers v1.0, MERIT-Basins v0.1, TDX-Hydro, GRIT v1.0 mainstem tree, and GRIT v1.0 bifurcating DAG — plus HydroSHEDS Levels 1–12 scale invariance, multi-fabric hydroclimatic out-of-distribution (OOD) dissimilarity, and spatially adjusted country correlations (Moran's $I$ and Clifford–Richardson–Hémon / Dutilleul $N_{\text{eff}}$).
2. **Global LSTM (`OpenHydroNet`) Cross-Validation Suite (`02_ohn_cross_validation/`)**
   - Reference training, fine-tuning, real-time data assimilation, and evaluation code for the 210-model global LSTM (`OpenHydroNet`) cross-validation experiments (*Random Ungauged*, *Hydrologically Separated*, and *Spatially In-Sample / Gauged*) across 8,811 post-2016 Caravan streamflow records (8,072 unique physical locations after coordinate deduplication; 8,075 distinct gauge IDs).
   - Because training 210 deep-learning models requires multi-GPU clusters, the resulting per-gauge CV evaluation CSVs across all 43 hydrological metrics and 4 modeling regimes are archived in Zenodo (`03_ohn_cv_metrics_and_splits.tar.gz`) so downstream analysis reproduces directly on CPU.
3. **Global Skill Predictors & Exact Group Shapley Attribution (`03_skill_predictors_and_shap/`)**
   - Rebuilds exact 1,034,083-watershed river-network gauge coverage and 88-feature HydroATLAS climatic dissimilarity (OOD) features across all institutional data-access tiers — including local universal gauging (no OOD clamp), the unsampled-climate global-training upper bound (HydroATLAS similarity scores floored at the Caravan training 75th percentile), and strict $\ge 2022$ daily-discharge DA (`stage_b1_tier_features_1034k.py`) — plus HydroBASINS L5–L12 scale invariance (`stage_b1a_scale_invariance_audit.py`).
   - Trains 171 strongly regularized stacked HistGradientBoosting + LightGBM + XGBoost + WeightedRidge skill-predictor meta-models (RidgeCV meta-learner) across 43 hydrological metrics and 4 regimes (`base`, `ft`, `da`, `ft_da`) using multi-level `MAIN_BAS`, `Station`, `Climate`, and `LOCO` cross-validation on physically deduplicated gauges and predicts global skill across all 1,034,083 HydroSHEDS Level 12 watersheds (`stage_c_train_meta_models.py`, `stage_d_counterfactuals_bootstrap_s16.py --part table_s3`).
   - Computes Exact $2^7 = 128$ Group Shapley attributions across seven predictor categories (with `ria_ha_usu` grouped in Topography) at $N = 1,000$ global medoid watersheds (`stage_e_global_kernelshap.py`).
4. **Publication Figures, Tables, Supplementary Data, and Verification (`04_figures_and_tables/`)**
   - Generates all public main figures (**Figs. 1, 2, 3, 6**), public supplementary figures (**Figs. S1–S6**), and public supplementary tables (**Tables S1–S5, S7–S8**, numbered strictly in order of first manuscript citation; 0 standalone main tables), archives the machine-readable supplementary datasets (**Data S1** and **Data S2** in `results/supplementary_data/`), and runs an automated quantitative verification suite (`verify_reproduction.py`).

---

## Third-Party Flood Hazard Licensing Notice (Fathom v3)

Fluvial flood-hazard inundation maps from **Fathom Global 3.0** are proprietary third-party data and **cannot be redistributed publicly**. Accordingly:

- **Zero raw Fathom rasters and zero watershed-level (`1,034,083`-basin) Fathom exposure tables** are hosted in this GitHub repository or in the public Zenodo archive (`10.5281/zenodo.21416986`). The only Fathom-derived figures included in the public archive are the published 175-territory aggregate summary columns in **Data S1** (`results/supplementary_data/Data_S1_175_territories.csv`), which accompany manuscript **Table S4**.
- Every non-Fathom analysis, figure, and table in the manuscript reproduces end-to-end from scratch using purely public code and data via `run_reproduce.sh` (public mode is enabled automatically whenever local Fathom inputs are absent).
- Optional Fathom-dependent outputs (**Figs. 4, 5**, **Figs. S7–S8**, **Table S6** and action-regime sensitivity table, Fathom exposure rows of **Table S3**, and Fathom exposure columns of **Table S4**) are skipped cleanly with an informative message when licensed Fathom data is not present locally. Users with a commercial or academic Fathom v3 license can place their basin-aggregated exposure shards in `$GEO_BIAS_DATA_DIR/$GEO_BIAS_ROUND_SUBDIR/exposure/shards/` (or `global_basin_exposure_defended_r2.parquet` in `$GEO_BIAS_DATA_DIR/$GEO_BIAS_ROUND_SUBDIR/exposure/`) and run the exact same `bash run_reproduce.sh` command to automatically execute the full 14-figure, 8-table reproduction including Recall & CSI ($\alpha = \text{CSI}/R$) accounting, $B=200$ stratified `MAIN_BAS` $\times$ climate block-bootstrap 95% CIs, OOD-scaled cross-domain uncertainty, and national $N=1\text{–}20$ gauge siting.

---

## Repository Structure

```text
geographical-bias-paper/
├── 01_hydrofabric_ungaugedness/          # Stage 1: Multi-hydrofabric ungaugedness benchmark
│   ├── configs/
│   │   ├── audit_reference/              # 73-archive license & streamflow audit tables + station sets
│   │   ├── benchmark_config.py           # Canonical paths, gauge tiers, and common-domain rules
│   │   └── merit_coastal_hillslopes_drive_ids.json
│   ├── scripts/
│   │   ├── audit_and_rebuild_gauge_catalog.py # Rebuilds 119,169 physical gauge catalog & tier flags
│   │   ├── 00_stage_gauges_and_population.py
│   │   ├── 01_download_hydrosheds.py     # Public HydroSHEDS BasinATLAS v10 & HydroRIVERS v1.0
│   │   ├── 02_download_tdx_geoglows.py   # Public TDX-Hydro & GEOGLOWS v2 (AWS S3)
│   │   ├── 03_download_merit_and_grades.py # Public MERIT-Basins v0.1 & GRADES v0.1
│   │   ├── 04_download_grit.py           # Public GRIT v1.0 (Zenodo)
│   │   ├── 11_run_hydrosheds.py          # HydroSHEDS L1–L12 + HydroRIVERS runs
│   │   ├── 12_run_tdx_geoglows.py        # TDX-Hydro + GEOGLOWS v2 run
│   │   ├── 13_run_merit_grades.py        # MERIT-Basins + GRADES v0.1 run
│   │   ├── 14_run_grit.py                # GRIT v1.0 mainstem tree + bifurcating DAG runs
│   │   ├── 15_run_ohn_ood_4fabrics.py    # 4-fabric cutoff-free hydroclimatic OOD analysis
│   │   └── 20_synthesize_and_plot.py     # Master 2D/1D/OOD synthesis across all 17 runs
│   ├── src/global_ungaugedness/          # Core graph snapping, Eq. 4, 1D, DAG & enrichment library
│   └── tests/test_mathematical_identities.py
├── 02_ohn_cross_validation/              # Stage 2: Reference OpenHydroNet CV & metrics code
│   ├── hydro_metrics/                    # Continuous & return-period flood metric calculators
│   ├── fast_multimet.py
│   ├── prepare_caravan_post2016.py
│   ├── run_cv_orchestrator.py
│   ├── run_ft_da_cv.py
│   ├── run_stage1_option_b_metrics.py
│   └── stitch_and_compute_ft_da_metrics.py
├── 03_skill_predictors_and_shap/         # Stage 3: 1,034,083-basin features, meta-models & SHAP
│   ├── stage_b0_merge_and_audit_exposure.py  # Optional (licensed Fathom v3 users only)
│   ├── stage_b1_tier_features_1034k.py   # Gauge coverage + 88-feature OOD across all tiers
│   ├── stage_b1a_scale_invariance_audit.py   # HydroBASINS L5–L12 scale invariance check
│   ├── stage_c_skill_predictor.py        # Regularized 4-learner stack (HGB+LGB+XGB+WeightedRidge, v2_strongreg)
│   ├── stage_c_train_meta_models.py      # 171 meta-models (43 metrics x 4 regimes, multi-level spatial CV)
│   ├── stage_d_counterfactuals_bootstrap_s16.py # Table S7 skill stats (public) + optional Fathom counterfactuals
│   └── stage_e_global_kernelshap.py      # Exact 2^7 = 128 Group Shapley at N = 1,000 global medoids
├── 04_figures_and_tables/                # Stage 4: Publication figures, tables & verification
│   ├── build_table_s1.py                 # Table S1 (72 physical stream-gauge repositories)
│   ├── build_tables.py                   # Tables S1–S8 (with optional Fathom exposure rows/cols & Table S6)
│   ├── generate_main_figures.py          # Figs. 1, 2, 3, 6 (+ optional Figs. 4–5 with Fathom)
│   ├── generate_supp_figures.py          # Figs. S1, S3–S6 (+ optional Figs. S7–S8 with Fathom)
│   └── verify_reproduction.py            # Automated quantitative verification suite
├── utils/                                # Shared path, I/O, CAP status, spatial stats & feature utils
├── results/                              # Lightweight non-Fathom reference summaries, Data S1–S2 & Fig. S2 schematic
├── run_reproduce.sh                      # One-command reproduction (public mode & auto Fathom mode)
├── DATA_DICTIONARY.md                    # Complete schema guide for Zenodo archives & outputs
├── environment.yml                       # Reproducible Conda/Mamba environment
├── pyproject.toml
├── CITATION.cff
└── .zenodo.json
```

---

## Quick Start: End-to-End Reproduction from Scratch

### System Requirements

- **OS:** Linux (tested on Debian 12 / Ubuntu 22.04+)
- **CPU / RAM:** 16–32 CPU cores and ≥64 GB RAM recommended (reproduces cleanly in ~2–3 hours on a 32-vCPU / 256 GB `n2-highmem-32` VM; Stage 1 hydrofabric geometry processing peaks near ~40 GB RAM for HydroRIVERS and GRIT).
- **Disk:** ~250 GB free SSD space (for downloading and extracting raw public global hydrofabrics: HydroSHEDS BasinATLAS v10, HydroRIVERS v1.0, MERIT-Basins v0.1, TDX-Hydro + GEOGLOWS v2, and GRIT v1.0).

### 1. Create the Conda / Mamba Environment

```bash
mamba env create -f environment.yml
conda activate geographical-bias-paper
```

### 2. Download and Extract the Public Zenodo Data Archive

Download the three public data tarballs (~405 MB total) from [Zenodo (`10.5281/zenodo.21416986`)](https://doi.org/10.5281/zenodo.21416986) into a local data directory (e.g., `./data`):

```bash
export GEO_BIAS_DATA_DIR="$PWD/data"
mkdir -p "$GEO_BIAS_DATA_DIR"

# Or let the reproduction script fetch them automatically with --download-zenodo:
bash run_reproduce.sh --download-zenodo
```

If downloading manually, extract all three archives into `$GEO_BIAS_DATA_DIR`:

```bash
tar -xzf 01_global_gauge_catalog.tar.gz -C "$GEO_BIAS_DATA_DIR"
tar -xzf 02_hydroatlas_worldpop_and_auxiliary.tar.gz -C "$GEO_BIAS_DATA_DIR"
tar -xzf 03_ohn_cv_metrics_and_splits.tar.gz -C "$GEO_BIAS_DATA_DIR"
```

> **Note on ReachHydro MERIT-Basins Google Drive rate limits:** If ReachHydro's public Google Drive folder hits its 24-hour anonymous download quota during `03_download_merit_and_grades.py`, download MERIT-Basins v0.1 from ReachHydro's official Globus mirror (`origin_id=f349971a-b759-4e38-8bc4-f6ccb057c499`) and export `MERIT_BASINS_LOCAL_DIR=/path/to/merit` before running `run_reproduce.sh`.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the exact file tree and column schemas inside each archive.

### 3. Run the End-to-End Pipeline

```bash
# Public mode (non-Fathom) — or Full mode automatically if licensed Fathom v3 exposure is staged locally:
bash run_reproduce.sh
```

`run_reproduce.sh` executes four sequential stages:

1. **Unit tests & Stage 1 (`01_hydrofabric_ungaugedness/`)**: Runs `pytest`, stages the clean 119,169-gauge physical catalog & WorldPop population table (plus optional locally staged Fathom exposure), downloads all four public global hydrofabric families from their official public hosts, runs all 17 hydrofabric configurations plus 4-fabric OOD dissimilarity, and synthesizes master 2D/1D/OOD tables.
2. **Stage 3 (`03_skill_predictors_and_shap/`)**: Computes 1,034,083-basin coverage & OOD features (including `tier5_universal_no_ood_clamp`, `tier5_universal_idealized_ceiling`, and strict `>=2022` daily-discharge DA), verifies HydroBASINS L5–L12 scale invariance, trains all 171 regularized 4-learner stacked skill-predictor meta-models (`HistGradientBoosting` + `LightGBM` + `XGBoost` + `WeightedRidge`, `RidgeCV` meta-learner) across 43 metrics and 4 regimes with multi-level spatial CV (`MAIN_BAS_5f`, `Station_5f`, `Climate_8f`, `LOCO_5cont`), builds the 4-regime Table S7 skill summary (plus full Recall & CSI exposure counterfactuals, $B=200$ stratified `MAIN_BAS` $\times$ climate block-bootstrap 95% CIs, OOD-scaled cross-domain uncertainty, and national $N=1\text{–}20$ gauge siting when licensed Fathom exposure is staged locally), and computes Exact $2^7 = 128$ Group Shapley attribution at $N = 1,000$ global medoids.
3. **Stage 4 (`04_figures_and_tables/`)**: Generates **Tables S1–S5, S7–S8**, **Figs. 1–3, 6**, and **Figs. S1, S3–S6** (plus **Table S6**, action-regime sensitivity table, **Figs. 4–5**, and **Figs. S7–S8** when licensed Fathom exposure is staged locally).
4. **Automated Verification**: Runs `04_figures_and_tables/verify_reproduction.py`, writing `$GEO_BIAS_DATA_DIR/$GEO_BIAS_ROUND_SUBDIR/verification/verification_report.{json,md}`.

---

## Manuscript Figures, Tables & Supplementary Data Mapping

| Manuscript Item | Script / File | Publicly Reproducible Without Fathom? |
| --- | --- | :---: |
| **Fig. 1** (Global gauge catalog, never-passes-a-gauge, L12 ungaugedness, OOD) | `04_figures_and_tables/generate_main_figures.py --only fig1` | Yes |
| **Fig. 2** (Polygon-size scaling across L1–L12 & 6 hydrofabrics + visual Table S3 ranges) | `04_figures_and_tables/generate_main_figures.py --only fig2` | Yes (excl. 2 Fathom rows) |
| **Fig. 3** (Continental land/pop, aridity, OOD, CAP-gap boxplots & INFORM 2026) | `04_figures_and_tables/generate_main_figures.py --only fig3` | Yes (excl. Fathom series) |
| **Fig. 4** (Global flood exposure, today's open ML coverage, & cumulative gain maps) | `04_figures_and_tables/generate_main_figures.py --only fig4` | Requires Fathom v3 license |
| **Fig. 5** (CAP-gap/Alerting/Global exposure synthesis, Recall vs. CSI, mechanisms, vulnerability) | `04_figures_and_tables/generate_main_figures.py --only fig5` | Requires Fathom v3 license |
| **Fig. 6** (Exact $2^7$ Group Shapley factor attribution across 12 metrics) | `04_figures_and_tables/generate_main_figures.py --only fig6` | Yes |
| **Fig. S1** (Gauge operational status, end years, record lengths, & latitudes) | `04_figures_and_tables/generate_supp_figures.py --only figs1` | Yes |
| **Fig. S2** (Latvia conceptual illustration of ungaugedness metrics) | `results/figures/Fig_S2_conceptual_ungaugedness.png` | Yes (pre-rendered schematic) |
| **Fig. S3** (National flood-alerting & WMO CAP coverage across 174 countries) | `04_figures_and_tables/generate_supp_figures.py --only figs3` | Yes |
| **Fig. S4** (Harmonized 2016–2023 $\Delta$NSE river-network vs. local gauge skill) | `04_figures_and_tables/generate_supp_figures.py --only figs4` | Yes |
| **Fig. S5** (5-year 2-day flood recall across OOD dissimilarity sextiles) | `04_figures_and_tables/generate_supp_figures.py --only figs5` | Yes |
| **Fig. S6** (Predicted skill distributions across 1,034,083 global watersheds) | `04_figures_and_tables/generate_supp_figures.py --only figs6` | Yes |
| **Fig. S7** (Precision–Recall curves & warning-to-action regime sensitivity) | `04_figures_and_tables/generate_supp_figures.py --only figs7` | Requires Fathom v3 license |
| **Fig. S8** (National $N = 1\text{–}20$ strategic gauge siting across 173 territories) | `04_figures_and_tables/generate_supp_figures.py --only figs8` | Requires Fathom v3 license |
| **Table S1** (72 physical stream-gauge repositories: counts, licenses, URLs) | `04_figures_and_tables/build_tables.py --only ts1` | Yes |
| **Table S2** (Operational status, observation types, & strict $\ge 2022$ DA counts) | `04_figures_and_tables/build_tables.py --only ts2` | Yes |
| **Table S3** (Multi-hydrofabric ungaugedness ranges across 6 hydrofabrics) | `04_figures_and_tables/build_tables.py --only ts3` | Yes (excl. 2 Fathom rows) |
| **Table S4** (175 territories: basins, CAP status, ungaugedness, INFORM, $N_{\text{eff}}$) | `04_figures_and_tables/build_tables.py --only ts4` | Yes (excl. Fathom cols) |
| **Table S5** (Complete catalog of 95 meta-model input features; `ria_ha_usu` in Topography) | `04_figures_and_tables/build_tables.py --only ts5` | Yes |
| **Table S6** (Forecast skill & exposure counterfactuals + action-regime sensitivity) | `04_figures_and_tables/build_tables.py --only ts6` | Requires Fathom v3 license |
| **Table S7** (43 hydrological metrics across 4 regimes: observed, multi-level CV $R^2$, pred) | `04_figures_and_tables/build_tables.py --only ts7` | Yes |
| **Table S8** (Global, regional, Strahler stream-order & tier ungaugedness) | `04_figures_and_tables/build_tables.py --only ts8` | Yes |
| **Data S1** (Machine-readable 175-territory summary dataset accompanying Table S4) | `results/supplementary_data/Data_S1_175_territories.csv` | Archived directly (summary CSV) |
| **Data S2** (Machine-readable 43-metric hydrological skill dataset accompanying Table S7) | `results/supplementary_data/Data_S2_43_hydrological_metrics.csv` | Yes |

---

## Citation

If you use this repository or the Zenodo dataset, please cite [CITATION.cff](CITATION.cff):

```bibtex
@dataset{nearing_2026_ungauged_watersheds,
  author       = {Nearing, Grey and Ruparell, Karan and Taccari, Maria Luisa and
                  Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and
                  Wing, Oliver and Bates, Paul and Rothenberg, Juliet and
                  Kratzert, Frederik and Cohen, Deborah and Matias, Yossi},
  title        = {What Fraction of the World's Watersheds Are Ungauged?},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.3.0},
  doi          = {10.5281/zenodo.21416986},
  url          = {https://doi.org/10.5281/zenodo.21416986}
}
```

## License

- **Code:** Apache License 2.0 ([LICENSE](LICENSE)).
- **Public Derived Data (Zenodo v3.3.1):** CC-BY-4.0 (subject to upstream source attributions listed in [DATA_DICTIONARY.md](DATA_DICTIONARY.md)).
