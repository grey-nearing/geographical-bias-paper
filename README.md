# What Fraction of the World's Watersheds Are Ungauged?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21416986.svg)](https://doi.org/10.5281/zenodo.21416986)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official code repository and public reproduction suite for:

> **What Fraction of the World's Watersheds Are Ungauged?**
> Grey Nearing, Karan Ruparell, Maria Luisa Taccari, Stephanie Rees, Martin Gauch, Justin Ginnetti, Oliver Wing, Juliet Rothenberg, Frederik Kratzert, Deborah Cohen, and Yossi Matias (2026).

Companion public data archive (**v3.0.1**): **https://doi.org/10.5281/zenodo.21416986**

---

## Overview

This repository reproduces the global hydrological ungaugedness, national flood-alerting gap, machine-learning flood-forecast skill modeling, and factor attribution analyses reported in the manuscript:

1. **Multi-Hydrofabric Global Ungaugedness (`01_hydrofabric_ungaugedness/`)**
   - Benchmarks global streamflow monitoring across **121,819 deduplicated physical stream gauges** from 73 public archives (nested into *Standardized Open* = 34,577, *Public Discharge* = 93,827, and *All Discoverable* = 121,819).
   - Evaluates continuous and binary ungaugedness (Eq. 4, *never passes a gauge*, and 1D flow-volume weighted river discharge) symmetrically across **six fine-scale global hydrofabric configurations** — HydroSHEDS BasinATLAS Level 12, HydroSHEDS Rivers v1.0, MERIT-Basins v0.1, TDX-Hydro, GRIT v0.6 mainstem tree, and GRIT v0.6 bifurcating DAG — plus HydroSHEDS Levels 1–12 scale invariance and multi-fabric hydroclimatic out-of-distribution (OOD) dissimilarity.
2. **OpenHydroNet Cross-Validation Suite (`02_ohn_cross_validation/`)**
   - Reference training, fine-tuning, real-time data assimilation, and evaluation code for the 300-model OpenHydroNet (OHN) cross-validation experiments (*Random Ungauged*, *Hydrologically Separated*, and *Spatially In-Sample / Gauged*) across 8,811 post-2016 Caravan streamflow records (8,075 physical stations).
   - Because training 300 deep-learning models requires multi-GPU clusters, the resulting per-gauge CV evaluation CSVs across all 43 hydrological metrics and 4 modeling regimes are archived in Zenodo (`03_ohn_cv_metrics_and_splits.tar.gz`) so downstream analysis reproduces directly on CPU.
3. **Global Skill Predictors & SHAP Attribution (`03_skill_predictors_and_shap/`)**
   - Rebuilds exact 1,034,083-watershed river-network gauge coverage and 88-feature HydroATLAS climatic dissimilarity (OOD) features across all institutional data-access tiers (`stage_b1_tier_features_1034k.py`) and HydroBASINS L5–L12 scale invariance (`stage_b1a_scale_invariance_audit.py`).
   - Trains 171 LightGBM skill-predictor meta-models across 43 hydrological metrics and 4 regimes (`base`, `ft`, `da`, `ft_da`) using 10-fold gauge-grouped cross-validation and predicts global skill across all 1,034,083 HydroSHEDS Level 12 watersheds (`stage_c_train_meta_models.py`, `stage_d_counterfactuals_bootstrap_s16.py --part table_s3`).
   - Computes Exact $2^7 = 128$ Group Shapley attributions across seven predictor categories and 200-path antithetic permutation SHAP across all 95 features at $N = 1,000$ global medoid watersheds (`stage_e_global_kernelshap.py`).
4. **Publication Figures, Tables, and Verification (`04_figures_and_tables/`)**
   - Generates all public main figures (**Figs. 1, 2, 3, 4, 6**), supplementary figures (**Figs. S1–S5**), main and supplementary tables (**Table 1**, **Tables S1–S5**), and runs an automated quantitative verification suite (`verify_reproduction.py`).

---

## Third-Party Flood Hazard Licensing Notice (Fathom v3)

Fluvial flood-hazard inundation maps from **Fathom Global 3.0** are proprietary third-party data and **cannot be redistributed publicly**. Accordingly:

- **Zero Fathom data** (neither raw inundation rasters nor derived expected annual flood-exposure values) is hosted in this GitHub repository or in the public Zenodo archive (`10.5281/zenodo.21416986`).
- Every non-Fathom analysis, figure, and table in the manuscript reproduces end-to-end from scratch using purely public code and data via `run_reproduce.sh` (public mode is enabled automatically whenever Fathom inputs are absent).
- Optional Fathom-dependent outputs (**Table 2**, **Fig. 5**, **Figs. S6–S8**, Fathom exposure rows of **Table 1**, and Fathom exposure columns of **Table S2**) are skipped cleanly with an informative message when licensed Fathom data is not present locally. Users with a commercial or academic Fathom v3 license can place their basin-aggregated exposure shards in `$GEO_BIAS_DATA_DIR/round2/exposure/shards/` (or `global_basin_exposure_defended_r2.parquet` in `$GEO_BIAS_DATA_DIR/round2/exposure/`) and run the exact same `bash run_reproduce.sh` command to automatically execute the full 14-figure, 7-table reproduction including bootstrap 95% CIs and national $N=1\text{–}20$ gauge siting.

---

## Repository Structure

```text
geographical-bias-paper/
├── 01_hydrofabric_ungaugedness/          # Stage 1: Multi-hydrofabric ungaugedness benchmark
│   ├── configs/
│   │   ├── benchmark_config.py           # Canonical paths, gauge tiers, and common-domain rules
│   │   └── merit_coastal_hillslopes_drive_ids.json
│   ├── scripts/
│   │   ├── 00_stage_gauges_and_population.py
│   │   ├── 01_download_hydrosheds.py     # Public HydroSHEDS BasinATLAS v10 & HydroRIVERS v1.0
│   │   ├── 02_download_tdx_geoglows.py   # Public TDX-Hydro & GEOGLOWS v2 (AWS S3)
│   │   ├── 03_download_merit_and_grades.py # Public MERIT-Basins v0.1 & GRADES v0.1
│   │   ├── 04_download_grit.py           # Public GRIT v0.6 (Zenodo)
│   │   ├── 11_run_hydrosheds.py          # HydroSHEDS L1–L12 + HydroRIVERS runs
│   │   ├── 12_run_tdx_geoglows.py        # TDX-Hydro + GEOGLOWS v2 run
│   │   ├── 13_run_merit_grades.py        # MERIT-Basins + GRADES v0.1 run
│   │   ├── 14_run_grit.py                # GRIT v0.6 mainstem tree + bifurcating DAG runs
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
│   ├── stage_c_skill_predictor.py        # LightGBM skill-predictor wrapper
│   ├── stage_c_train_meta_models.py      # 171 meta-models (43 metrics x 4 regimes, 10-fold CV)
│   ├── stage_d_counterfactuals_bootstrap_s16.py # Table S3 skill stats (public) + optional Fathom
│   └── stage_e_global_kernelshap.py      # Exact 2^7 Group Shapley + 200-path permutation SHAP
├── 04_figures_and_tables/                # Stage 4: Publication figures, tables & verification
│   ├── build_tables.py                   # Table 1, optional Table 2, and Tables S1–S5
│   ├── generate_main_figures.py          # Figs. 1, 2, 3, 4, 6 (+ optional Fig. 5)
│   ├── generate_supp_figures.py          # Figs. S1–S5 (+ optional Figs. S6–S8)
│   └── verify_reproduction.py            # Automated quantitative verification suite
├── utils/                                # Shared path, I/O, CAP status, coverage & feature utils
├── results/                              # Lightweight non-Fathom reference summaries & Fig. S3
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
- **Disk:** ~250 GB free SSD space (for downloading and extracting raw public global hydrofabrics: HydroSHEDS BasinATLAS v10, HydroRIVERS v1.0, MERIT-Basins v0.1, TDX-Hydro + GEOGLOWS v2, and GRIT v0.6).

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

1. **Unit tests & Stage 1 (`01_hydrofabric_ungaugedness/`)**: Runs `pytest`, stages the clean 121,819-gauge catalog & WorldPop population table (plus optional locally staged Fathom exposure), downloads all four public global hydrofabric families from their official public hosts, runs all 17 hydrofabric configurations plus 4-fabric OOD dissimilarity, and synthesizes master 2D/1D/OOD tables.
2. **Stage 3 (`03_skill_predictors_and_shap/`)**: Computes 1,034,083-basin coverage & OOD features, verifies HydroBASINS L5–L12 scale invariance, trains all 171 LightGBM skill-predictor meta-models across 43 metrics and 4 regimes, builds the 4-regime Table S3 skill summary (plus full exposure counterfactuals, $B=20$ gauge-clustered bootstrap 95% CIs, and national $N=1\text{–}20$ gauge siting when licensed Fathom exposure is staged locally), and computes Exact $2^7$ Group Shapley + 200-path antithetic permutation SHAP at $N = 1,000$ global medoids.
3. **Stage 4 (`04_figures_and_tables/`)**: Generates **Table 1**, **Tables S1–S5**, **Figs. 1–4, 6**, and **Figs. S1–S5** (plus **Table 2**, **Fig. 5**, and **Figs. S6–S8** when licensed Fathom exposure is staged locally).
4. **Automated Verification**: Runs `04_figures_and_tables/verify_reproduction.py`, writing `$GEO_BIAS_DATA_DIR/round2/verification/verification_report.{json,md}`.

---

## Manuscript Figures & Tables Mapping

| Manuscript Item | Script | Publicly Reproducible Without Fathom? |
| --- | --- | :---: |
| **Fig. 1** (Global gauge catalog, never-passes-a-gauge, L12 ungaugedness, OOD) | `04_figures_and_tables/generate_main_figures.py --only fig1` | Yes |
| **Fig. 2** (Polygon-size dependence across HydroSHEDS L1–L12 & 6 fine maps) | `04_figures_and_tables/generate_main_figures.py --only fig2` | Yes |
| **Fig. 3** (Continental land/pop ungaugedness, aridity zones, OOD percentiles) | `04_figures_and_tables/generate_main_figures.py --only fig3` | Yes |
| **Fig. 4** (National CAP-gap coverage, country ungaugedness & INFORM 2026) | `04_figures_and_tables/generate_main_figures.py --only fig4` | Yes |
| **Fig. 5** (Flood exposure counterfactuals across gauge tiers & mechanisms) | `04_figures_and_tables/generate_main_figures.py --only fig5` | Requires Fathom v3 license |
| **Fig. 6** (Exact $2^7$ Group Shapley factor attribution across 12 metrics) | `04_figures_and_tables/generate_main_figures.py --only fig6` | Yes |
| **Table 1** (Multi-hydrofabric ungaugedness ranges: area, runoff, pop, 1D Q) | `04_figures_and_tables/build_tables.py --only t1` | Yes (excl. 2 Fathom rows) |
| **Table 2** (Forecast skill & global/CAP-gap annual flood exposure counterfactuals) | `04_figures_and_tables/build_tables.py --only t2` | Requires Fathom v3 license |
| **Fig. S1** (Harmonized 2016–2023 $\Delta$NSE river-network vs. local gauge skill) | `04_figures_and_tables/generate_supp_figures.py --only figs1` | Yes |
| **Fig. S2** (Top-35 individual features via 200-path permutation SHAP) | `04_figures_and_tables/generate_supp_figures.py --only figs2` | Yes |
| **Fig. S3** (Latvia conceptual illustration of ungaugedness metrics) | `results/figures/Fig_S3_conceptual_ungaugedness.png` | Yes (pre-rendered schematic) |
| **Fig. S4** (5-year 2-day flood recall across OOD dissimilarity sextiles) | `04_figures_and_tables/generate_supp_figures.py --only figs4` | Yes |
| **Fig. S5** (Predicted skill distributions across 1,034,083 global watersheds) | `04_figures_and_tables/generate_supp_figures.py --only figs5` | Yes |
| **Figs. S6–S8** (Flood exposure maps, bias-gap correlation, national siting) | `04_figures_and_tables/generate_supp_figures.py --only figs6,figs7,figs8` | Requires Fathom v3 license |
| **Table S1** (Operational & temporal status of 121,819 stream gauges) | `04_figures_and_tables/build_tables.py --only ts1` | Yes |
| **Table S2** (175 territories: basins, CAP status, ungaugedness, open gauges, INFORM) | `04_figures_and_tables/build_tables.py --only ts2` | Yes (excl. Fathom cols) |
| **Table S3** (43 hydrological metrics across 4 regimes: observed, CV $R^2$, global pred) | `04_figures_and_tables/build_tables.py --only ts3` | Yes |
| **Table S4** (Complete catalog of 95 meta-model input features) | `04_figures_and_tables/build_tables.py --only ts4` | Yes |
| **Table S5** (Global, regional, Strahler stream-order & tier ungaugedness) | `04_figures_and_tables/build_tables.py --only ts5` | Yes |

---

## Citation

If you use this repository or the Zenodo dataset, please cite [CITATION.cff](CITATION.cff):

```bibtex
@dataset{nearing_2026_ungauged_watersheds,
  author       = {Nearing, Grey and Ruparell, Karan and Taccari, Maria Luisa and
                  Rees, Stephanie and Gauch, Martin and Ginnetti, Justin and
                  Wing, Oliver and Rothenberg, Juliet and Kratzert, Frederik and
                  Cohen, Deborah and Matias, Yossi},
  title        = {What Fraction of the World's Watersheds Are Ungauged?},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.0.1},
  doi          = {10.5281/zenodo.21416986},
  url          = {https://doi.org/10.5281/zenodo.21416986}
}
```

## License

- **Code:** Apache License 2.0 ([LICENSE](LICENSE)).
- **Public Derived Data (Zenodo v3.0.1):** CC-BY-4.0 (subject to upstream source attributions listed in [DATA_DICTIONARY.md](DATA_DICTIONARY.md)).
