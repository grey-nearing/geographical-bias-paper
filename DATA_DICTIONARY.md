# Data Dictionary — Public Zenodo Archive (v3.0.0) & Upstream Hydrofabrics

Companion archive DOI: **https://doi.org/10.5281/zenodo.21416986** (v3.0.0)

> **Proprietary Data Policy:** No Fathom Global 3.0 flood-hazard rasters and no Fathom-derived expected annual flood exposure (`eae_*`, `exposed_pop_*`) columns or tables are included in any archive below.

---

## 1. Zenodo Archive Contents (`$GEO_BIAS_DATA_DIR`)

Extracting the three public data tarballs into `$GEO_BIAS_DATA_DIR` produces the following layout:

### Archive 1: `01_global_gauge_catalog.tar.gz`

Global deduplicated stream-gauge inventory compiled from 73 public national, regional, and international archives (121,819 distinct physical stations across 163 countries).

| Path (relative to `$GEO_BIAS_DATA_DIR`) | Rows | Key Columns / Purpose |
| --- | ---: | --- |
| `attributes/global_gauges_deduplicated.parquet` | 121,819 | `canonical_gauge_id`, `latitude`, `longitude`, `country`, `continent`, `drainage_area_km2`, `all_source_datasets`, `best_observation_type`, `best_data_access_policy`, `verified_record_end_year`, `is_active_opt1`, `active_reason`, `snapped_hybas_id`, `in_canonical_38069`. |
| `attributes/global_gauges_by_source_archive.parquet` | 365,290 | Un-deduplicated archive-level provenance table linking every `native_station_id` and `source_dataset` to its `canonical_gauge_id`. |
| `attributes/global_gauges.parquet` | 121,819 | Hydrofabric-agnostic gauge table (HydroBASINS-snapped fields removed) with nested boolean tier flags (`in_tier_baseline`, `in_tier_1`, `in_tier_2`, `in_tier_3`, `is_active`). |
| `attributes/grdc_attributes.csv` | 8,811 | Post-2016 Caravan training station identifiers and static catchment attributes used to anchor the baseline OpenHydroNet training subset (8,075 distinct physical stations). |
| `attributes/caravan_attributes.csv` | 121,819 | 88 HydroATLAS upstream catchment attributes mapped to every `canonical_gauge_id` for tier-specific OOD scoring in Stage 3 (`stage_b1_tier_features_1034k.py`). |

### Archive 2: `02_hydroatlas_worldpop_and_auxiliary.tar.gz`

HydroSHEDS Level 12 topology, WorldPop 2020 basin population totals, Natural Earth country polygons, INFORM 2026 Risk Index, pre-aggregated GRADES runoff, and the 95-feature matrices & 38k polygon-overlap lookup tables required by Stage 3.

| Path (relative to `$GEO_BIAS_DATA_DIR`) | Rows / Shape | Key Columns / Purpose |
| --- | ---: | --- |
| `attributes/hybas_l12_topology_and_runoff.parquet` | 1,034,083 | `HYBAS_ID`, `NEXT_DOWN`, `continent`, `area_m2`, `SUB_AREA`, `PFAF_ID`, `run_mm_syr` (extracted from HydroSHEDS BasinATLAS v10 Level 12; replaces multi-GB raw CSV). |
| `attributes/hybas_routing_topology.csv` | 1,034,083 | `HYBAS_ID`, `NEXT_DOWN`, `continent`, `area_m2` routing graph across all 9 HydroBASINS regional archives. |
| `attributes/hybas_latlon.csv` | 1,034,083 | Basin centroid coordinates (`lat`, `lon`) keyed by `hybas_<HYBAS_ID>`. |
| `attributes/hydrosheds_l12_worldpop_population.parquet` | 1,034,083 | Strictly 2 columns: `HYBAS_ID` and `total_population` (WorldPop 2020 100 m unconstrained population summed per HydroSHEDS L12 polygon; global total 7.963B). **Zero Fathom columns.** |
| `saved_predictions/country_mappings/hybas_to_country_mapping.csv` | 1,034,083 | Maps each HydroSHEDS L12 basin to its Natural Earth territory (`175` mapped territories; `174` excluding Greenland). |
| `naturalearth_lowres.geojson` | 177 | Natural Earth 1:110m country boundaries used for territory mapping, spatial open-gauge border counts (`Table S2`), and cartography (`Figs. 1, 4`). |
| `risk_data/INFORM_Risk_2026_v072 - INFORM Risk 2026 (a-z).csv` | 191 | Official INFORM Risk Index 2026 v0.7.2 (`ISO3`, `COUNTRY`, `VULNERABILITY`, `LACK OF COPING CAPACITY`; 166 matched countries in `Fig. 4` & `Table S2`). |
| `flow_models/grades_mean_annual_runoff_1979_2013.nc` | 2.94M reaches | Pre-aggregated 1979–2013 mean annual reach discharge (`Q_mean_m3s`) from GRADES v0.1 for MERIT-Basins (~6.7 MB, avoiding multi-hundred-GB daily NetCDF downloads). |
| `generic_training_data/ungauged_generic_training_data.parquet` | 26,433 × 95 | Stacked 3-split (*Gauged*, *Random Ungauged*, *Hydrologically Separated*) × 8,811 Caravan training gauges × 95 meta-model features (`float32`). |
| `generic_training_data/hybas_generic_inference_data.parquet` | 1,034,083 × 95 | Global HydroSHEDS Level 12 inference matrix across the exact same 95 meta-model features (`float32`). |
| `coverage_fractions/train_fold_ml_features.csv` | 17,622 | Per-fold river-network coverage features (`total_upstream_coverage`, `max_upstream_coverage`, `max_downstream_coverage`, `upstream_gauge_count`) for the Random Ungauged & Hydrologically Separated CV splits (used in `Fig. S1`). |
| `coverage_fractions/polygon_overlaps_38k/*.parquet` | 20 files | Compact Parquet polygon-overlap lookup tables (`caravan_attributes_38k.parquet`, `caravan_pairwise_overlap_areas.parquet`, and per-continent `infer_pairwise_overlaps_{AF..SI}.parquet` + `infer_ml_features_{AF..SI}.parquet`) enabling exact sub-basin polygon-overlap river-network coverage for the 38,069 Caravan/EStreams/GRDC polygons in `stage_b1_tier_features_1034k.py`. |
| `anomaly_scores/{gauged,random,separated}_anomaly_scores.csv` | 8,811 each | Per-split Isolation Forest, One-Class SVM, and Local Outlier Factor OOD scores at Caravan CV gauges (used in `Fig. S4`). |

### Archive 3: `03_ohn_cv_metrics_and_splits.tar.gz`

Out-of-sample evaluation metrics from the 300-model OpenHydroNet cross-validation experiments and the 10-fold CV split definitions.

| Path (relative to `$GEO_BIAS_DATA_DIR`) | Contents |
| --- | --- |
| `round2/hydro_model_metrics/` | Per-gauge lead-time-1 evaluation CSVs across all **43 hydrological metrics** (8 harmonized 2016–2023 continuous hydrograph metrics, 32 return-period recall/precision metrics over 1986–2023, and 3 2-day F1 metrics) across all 3 CV splits (`gauged`, `random`, `separated`) and 4 modeling regimes (`base`, `_ft`, `_da`, `_ft_da`). |
| `cross_validation_splits/` | Deterministic 10-fold gauge-grouped CV split assignments (`random`, `separated`, and `gauged` splits). |

### Archive 4: `geographical-bias-paper-v3.0.0.zip`

Frozen snapshot of the GitHub code repository at release tag `v3.0.0`.

---

## 2. Public Upstream Hydrofabrics (Downloaded Automatically in Stage 1)

Stage 1 (`01_hydrofabric_ungaugedness/scripts/01..04_download_*.py`) fetches four publicly hosted global river-network families directly from their authoritative upstream providers (not mirrored in Zenodo due to size):

| Hydrofabric Family | Upstream Source | Approx. Download |
| --- | --- | ---: |
| **HydroSHEDS BasinATLAS v10** (Levels 1–12, WaterGAP v2.2 runoff) & **HydroRIVERS v1.0** | `data.hydrosheds.org` (HydroSHEDS official CDN) | ~4.2 GB |
| **TDX-Hydro** & **GEOGLOWS v2** mean annual discharge | Public AWS S3 (`geoglows-v2` / NGA TDX-Hydro open buckets) | ~18 GB |
| **MERIT-Basins v0.1** (cat + riv shapefiles & coastal hillslopes) | Public MERIT-Basins Google Drive mirror (300 indexed public IDs) | ~8.5 GB |
| **GRIT v0.6** (mainstem tree & bifurcating DAG) | Public Zenodo record (GRIT v0.6 official release) | ~12 GB |

---

## 3. Lightweight Reference Summaries Committed in `results/`

The GitHub repository includes strictly non-Fathom reference summaries so readers can inspect headline outputs without running the full pipeline:

- `results/hydrofabric_summary/master_cross_hydrofabric_summary_nonfathom.csv`: Cross-hydrofabric summary across all 17 runs and 4 gauge tiers (all Fathom exposure columns removed).
- `results/meta_models/stage_c_cv_summary.csv` & `features_95.json`: 10-fold gauge-grouped CV $R^2$ for all 171 LightGBM meta-models and exact 95-feature list.
- `results/tables/Table_S3_4regimes_skill.csv`: All 43 metrics × 4 regimes summary (`Table S3`).
- `results/shap/shap_{category_pct,category_mean,top35_features}_r2.csv` & `shap_method_crosscheck_r2.json`: Exact $2^7$ Group Shapley and 200-path permutation SHAP attribution summaries (`Fig. 6`, `Fig. S2`).
- `results/figures/Fig_S3_conceptual_ungaugedness.png`: Pre-rendered conceptual Latvia watershed schematic (`Fig. S3`).
