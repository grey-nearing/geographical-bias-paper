# Data Dictionary & Zenodo Archive Reference (`10.5281/zenodo.21416987`)

This document describes the directory hierarchy, archive contents, file schemas, and exact column definitions for all datasets archived in **[https://doi.org/10.5281/zenodo.21416987](https://doi.org/10.5281/zenodo.21416987)** and used by the scripts in [`https://github.com/grey-nearing/geographical-bias-paper`](https://github.com/grey-nearing/geographical-bias-paper).

---

## 1. Zenodo Archive Inventory (`10.5281/zenodo.21416987`)

When extracted into `./data` (`GEO_BIAS_DATA_DIR`), the Zenodo archives populate the exact directory tree expected by `utils/file_paths.py`:

| Zenodo File | Extracted Path in `GEO_BIAS_DATA_DIR` | Description |
| :--- | :--- | :--- |
| `geographical-bias-paper-v1.0.1.tar.gz` | Standalone repository source release | Frozen `v1.0.1` source code, scripts, and pre-rendered figures/tables. |
| `v2_exposure_blockers.tar.gz` | `v2_exposure_blockers/` | Master 964,084-basin exposure & blocker table (`basin_exposure_and_blockers_v2.parquet`), country summaries (`173` mapped countries), 5-factor counterfactual & 96-feature TreeSHAP tables, $N$-gauge evaluation summaries, candidate gauge lists, and all paper figures (`Figure_1.png`–`Figure_5.png`, `Extended_Data_Figure_1.png`, `Extended_Data_Figure_3.png`–`Extended_Data_Figure_6.png`) and tables (`Table_1.csv`, `Extended_Data_Table_1.csv`–`Extended_Data_Table_3.csv`). |
| `hydro_model_metrics_1986_2023.tar.gz` | `hydro_model_metrics/` & `cross_validation_splits/` | Full-period (`1986–2023`) Dual-LSTM hydrological model evaluation metrics across `gauged` (`8,811` gauges), `random` 10-fold CV (`8,811` gauges), and `separated` 10-fold CV (`8,811` gauges) for all 38 target metrics and diagnostic variants (`204` CSV files), plus the 10-fold CV split definitions (`20` text files). |
| `anomaly_scores_and_coverage_fractions.tar.gz` | `anomaly_scores/` & `coverage_fractions/` | Precomputed Isolation Forest (`anomaly_score_if`), One-Class SVM (`anomaly_score_ocsvm`), and Local Outlier Factor (`anomaly_score_lof`) anomaly scores (`gauged`, `random`, `separated`, and `infer_hybas_caravan`) and topological upstream/downstream gauge coverage fractions for training gauges and all 9 continental HydroBASINS Level 12 partitions. |
| `attributes_and_exposure_inputs.tar.gz` | `attributes/`, `exposure/`, `risk_data/`, `naturalearth_lowres.geojson` | HydroATLAS static attributes, coordinates, record lengths, and routing topology (`NEXT_DOWN`) for GRDC, Caravan, and 964,084 global HydroBASINS Level 12 sub-basins; basin-level flood exposure by return period (`global_basin_exposure_by_return_period.csv`); INFORM risk/vulnerability indicators and World Bank classifications; and Natural Earth country boundaries. |
| `generic_training_and_inference_data.tar.gz` | `generic_training_data/` | Compiled feature matrices for `gauged` training (`8,811 × 92`), `ungauged` training (`17,622 × 96`, combining `8,811` random CV + `8,811` separated CV rows), and global HydroBASINS Level 12 inference (`1,034,083 × 96` total HydroBASINS records; `964,084` valid non-NaN global land sub-basins, provided in both CSV and Zarr v2 format). |
| `saved_predictions.tar.gz` | `saved_predictions/` | Out-of-fold 10-fold cross-validation predictions (`saved_predictions/cv_preds/`), global 964,084-basin HydroBASINS predictions (`saved_predictions/hybas_preds/`) for `gauged` and `ungauged` regimes across all 38 metrics (`76` CSVs each), `hybas_to_country_mapping.csv`, and stratified SHAP score matrices (`saved_predictions/shap/`). |
| `saved_models_return_period.tar.gz` | `saved_models/` | Trained full-archive `SkillPredictor` Random Forest meta-models (`.pkl`, loaded via `utils.skill_predictor.load_skill_predictor`) for all 30 return-period flood precision and recall metrics (`60` model files: `gauged` and `ungauged` across $T \in \{1.01, 2.0, 5.0, 10.0, 20.0\}\text{ yr}$ and windows $W \in \{0.0, 1.0, 2.0\}\text{ days}$). |
| `saved_models_classic.tar.gz` | `saved_models/` | Trained full-archive `SkillPredictor` Random Forest meta-models (`.pkl`) for all 8 continuous hydrograph metrics (`16` model files: `gauged` and `ungauged` for `NSE`, `KGE`, `log-NSE`, `log-KGE`, `Beta-NSE`, `Beta-KGE`, `Alpha-NSE`, `pearson-r`). |

In addition, the Zenodo record provides standalone copies of the primary summary tables at the top level (`basin_exposure_and_blockers_v2.parquet`, `country_exposure_and_blockers_v2.csv`, `country_joint_n_gauge_exposure_reduction_v2.csv`, `n_gauge_sets_evaluation_summary_v2.csv`, `five_factor_exposure_interventions_v2.json`, `five_factor_exposure_shap_features_v2.csv`) so researchers can inspect and analyze the results immediately without extracting any `.tar.gz` archives.

---

## 2. Schema of Core Datasets

### 2.1 Master Global Basin Table (`v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet`)
Contains **`964,084` rows** and **`16` columns**, indexed by string basin ID (`hybas_1120000010`, etc., corresponding to HydroBASINS Level 12 sub-basins).

| Column | Type | Global Sum / Summary | Description |
| :--- | :--- | :--- | :--- |
| Index (`hybas_<id>`) | `str` | `964,084` unique sub-basins | HydroBASINS Level 12 identifier prefixed with `hybas_`. |
| `country_name` | `str` | `173` mapped countries + `Unmapped` | Sovereign country or territory assigned via spatial join (`Unmapped` for `27,064` coastal/island sub-basins outside polygon bounds). |
| `has_system` | `bool` | `True` (`60` countries) / `False` | `True` if the basin's country has operational national flood warning capacity (`Tier 1` or `Tier 3` in `utils/cap_utils.py`); `False` otherwise (`113` countries + `Unmapped`). |
| `E_total` | `float64` | `221,567,096.30` people/yr | Expected Annual Riverine Flood Exposure integrated across $T \in \{5, 10, 20, 50, 100\}\text{ yr}$ return periods from 30 m defended Fathom v3 fluvial hazard maps $\times$ 100 m WorldPop population counts. |
| `warned_ungauged` | `float64` | `64,037,728.86` people/yr | Baseline warned flood exposure under `M_ungauged` ($\alpha = 1.0$, `28.90%` of `E_total`). |
| `unwarned_ungauged` | `float64` | `157,529,367.44` people/yr | Baseline unwarned flood exposure under `M_ungauged` ($\alpha = 1.0$, `71.10%` of `E_total`). |
| `recall_5y_u` | `float64` | Median `0.2757` | Predicted `M_ungauged` 5-year flood event recall (`5.0-year-2.0-window-recall`, $\in [0, 1]$). |
| `recall_10y_u` | `float64` | Median `0.2635` | Predicted `M_ungauged` 10-year flood event recall (`10.0-year-2.0-window-recall`, $\in [0, 1]$). |
| `recall_20y_u` | `float64` | Median `0.2074` | Predicted `M_ungauged` 20-year flood event recall (`20.0-year-2.0-window-recall`, $\in [0, 1]$). |
| `cf_factor3_ood` | `float64` | `+16,277,248.76` people/yr | Counterfactual reduction in unwarned exposure from eliminating out-of-distribution representation bias (`Factor 3`, setting `anomaly_score_if`, `anomaly_score_ocsvm`, `anomaly_score_lof` to their 10th-percentile in-distribution reference values). |
| `cf_factor2_gauge_cov` | `float64` | `+5,535,831.13` people/yr | Counterfactual reduction in unwarned exposure from universal `100%` river-network gauge coverage (`Factor 2`, setting `total_upstream_coverage`, `max_upstream_coverage`, `max_downstream_coverage` to `1.0` and `upstream_gauge_count >= 1`). |
| `cf_factor2_dense_corridor` | `float64` | `+5,437,981.89` people/yr | Counterfactual reduction in unwarned exposure under multi-gauge nested corridor coverage (`75th` percentile upstream gauge count). |
| `cf_factor1a_climate` | `float64` | `+4,750,213.73` people/yr | Counterfactual benchmark reduction in unwarned exposure when shifting arid/flashy climate features (`Factor 1a`) to humid well-observed reference values. |
| `cf_factor1b_regulation` | `float64` | `+1,178,106.88` people/yr | Counterfactual benchmark reduction in unwarned exposure when removing anthropogenic reservoir and irrigation regulation (`Factor 1b`). |
| `cf_factor1c_landcover` | `float64` | `+2,883,044.24` people/yr | Counterfactual benchmark reduction in unwarned exposure when shifting floodplain, wetland, and lake storage features (`Factor 1c`) to low-attenuation reference values. |
| `cf_obs_ceiling` | `float64` | `+20,355,848.14` people/yr | Combined Observational Ceiling (`Factors 2 + 3` applied jointly in `M_ungauged`, `12.92%` of global unwarned exposure). |
| `delta_mgauged_confirm_rp_clipped` | `float64` | `+4,019,704.48` people/yr | Independent confirmation check: incremental exposure warned when evaluating the separate in-sample `M_gauged` meta-model relative to baseline `M_ungauged` (`2.55%` of global unwarned exposure). |

---

### 2.2 Country-Level & Summary Tables (`v2_exposure_blockers/statistics/`)

- **`country_exposure_and_blockers_v2.csv`** (`173` rows $\times$ `22` columns):
  Country-level aggregation across all `173` mapped countries (`160` with `E_total > 0`):
  `['country_name', 'n_basins', 'has_system', 'cap_tier', 'E_total', 'warned_ungauged', 'unwarned_ungauged', 'median_recall_5y', 'median_recall_10y', 'median_recall_20y', 'cf_factor3_ood', 'cf_factor2_gauge_cov', 'cf_factor2_dense_corridor', 'cf_factor1a_climate', 'cf_factor1b_regulation', 'cf_factor1c_landcover', 'cf_factor1d_topography', 'cf_factor1e_soil', 'cf_factor4_reclen', 'cf_obs_ceiling', 'delta_mgauged_confirm', 'pct_unwarned']`.
- **`n_gauge_sets_evaluation_summary_v2.csv`** (`12` rows $\times$ `8` columns):
  Global joint $N$-gauge optimization results across `Bucket 1 (Representativeness)`, `Bucket 2A (Global Exposure)`, and `Bucket 2B (Exposure-Weighted OOD)` for $N \in \{10, 50, 100, 1000\}$:
  `['Bucket', 'N', 'Global_Risk_Reduced', 'Global_Risk_Reduced_With_OOD_Transfer', 'Global_Risk_Reduced_MGauged_Confirm', 'N_River_Network_Basins', 'N_Global_OOD_Neighbor_Basins', 'Elapsed_Seconds']`.
- **`country_joint_n_gauge_exposure_reduction_v2.csv`** (`160` rows $\times$ `17` columns) & **`country_gauge_placement_exposure_reduction.csv`** (`160` rows $\times$ `19` columns):
  Per-country joint exposure reduction achieved across all `160` flood-exposed countries by adding $N \in \{1, 5, 10, 20\}$ optimal domestic gauges:
  `['Country', 'Baseline Exposure (people/yr)', 'Unwarned Exposure (people/yr)', 'Max_Evaluated_Candidates', 'N=1 Exposure Reduced (people/yr)', 'N=1 Exposure Reduced (%)', 'N=1 Unwarned Reduced (%)', 'N=5 Exposure Reduced (people/yr)', 'N=5 Exposure Reduced (%)', 'N=5 Unwarned Reduced (%)', 'N=10 Exposure Reduced (people/yr)', 'N=10 Exposure Reduced (%)', 'N=10 Unwarned Reduced (%)', 'N=20 Exposure Reduced (people/yr)', 'N=20 Exposure Reduced (%)', 'N=20 Unwarned Reduced (%)', 'N=20 Exposure Reduced MGauged Confirm (people/yr)', 'CAP Warning System Tier', 'Operational System Status']`.
- **`five_factor_exposure_shap_features_v2.csv`** (`100` rows $\times$ `9` columns):
  Feature-level exposure-weighted TreeSHAP attribution (`['feature', 'factor', 'subcategory', 'remedy', 'abs_exposure_people_yr', 'deficit_exposure_people_yr', 'deficit_with_sys_people_yr', 'deficit_no_sys_people_yr', 'net_exposure_people_yr']`).
- **`five_factor_exposure_interventions_v2.json`**:
  JSON dictionary containing `cv_r2_ungauged`, `cv_r2_gauged_confirm`, `baseline_summary` (global, mapped, `60` countries with warning capacity, and `113` countries without effective systems), and `interventions` (full counterfactual exposure reductions across Factors 1a–1e, 2, 3, 4, Combined Observational Ceiling, Full Catchment Ceiling, and `M_gauged` confirmation check).
- **`Table_1.csv`** (`20` rows $\times$ `7` columns):
  Formatted main-text Table 1 (`['Group', 'Exposure Metric or Intervention Lever', 'Global Total (people/yr)', 'With Warning Capacity (60 Countries)', 'Without Effective System (113 Countries)', '% of Global Unwarned', 'Primary Engineering or Policy Lever']`).
- **`Extended_Data_Table_1.csv`** (`96` rows $\times$ `3` columns):
  Complete inventory of the `96` catchment and network predictors (`['Variable Symbol', 'Description', 'Subcategory']`).
- **`Extended_Data_Table_2.csv`** (`38` rows $\times$ `9` columns) & **`main_text_table_1_summary_statistics.csv`** (`38` rows $\times$ `29` columns):
  Cross-validation performance ($R^2$), empirical medians/IQRs, and global extrapolation medians across all `38` target metrics (`['Metric ID', 'Metric', 'Sample Size N (Ungauged CV)', 'Empirical Ungauged Median (IQR)', 'Skill Predictor CV R2 (M_ungauged)', 'Global Extrapolation Median (IQR)', 'Global Median (With Warning System)', 'Global Median (Without Effective System)', 'Secondary Check: M_gauged Median (CV R2)']`).
- **`Extended_Data_Table_3.csv`** (`173` rows $\times$ `12` columns):
  Country-level exposure, warning system status, CAP tier, $N=20$ gauge gain, Factor 2 and Factor 3 ceilings, Combined Observational Ceiling, and INFORM Vulnerability Index (`['Country', 'Sub-Basins', 'Warning System Status', 'CAP Tier', 'Baseline Flood Exposure (people/yr)', 'Unwarned Flood Exposure (people/yr)', '% Unwarned', 'N=20 National Gauges Reduction (people/yr)', 'Factor 2 Universal Gauge Ceiling (people/yr)', 'Factor 3 OOD Representation Ceiling (people/yr)', 'Combined Observational Ceiling (people/yr)', 'INFORM Vulnerability Index']`).

---

### 2.3 Hydrological Model Performance Metrics (`hydro_model_metrics/{gauged,random,separated}/*.csv`)
Each CSV is indexed by `gauge_id` (`GRDC_<id>` or Caravan ID `camels_...`, `hysets_...`, `lamah_...`, etc.) with a single value column calculated over the full 38-year evaluation window (`1986–2023`). The `38` target metrics in `utils/file_paths.py::ALL_METRICS` are:
- **`8` continuous hydrograph metrics**:
  `NSE`, `KGE`, `log-NSE`, `log-KGE`, `Beta-NSE`, `Beta-KGE`, `Alpha-NSE`, `pearson-r`.
- **`30` flood return-period event metrics**:
  `{T}-year-{W}-window-{precision|recall}` for return periods $T \in \{1.01, 2.0, 5.0, 10.0, 20.0\}\text{ yr}$ and temporal matching windows $W \in \{0.0, 1.0, 2.0\}\text{ days}$.

---

### 2.4 Generic Feature Matrices (`generic_training_data/`)
Each table is indexed by basin ID (`gauge_id` for `gauged_generic_training_data.csv` [`8,811 × 92`] and `ungauged_generic_training_data.csv` [`17,622 × 96`]; `hybas_id` for `hybas_generic_inference_data.csv` and `.zarr` [`964,084` valid global land sub-basins $\times$ `96` features]) and contains:
- **`88` HydroATLAS static physical and anthropogenic catchment attributes**: Topography & Drainage Scale (`calculated_drain_area`, `ele_mt_uav`, `slp_dg_uav`, `riv_tc_usu`, `sgr_dk_rav`), Climate & Aridity (`pre_mm_uyr`, `pet_mm_uyr`, `aet_mm_uyr`, `ari_ix_uav`, `cmi_ix_uyr`, `tmp_dc_uyr`, `snw_pc_uyr`, `swc_pc_uyr`), Soil & Hydrogeology (`cly_pc_uav`, `slt_pc_uav`, `snd_pc_uav`, `soc_th_uav`, `ero_kh_uav`, `kar_pc_use`), Land Cover, Floodplains, Wetlands & Lakes (`for_pc_use`, `crp_pc_use`, `pst_pc_use`, `gla_pc_use`, `prm_pc_use`, `pac_pc_use`, `lka_pc_use`, `lkv_mc_usu`, `inu_pc_umn`, `inu_pc_umx`, `inu_pc_ult`, `glc_pc_u01`–`u22`, `pnv_pc_u01`–`u15`, `wet_pc_u01`–`u09`, `wet_pc_ug1`–`ug2`), and Anthropogenic Regulation & Infrastructure (`pop_ct_usu`, `ppd_pk_uav`, `urb_pc_use`, `nli_ix_uav`, `rdd_mk_uav`, `hft_ix_u93`, `hft_ix_u09`, `gdp_ud_usu`, `ire_pc_use`, `rev_mc_usu`, `ria_ha_usu`, `dor_pc_pva`). All human-readable descriptions and category mappings are defined in `utils/feature_utils.py` and `Extended_Data_Table_1.csv`.
- **`3` Multivariate Out-of-Distribution (OOD) Anomaly Scores (`Factor 3`)**: `anomaly_score_if` (Isolation Forest), `anomaly_score_ocsvm` (One-Class SVM), `anomaly_score_lof` (Local Outlier Factor).
- **`4` Topological Gauge Coverage Features (`Factor 2`, included in `ungauged` and `hybas` `96`-feature matrices)**: `total_upstream_coverage` ($\in [0, 1]$), `max_upstream_coverage` ($\in [0, 1]$), `max_downstream_coverage` ($\in [0, 1]$), and `upstream_gauge_count` (integer count of maximal independent upstream gauges).
- **`1` Observation Record Length Feature (`Factor 4`)**: `record_length` (years of valid daily streamflow observations; set to `50` years during global counterfactual inference).

---

## 3. External Third-Party Data Sources

1. **Global Runoff Data Centre (GRDC)**: Historical daily river discharge observations (BfG, Koblenz, Germany; [https://www.bafg.de/GRDC](https://www.bafg.de/GRDC)).
2. **Caravan Global Community Dataset**: Standardized meteorological forcings, HydroATLAS attributes, and daily streamflow observations (Kratzert et al., *Scientific Data*, 2023; [https://doi.org/10.1038/s41597-023-01975-w](https://doi.org/10.1038/s41597-023-01975-w)).
3. **HydroBASINS & HydroATLAS v1.0 (Level 12)**: Global watershed boundaries, Pfafstetter river network topology (`NEXT_DOWN`), and hydro-environmental attributes across 964,084 Level 12 sub-basins (Lehner & Grill, *Hydrological Processes*, 2013; Linke et al., *Scientific Data*, 2019; [https://www.hydrosheds.org](https://www.hydrosheds.org)).
4. **WorldPop Global High Resolution Population Denominators (2020)**: 100 m population counts ([https://www.worldpop.org](https://www.worldpop.org)).
5. **Fathom Global Flood Hazard v3 (2020 Fluvial Defended)**: 1-arcsecond (~30 m) defended fluvial flood depth grids for return periods $T \in \{5, 10, 20, 50, 100\}\text{ yr}$ ([https://www.fathom.global](https://www.fathom.global)). Raw commercial Fathom v3 rasters are subject to Fathom licensing; derived basin-level flood exposure totals (`global_basin_exposure_by_return_period.csv`) are provided in `attributes_and_exposure_inputs.tar.gz`.
6. **WMO Common Alerting Protocol (CAP) Alert Hub & Country Warning Inventories**: Multi-year operational CAP alert logs and domestic flood early warning system classifications (`utils/cap_utils.py`).
7. **INFORM Risk Index & World Bank Country Classifications**: National socioeconomic vulnerability, lack of coping capacity, and income group classifications (`risk_data/`).
