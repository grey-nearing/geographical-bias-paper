# Data Dictionary & Zenodo Archive Reference (`10.5281/zenodo.21416987`)

This document describes the directory hierarchy, archive contents, file schemas, and column definitions for all datasets archived in **[https://doi.org/10.5281/zenodo.21416987](https://doi.org/10.5281/zenodo.21416987)** and used by the scripts in [`https://github.com/grey-nearing/geographical-bias-paper`](https://github.com/grey-nearing/geographical-bias-paper).

---

## 1. Zenodo Archive Inventory (`10.5281/zenodo.21416987`)

When extracted into `./data` (`GEO_BIAS_DATA_DIR`), the Zenodo archives populate the exact directory tree expected by `utils/file_paths.py`:

| Zenodo File | Extracted Path in `GEO_BIAS_DATA_DIR` | Description |
| :--- | :--- | :--- |
| `geographical-bias-paper-v1.0.0.tar.gz` | Standalone repository source release | Frozen `v1.0.0` source code, scripts, and pre-rendered figures/tables. |
| `v2_exposure_blockers.tar.gz` | `v2_exposure_blockers/` | Master 964,084-basin exposure & blocker table (`basin_exposure_and_blockers_v2.parquet`), country/regional/income/CAP-tier summaries, 5-factor counterfactual & 96-feature TreeSHAP tables, $N$-gauge evaluation summaries, candidate gauge lists, and all paper figures (`Figure_1.png`–`Figure_5.png`, `Extended_Data_Figure_1.png`, `Extended_Data_Figure_3.png`–`Extended_Data_Figure_6.png`) and tables (`Table_1.csv`, `Extended_Data_Table_1.csv`–`Extended_Data_Table_3.csv`). |
| `hydro_model_metrics_1986_2023.tar.gz` | `hydro_model_metrics/` & `cross_validation_splits/` | Full-period (`1986–2023`) Dual-LSTM hydrological model evaluation metrics across `gauged` (`8,354` gauges), `random` 10-fold CV (`5,049` GRDC gauges), and `separated` 10-fold CV (`5,049` GRDC gauges) for all 68 performance metrics (`204` CSV files), plus the 10-fold CV split definitions (`20` text files). |
| `anomaly_scores_and_coverage_fractions.tar.gz` | `anomaly_scores/` & `coverage_fractions/` | Precomputed Isolation Forest, One-Class SVM, and Local Outlier Factor anomaly scores (`gauged`, `random`, `separated`, and `infer_hybas_caravan`) and topological upstream/downstream gauge coverage fractions for training gauges and all 9 continental HydroBASINS Level 12 partitions. |
| `attributes_and_exposure_inputs.tar.gz` | `attributes/`, `exposure/`, `risk_data/`, `naturalearth_lowres.geojson` | HydroATLAS static attributes, coordinates, record lengths, and routing topology (`NEXT_DOWN`) for GRDC, Caravan, and 964,084 global HydroBASINS Level 12 catchments; basin-level flood exposure by return period (`global_basin_exposure_by_return_period.csv`); INFORM risk/vulnerability indicators and World Bank income classifications; and Natural Earth country boundaries. |
| `generic_training_and_inference_data.tar.gz` | `generic_training_data/` | Compiled 95-feature matrices for `gauged` training (`4,672 × 95`), `ungauged` training (`9,324 × 95`), and global HydroBASINS Level 12 inference (`964,084 × 95`, provided in both CSV and fast Zarr v2 format). |
| `saved_predictions.tar.gz` | `saved_predictions/` | Out-of-fold 10-fold cross-validation predictions (`saved_predictions/cv_preds/`), global 964,084-basin HydroBASINS predictions (`saved_predictions/hybas_preds/`) for `gauged` and `ungauged` regimes across all 38 metrics (`76` CSVs each), `hybas_to_country_mapping.csv`, and stratified KernelSHAP score matrices (`saved_predictions/shap/`). |
| `saved_models_return_period.tar.gz` | `saved_models/` | Trained full-archive `SkillPredictor` stacked ensembles (`.pkl`, loaded via `utils.skill_predictor.load_skill_predictor`) for all 30 return-period flood precision and recall metrics (`60` model files: `gauged` and `ungauged` across $T \in \{1.01, 2, 5, 10, 20\}\text{ yr}$ and windows $W \in \{0, 1, 2\}\text{ days}$). |
| `saved_models_classic.tar.gz` | `saved_models/` | Trained full-archive `SkillPredictor` stacked ensembles (`.pkl`) for all 8 continuous hydrograph metrics (`16` model files: `gauged` and `ungauged` for `NSE`, `KGE`, `alpha-NSE`, `beta-NSE`, `beta-KGE`, `FHV`, `FMS`, `FLV`). |

In addition, the Zenodo record provides standalone copies of the primary summary tables at the top level (`basin_exposure_and_blockers_v2.parquet`, `country_exposure_and_blockers_v2.csv`, `country_joint_n_gauge_exposure_reduction_v2.csv`, `n_gauge_sets_evaluation_summary_v2.csv`, `five_factor_exposure_interventions_v2.json`, `five_factor_exposure_shap_features_v2.csv`) so researchers can inspect and analyze the results immediately without extracting any `.tar.gz` archives.

---

## 2. Schema of Core Datasets

### 2.1 Master Global Basin Table (`v2_exposure_blockers/statistics/basin_exposure_and_blockers_v2.parquet`)
Contains **964,084 rows** (indexed by `hybas_id`, HydroBASINS Level 12) covering all ice-free global land areas with valid HydroATLAS attributes.

| Column | Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `hybas_id` (index) | `int64` | HydroBASINS Level 12 ID | Unique 10-digit Pfafstetter basin identifier. |
| `country_name` | `str` | Harmonized country name | Sovereign country or territory assigned via Natural Earth spatial join (`Unmapped` for small coastal/island catchments outside polygon bounds). |
| `region` | `str` | 7 World Bank regions | Geographic region (`East Asia & Pacific`, `Europe & Central Asia`, `Latin America & Caribbean`, `Middle East & North Africa`, `North America`, `South Asia`, `Sub-Saharan Africa`). |
| `income_group` | `str` | 4 World Bank income tiers | `High income`, `Upper middle income`, `Lower middle income`, or `Low income`. |
| `cap_tier` | `str` | 4-tier operational taxonomy | Operational flood warning classification (`Tier 1: Comprehensive CAP Alerting`, `Tier 2: Proportionality Gap (Under-Alerting)`, `Tier 3: Non-CAP Domestic Architecture`, `Tier 4: Complete Systemic Gap (No CAP)`). |
| `cap_binary` | `str` | Binary warning status | `With Effective Warning System` (Tier 1 + Tier 3, 65 countries) or `Without Effective Warning System` (Tier 2 + Tier 4, 99 countries). |
| `total_pop` | `float64` | People | Total basin population from WorldPop 2020 (1 km). |
| `exp_2yr`, `exp_5yr`, `exp_10yr`, `exp_20yr`, `exp_50yr`, `exp_100yr` | `float64` | People | Population residing inside the defended Fathom v3 fluvial inundation zone (depth $\ge 10\text{ cm}$) for return period $T \in \{2, 5, 10, 20, 50, 100\}\text{ yr}$. |
| `EAE_people_yr` | `float64` | People / year | Expected Annual Flood Exposure ($E_b$) integrated across return periods via trapezoidal rule over exceedance probability $p = 1/T$ (`21,168,266` people/yr globally). |
| `R_gauged_2yr`, `R_gauged_5yr`, `R_gauged_10yr`, `R_gauged_20yr` | `float64` | $[0, 1]$ | Predicted flood recall (2-day window) under the `gauged` model regime for $T \in \{2, 5, 10, 20\}\text{ yr}$. |
| `R_ungauged_2yr`, `R_ungauged_5yr`, `R_ungauged_10yr`, `R_ungauged_20yr` | `float64` | $[0, 1]$ | Predicted flood recall (2-day window) under the `ungauged` model regime for $T \in \{2, 5, 10, 20\}\text{ yr}$. |
| `R_gauged_bar` | `float64` | $[0, 1]$ | Exposure-weighted expected annual flood recall under the `gauged` regime ($R_{b,\text{gauged}}$). |
| `R_ungauged_bar` | `float64` | $[0, 1]$ | Exposure-weighted expected annual flood recall under the `ungauged` regime ($R_{b,\text{ungauged}}$). |
| `R_current_bar` | `float64` | $[0, 1]$ | Current baseline expected annual flood recall ($R_{b,\text{current}}$), equal to $R_{b,\text{gauged}}$ where existing gauge coverage $c_b = 1$ and interpolated via $c_b = \max(\text{total\_upstream\_coverage}_b, \text{max\_downstream\_coverage}_b)$ otherwise. |
| `R_perfect_network_bar` | `float64` | $[0, 1]$ | Counterfactual ungauged recall under complete global network topology and zero OOD anomaly ($R_{b,\text{ungauged}}^*$). |
| `E_covered_current` | `float64` | People / year | Currently covered flood exposure ($E_b \cdot R_{b,\text{current}}$; **8.39M people/yr** globally). |
| `E_covered_local_gauged` | `float64` | People / year | Flood exposure covered in local and nested-upstream gauged basins (**7.46M people/yr** globally). |
| `E_covered_regionalization` | `float64` | People / year | Additional flood exposure covered via spatial regionalization into ungauged basins (**0.93M people/yr** globally). |
| `E_uncovered_current` | `float64` | People / year | Total uncovered flood exposure ($E_b \cdot (1 - R_{b,\text{current}})$; **12.78M people/yr** globally). |
| `B_local_people_yr` | `float64` | People / year | **Blocker 1 (Local Gauge Gap)**: $E_b \cdot (R_{b,\text{gauged}} - R_{b,\text{current}})$ (**0.65M people/yr** globally). |
| `B_network_people_yr` | `float64` | People / year | **Blocker 2 (Global Gauge Network Gap)**: $E_b \cdot (R_{b,\text{ungauged}}^* - R_{b,\text{gauged}})$ (**1.23M people/yr** globally). |
| `B_model_people_yr` | `float64` | People / year | **Blocker 3 (Gauged Model Error)**: $E_b \cdot (1 - R_{b,\text{ungauged}}^*)$ (**10.90M people/yr** globally). |

---

### 2.2 Country-Level Summary Tables (`v2_exposure_blockers/statistics/`)
- **`country_exposure_and_blockers_v2.csv`** (`166` rows: 164 countries + `Unmapped` + `Global Total`):
  Aggregates `EAE_people_yr`, `E_covered_current`, `E_uncovered_current`, `B_local_people_yr`, `B_network_people_yr`, `B_model_people_yr`, fractional shares (`pct_covered_current`, `pct_B_local`, `pct_B_network`, `pct_B_model`), and exposure-weighted mean recall (`R_current_mean`, `R_gauged_mean`, `R_ungauged_mean`) by country alongside `region`, `income_group`, `cap_tier`, and `cap_binary`.
- **`country_joint_n_gauge_exposure_reduction_v2.csv`** (`164` countries):
  Reports the joint topological + OOD exposure reduction achieved in each country by adding $N \in \{1, 5, 10, 20\}$ optimal domestic gauges (`joint_gain_N1_people_yr`, `joint_gain_N5_people_yr`, `joint_gain_N10_people_yr`, `joint_gain_N20_people_yr` and corresponding percentages of national uncovered exposure `pct_uncovered_reduced_N1`..`N20`).
- **`n_gauge_sets_evaluation_summary_v2.csv`** (`19` rows):
  Global joint $N$-gauge optimization results across `exposure_global` ($N \in \{10, 25, 50, 100, 250, 500, 1000\}$), `exposure_ood` ($N \in \{10, 25, 50, 100, 250, 500, 1000\}$), `representativeness` ($N \in \{10, 25, 50, 100, 1000\}$), and `exposure_country` ($N=20\text{/country}$, up to 1,000 gauges globally), decomposing total people-years/yr gained into `local_and_routing_gain_people_yr` and `ood_network_gain_people_yr`.
- **`five_factor_exposure_interventions_v2.json` & `five_factor_exposure_shap_features_v2.csv`**:
  Complete five-factor counterfactual intervention chain (Local Gauge Presence, Record Length Saturation at 76 yr, Upstream/Downstream Nested Routing Coverage, Multivariate Hydroclimatic OOD Elimination, and Residual Gauged Model Error) and 96-feature exposure-weighted TreeSHAP attribution across all 964,084 global basins.

---

### 2.3 Hydrological Model Performance Metrics (`hydro_model_metrics/{gauged,random,separated}/*.csv`)
Each CSV is indexed by `gauge_id` (`GRDC_<id>` or Caravan ID `camels_...`, `hysets_...`, `lamah_...`, etc.) with a single value column `0` calculated over the full 38-year evaluation window (`1986–2023`):
- **Continuous hydrograph metrics (`8` used in SkillPredictor + diagnostic variants)**:
  `NSE`, `KGE`, `alpha-NSE`, `beta-NSE`, `beta-KGE`, `FHV` (top 2% peak flow bias), `FMS` (flow duration curve mid-segment slope bias), `FLV` (bottom 30% low-flow volume bias).
- **Flood return-period event metrics (`30` used in SkillPredictor)**:
  `{T}-year-{W}-window-{precision|recall}.csv` for return periods $T \in \{1.01, 2.0, 5.0, 10.0, 20.0\}\text{ yr}$ and temporal matching windows $W \in \{0.0, 1.0, 2.0\}\text{ days}$.

---

### 2.4 Generic Feature Matrices (`generic_training_data/`)
Each table is indexed by basin ID (`gauge_id` for `gauged_generic_training_data.csv` and `ungauged_generic_training_data.csv`; `hybas_id` for `hybas_generic_inference_data.csv` and `.zarr`) and contains:
- **87 HydroATLAS static catchment attributes**: Topography (`calculated_drain_area`, `ele_mt_uav`, `slp_dg_uav`, `riv_tc_usu`), Climate & Hydrology (`pre_mm_uyr`, `pet_mm_uyr`, `aet_mm_uyr`, `ari_ix_uav`, `cmi_ix_uyr`, `tmp_dc_uyr`, `snw_pc_uyr`, `swc_pc_uyr`), Soil & Geology (`cly_pc_uav`, `slt_pc_uav`, `snd_pc_uav`, `soc_th_uav`, `ero_kh_uav`, `kar_pc_use`), Land Cover (`for_pc_use`, `crp_pc_use`, `pst_pc_use`, `gla_pc_use`, `prm_pc_use`, `pac_pc_use`, `lka_pc_use`, `lkv_mc_usu`, `inu_pc_umn`, `inu_pc_umx`, `inu_pc_ult`, `glc_pc_u01`–`u22`, `pnv_pc_u01`–`u15`, `wet_pc_u01`–`u09`, `wet_pc_ug1`–`ug2`), and Anthropogenic Factors (`pop_ct_usu`, `ppd_pk_uav`, `urb_pc_use`, `nli_ix_uav`, `rdd_mk_uav`, `hft_ix_u93`, `hft_ix_u09`, `gdp_ud_usu`, `ire_pc_use`, `rev_mc_usu`, `ria_ha_usu`). All human-readable descriptions and category mappings are defined in `utils/feature_utils.py`.
- **3 Multivariate Anomaly Scores**: `anomaly_score_if` (Isolation Forest), `anomaly_score_ocsvm` (One-Class SVM), `anomaly_score_lof` (Local Outlier Factor).
- **4 Topological Gauge Coverage Features** (included in `ungauged` and `hybas` matrices): `total_upstream_coverage` ($\in [0, 1]$), `max_upstream_coverage` ($\in [0, 1]$), `max_downstream_coverage` ($\in [0, 1]$), and `upstream_gauge_count` (integer count of maximal independent upstream gauges).
- **1 Observation Record Length Feature**: `record_length` (number of valid daily streamflow observations; set to the training maximum of `27,759` days $\approx 76\text{ yr}$ during global inference).

---

## 3. External Third-Party Data Sources

1. **Global Runoff Data Centre (GRDC)**: Historical daily river discharge observations (BfG, Koblenz, Germany; [https://www.bafg.de/GRDC](https://www.bafg.de/GRDC)).
2. **Caravan Global Community Dataset**: Standardized meteorological forcings, HydroATLAS attributes, and daily streamflow observations (Kratzert et al., *Scientific Data*, 2023; [https://doi.org/10.1038/s41597-023-01975-w](https://doi.org/10.1038/s41597-023-01975-w)).
3. **HydroBASINS & HydroATLAS v1.0 (Level 12)**: Global watershed boundaries, Pfafstetter river network topology (`NEXT_DOWN`), and 281 hydro-environmental attributes across 964,084 Level 12 sub-basins (Lehner & Grill, *Hydrological Processes*, 2013; Linke et al., *Scientific Data*, 2019; [https://www.hydrosheds.org](https://www.hydrosheds.org)).
4. **WorldPop Global High Resolution Population Denominators (2020)**: 1 km UN-adjusted population counts (`ppp_2020_1km_Aggregated.tif`; [https://www.worldpop.org](https://www.worldpop.org)).
5. **Fathom Global Flood Hazard v3 (2020 Fluvial Defended)**: 1-arcsecond (~30 m) defended fluvial flood depth grids for return periods $T \in \{2, 5, 10, 20, 50, 100\}\text{ yr}$ ([https://www.fathom.global](https://www.fathom.global)). Raw commercial Fathom v3 rasters are subject to Fathom licensing; derived basin-level flood exposure totals (`global_basin_exposure_by_return_period.csv`) are provided in `attributes_and_exposure_inputs.tar.gz`.
6. **WMO Common Alerting Protocol (CAP) Alert Hub & Country Warning Inventories**: Multi-year operational CAP alert logs and domestic flood early warning system classifications (`utils/cap_utils.py`).
7. **INFORM Risk Index & World Bank Country Classifications**: National socioeconomic vulnerability, lack of coping capacity, and income group classifications (`risk_data/`).
