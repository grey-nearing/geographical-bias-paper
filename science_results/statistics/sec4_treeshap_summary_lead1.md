# Section 4: TreeSHAP Feature Attribution Summary at Lead Time 1 ($t_0 + 1$ Day)

Computed on $N = 1,000$ stratified representative global watersheds across all 12 core Lead-Time-1 `M_base` (`ungauged`) `SkillPredictor` meta-models (`NSE`, `KGE`, and `1.01, 2.0, 5.0, 10.0, 20.0`-year `2d` flood recall and precision).

## 1. Category-Level Relative Importance (% of Total Category |SHAP|)

| Feature Category | Mean Across All 12 Metrics (%) | Classic (`NSE`, `KGE`) Mean (%) | Return-Period (`Recall` & `Precision`) Mean (%) | `NSE` (%) | `KGE` (%) | `5y Recall (2d)` (%) | `5y Precision (2d)` (%) | `20y Recall (2d)` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climate & Hydrology** | **33.49%** | 41.69% | 31.85% | 37.03% | 46.36% | 27.26% | 30.00% | 20.31% |
| **Land Cover** | **22.25%** | 12.11% | 24.28% | 13.97% | 10.26% | 24.21% | 23.70% | 29.87% |
| **Anomaly Score** | **13.00%** | 4.55% | 14.69% | 4.46% | 4.63% | 19.52% | 18.86% | 16.07% |
| **Topography** | **9.92%** | 10.40% | 9.82% | 10.78% | 10.01% | 10.19% | 9.08% | 8.18% |
| **Soil and Geology** | **8.70%** | 12.19% | 8.00% | 13.96% | 10.41% | 7.90% | 7.08% | 12.66% |
| **Anthropogenic Factors** | **7.57%** | 5.54% | 7.98% | 5.96% | 5.12% | 7.89% | 7.99% | 10.17% |
| **Coverage/Overlap** | **4.35%** | 9.16% | 3.39% | 9.27% | 9.05% | 3.04% | 3.28% | 2.73% |
| **Record Length** | **0.73%** | 4.37% | 0.00% | 4.58% | 4.16% | 0.00% | 0.00% | 0.00% |

## 2. Category-Level Raw Mean |SHAP| Across All 12 Metrics

| Feature Category | `NSE` | `KGE` | `Rec 1y` | `Rec 2y` | `Rec 5y` | `Rec 10y` | `Rec 20y` | `Prec 1y` | `Prec 2y` | `Prec 5y` | `Prec 10y` | `Prec 20y` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Climate & Hydrology** | 0.1535 | 0.2050 | 0.0457 | 0.0348 | 0.0337 | 0.0325 | 0.0297 | 0.0929 | 0.0425 | 0.0375 | 0.0344 | 0.0303 |
| **Land Cover** | 0.0579 | 0.0454 | 0.0249 | 0.0257 | 0.0300 | 0.0306 | 0.0437 | 0.0282 | 0.0281 | 0.0296 | 0.0309 | 0.0390 |
| **Anomaly Score** | 0.0185 | 0.0205 | 0.0033 | 0.0169 | 0.0242 | 0.0224 | 0.0235 | 0.0069 | 0.0224 | 0.0235 | 0.0185 | 0.0247 |
| **Topography** | 0.0447 | 0.0443 | 0.0192 | 0.0149 | 0.0126 | 0.0120 | 0.0120 | 0.0118 | 0.0129 | 0.0113 | 0.0089 | 0.0077 |
| **Soil and Geology** | 0.0579 | 0.0460 | 0.0128 | 0.0068 | 0.0098 | 0.0119 | 0.0185 | 0.0067 | 0.0059 | 0.0088 | 0.0100 | 0.0110 |
| **Anthropogenic Factors** | 0.0247 | 0.0226 | 0.0064 | 0.0065 | 0.0098 | 0.0141 | 0.0149 | 0.0096 | 0.0079 | 0.0100 | 0.0105 | 0.0133 |
| **Coverage/Overlap** | 0.0384 | 0.0400 | 0.0021 | 0.0030 | 0.0038 | 0.0066 | 0.0040 | 0.0059 | 0.0052 | 0.0041 | 0.0053 | 0.0037 |
| **Record Length** | 0.0190 | 0.0184 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 3. Top-8 Features Per Metric (Raw Mean |SHAP| & Category, Matching `Fig. 5`)

### `NSE`
1. **`ari_ix_uav`** (Aridity index, *Climate & Hydrology*): Mean |SHAP| = **0.0498**
2. **`swc_pc_uyr`** (Soil water content (%), *Climate & Hydrology*): Mean |SHAP| = **0.0428**
3. **`soc_th_uav`** (Soil organic carbon (t/ha), *Soil and Geology*): Mean |SHAP| = **0.0352**
4. **`max_downstream_coverage`** (Max downstream gauge cov., *Coverage/Overlap*): Mean |SHAP| = **0.0249**
5. **`pre_mm_uyr`** (Mean annual precipitation, *Climate & Hydrology*): Mean |SHAP| = **0.0246**
6. **`slp_dg_uav`** (Mean terrain slope (deg), *Topography*): Mean |SHAP| = **0.0234**
7. **`ele_mt_uav`** (Mean elevation (m), *Topography*): Mean |SHAP| = **0.0226**
8. **`slt_pc_uav`** (Soil silt fraction (%), *Soil and Geology*): Mean |SHAP| = **0.0216**

### `KGE`
1. **`ari_ix_uav`** (Aridity index, *Climate & Hydrology*): Mean |SHAP| = **0.0877**
2. **`cmi_ix_uyr`** (Climate moisture index, *Climate & Hydrology*): Mean |SHAP| = **0.0439**
3. **`swc_pc_uyr`** (Soil water content (%), *Climate & Hydrology*): Mean |SHAP| = **0.0419**
4. **`soc_th_uav`** (Soil organic carbon (t/ha), *Soil and Geology*): Mean |SHAP| = **0.0339**
5. **`riv_tc_usu`** (Total river channel volume, *Topography*): Mean |SHAP| = **0.0208**
6. **`calculated_drain_area`** (Drainage area, *Topography*): Mean |SHAP| = **0.0206**
7. **`pre_mm_uyr`** (Mean annual precipitation, *Climate & Hydrology*): Mean |SHAP| = **0.0199**
8. **`record_length`** (Gauge record length (yr), *Record Length*): Mean |SHAP| = **0.0184**

### `5.0-year-2.0-window-recall`
1. **`tmp_dc_uyr`** (Mean annual air temperature, *Climate & Hydrology*): Mean |SHAP| = **0.0171**
2. **`anomaly_score_ocsvm`** (Climate OOD (One-Class SVM), *Anomaly Score*): Mean |SHAP| = **0.0154**
3. **`anomaly_score_if`** (Climate OOD (Isolation Forest), *Anomaly Score*): Mean |SHAP| = **0.0136**
4. **`aet_mm_uyr`** (Actual evapotranspiration, *Climate & Hydrology*): Mean |SHAP| = **0.0128**
5. **`slp_dg_uav`** (Mean terrain slope (deg), *Topography*): Mean |SHAP| = **0.0108**
6. **`ele_mt_uav`** (Mean elevation (m), *Topography*): Mean |SHAP| = **0.0078**
7. **`pre_mm_uyr`** (Mean annual precipitation, *Climate & Hydrology*): Mean |SHAP| = **0.0076**
8. **`pnv_pc_u01`** (PNV: Tropical/Subtropical Moist Broadleaf Forests…, *Land Cover*): Mean |SHAP| = **0.0066**

### `10.0-year-2.0-window-recall`
1. **`anomaly_score_if`** (Climate OOD (Isolation Forest), *Anomaly Score*): Mean |SHAP| = **0.0185**
2. **`pre_mm_uyr`** (Mean annual precipitation, *Climate & Hydrology*): Mean |SHAP| = **0.0131**
3. **`tmp_dc_uyr`** (Mean annual air temperature, *Climate & Hydrology*): Mean |SHAP| = **0.0099**
4. **`aet_mm_uyr`** (Actual evapotranspiration, *Climate & Hydrology*): Mean |SHAP| = **0.0093**
5. **`anomaly_score_ocsvm`** (Climate OOD (One-Class SVM), *Anomaly Score*): Mean |SHAP| = **0.0074**
6. **`slp_dg_uav`** (Mean terrain slope (deg), *Topography*): Mean |SHAP| = **0.0071**
7. **`ele_mt_uav`** (Mean elevation (m), *Topography*): Mean |SHAP| = **0.0065**
8. **`for_pc_use`** (Forest cover extent (%), *Land Cover*): Mean |SHAP| = **0.0055**

### `20.0-year-2.0-window-recall`
1. **`anomaly_score_ocsvm`** (Climate OOD (One-Class SVM), *Anomaly Score*): Mean |SHAP| = **0.0182**
2. **`tmp_dc_uyr`** (Mean annual air temperature, *Climate & Hydrology*): Mean |SHAP| = **0.0093**
3. **`slp_dg_uav`** (Mean terrain slope (deg), *Topography*): Mean |SHAP| = **0.0082**
4. **`anomaly_score_if`** (Climate OOD (Isolation Forest), *Anomaly Score*): Mean |SHAP| = **0.0080**
5. **`cly_pc_uav`** (Soil clay fraction (%), *Soil and Geology*): Mean |SHAP| = **0.0079**
6. **`pet_mm_uyr`** (Potential evapotranspiration, *Climate & Hydrology*): Mean |SHAP| = **0.0077**
7. **`aet_mm_uyr`** (Actual evapotranspiration, *Climate & Hydrology*): Mean |SHAP| = **0.0077**
8. **`ele_mt_uav`** (Mean elevation (m), *Topography*): Mean |SHAP| = **0.0073**

### `5.0-year-2.0-window-precision`
1. **`anomaly_score_ocsvm`** (Climate OOD (One-Class SVM), *Anomaly Score*): Mean |SHAP| = **0.0161**
2. **`tmp_dc_uyr`** (Mean annual air temperature, *Climate & Hydrology*): Mean |SHAP| = **0.0153**
3. **`aet_mm_uyr`** (Actual evapotranspiration, *Climate & Hydrology*): Mean |SHAP| = **0.0153**
4. **`anomaly_score_if`** (Climate OOD (Isolation Forest), *Anomaly Score*): Mean |SHAP| = **0.0102**
5. **`pre_mm_uyr`** (Mean annual precipitation, *Climate & Hydrology*): Mean |SHAP| = **0.0097**
6. **`slp_dg_uav`** (Mean terrain slope (deg), *Topography*): Mean |SHAP| = **0.0092**
7. **`for_pc_use`** (Forest cover extent (%), *Land Cover*): Mean |SHAP| = **0.0082**
8. **`soc_th_uav`** (Soil organic carbon (t/ha), *Soil and Geology*): Mean |SHAP| = **0.0078**

## 4. Top-35 Features Across All 12 Metrics (Sorted by Mean Normalized Importance, Matching `Fig. S6`)

| Rank | Feature Code | Description | Category | Mean Norm. |SHAP| | `NSE` | `KGE` | `5y Recall (2d)` | `10y Recall (2d)` | `20y Recall (2d)` | `5y Precision (2d)` |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `ari_ix_uav` | Aridity index | Climate & Hydrology | **0.268** | 0.0498 | 0.0877 | 0.0012 | 0.0030 | 0.0027 | 0.0010 |
| 2 | `cmi_ix_uyr` | Climate moisture index | Climate & Hydrology | **0.208** | 0.0208 | 0.0439 | 0.0018 | 0.0019 | 0.0029 | 0.0029 |
| 3 | `swc_pc_uyr` | Soil water content (%) | Climate & Hydrology | **0.333** | 0.0428 | 0.0419 | 0.0018 | 0.0026 | 0.0028 | 0.0020 |
| 4 | `soc_th_uav` | Soil organic carbon (t/ha) | Soil and Geology | **0.254** | 0.0352 | 0.0339 | 0.0035 | 0.0043 | 0.0054 | 0.0078 |
| 5 | `pre_mm_uyr` | Mean annual precipitation | Climate & Hydrology | **0.612** | 0.0246 | 0.0199 | 0.0076 | 0.0131 | 0.0068 | 0.0097 |
| 6 | `max_downstream_coverage` | Max downstream gauge cov. | Coverage/Overlap | **0.151** | 0.0249 | 0.0171 | 0.0016 | 0.0033 | 0.0023 | 0.0021 |
| 7 | `slp_dg_uav` | Mean terrain slope (deg) | Topography | **0.366** | 0.0234 | 0.0172 | 0.0108 | 0.0071 | 0.0082 | 0.0092 |
| 8 | `ele_mt_uav` | Mean elevation (m) | Topography | **0.383** | 0.0226 | 0.0147 | 0.0078 | 0.0065 | 0.0073 | 0.0061 |
| 9 | `slt_pc_uav` | Soil silt fraction (%) | Soil and Geology | **0.246** | 0.0216 | 0.0123 | 0.0025 | 0.0043 | 0.0063 | 0.0018 |
| 10 | `riv_tc_usu` | Total river channel volume | Topography | **0.162** | 0.0215 | 0.0208 | 0.0021 | 0.0038 | 0.0037 | 0.0016 |
| 11 | `calculated_drain_area` | Drainage area | Topography | **0.274** | 0.0155 | 0.0206 | 0.0036 | 0.0043 | 0.0038 | 0.0034 |
| 12 | `aet_mm_uyr` | Actual evapotranspiration | Climate & Hydrology | **0.597** | 0.0165 | 0.0111 | 0.0128 | 0.0093 | 0.0077 | 0.0153 |
| 13 | `pet_mm_uyr` | Potential evapotranspiration | Climate & Hydrology | **0.391** | 0.0190 | 0.0165 | 0.0046 | 0.0052 | 0.0077 | 0.0068 |
| 14 | `record_length` | Gauge record length (yr) | Record Length | **0.049** | 0.0190 | 0.0184 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 15 | `anomaly_score_if` | Climate OOD (Isolation Forest) | Anomaly Score | **0.452** | 0.0045 | 0.0038 | 0.0136 | 0.0185 | 0.0080 | 0.0102 |
| 16 | `anomaly_score_ocsvm` | Climate OOD (One-Class SVM) | Anomaly Score | **0.579** | 0.0129 | 0.0153 | 0.0154 | 0.0074 | 0.0182 | 0.0161 |
| 17 | `total_upstream_coverage` | Upstream gauge coverage (%) | Coverage/Overlap | **0.093** | 0.0171 | 0.0174 | 0.0012 | 0.0019 | 0.0010 | 0.0010 |
| 18 | `tmp_dc_uyr` | Mean annual air temperature | Climate & Hydrology | **0.575** | 0.0081 | 0.0085 | 0.0171 | 0.0099 | 0.0093 | 0.0153 |
| 19 | `anomaly_score_lof` | Climate OOD (Local Outlier Factor) | Anomaly Score | **0.165** | 0.0164 | 0.0109 | 0.0024 | 0.0045 | 0.0049 | 0.0025 |
| 20 | `snw_pc_uyr` | Snow cover extent (%) | Climate & Hydrology | **0.210** | 0.0164 | 0.0131 | 0.0033 | 0.0044 | 0.0027 | 0.0037 |
| 21 | `pnv_pc_u12` | PNV: Mediterranean Forests/Woodlands (%) | Land Cover | **0.120** | 0.0097 | 0.0111 | 0.0007 | 0.0021 | 0.0071 | 0.0007 |
| 22 | `pst_pc_use` | Pasture extent (%) | Land Cover | **0.154** | 0.0081 | 0.0103 | 0.0023 | 0.0039 | 0.0070 | 0.0034 |
| 23 | `snd_pc_uav` | Soil sand fraction (%) | Soil and Geology | **0.171** | 0.0100 | 0.0073 | 0.0025 | 0.0037 | 0.0054 | 0.0027 |
| 24 | `pnv_pc_u01` | PNV: Tropical/Subtropical Moist Broadleaf Forests (%) | Land Cover | **0.240** | 0.0043 | 0.0024 | 0.0066 | 0.0040 | 0.0060 | 0.0047 |
| 25 | `lkv_mc_usu` | Total lake volume | Land Cover | **0.240** | 0.0091 | 0.0048 | 0.0049 | 0.0053 | 0.0027 | 0.0043 |
| 26 | `lka_pc_use` | Lake area extent (%) | Land Cover | **0.172** | 0.0084 | 0.0044 | 0.0035 | 0.0017 | 0.0026 | 0.0024 |
| 27 | `for_pc_use` | Forest cover extent (%) | Land Cover | **0.271** | 0.0076 | 0.0038 | 0.0052 | 0.0055 | 0.0050 | 0.0082 |
| 28 | `wet_pc_u01` | Wetland: Lakes (percentage) | Land Cover | **0.119** | 0.0083 | 0.0050 | 0.0013 | 0.0011 | 0.0007 | 0.0018 |
| 29 | `cly_pc_uav` | Soil clay fraction (%) | Soil and Geology | **0.167** | 0.0058 | 0.0068 | 0.0035 | 0.0045 | 0.0079 | 0.0024 |
| 30 | `max_upstream_coverage` | Max single-gauge upstream cov. | Coverage/Overlap | **0.052** | 0.0052 | 0.0076 | 0.0005 | 0.0006 | 0.0011 | 0.0007 |
| 31 | `upstream_gauge_count` | Upstream gauge count | Coverage/Overlap | **0.075** | 0.0064 | 0.0072 | 0.0013 | 0.0021 | 0.0013 | 0.0008 |
| 32 | `pnv_pc_u10` | PNV: Montane Grasslands/Shrublands (%) | Land Cover | **0.114** | 0.0069 | 0.0063 | 0.0019 | 0.0017 | 0.0015 | 0.0029 |
| 33 | `wet_pc_ug1` | Wetlands Global Class: Lake/Reservoir (percentage) | Anthropogenic Factors | **0.144** | 0.0029 | 0.0029 | 0.0031 | 0.0026 | 0.0069 | 0.0015 |
| 34 | `ero_kh_uav` | Soil erosion rate (kg/ha/yr) | Soil and Geology | **0.145** | 0.0065 | 0.0046 | 0.0030 | 0.0037 | 0.0045 | 0.0019 |
| 35 | `hft_ix_u93` | Human footprint index (1993) | Anthropogenic Factors | **0.125** | 0.0065 | 0.0042 | 0.0021 | 0.0043 | 0.0031 | 0.0023 |
