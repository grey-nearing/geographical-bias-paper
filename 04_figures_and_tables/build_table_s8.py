#!/usr/bin/env python3
"""Builds `Table_S8_72_gauge_repositories.csv` (all 72 physical-gauge repositories)."""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.environ.get('GEO_BIAS_REPO_DIR', str(SCRIPT_DIR.parent)))
CANON_DIR = Path(os.environ.get('GEO_BIAS_DATA_DIR', str(REPO_DIR / 'data')))

REPO_META = {
    'GSIM': ('GSIM', 'Do et al. (2018); Gudmundsson et al. (2018) (20, 21)', 'https://doi.org/10.1594/PANGAEA.887477'),
    'Caravan': ('Caravan (Core + Extensions)', 'Kratzert et al. (2023) (17)', 'https://doi.org/10.5281/zenodo.10006322'),
    'GSHA': ('GSHA v1.1', 'Yin et al. (2024) (22)', 'https://doi.org/10.5281/zenodo.10433905'),
    'GRDC': ('Global Runoff Data Centre (GRDC)', 'Hannah et al. (2011); WMO/BfG (13)', 'https://portal.grdc.bafg.de/'),
    'Caravan (GRDC Extension)': ('GRDC-Caravan Extension', 'Färber et al. (2025) (18)', 'https://doi.org/10.5281/zenodo.10006322'),
    'ROBIN_v1': ('ROBIN v1', 'Turner et al. (2025) (76)', 'https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9'),
    'Caravan (Regional CAMELS Extension)': ('Caravan Regional CAMELS Extensions', 'Kratzert et al. (2023) (17)', 'https://doi.org/10.5281/zenodo.15530021'),
    'EStreams': ('EStreams (Europe)', 'do Nascimento et al. (2024) (19)', 'https://doi.org/10.5281/zenodo.13255133'),
    'HYSETS': ('HYSETS (North America)', 'Arsenault et al. (2020) (77)', 'https://doi.org/10.17605/OSF.IO/RPC3W'),
    'R_ArcticNET_v4': ('R-ArcticNET v4.0', 'Lammers et al. (2001) (78)', 'https://www.r-arcticnet.sr.unh.edu/v4.0/'),
    'CIS_Watersheds': ('CIS Watersheds (Eurasia)', 'Abramov et al. (2023)', 'https://doi.org/10.5281/zenodo.8432070'),
    'ADHI': ('ADHI (Africa)', 'Tramblay et al. (2021) (79)', 'https://doi.org/10.23708/LXGXQ9'),
    'CA_discharge': ('CA-discharge (Central Asia)', 'Marti et al. (2023)', 'https://doi.org/10.5281/zenodo.8147591'),
    'Andean-GC': ('Andean-GC (Andes)', 'Aguayo et al. (2026)', 'https://doi.org/10.5281/zenodo.18035801'),
    'MRC': ('Mekong River Commission (MRC)', 'MRC Data & Information Services', 'https://portal.mrcmekong.org/'),
    'SSA-HCD': ('SSA-HCD (Sub-Saharan Africa)', 'Mo et al. (2026)', 'https://doi.org/10.5281/zenodo.15260974'),
    'North_Africa_98_Basins': ('North Africa Flood Database', 'Tramblay (2023)', 'https://doi.org/10.5281/zenodo.10277968'),
    'ArcticGRO': ('ArcticGRO (Pan-Arctic)', 'McClelland et al. (2023)', 'https://arcticgreatrivers.org/discharge/'),
    'CCAM': ('CCAM (China)', 'Hao et al. (2021) (81)', 'https://doi.org/10.5281/zenodo.5729444'),
    'CAMELS-BR': ('CAMELS-BR (Brazil)', 'Chagas et al. (2020) (82)', 'https://doi.org/10.5281/zenodo.15025488'),
    'GHI_v2': ('GHI v2 (Indian Subcontinent)', 'Goteti (2024)', 'https://doi.org/10.5281/zenodo.13852439'),
    'CNS_CAMELS_Extra': ('CAMELS-SPAT & Extra National Syntheses', 'Knoben et al. (2025) (80)', 'https://doi.org/10.20383/103.01306'),
    'CDR2': ('CDR² (China)', 'Yong et al. (2026)', 'https://doi.org/10.5281/zenodo.22231453'),
    'JPDH': ('JPDH (Japan)', 'MLIT / JPDH Benchmark (2025)', 'https://huggingface.co/datasets/prediction2/jpdh-dataset'),
    'CAMELS-PL': ('CAMELS-PL (Poland)', 'Brzezińska et al. (2026)', 'https://doi.org/10.5281/zenodo.23275336'),
    'CABra': ('CABra (Brazil)', 'Almagro et al. (2023)', 'https://doi.org/10.5281/zenodo.7612350'),
    'GHI_v1': ('GHI v1 (India)', 'Goteti (2023)', 'https://doi.org/10.5281/zenodo.7563600'),
    'FOCA_ISPRA_Italy': ('FOCA / ISPRA (Italy)', 'Claps et al. (2024)', 'https://doi.org/10.5281/zenodo.11217292'),
    'CAMELS-US': ('CAMELS-US (United States & Canada)', 'Addor et al. (2017) (80)', 'https://gdex.ucar.edu/dataset/camels.html'),
    'CAMELS-CL': ('CAMELS-CL (Chile)', 'Alvarez-Garreton et al. (2018)', 'https://doi.org/10.1594/PANGAEA.894885'),
    'CAMELS_IND': ('CAMELS-IND (India)', 'Mangukiya et al. (2025)', 'https://doi.org/10.5281/zenodo.14999580'),
    'CAMELS_NZ': ('CAMELS-NZ (New Zealand)', 'Bushra et al. (2025)', 'https://doi.org/10.26021/canterburynz.28827644'),
    'CAMELS-COL': ('CAMELS-COL (Colombia)', 'Jimenez et al. (2025)', 'https://doi.org/10.5281/zenodo.15554735'),
    'CAMELS-FI': ('CAMELS-FI (Finland)', 'Seppä et al. (2026)', 'https://doi.org/10.5281/zenodo.15853357'),
    'CAMELS-CH': ('CAMELS-CH (Switzerland)', 'Höge et al. (2023)', 'https://doi.org/10.5281/zenodo.7784633'),
    'CAMELS_KR': ('CAMELS-KR (South Korea)', 'Lee et al. (2026)', 'https://doi.org/10.5281/zenodo.23549490'),
    'CAMELSH_KR': ('CAMELSH-KR (South Korea)', 'Tran et al. (2025)', 'https://doi.org/10.5281/zenodo.15073264'),
    'CAMELS-PE': ('CAMELS-PE (Peru)', 'Llauca et al. (2026)', 'https://doi.org/10.5281/zenodo.21195425'),
    'PMET-obs': ('PMET-obs (Patagonia)', 'Aguayo et al. (2024)', 'https://doi.org/10.5281/zenodo.8374194'),
    'LamaH-Ice': ('LamaH-Ice (Iceland)', 'Helgason & Nijssen (2024)', 'https://doi.org/10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91'),
    'HydroMLYR': ('HydroMLYR (Yellow River, China)', 'Hao et al. (2021) (81)', 'https://doi.org/10.5281/zenodo.5729444'),
    'CAMELS-LUX': ('CAMELS-LUX (Luxembourg)', 'Nijzink et al. (2026)', 'https://doi.org/10.5281/zenodo.13846619'),
    'GHSA_v3': ('GHSA v3 (Pakistan)', 'Goteti (2025)', 'https://doi.org/10.5281/zenodo.15258773'),
    'SIREN_Italy': ('SIREN (Italy)', 'Mazzoglio et al. (2025)', 'https://doi.org/10.5281/zenodo.14996263'),
    'RivRetrieve': ('RivRetrieve Multi-Agency APIs (39 providers)', 'Riggs et al. (2026); Moulds et al. (2026) (23, 24)', 'https://github.com/RivRetrieve'),
    'USGS NWIS': ('USGS NWIS (United States)', 'U.S. Geological Survey (83)', 'https://waterdata.usgs.gov/nwis'),
    'BoM Water Data Online': ('BoM Water Data Online (Australia & Papua New Guinea)', 'Australian Bureau of Meteorology', 'http://www.bom.gov.au/waterdata/'),
    'ANA HidroWeb': ('ANA HidroWeb (Brazil)', 'Agência Nacional de Águas e Saneamento Básico', 'https://www.snirh.gov.br/hidroweb'),
    'NVE HydAPI': ('NVE HydAPI (Norway)', 'Norwegian Water Resources & Energy Directorate', 'https://hydapi.nve.no/'),
    'IDEAM_Colombia': ('IDEAM National Station Registry (Colombia)', 'IDEAM Datos Abiertos Colombia', 'https://www.datos.gov.co/resource/hp9r-jxuu.csv'),
    'UK Environment Agency': ('UK Environment Agency Hydrology API', 'Environment Agency (England)', 'https://environment.data.gov.uk/hydrology/landing'),
    'CWC India': ('Central Water Commission (CWC India)', 'Central Water Commission Flood Forecast Portal', 'https://ffs.india-water.gov.in/'),
    'IDEAM DHIME': ('IDEAM DHIME Portal (Colombia)', 'IDEAM Colombia', 'http://dhime.ideam.gov.co/'),
    'CONAGUA BANDAS': ('CONAGUA BANDAS (Mexico)', 'Comisión Nacional del Agua (Mexico)', 'https://www.gob.mx/conagua/acciones-y-programas/bandas'),
    'DWS South Africa': ('DWS Verified Hydrology (South Africa)', 'Dept. of Water and Sanitation (South Africa)', 'https://www.dws.gov.za/Hydrology/Verified/'),
    'CR2_Caudales_Chile': ('CR2 Explorador / Caudales (Chile)', 'Center for Climate and Resilience Research (CR2)', 'https://explorador.cr2.cl/'),
    'JPDH_MLIT_Network_Extra': ('MLIT WIS / JPDH Network Stations (Japan)', 'Ministry of Land, Infrastructure, Transport & Tourism', 'https://www1.river.go.jp/'),
    'DINAGUA Uruguay': ('DINAGUA (Uruguay)', 'Dirección Nacional de Aguas (Uruguay)', 'https://www.gub.uy/ministerio-ambiente'),
    'CHMI (ČHMÚ)': ('CHMI Open Data (Czechia)', 'Czech Hydrometeorological Institute', 'https://opendata.chmi.cz/'),
    'BWDB FFWC': ('BWDB Flood Forecasting & Warning Centre', 'Bangladesh Water Development Board', 'https://ffwc.gov.bd/'),
    'MAWLR Namibia': ('MAWLR Hydrological Services (Namibia)', 'Ministry of Agriculture, Water & Land Reform', 'https://mawlr.gov.na/'),
    'DWR Thailand': ('Department of Water Resources (Thailand)', 'Dept. of Water Resources Thailand', 'https://www.dwr.go.th/'),
    'DGA Chile': ('DGA SNIA (Chile)', 'Dirección General de Aguas (Chile)', 'https://snia.mop.gob.cl/'),
    'PUPR Indonesia': ('PUPR Water Resources (Indonesia)', 'Ministry of Public Works & Housing (Indonesia)', 'https://sda.pu.go.id/'),
    'ICE Costa Rica': ('ICE Hydrometeorology (Costa Rica)', 'Instituto Costarricense de Electricidad', 'https://www.grupoice.com/'),
    'KRC / WRA Kenya': ('Water Resources Authority / KRC (Kenya)', 'Water Resources Authority (Kenya)', 'https://wra.go.ke/'),
    'DNAAS Mozambique': ('DNGRH / DNAAS (Mozambique)', 'Direcção Nacional de Gestão de Recursos Hídricos', 'https://www.dngrh.gov.mz/'),
    'PID Pakistan': ('Punjab Irrigation Department (Pakistan)', 'Punjab Irrigation Department', 'https://irrigation.punjab.gov.pk/'),
    'NIHSA Nigeria': ('NIHSA (Nigeria)', 'Nigeria Hydrological Services Agency', 'https://nihsa.gov.ng/'),
    'MOI Sri Lanka': ('Irrigation Department (Sri Lanka)', 'Department of Irrigation (Sri Lanka)', 'https://www.irrigation.gov.lk/'),
    'VNMHA Vietnam': ('VNMHA / NCHMF (Vietnam)', 'Vietnam Meteorological & Hydrological Admin.', 'https://nchmf.gov.vn/'),
    'SENAMHI Peru': ('SENAMHI (Peru)', 'Servicio Nacional de Meteorología e Hidrología', 'https://www.senamhi.gob.pe/'),
}


def classify_station_temporal(g_dedup: pd.DataFrame) -> pd.Series:
  ey = pd.to_numeric(g_dedup['verified_record_end_year'], errors='coerce')
  ar = g_dedup['active_reason'].astype(str)
  status = pd.Series('undated', index=g_dedup.index)
  is_act = (ey >= 2016) | (ey.isna() & ar.isin(['1_cv8811_verified_active', '2_end_year_ge_2016', '3_portal_status_active']))
  is_inact = (~is_act) & ((ey <= 2015) | ar.isin(['5_portal_status_inactive', '6_end_year_lt_2016']))
  status[is_act] = 'active'
  status[is_inact] = 'inactive'
  return status


def build_table_s8(out_dir: Path) -> Path:
  attr_dir = CANON_DIR / 'attributes'
  arch = pd.read_parquet(attr_dir / 'global_gauges_by_source_archive.parquet')
  dedup = pd.read_parquet(attr_dir / 'global_gauges_deduplicated.parquet')
  dedup['temp_status'] = classify_station_temporal(dedup)
  cid_to_temp = dict(zip(dedup['canonical_gauge_id'], dedup['temp_status']))

  cat_map = {}
  for x in ['Caravan', 'Caravan (GRDC Extension)', 'Caravan (Regional CAMELS Extension)', 'GRDC', 'ROBIN_v1', 'GSIM', 'GSHA']:
    cat_map[x] = 'Part A: Global Multi-Country Compilations and Index Archives (7 repositories)'
  for x in ['EStreams', 'HYSETS', 'R_ArcticNET_v4', 'ArcticGRO', 'CIS_Watersheds', 'ADHI', 'SSA-HCD', 'North_Africa_98_Basins', 'CA_discharge', 'Andean-GC', 'MRC']:
    cat_map[x] = 'Part B: Continental and Transboundary Regional Compilations (11 repositories)'
  for x in [
      'CNS_CAMELS_Extra', 'CCAM', 'CDR2', 'HydroMLYR', 'CAMELS-BR', 'CABra', 'GHI_v1', 'GHI_v2', 'GHSA_v3', 'CAMELS_IND',
      'JPDH', 'CAMELS-PL', 'FOCA_ISPRA_Italy', 'SIREN_Italy', 'CAMELS-US',
      'CAMELS-CL', 'PMET-obs', 'CAMELS_NZ', 'CAMELS-COL', 'CAMELS-CH', 'CAMELS-FI', 'CAMELS_KR',
      'CAMELSH_KR', 'CAMELS-PE', 'LamaH-Ice', 'CAMELS-LUX',
  ]:
    cat_map[x] = 'Part C: National CAMELS and Academic Catchment Syntheses (26 repositories)'
  for x in sorted(set(arch['source_dataset'].unique()) - set(cat_map.keys())):
    cat_map[x] = 'Part D: National Hydrometeorological Agency Portals and Operational APIs (28 repositories)'

  arch['is_q'] = arch['observation_type'].isin(['daily_discharge', 'monthly_discharge'])
  arch['is_open_q'] = arch['row_in_tier_1'].astype(bool) | (
      arch['is_q'] & arch['data_access_policy'].isin(['open_academic_compilation', 'open_public_api_or_bulk'])
  )
  arch['is_rest_q'] = arch['is_q'] & (~arch['is_open_q'])
  arch['is_stage'] = arch['observation_type'] == 'stage_or_fluviometric'

  rows = []
  for ds, g in arch.groupby('source_dataset'):
    disp_name, pub_ref, url = REPO_META[ds]
    ctries = sorted([str(c) for c in g['country'].dropna().unique() if str(c).strip()])
    dom = f'{len(ctries)} countries' if len(ctries) > 2 else (', '.join(ctries) if ctries else 'Global')
    g_by_cid = g.groupby('canonical_gauge_id').agg(
        has_t1=('row_in_tier_1', 'any'),
        has_open_q=('is_open_q', 'any'),
        has_rest_q=('is_rest_q', 'any'),
        has_stage=('is_stage', 'any'),
    )
    n_unique = len(g_by_cid)
    n_t1 = int(g_by_cid['has_t1'].sum())
    n_open_q = int(g_by_cid['has_open_q'].sum())
    n_rest_q = int((~g_by_cid['has_open_q'] & g_by_cid['has_rest_q']).sum())
    n_stage = int((~g_by_cid['has_open_q'] & ~g_by_cid['has_rest_q'] & g_by_cid['has_stage']).sum())
    n_meta = int((~g_by_cid['has_open_q'] & ~g_by_cid['has_rest_q'] & ~g_by_cid['has_stage']).sum())

    ts_vals = pd.Series([cid_to_temp[cid] for cid in g_by_cid.index]).value_counts().to_dict()
    n_act = int(ts_vals.get('active', 0))
    n_inact = int(ts_vals.get('inactive', 0))
    n_und = int(ts_vals.get('undated', 0))

    raw_l = str(g['license'].dropna().iloc[0]) if not g['license'].dropna().empty else 'Unspecified'
    rows.append({
        'category': cat_map[ds],
        'source_key': ds,
        'display_name': disp_name,
        'domain': dom,
        'archive_rows': int(len(g)),
        'unique_gauges': n_unique,
        'tier1_open_dataset_q': n_t1,
        'open_discharge_q': n_open_q,
        'restricted_discharge_q': n_rest_q,
        'stage_only': n_stage,
        'metadata_or_indices_only': n_meta,
        'active_ge2016_or_portal': n_act,
        'inactive_le2015': n_inact,
        'undated_unknown': n_und,
        'license_summary': raw_l,
        'publication': pub_ref,
        'url': url,
    })

  df = pd.DataFrame(rows).sort_values(['category', 'unique_gauges'], ascending=[True, False]).reset_index(drop=True)
  df.insert(0, 'repo_id', range(1, len(df) + 1))
  out_path = out_dir / 'Table_S8_72_gauge_repositories.csv'
  df.to_csv(out_path, index=False)
  print(f'  Saved {out_path} ({len(df)} repositories; {df["archive_rows"].sum():,} archive rows)', flush=True)
  return out_path
