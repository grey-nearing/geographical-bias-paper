#!/usr/bin/env python3
"""Stage B0 Post-Merge & Exposure Audit.

Merges sharded per-basin exposure CSVs onto the canonical 1,034,083
HydroBASINS v1c Level 12 index, writes `global_basin_exposure_{defense}_r2.parquet`,
and runs a scientific verification audit. Requires licensed Fathom v3 hazard inputs.
"""

import argparse
import glob
import os
import sys

REPO_DIR = os.environ.get(
    'GEO_BIAS_REPO_DIR',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if REPO_DIR not in sys.path:
  sys.path.insert(0, REPO_DIR)

CANON_DIR = os.environ.get('GEO_BIAS_DATA_DIR', './data')
os.environ.setdefault('GEO_BIAS_DATA_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_THIRD_PARTY_DIR', CANON_DIR)
os.environ.setdefault('GEO_BIAS_OUTPUT_DIR', CANON_DIR)

import numpy as np
import pandas as pd
from utils import cap_utils, file_paths

OUT_DIR = os.path.join(CANON_DIR, 'round2', 'exposure')
SHARDS_LOCAL_DIR = os.environ.get('GEO_BIAS_EXPOSURE_SHARDS_DIR', os.path.join(OUT_DIR, 'shards'))
OLD_EXP_CSV = os.path.join(CANON_DIR, 'exposure', 'global_basin_exposure_defended_by_return_period.csv')
HYBAS_PARQUET = os.path.join(CANON_DIR, 'round2', 'meta_models', 'hybas_inference_95_float32.parquet')

OLD_W = {5: 0.05, 10: 0.075, 20: 0.04, 50: 0.02, 100: 0.005}
NEW_W = {5: 0.05, 10: 0.075, 20: 0.04, 50: 0.02, 100: 0.015}


def _norm_hybas(s: pd.Series) -> pd.Series:
  s2 = s.astype(str).str.replace(r'\.0$', '', regex=True)
  return np.where(s2.str.startswith('hybas_'), s2, 'hybas_' + s2)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--defense', choices=['defended', 'undefended'], default='defended')
  args = parser.parse_args()

  dt = args.defense.lower()
  os.makedirs(SHARDS_LOCAL_DIR, exist_ok=True)

  shard_files = sorted(glob.glob(os.path.join(SHARDS_LOCAL_DIR, f'global_basin_exposure_{dt}_all_rps_r2.csv-*')))
  assert len(shard_files) > 0, f'No shards found matching {SHARDS_LOCAL_DIR}/global_basin_exposure_{dt}_all_rps_r2.csv-*'
  print(f'Reading {len(shard_files)} local CSV shards...', flush=True)
  dfs = [pd.read_csv(f, dtype={'HYBAS_ID': str}) for f in shard_files]
  df_raw = pd.concat(dfs, ignore_index=True)
  df_raw['hybas_id'] = _norm_hybas(df_raw['HYBAS_ID'])
  n_raw = len(df_raw)
  n_uniq = df_raw['hybas_id'].nunique()
  print(f'Raw emitted basins: {n_raw:,} rows ({n_uniq:,} unique)', flush=True)

  # Sum duplicates just in case (should be 0 duplicates due to CombinePerKey)
  num_cols = [c for c in df_raw.columns if c not in ('HYBAS_ID', 'hybas_id')]
  df_agg = df_raw.groupby('hybas_id', as_index=True)[num_cols].sum()

  # Canonical 1,034,083 basin universe
  all_ids = pd.read_parquet(HYBAS_PARQUET, columns=['pre_mm_uyr']).index
  assert len(all_ids) == 1034083
  in_both = all_ids.isin(df_agg.index).sum()
  missing_from_raw = len(all_ids) - in_both
  extra_in_raw = (~df_agg.index.isin(all_ids)).sum()
  print(f'Alignment vs 1,034,083 canonical index: matched={in_both:,}, zero-pixel/missing={missing_from_raw:,}, extra={extra_in_raw:,}', flush=True)

  df = df_agg.reindex(all_ids).fillna(0.0)
  for T in (5, 10, 20, 50, 100):
    df[f'exposed_pop_{T}'] = df[f'exposed_pop_gt10cm_{T}']
    df[f'exposed_pop_gt0_{T}'] = df[f'exposed_pop_gt0cm_{T}']

  # Compute exact trapezoidal EAE (>10 cm primary & >0 cm sensitivity)
  df['eae_gt10cm'] = sum(NEW_W[T] * df[f'exposed_pop_gt10cm_{T}'] for T in NEW_W)
  df['eae_gt0cm'] = sum(NEW_W[T] * df[f'exposed_pop_gt0cm_{T}'] for T in NEW_W)
  df['eae_gt10cm_oldw'] = sum(OLD_W[T] * df[f'exposed_pop_gt10cm_{T}'] for T in OLD_W)

  out_pq = os.path.join(OUT_DIR, f'global_basin_exposure_{dt}_r2.parquet')
  df.to_parquet(out_pq)
  print(f'Saved canonical Round-2 exposure parquet: {out_pq} ({df.shape})', flush=True)

  # Scientific verification checks
  print('\n=== SCIENTIFIC VALIDATION CHECKS ===')
  tot_pop = df['total_population'].sum()
  print(f'1. Global WorldPop 2020 total_population = {tot_pop / 1e9:.4f} B ({tot_pop:,.0f})')
  for T in (5, 10, 20, 50, 100):
    p10 = df[f'exposed_pop_gt10cm_{T}'].sum() / 1e6
    p0 = df[f'exposed_pop_gt0cm_{T}'].sum() / 1e6
    print(f'   1-in-{T:3d} yr exposed pop: >10cm = {p10:8.2f} M | >0cm = {p0:8.2f} M')
  print(f'   Global EAE (w_100=0.015, sum=0.200): >10cm = {df["eae_gt10cm"].sum()/1e6:.2f} M/yr | >0cm = {df["eae_gt0cm"].sum()/1e6:.2f} M/yr')
  print(f'   Global EAE (old w_100=0.005, sum=0.190): >10cm = {df["eae_gt10cm_oldw"].sum()/1e6:.2f} M/yr')

  # Monotonicity check across RPs
  rps = [5, 10, 20, 50, 100]
  for mode, pfx in [('>10cm', 'exposed_pop_gt10cm_'), ('>0cm', 'exposed_pop_gt0cm_')]:
    viol = 0
    max_drop = 0.0
    for a, b in zip(rps[:-1], rps[1:]):
      diff = df[f'{pfx}{a}'] - df[f'{pfx}{b}']
      v = (diff > 1e-6).sum()
      viol += int(v)
      if v > 0:
        max_drop = max(max_drop, float(diff.max()))
    print(f'2. RP Monotonicity ({mode}): violations={viol:,}, max_drop={max_drop:.4f}')

  # Country mapping check
  file_paths.init('5.0-year-2.0-window-recall')
  df_map = pd.read_csv(file_paths.HYBAS_MAPPING_PATH)
  idc = 'hybas_id' if 'hybas_id' in df_map.columns else df_map.columns[0]
  df_map['hid'] = _norm_hybas(df_map[idc])
  df_map['country_name_clean'] = df_map['country_name'].replace(file_paths.country_name_map)
  df_map = df_map.drop_duplicates('hid').set_index('hid')
  df['country_name'] = df_map['country_name_clean'].reindex(all_ids)
  df['has_system'] = df['country_name'].map(cap_utils.has_effective_flood_warning_system).fillna(False).astype(bool)

  is_unm = df['country_name'].isna()
  print(
      f'3. Unmapped coastal micro-polygons ({is_unm.sum():,} basins): '
      f'total_pop={df.loc[is_unm, "total_population"].sum()/1e6:.2f} M | '
      f'EAE(>10cm)={df.loc[is_unm, "eae_gt10cm"].sum()/1e6:.3f} M/yr '
      f'({100.0 * df.loc[is_unm, "eae_gt10cm"].sum() / df["eae_gt10cm"].sum():.2f}% of global EAE)'
  )

  # Continent check via HYBAS first digit
  df['cont_digit'] = pd.Series(all_ids).str.replace('hybas_', '').str[0].values
  cont_names = {'1': 'af', '2': 'eu', '3': 'si', '4': 'as', '5': 'au', '6': 'sa', '7': 'na', '8': 'ar', '9': 'gr'}
  print('\n4. Per-continent breakdown (all 9 HydroBASINS v1c continents):')
  for d_code, cname in cont_names.items():
    sub = df[df['cont_digit'] == d_code]
    print(
        f'   {cname} ({d_code}): basins={len(sub):7,} | pop={sub["total_population"].sum()/1e6:8.2f} M | '
        f'EAE_10cm={sub["eae_gt10cm"].sum()/1e6:7.2f} M/yr | EAE_0cm={sub["eae_gt0cm"].sum()/1e6:7.2f} M/yr'
    )

  if dt != 'defended' or not os.path.exists(OLD_EXP_CSV):
    return

  # Compare directly against OLD defended CSV (before 4-bug fix)
  print('\n=== BEFORE vs AFTER 4-BUG EXPOSURE AUDIT (DEFENDED) ===')
  df_old = pd.read_csv(OLD_EXP_CSV, dtype={'hybas_id': str})
  df_old['hid'] = _norm_hybas(df_old['hybas_id'])
  df_old = df_old.drop_duplicates('hid').set_index('hid').reindex(all_ids).fillna(0.0)
  df['old_eae'] = sum(OLD_W[T] * df_old[f'exposed_pop_{T}'] for T in OLD_W)
  df['old_eae_div9'] = df['old_eae'] / 9.0
  df['old_pop100'] = df_old['exposed_pop_100']
  df['old_pop100_div9'] = df['old_pop100'] / 9.0

  print(f'Global OLD EAE (buggy 9x, 1-deg shifted, >0cm, 7-tile, w100=0.005): {df["old_eae"].sum()/1e6:7.2f} M/yr')
  print(f'Global OLD EAE / 9:                                                 {df["old_eae_div9"].sum()/1e6:7.2f} M/yr')
  print(f'Global NEW EAE (>0 cm, fixed align/9-tile/w100=0.015):              {df["eae_gt0cm"].sum()/1e6:7.2f} M/yr')
  print(f'Global NEW EAE (>10 cm PRIMARY, fixed align/9-tile/w100=0.015):     {df["eae_gt10cm"].sum()/1e6:7.2f} M/yr')
  print(f'Global OLD 100-yr / 9 vs NEW 100-yr (>0cm | >10cm): {df["old_pop100_div9"].sum()/1e6:.2f} M | {df["exposed_pop_gt0cm_100"].sum()/1e6:.2f} M | {df["exposed_pop_gt10cm_100"].sum()/1e6:.2f} M')

  m_sys = df['has_system'] & df['country_name'].notna()
  m_nos = (~df['has_system']) & df['country_name'].notna()
  print(
      f'CAP split NEW >10cm: 60 Sys={df.loc[m_sys, "eae_gt10cm"].sum()/1e6:.2f} M/yr '
      f'({100*df.loc[m_sys, "eae_gt10cm"].sum()/df["eae_gt10cm"].sum():.1f}%) | '
      f'115 NoSys={df.loc[m_nos, "eae_gt10cm"].sum()/1e6:.2f} M/yr '
      f'({100*df.loc[m_nos, "eae_gt10cm"].sum()/df["eae_gt10cm"].sum():.1f}%)'
  )

  # Top 30 countries + Canada, Greenland, Fiji, Jamaica, Bahamas, Trinidad & Tobago
  ctry = df[df['country_name'].notna()].groupby('country_name').agg(
      n_basins=('total_population', 'size'),
      has_system=('has_system', 'first'),
      total_pop=('total_population', 'sum'),
      old_eae=('old_eae', 'sum'),
      old_eae_div9=('old_eae_div9', 'sum'),
      new_eae_gt0=('eae_gt0cm', 'sum'),
      new_eae_gt10=('eae_gt10cm', 'sum'),
      new_pop100_gt10=('exposed_pop_gt10cm_100', 'sum'),
  )
  ctry['ratio_new10_to_old'] = np.where(ctry['old_eae'] > 0, ctry['new_eae_gt10'] / ctry['old_eae'], np.nan)
  ctry['ratio_new0_to_olddiv9'] = np.where(ctry['old_eae_div9'] > 0, ctry['new_eae_gt0'] / ctry['old_eae_div9'], np.nan)
  ctry = ctry.sort_values('new_eae_gt10', ascending=False)

  audit_csv = os.path.join(OUT_DIR, 'exposure_4bug_before_after_country_audit.csv')
  ctry.to_csv(audit_csv)
  print(f'\nSaved full 175-territory audit table to {audit_csv}')

  print('\nTop 25 territories by NEW EAE (>10cm):')
  show = ctry.head(25).copy()
  for col in ('total_pop', 'old_eae', 'old_eae_div9', 'new_eae_gt0', 'new_eae_gt10', 'new_pop100_gt10'):
    show[col] = (show[col] / 1e6).round(3)
  print(show[['has_system', 'total_pop', 'old_eae', 'old_eae_div9', 'new_eae_gt0', 'new_eae_gt10', 'ratio_new0_to_olddiv9']].to_string())

  print('\nSpecial check (formerly missing/shifted islands, Canada, Greenland, Sudan):')
  for nm in ['Canada', 'Greenland', 'Fiji', 'Jamaica', 'Bahamas', 'Trinidad and Tobago', 'Barbados', 'Bangladesh', 'Sudan', 'China', 'India', 'United States of America']:
    if nm in ctry.index:
      r = ctry.loc[nm]
      print(
          f'  {nm:26s}: pop={r["total_pop"]/1e6:7.2f}M | old={r["old_eae"]/1e6:7.3f}M | '
          f'old/9={r["old_eae_div9"]/1e6:6.3f}M | new>0={r["new_eae_gt0"]/1e6:6.3f}M | '
          f'new>10={r["new_eae_gt10"]/1e6:6.3f}M | ratio(>0 / old/9)={r["ratio_new0_to_olddiv9"]:.2f}'
      )


if __name__ == '__main__':
  main()
