# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calculates exposed population per HydroBASINS sub-basin at return periods."""

import collections
import concurrent.futures
import io
import os
import re
import sys

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
import absl.app
import absl.flags
import absl.logging
from utils import gfile_compat as gfile
import multiprocessing
import geopandas
import numpy as np
import pandas as pd
from scipy import ndimage
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_integer(
    'num_workers', 20, 'Number of parallel worker processes.'
)
_DATA_ROOT = os.environ.get('GEO_BIAS_DATA_DIR', os.path.join(os.getcwd(), 'data'))

absl.flags.DEFINE_string(
    'fathom_base_dir',
    os.path.join(_DATA_ROOT, 'external_rasters', 'fathom'),
    'Base directory for Fathom flood maps.',
)
absl.flags.DEFINE_string(
    'output_path',
    os.path.join(_DATA_ROOT, 'exposure', 'global_basin_exposure.csv'),
    'Output path for the global basin exposure CSV.',
)
absl.flags.DEFINE_string(
    'test_tile', '', 'If specified, process only this tile and print results.'
)
absl.flags.DEFINE_string(
    'worldpop_vrt_path',
    os.path.join(_DATA_ROOT, 'external_rasters', 'worldpop', '2020', 'worldpop_global.vrt'),
    'Path to the WorldPop VRT.',
)

RETURN_PERIODS = [5, 10, 20, 50, 100]

CONTIENT_PATHS = [
    os.path.join(_DATA_ROOT, 'geoparquet', c, f'hybas_{c}_lev12_v1c.geoparquet')
    for c in ['af', 'ar', 'as', 'au', 'eu', 'gr', 'na', 'sa', 'si']
]


_GLOBAL_SUB_BASINS = None


def init_worker(continent_paths: list[str]):
  """Initializes the worker process by pre-loading Level 12 sub-basins."""
  global _GLOBAL_SUB_BASINS
  absl.logging.info('Worker pre-loading global Level 12 sub-basins shapefiles...')
  gdf_list = []
  for path in continent_paths:
    try:
      with gfile.GFile(path, 'rb') as f:
        gdf = geopandas.read_parquet(f)
      if 'HYBAS_ID' in gdf.columns:
        gdf.rename(columns={'HYBAS_ID': 'hybas_id'}, inplace=True)
      gdf_list.append(gdf[['hybas_id', 'geometry']])
    except Exception as e:
      absl.logging.warning('Failed to load geoparquet %s: %s', path, e)

  if not gdf_list:
    absl.logging.error('Failed to load any sub-basin shapefiles in worker.')
    sys.exit(1)

  _GLOBAL_SUB_BASINS = geopandas.GeoDataFrame(
      pd.concat(gdf_list, ignore_index=True)
  )
  absl.logging.info(
      'Worker loaded %d sub-basins successfully.', len(_GLOBAL_SUB_BASINS)
  )


def parse_tile_coords(tile_name: str) -> tuple[float, float, float, float]:
  """Parses a tile name like 'n24w081' to get bbox (min_lon, min_lat, max_lon, max_lat)."""
  match = re.match(r'([ns])(\d+)([ew])(\d+)', tile_name)
  if not match:
    raise ValueError(f'Invalid tile name format: {tile_name}')

  lat_dir, lat_val, lon_dir, lon_val = match.groups()
  lat = int(lat_val)
  if lat_dir == 's':
    lat = -lat
  lon = int(lon_val)
  if lon_dir == 'w':
    lon = -lon

  return float(lon), float(lat), float(lon + 1), float(lat + 1)


def process_single_tile(
    tile_name: str,
    base_dir: str,
    worldpop_vrt_path: str,
) -> dict[int, dict[int, float]]:
  """Processes a single tile across all return periods.

  Returns a mapping of hybas_id -> return_period -> exposed_population sum.
  """
  global _GLOBAL_SUB_BASINS
  if _GLOBAL_SUB_BASINS is None:
    raise RuntimeError('Global sub-basins not initialized in worker.')

  tile_results = collections.defaultdict(dict)
  try:
    min_lon, min_lat, max_lon, max_lat = parse_tile_coords(tile_name)
  except ValueError as e:
    absl.logging.error(str(e))
    return tile_results

  # 1. Extract overlapping Level 12 sub-basins in-memory
  tile_sub_basins = _GLOBAL_SUB_BASINS.cx[min_lon:max_lon, min_lat:max_lat]
  if tile_sub_basins.empty:
    absl.logging.info('No sub-basins found overlapping tile %s', tile_name)
    return tile_results

  tile_sub_basins = geopandas.GeoDataFrame(tile_sub_basins)

  # 2. Open return period 5 Fathom map to get shape/transform reference
  ref_fathom_path = os.path.join(
      base_dir,
      'FLOOD_MAP-1ARCSEC-NW_OFFSET-1in5-FLUVIAL-DEFENDED-DEPTH-2020-PERCENTILE50-v3.1',
      f'{tile_name}.tif',
  )
  if not gfile.Exists(ref_fathom_path):
    absl.logging.error('Reference Fathom file not found: %s', ref_fathom_path)
    return tile_results

  temp_ref_path = f'/tmp/ref_{tile_name}.tif'
  try:
    gfile.Copy(ref_fathom_path, temp_ref_path, overwrite=True)
    with rasterio.open(temp_ref_path) as src:
      fathom_shape = src.shape
      fathom_transform = src.transform
      fathom_crs = src.crs
  except Exception as e:
    absl.logging.error(
        'Failed to stage or read reference Fathom tile %s: %s', tile_name, e
    )
    return tile_results
  finally:
    if gfile.Exists(temp_ref_path):
      gfile.Remove(temp_ref_path)

  # 3. Read WorldPop 100m count raster for this bounding box from local VRT
  try:
    with rasterio.open(worldpop_vrt_path) as pop_src:
      from rasterio.windows import from_bounds
      window = from_bounds(
          min_lon, min_lat, max_lon, max_lat, transform=pop_src.transform
      )
      pop_data = pop_src.read(1, window=window)
      pop_transform = pop_src.window_transform(window)
      pop_crs = pop_src.crs
  except Exception as e:
    absl.logging.error(
        'Failed to read WorldPop from VRT for tile %s: %s', tile_name, e
    )
    return tile_results

  # Replace nodata values with 0
  pop_data = np.where(pop_data < 0, 0.0, pop_data)

  # 4. Upsample population array to match Fathom grid
  pop_upsampled = np.zeros(fathom_shape, dtype=np.float32)
  reproject(
      source=pop_data,
      destination=pop_upsampled,
      src_transform=pop_transform,
      src_crs=pop_crs,
      dst_transform=fathom_transform,
      dst_crs=fathom_crs,
      resampling=Resampling.nearest,
  )
  # Scale to preserve counts
  cell_area_ratio = abs(
      (fathom_transform.a * fathom_transform.e)
      / (pop_transform.a * pop_transform.e)
  )
  pop_upsampled *= cell_area_ratio

  # 5. Rasterize sub-basin polygons onto the Fathom grid
  shapes = [(geom, idx) for idx, geom in enumerate(tile_sub_basins.geometry)]
  basin_index_grid = rasterize(
      shapes,
      out_shape=fathom_shape,
      transform=fathom_transform,
      fill=-1,
      all_touched=False,
      dtype=np.int32,
  )

  # 6. For each return period, load flood map, compute mask, and sum pop
  for rp in RETURN_PERIODS:
    fathom_path = os.path.join(
        base_dir,
        f'FLOOD_MAP-1ARCSEC-NW_OFFSET-1in{rp}-FLUVIAL-DEFENDED-DEPTH-2020-PERCENTILE50-v3.1',
        f'{tile_name}.tif',
    )
    if not gfile.Exists(fathom_path):
      absl.logging.warning(
          'Fathom map missing for tile %s, rp %d', tile_name, rp
      )
      continue

    temp_fathom_path = f'/tmp/fathom_{tile_name}_{rp}.tif'
    try:
      gfile.Copy(fathom_path, temp_fathom_path, overwrite=True)
      with rasterio.open(temp_fathom_path) as src:
        depth_data = src.read(1)
    except Exception as e:
      absl.logging.error(
          'Failed to stage or read Fathom tile %s, rp %d: %s', tile_name, rp, e
      )
      continue
    finally:
      if gfile.Exists(temp_fathom_path):
        gfile.Remove(temp_fathom_path)

    # Flood threshold is 0.1m, and mask out nodata (values > 999.0)
    flooded_mask = (depth_data > 0.1) & (depth_data < 990.0)
    flooded_pop = pop_upsampled * flooded_mask

    # Fast zonal sum using scipy.ndimage.sum (C-extension, GIL-free)
    unique_indices = np.unique(basin_index_grid)
    valid_indices = unique_indices[unique_indices != -1]
    if len(valid_indices) > 0:
      sums = ndimage.sum(
          flooded_pop, labels=basin_index_grid, index=valid_indices
      )
      if isinstance(sums, (int, float, np.float32, np.float64)):
        sums = [sums]
      for idx, exposed_pop in zip(valid_indices, sums):
        if exposed_pop > 0:
          hybas_id = int(tile_sub_basins.iloc[idx]['hybas_id'])
          tile_results[hybas_id][rp] = float(exposed_pop)

  absl.logging.info('Successfully processed tile %s', tile_name)
  return tile_results


def main(argv):
  if len(argv) > 1:
    raise absl.app.UsageError('Too many command-line arguments.')

  # Ensure the output directory exists
  out_dir = os.path.dirname(FLAGS.output_path)
  if not gfile.Exists(out_dir):
    gfile.MakeDirs(out_dir)

  # Find all tiles available in fluvial 5-year directory
  fluvial_5y_dir = os.path.join(
      FLAGS.fathom_base_dir,
      'FLOOD_MAP-1ARCSEC-NW_OFFSET-1in5-FLUVIAL-DEFENDED-DEPTH-2020-PERCENTILE50-v3.1',
  )
  if not gfile.Exists(fluvial_5y_dir):
    absl.logging.error('Base Fathom folder does not exist: %s', fluvial_5y_dir)
    sys.exit(1)

  if FLAGS.test_tile:
    absl.logging.info(
        'Running in test mode for single tile: %s', FLAGS.test_tile
    )
    # In test mode, initialize global sub-basins first (runs in parent process)
    init_worker(CONTIENT_PATHS)
    tile_res = process_single_tile(
        FLAGS.test_tile,
        FLAGS.fathom_base_dir,
        FLAGS.worldpop_vrt_path,
    )
    print(f'Test tile {FLAGS.test_tile} results (first 10 basins):')
    for hybas_id, rp_map in list(tile_res.items())[:10]:
      print(f'  Basin {hybas_id}: {rp_map}')
    return

  tile_files = [f for f in gfile.ListDir(fluvial_5y_dir) if f.endswith('.tif')]
  tile_names = [os.path.splitext(f)[0] for f in tile_files]
  absl.logging.info('Found %d tiles to process.', len(tile_names))

  # Accumulator for global results: hybas_id -> return_period -> sum
  global_results = collections.defaultdict(
      lambda: {rp: 0.0 for rp in RETURN_PERIODS}
  )

  # Execute tiles in parallel using ProcessPoolExecutor
  mp_context = multiprocessing.get_context('spawn')
  with concurrent.futures.ProcessPoolExecutor(
      max_workers=FLAGS.num_workers,
      mp_context=mp_context,
      initializer=init_worker,
      initargs=(CONTIENT_PATHS,),
  ) as executor:
    future_to_tile = {
        executor.submit(
            process_single_tile,
            tile,
            FLAGS.fathom_base_dir,
            FLAGS.worldpop_vrt_path,
        ): tile
        for tile in tile_names
    }
    completed_count = 0
    total_tiles = len(tile_names)
    for future in concurrent.futures.as_completed(future_to_tile):
      tile = future_to_tile[future]
      completed_count += 1
      print(
          f'Completed tile {tile} ({completed_count}/{total_tiles})', flush=True
      )
      try:
        tile_res = future.result()
        # Merge tile results into global accumulator
        for hybas_id, rp_map in tile_res.items():
          for rp, val in rp_map.items():
            global_results[hybas_id][rp] += val

        if completed_count % 10 == 0 or completed_count == total_tiles:
          print(
              f'Progress: {completed_count}/{total_tiles} tiles completed'
              f' ({(completed_count / total_tiles) * 100:.1f}%)',
              flush=True,
          )
      except Exception as exc:
        absl.logging.error('Tile %s generated an exception: %s', tile, exc)

  # Convert accumulator to DataFrame and save
  rows = []
  for hybas_id, rp_map in global_results.items():
    row = {'hybas_id': hybas_id}
    for rp in RETURN_PERIODS:
      row[f'exposed_pop_1in{rp}'] = rp_map[rp]
    rows.append(row)

  df_out = pd.DataFrame(rows)
  df_out.sort_values('hybas_id', inplace=True)
  with gfile.GFile(FLAGS.output_path, 'w') as f:
    df_out.to_csv(f, index=False)
  absl.logging.info(
      'Successfully saved global exposure results to %s', FLAGS.output_path
  )


if __name__ == '__main__':
  app.run(main)
