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

# Author: Grey Nearing
# Created: 2026-07-14


"""Create HyBAS to Country Mapping.

This script creates the mapping from HyBAS IDs to countries globally by
performing a spatial join with Natural Earth country boundaries.

Main Steps:
1. Load Natural Earth geographic boundary data.
2. Load HyBAS geographic coordinates.
3. Perform a spatial intersection.
4. Store the basin to country relationship.

Required Input Files:
- Natural Earth Geometries: file_paths.NATURAL_EARTH_PATH
- HyBAS Latitudes/Longitudes: file_paths.HYBAS_LATLON_PATH

Saved Output Files:
- HyBAS Mapping File: file_paths.HYBAS_MAPPING_FILE
"""

import importlib
import os
import time

from absl import app
import geopandas as gpd
import pandas as pd

from utils import file_paths
from utils.io_utils import load_csv
from utils import gfile_compat as gfile


def load_natural_earth_data(path):
  """Loads the Natural Earth country boundaries dataset.

  Args:
    path: The path string pointing to the Natural Earth geographic data file.

  Returns:
    A geopandas.GeoDataFrame containing valid country geometries.
  """
  # We track execution time to give visibility into how long data loading takes.
  start = time.time()
  
  # gfile is used for reading from Google's distributed file system.
  with gfile.Open(path, 'rb') as f:
    world_gdf = gpd.read_file(f)
    
  # Remove any rows where the geometric shape itself is missing.
  world_gdf = world_gdf.dropna(subset=[world_gdf.geometry.name])
  
  # Safety check: keep only geometries that are valid (topologically sound) 
  # and not completely empty.
  world_gdf = world_gdf[world_gdf.is_valid & ~world_gdf.is_empty]
  
  print(
      f"[{time.strftime('%H:%M:%S')}] Loaded Natural Earth Data in"
      f' {(time.time() - start):.2f}s. Shape: {world_gdf.shape}'
  )
  return world_gdf


def load_hybas_data(path):
  """Loads the HyBAS basin coordinate data.

  Args:
    path: The file path pointing to a CSV file of HyBAS latitudes and longitudes.

  Returns:
    A pandas.DataFrame containing the raw HyBAS coordinates data.
  """
  # Time the data loading step to help diagnose any performance bottlenecks.
  start = time.time()
  
  # Read the CSV from distributed storage into a standard Pandas DataFrame.
  hybas_latlon = load_csv(path)
  
  print(
      f"[{time.strftime('%H:%M:%S')}] Loaded HyBAS coordinates in"
      f' {(time.time() - start):.2f}s. Shape: {hybas_latlon.shape}'
  )
  return hybas_latlon


def create_geodataframe(hybas_latlon, crs):
  """Converts raw HyBAS tabular coordinates into a spatial GeoDataFrame.

  Args:
    hybas_latlon: A pandas.DataFrame with HyBAS latitude/longitude columns.
    crs: Coordinate Reference System (CRS) matching the country boundaries, 
      so both datasets overlay correctly.

  Returns:
    A geopandas.GeoDataFrame where each row is a spatial Point representing a basin.
  """
  latlon_cols = {
      'HYDRO_ATLAS:latitude': 'latitude',
      'HYDRO_ATLAS:longitude': 'longitude',
  }
  
  # Rename verbose columns to simple names and pare down the DataFrame 
  # to focus only on coordinate columns.
  geo_df = hybas_latlon.rename(columns=latlon_cols)[['latitude', 'longitude']]
  print(
      f"[{time.strftime('%H:%M:%S')}] Selected latitude/longitude columns."
      f' Shape: {geo_df.shape}'
  )

  # Track how long the geometric conversion takes.
  start = time.time()
  
  # Convert standard x/y columns into formal spatial Point objects.
  # We apply the passed-in Coordinate Reference System (CRS) ensuring compatibility.
  gdf_points = gpd.GeoDataFrame(
      geo_df,
      geometry=gpd.points_from_xy(geo_df.longitude, geo_df.latitude),
      crs=crs,
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Created GeoDataFrame for points in"
      f' {(time.time() - start):.2f}s. Shape: {gdf_points.shape}'
  )
  return gdf_points


def perform_spatial_join(gdf_points, world_gdf):
  """Intersects basin points with country boundaries to map basins to countries.

  Args:
    gdf_points: A geopandas.GeoDataFrame containing Point geometries for basins.
    world_gdf: A geopandas.GeoDataFrame containing Polygon boundaries for countries.

  Returns:
    A pandas.DataFrame mapping each HyBAS basin ID to its containing country name.
  """
  # Time the spatial join since it can be a computationally heavy operation.
  start = time.time()
  
  # Performs a spatial join (sjoin) to find which country polygon each basin point 
  # falls into. We use an 'inner' join and look for an 'intersects' relationship.
  points_with_country = gpd.sjoin(
      gdf_points, world_gdf, how='inner', predicate='intersects'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Performed spatial join in"
      f' {(time.time() - start):.2f}s. Shape: {points_with_country.shape}'
  )

  # Track the time needed to format the output mapping.
  start = time.time()
  
  # Extract the single column identifying the country ('name'). The original index 
  # holds our basin ID, so we bring it back as a regular column named 'hybas_id'.
  mapping_df = points_with_country[['name']].reset_index()
  mapping_df.columns = ['hybas_id', 'country_name']
  print(
      f"[{time.strftime('%H:%M:%S')}] Created mapping dataframe in"
      f' {(time.time() - start):.2f}s. Shape: {mapping_df.shape}'
  )
  return mapping_df


def save_mapping(mapping_df, hybas_mapping_file):
  """Saves the basin-to-country mapping results as a CSV file.

  Args:
    mapping_df: The fully prepared mapping pandas.DataFrame to save.
    hybas_mapping_file: File path string indicating where to write the output CSV.
    
  Returns:
    None
  """
  # Make sure the target folder actually exists before writing to it to avoid IO errors.
  file_paths.ensure_directories_exist()

  # Write the generated dataframe back out to Google's file system, avoiding index saving.
  with gfile.Open(hybas_mapping_file, 'wt') as f:
    mapping_df.to_csv(f, index=False)


def main(argv):
  """Entry point orchestrating the creation of the HyBAS-to-country mapping.

  Args:
    argv: List of command line arguments array (not used).
    
  Returns:
    None
  """
  del argv
  
  # We dynamically reload the file_paths module here. This guarantees that if 
  # config paths change during active development in interactive environments 
  # like notebooks, we grab the absolute latest paths.
  importlib.reload(file_paths)

  print(f"[{time.strftime('%H:%M:%S')}] Environment setup complete.")

  mapping_start_time = time.time()
  hybas_mapping_file = file_paths.HYBAS_MAPPING_PATH

  print(
      f"[{time.strftime('%H:%M:%S')}] Starting spatial join to map HyBAS to"
      ' countries...'
  )

  # 1. Fetch our base geographical map of countries.
  world_gdf = load_natural_earth_data(file_paths.NATURAL_EARTH_PATH)

  # 2. Fetch our coordinate definitions for all HyBAS basins.
  hybas_latlon = load_hybas_data(file_paths.HYBAS_LATLON_PATH)

  # 3. Marry the basin spatial points with the country's coordinate system.
  gdf_points = create_geodataframe(hybas_latlon, world_gdf.crs)

  # 4. Overlap basins against the country outlines.
  mapping_df = perform_spatial_join(gdf_points, world_gdf)

  # 5. Flush the finalized mapping back to disk.
  save_mapping(mapping_df, hybas_mapping_file)

  print(
      f"[{time.strftime('%H:%M:%S')}] Created mapping for {len(mapping_df)}"
      f' basins in {time.time() - mapping_start_time:.2f}s.'
  )


if __name__ == '__main__':
  app.run(main)
