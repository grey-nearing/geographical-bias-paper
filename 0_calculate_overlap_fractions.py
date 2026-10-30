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


"""Calculate Spatial Overlap Fractions.

This script calculates the spatial overlap fractions between hydrological
basins.
These overlaps are used as features in a machine learning model to predict the
skill
of a rainfall-runoff model in ungauged basins.

Main Steps:
1. Load random and separated cross-validation splits.
2. Load and clean GRDC training basin geometries.
3. Pre-calculate global pairwise overlaps among training basins.
4. Extract fold-aware upstream and downstream features for training basins.
5. Load Caravan and global HyBAS geometries continuously for inference.
6. Calculate global pairwise overlaps between HyBAS and Caravan.
7. Extract upstream and downstream coverage features for global inference.

Required Input Files:
- Cross-Validation Splits: file_paths.CROSS_VALIDATION_DIR
- GRDC Polygons: file_paths.GRDC_POLYGONS_FILE
- Caravan Polygons: file_paths.CARAVAN_POLYGONS_FILE
- HyBAS Polygons: file_paths.HYBAS_POLYGONS_DIR

Saved Output Files:
- TRAIN ML Features: file_paths.TRAIN_ML_FEATURES_PATH
- TRAIN Pairwise Overlaps: file_paths.TRAIN_PAIRWISE_OVERLAPS_PATH
- INFER ML Features: file_paths.HYBAS_COVERAGE_DIR/infer_ml_features_*.csv
- INFER Pairwise Overlaps:
file_paths.HYBAS_COVERAGE_DIR/infer_pairwise_overlaps_*.csv
- CARAVAN Pairwise Overlaps: file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH
- FULL DRAINAGE Polygons: file_paths.HYBAS_FULL_DRAINAGE_DIR/hybas_*_full_drainage.geojson
"""

import os
import random
import time

from absl import app
from absl import flags
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import box
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import seaborn as sns
from tqdm.auto import tqdm

from utils import file_paths
from utils.io_utils import load_gdf_from_cns
from utils import gfile_compat as gfile

FLAGS = flags.FLAGS

def build_full_drainage_polygons(hybas_gdf):
  """Constructs full drainage polygons by recursively merging upstream contributing catchments.
  
  Args:
    hybas_gdf: A GeoDataFrame containing baseline Level 12 HydroBASIN geometries and a 'NEXT_DOWN' column.
    
  Returns:
    A GeoDataFrame containing the newly stitched multi-basin aggregate polygons.
  """
  from collections import defaultdict
  adj = defaultdict(list)
  # Build topology pointers upwards
  for child, row in hybas_gdf.iterrows():
    parent = row.get('NEXT_DOWN', 0)
    if pd.notna(parent) and parent != '0' and parent != 0:
      adj[parent].append(child)
  
  def get_upstream(node):
    upstreams = [node]
    queue = [node]
    while queue:
      curr = queue.pop(0)
      for c in adj[curr]:
        upstreams.append(c)
        queue.append(c)
    return upstreams

  drainage_geoms = []
  for idx in hybas_gdf.index:
    up_ids = get_upstream(idx)
    # Physically merge boundaries for all upstream components
    poly = hybas_gdf.loc[hybas_gdf.index.isin(up_ids)].geometry.unary_union
    drainage_geoms.append(poly)
    
  drainage_gdf = gpd.GeoDataFrame({'HYBAS_ID': hybas_gdf.index}, geometry=drainage_geoms, crs=hybas_gdf.crs)
  return drainage_gdf


flags.DEFINE_integer(
    'n_sample_basins',
    -1,
    'Number of basins to randomly sample for testing (set to None or -1 to use'
    ' all basins)',
)

flags.DEFINE_bool(
    'recalculate_train_overlaps',
    False,
    'Set to True to force recalculation of training overlap fractions',
)

flags.DEFINE_bool(
    'recalculate_hybas_overlaps',
    False,
    'Set to True to force recalculation of HyBAS overlap fractions',
)


def load_cross_validation_splits():
  """Parses cross-validation text files to determine training and test sets.

  Reads all text files defining the testing basins for each cross-validation
  fold. By aggregating all testing basins together, we are able to determine
  the absolute universe of data we are working with. Then, we deduce the
  training set for each fold by taking the whole universe of basins and
  subtracting that specific fold's test basins.

  Returns:
      tuple: A two-element tuple containing:
          - fold_gauges (dict): A nested dictionary structured as
            {fold_index: {'train': [list of string IDs], 'test': [list of string
            IDs]}}.
          - all_train_gauges (list): A list containing all unique gauge IDs
          found
            across all folds combined.
  """
  # Store the current time so we can calculate how long this function takes to run.
  step_start = time.time()

  # Find all '.txt' files in our cross-validation directory. Each file represents a single fold.
  xval_split_files = gfile.Glob(file_paths.RANDOM_CV_DIR + '/*.txt')

  # Prepare an empty dictionary to hold the training and test IDs for each fold.
  fold_gauges = {}

  # Go through each text file one by one.
  for fold in range(len(xval_split_files)):
    fold_gauges[fold] = {}

    # Open the text file and read its contents line by line.
    with gfile.Open(xval_split_files[fold], 'rt') as f:
      lines = f.readlines()

      # Clean up the hidden newline character ('\n') from the end of each line to get clean string IDs.
      # These represent the basins we will 'test' on for this specific fold.
      fold_gauges[fold]['test'] = [line.split('\n')[0] for line in lines]

  # Aggregate all the test gauges from every fold into one massive list.
  # This combined list represents the complete "universe" of all gauges we will ever use.
  all_train_gauges = [g for fold in fold_gauges.values() for g in fold['test']]
  print(
      f"[{time.strftime('%H:%M:%S')}] Found {len(all_train_gauges)} gauges"
      ' across all random xval splits.'
  )

  # Now, we circle back to figure out the "training" set for each fold.
  for fold in fold_gauges:
    fold_test_gauges = fold_gauges[fold]['test']

    # A set allows us to easily subtract one group from another.
    # We take all possible gauges and subtract the ones we are testing on right now.
    # What's left over becomes our training set for this fold.
    fold_gauges[fold]['train'] = list(
        set(all_train_gauges) - set(fold_test_gauges)
    )

  print(
      f"[{time.strftime('%H:%M:%S')}] Splits processed in"
      f' {time.time() - step_start:.2f} seconds.'
  )
  return fold_gauges, all_train_gauges


def load_and_clean_training_geometries(all_train_gauges, n_sample_basins):
  """Loads raw geographic polygons and ensures they match our splits.

  Loads raw GeoJSON polygons into a GeoPandas DataFrame. The loader utility
  automatically reprojects coordinates to Equal Earth (EPSG:6933) to allow for
  accurate square meter area calculations and fixes broken shapes. This function
  subsequently filters the geometries so we only retain basins that are actually
  used in our cross-validation splits. By calculating and caching the area of
  each
  basin here, we save computation time in downstream steps.

  Args:
      all_train_gauges (list): The complete list of basin IDs expected from the
        cross-validation split files.
      n_sample_basins (int): An optional number of basins to randomly sample. If
        greater than 0, it reduces the size of the dataset for faster testing.

  Returns:
      tuple: A three-element tuple containing:
          - all_train_gdf (gpd.GeoDataFrame): A dataframe containing the cleaned
            and filtered spatial geometries for the training basins.
          - all_train_gauges (list): An updated list of the valid string basin
          IDs
            that were successfully matched.
          - basin_areas (pd.Series): A pandas Series mapping the basin ID to its
            calculated physical area in square meters.
  """
  # Record the start time so we can measure loading performance.
  t0_load_grdc = time.time()

  # Load the raw geographic polygon data into a GeoPandas DataFrame.
  # The utility specifically cleans topological anomalies so our program doesn't crash later.
  all_train_gdf = load_gdf_from_cns(
      file_paths.GRDC_POLYGONS_FILE, index_col='unique_gauge_id'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Loaded GRDC Polygons in"
      f' {time.time() - t0_load_grdc:.2f}s. Shape: {all_train_gdf.shape}'
  )

  # Convert our lists to sets so we can easily compare which IDs exist in both places.
  set_cv_gauges = set(all_train_gauges)
  set_gdf_gauges = set(all_train_gdf.index)

  # We identify the overlap between exactly what the split files ask for, and what polygons we actually have.
  # Discarding instances where a basin appears in one file but not the other avoids missing data errors later.
  both = set_cv_gauges.intersection(set_gdf_gauges)
  print(f"[{time.strftime('%H:%M:%S')}] Size of set in BOTH: {len(both)}")
  print(
      f"[{time.strftime('%H:%M:%S')}] Size of set ONLY in cross-validation"
      f' splits: {len(set_cv_gauges - set_gdf_gauges)}'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Size of set ONLY in GeoDataFrame:"
      f' {len(set_gdf_gauges - set_cv_gauges)}'
  )

  # Overwrite our master list of gauges to only contain the guaranteed matches.
  all_train_gauges = list(both)

  # Filter the geographic dataframe down to just these matching basin IDs.
  all_train_gdf = all_train_gdf.loc[all_train_gauges]
  print(
      f"[{time.strftime('%H:%M:%S')}] Data filtered. all_train_gdf shape:"
      f' {all_train_gdf.shape}'
  )

  # If a specific sample size is provided (e.g. for rapid debugging), we randomly pick that many basins.
  # We check that the requested sample size is larger than zero but smaller than our entire dataset.
  if (
      n_sample_basins
      and n_sample_basins > 0
      and n_sample_basins < len(all_train_gauges)
  ):
    print(
        f"[{time.strftime('%H:%M:%S')}] Subsampling training dataset to"
        f' {n_sample_basins} basins...'
    )
    # Randomly pick a smaller subset of basin IDs.
    all_train_gauges = random.sample(all_train_gauges, n_sample_basins)
    # Filter the dataframe again to only include this smaller, sampled subset.
    all_train_gdf = all_train_gdf.loc[all_train_gauges]
    print(
        f"[{time.strftime('%H:%M:%S')}] After sampling, there are"
        f' {all_train_gdf.shape[0]} valid training basins.'
    )

  # Calculate the physical area in square meters for every basin now, and save the result.
  # Doing this outside of loops makes future area lookups instantaneous.
  print(
      f"[{time.strftime('%H:%M:%S')}] Caching projected training basin areas..."
  )
  basin_areas = all_train_gdf.geometry.area

  return all_train_gdf, all_train_gauges, basin_areas


def compute_all_overlaps(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame = None
) -> pd.DataFrame:
  """Calculates intersecting areas between geographic polygons in two dataframes.

  This function identifies where basins physically overlap with each other and
  calculates
  the exact square meter area of that intersection. Geopandas intersections can
  be
  computationally expensive, so this function begins with a fast spatial join to
  find
  basins that appear to be overlapping based on their simple bounding boxes. It
  then
  performs exact, dense geometric intersections only on those paired geometries
  to
  save significant processing time.

  If a second dataframe is omitted, it assumes we are comparing basins from the
  primary
  dataframe against other basins in the very same dataframe (i.e. finding
  overlapping
  training basins).

  Args:
      gdf1 (gpd.GeoDataFrame): A dataframe containing the first set of polygons
        (e.g., training basins).
      gdf2 (gpd.GeoDataFrame, optional): A dataframe containing the second set
        of polygons (e.g., inference basins). If None, calculates intersections
        of basins within gdf1 against each other.

  Returns:
      pd.DataFrame: A flat pandas DataFrame with columns 'basin1', 'basin2', and
        'overlap_area' storing every non-zero geometric overlap fraction found.
  """
  # If we were not provided a second dataframe to compare against, we are comparing the first dataframe against itself.
  if gdf2 is None:
    gdf2 = gdf1
    is_self_intersection = True
  else:
    is_self_intersection = False

  print(
      f"[{time.strftime('%H:%M:%S')}] Precalculating structural geometric"
      ' intersections...'
  )
  calc_start = time.time()

  # Apply a "fast spatial join" using bounding boxes. This very quickly eliminates
  # pairs of basins that are on opposite sides of the world and could never intersect anyway.
  left_gdf = gpd.GeoDataFrame(geometry=gdf1.geometry)
  left_gdf.index.name = None
  right_gdf = gpd.GeoDataFrame(geometry=gdf2.geometry)
  right_gdf.index.name = None
  sjoin_df = gpd.sjoin(
      left_gdf,
      right_gdf,
      how='inner',
      predicate='intersects',
  )

  # Extract the paired ID values resulting from our fast spatial join.
  idx1 = sjoin_df.index.values
  idx2 = sjoin_df['index_right'].values

  if is_self_intersection:
    # If comparing a dataframe against itself, we don't want to calculate Basin A overlapping Basin B,
    # and then separately calculate Basin B overlapping Basin A, because that is mathematically identical.
    # Therefore, we only map unique pairs (where ID of 1 <= ID of 2) to halve the processing load.
    mask = idx1 <= idx2
    idx1_sub = idx1[mask]
    idx2_sub = idx2[mask]

    # Convert the paired IDs back into actual geographic polygons.
    geoms1 = gpd.GeoSeries(gdf1.loc[idx1_sub].geometry.values, crs=gdf1.crs)
    geoms2 = gpd.GeoSeries(gdf2.loc[idx2_sub].geometry.values, crs=gdf2.crs)

    # Execute the expensive geometric intersection engine and fetch the resulting area in square meters.
    inter_areas = geoms1.intersection(geoms2).area.values
    df_sub = pd.DataFrame(
        {'basin1': idx1_sub, 'basin2': idx2_sub, 'overlap_area': inter_areas}
    )

    # To satisfy expected machine learning layouts later, we dynamically recreate the A->B and B->A
    # symmetry we skipped earlier. We simply reflect the rows by renaming the columns and stacking them.
    df_sym = df_sub[df_sub['basin1'] != df_sub['basin2']].copy()
    df_sym = df_sym.rename(columns={'basin1': 'basin2', 'basin2': 'basin1'})
    overlap_df = pd.concat([df_sub, df_sym], ignore_index=True)
  else:
    # If we have two separate dataframes (e.g. comparing model inference basins to training basins),
    # the symmetry trick does not work, so we just run intersections for every identified pair.
    geoms1 = gpd.GeoSeries(gdf1.loc[idx1].geometry.values, crs=gdf1.crs)
    geoms2 = gpd.GeoSeries(gdf2.loc[idx2].geometry.values, crs=gdf2.crs)

    # Execute the intersection calculation.
    inter_areas = geoms1.intersection(geoms2).area.values
    overlap_df = pd.DataFrame(
        {'basin1': idx1, 'basin2': idx2, 'overlap_area': inter_areas}
    )

  # If a calculated interaction resulted in zero area (or microscopically small), discard it.
  overlap_df = overlap_df[overlap_df['overlap_area'] > 1e-6]

  print(
      f"[{time.strftime('%H:%M:%S')}] Geometric calculation finished in"
      f' {time.time() - calc_start:.2f} seconds.'
  )
  return overlap_df


def run_or_load_global_overlaps(all_train_gdf, calc_new_train_overlaps):
  """Computes overlaps from scratch or loads them directly from disk.

  Because computing overlaps between every possible pair of training basins can
  take hours depending on the dataset size, this function will first check if
  it is allowed to skip the computation and just load previously saved results
  from a file. If it must compute them from scratch, it evaluates the overlap
  areas and aggressively saves the output back to disk to speed up the next run.

  Args:
      all_train_gdf (gpd.GeoDataFrame): The spatial geometries associated with
        all valid training basins.
      calc_new_train_overlaps (bool): A boolean flag denoting whether or not the
        script should force recalculation natively (True), or if it is allowed
        to load from disk (False).

  Returns:
      pd.DataFrame: The resulting dataframe outlining precisely which basin ID
        overlaps with which, and the total square meter area of that
        intersection.
  """
  # If we have requested or been forced to recalculate everything from scratch...
  if calc_new_train_overlaps:
    step_start = time.time()

    # Instruct our geometry calculator to compute all pairwise overlaps for the training DataFrame.
    train_overlap_df = compute_all_overlaps(all_train_gdf)
    print(
        f"[{time.strftime('%H:%M:%S')}] train_overlap_df computed. Shape:"
        f' {train_overlap_df.shape}'
    )

    print(
        f"[{time.strftime('%H:%M:%S')}] Saving pairwise overlaps to"
        f' {file_paths.TRAIN_PAIRWISE_OVERLAPS_PATH}'
    )

    # Safely write the resulting pandas dataframe to disk as a CSV so we can bypass this next time.
    with gfile.Open(file_paths.TRAIN_PAIRWISE_OVERLAPS_PATH, 'w') as f:
      train_overlap_df.to_csv(f, index=False)
    print(
        f"[{time.strftime('%H:%M:%S')}] Step 3 completed in"
        f' {time.time() - step_start:.2f} seconds.'
    )
  else:
    # If recalculation is not required, we can simply read the past history from the disk as a CSV.
    print(
        f"[{time.strftime('%H:%M:%S')}] Loading cached pairwise overlaps from"
        f' {file_paths.TRAIN_PAIRWISE_OVERLAPS_PATH}'
    )
    t0_load_overlaps = time.time()

    # Open the file and parse it natively into a pandas dataframe.
    with gfile.Open(file_paths.TRAIN_PAIRWISE_OVERLAPS_PATH, 'r') as f:
      train_overlap_df = pd.read_csv(f)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded train_overlap_df in"
        f' {time.time() - t0_load_overlaps:.2f}s. Shape:'
        f' {train_overlap_df.shape}'
    )

  return train_overlap_df


def verify_global_overlaps(train_overlap_df, basin_areas):
  """Plots the geometric areas to verify overlaps calculated successfully.

  This function plots the distribution of pairwise overlaps against the primary
  distribution of total basin areas. This visual check ensures that the
  geometric
  intersection engine produced sensible numbers (e.g., overlaps should be
  smaller
  than or roughly similar in scale to whole basins), and gives engineers
  confidence
  the calculations didn't fail silently.

  Args:
      train_overlap_df (pd.DataFrame): The dataframe containing the paired
        combinations of basins and their intersecting square meter area.
      basin_areas (pd.Series): A mapped series indicating the total area of each
        basin.

  Returns:
      None: This function solely invokes matplotlib to draw a graph to the
      screen.
  """
  # Initialize the plotting layout for a clean, readable chart.
  sns.set_theme(style='whitegrid', palette='muted')
  plt.figure(figsize=(10, 6))

  # We filter out instances where a basin overlaps itself, because a basin is guaranteed to
  # 100% overlap itself, which would distort what we are actually trying to evaluate (basin-to-other-basin).
  actual_overlaps = train_overlap_df[
      train_overlap_df['basin1'] != train_overlap_df['basin2']
  ]

  # Plot a histogram representing how large the geometric intersections trend to be.
  sns.histplot(
      actual_overlaps['overlap_area'],
      bins=30,
      kde=True,
      color='teal',
      log_scale=True,
      label='Actual Overlap Areas',
      alpha=0.6,
  )

  # Layer a second histogram over it representing the raw area distribution of original basins.
  sns.histplot(
      basin_areas,
      bins=30,
      kde=True,
      color='orange',
      log_scale=True,
      label='Total Basin Areas',
      alpha=0.4,
  )

  # Finalize labels, title, and display the plot to the user viewing the console/notebook.
  plt.title(
      'Distribution of Basin Areas vs. Actual Overlaps (Train)',
      fontsize=15,
      fontweight='bold',
  )
  plt.xlabel('Area (m$^2$)', fontsize=12)
  plt.ylabel('Frequency', fontsize=12)
  plt.legend()
  plt.tight_layout()
  plt.show()


def calculate_total_upstream_coverage(
    infer_ids: list,
    tt_df: pd.DataFrame,
    ti_df: pd.DataFrame,
    basin_areas: pd.Series,
    containment_threshold: float = 0.98,
) -> pd.DataFrame:
  """Calculates predictive upstream geospatial features for inference basins.

  Determines how thoroughly a particular downstream inference basin has its upstream 
  drainage areas covered by known, gauged training basins. If an inference basin 
  contains several nested (child inside parent) upstream training basins, we 
  isolate only the largest ones ("maximal train IDs") so that we don't accidentally 
  count overlapping square meterage twice.

  Args:
      infer_ids (list): The complete list of target geometries (basins) we want to 
        calculate features for.
      tt_df (pd.DataFrame): Training-to-Training overlaps, outlining which known 
        training basins sit inside other known training basins.
      ti_df (pd.DataFrame): Training-to-Inference overlaps, mapping which known 
        training basins intersect our target inference basins.
      basin_areas (pd.Series): The precalculated dictionary mapping basin IDs to 
        their geographic area in square meters.
      containment_threshold (float, optional): Area fraction required to officially 
        declare one basin fully "contained" by another. Defaults to 0.98.

  Returns:
      pd.DataFrame: A DataFrame identifying three newly calculated numerical columns 
        for each evaluated target basin: 'total_upstream_coverage', 
        'max_upstream_coverage', and 'upstream_gauge_count'.
  """
  # We establish a containment map of which training basin contains which other 
  # training basin. This allows us to exclude highly redundant nested sub-basins.
  containment_dict = {}
  if not tt_df.empty:
    # First, append the physical areas of both basins in the pair...
    child_areas = tt_df['basin1'].map(basin_areas).values
    parent_areas = tt_df['basin2'].map(basin_areas).values

    # Determine if basin 1 area is mostly inside basin 2 area using the threshold condition.
    is_contained = (
        tt_df['overlap_area'].values / child_areas
    ) >= containment_threshold
    
    # Establish a hierarchy: True parents must logically have larger areas than children.
    is_strictly_larger = parent_areas > child_areas
    
    # In extremely rare cases where geometries are exactly the same footprint, simply pick a tie-breaker.
    is_tie_breaker = (parent_areas == child_areas) & (
        tt_df['basin2'].values > tt_df['basin1'].values
    )

    # Compile all valid parent-child relationships where the child lives inside the parent.
    valid = is_contained & (is_strictly_larger | is_tie_breaker)
    contained_df = tt_df[valid]
    
    # Group the pairs up and save them in our lookup dictionary mapping for fast reference.
    containment_dict = (
        contained_df.groupby('basin1')['basin2'].apply(set).to_dict()
    )

  # Filter the Training-to-Inference dataset down exclusively to "Upstream" dynamics.
  # A training basin is considered an "Upstream" component of the target inference basin 
  # if its physical area is smaller than the target inference basin's total area.
  up_ti_df = ti_df.copy()
  up_ti_df['area1'] = up_ti_df['basin1'].map(basin_areas)
  up_ti_df['area2'] = up_ti_df['basin2'].map(basin_areas)
  up_ti_df = up_ti_df[up_ti_df['area2'] < up_ti_df['area1']]

  # Prepare a dictionary structure loaded with zeroes for our baseline state.
  results_dict = {
      infer_id: {
          'total_upstream_coverage': 0.0,
          'max_upstream_coverage': 0.0,
          'upstream_gauge_count': 0,
      }
      for infer_id in infer_ids
  }

  # For every distinct inference basin, inspect all potential upstream sub-basins overlapping it.
  for infer_id, group in tqdm(up_ti_df.groupby('basin1'), desc="Upstream coverage", leave=False):
    active_train = set(group['basin2'])
    maximal_train_ids = set()

    for t_id in active_train:
      # Use our dictionary from earlier to check if this training basin is swallowed by a bigger one.
      parents = containment_dict.get(t_id, set())
      
      # If this basin has NO overlapping parents in this same specific target, it is an independent, 
      # "maximal" entity. We can count its total area without worrying about double-counting.
      if not (parents & active_train):
        maximal_train_ids.add(t_id)

    # Having isolated the maximal components, just add their independent overlap areas together.
    maximal_overlap_sum = group[group['basin2'].isin(maximal_train_ids)][
        'overlap_area'
    ].sum()
    
    # Calculate coverage by dividing overlapping area by total possible inference basin area.
    fraction = maximal_overlap_sum / basin_areas.loc[infer_id]
    
    # Calculate the largest single upstream continuous contributor.
    max_fraction = group['overlap_area'].max() / basin_areas.loc[infer_id]

    # Assign calculated metrics back to the dictionary, capped cleanly at 1.0 (100%).
    results_dict[infer_id]['total_upstream_coverage'] = min(fraction, 1.0)
    results_dict[infer_id]['max_upstream_coverage'] = min(max_fraction, 1.0)
    results_dict[infer_id]['upstream_gauge_count'] = len(maximal_train_ids)

  return pd.DataFrame.from_dict(results_dict, orient='index')


def calculate_max_downstream_coverage(
    infer_ids: list, ti_df: pd.DataFrame, basin_areas: pd.Series
) -> pd.Series:
  """Calculates how much downstream territory is mapped by our training dataset.

  We want to know if our target inference basin is eventually draining into a 
  much larger basin system that the machine learning model has already studied 
  during training. This function determines which intersection partners are 
  geographically larger (thus implying they are downstream entities) and reports 
  what percentage of that downstream environment our target basin represents.

  Args:
      infer_ids (list): A list strings identifying the specific basins we are targeting.
      ti_df (pd.DataFrame): The dataframe mapping overlaps between target basins 
        and available training basins. 
      basin_areas (pd.Series): The precalculated dictionary mapping basin IDs to 
        their geographic area in square meters.

  Returns:
      pd.Series: A pandas Series identifying a single numerical score 
        ('max_downstream_coverage') between 0.0 and 1.0 for each target ID.
  """
  # If there is no overlap data whatsoever, everybody gets a clean score of 0.
  if ti_df.empty:
    return pd.Series(0.0, index=infer_ids, name='max_downstream_coverage')

  # Create a working copy of our overlaps specifically targeting downstream dynamics.
  down_ti_df = ti_df.copy()
  
  # Inject the physical area metric for both participants into the dataframe.
  down_ti_df['area1'] = down_ti_df['basin1'].map(basin_areas)
  down_ti_df['area2'] = down_ti_df['basin2'].map(basin_areas)

  # An overlapping basin is only practically 'downstream' of our target basin 
  # if its total physical drainage area footprint is strictly larger than our target's.
  is_downstream = down_ti_df['area2'] >= down_ti_df['area1']
  
  # Shrink our frame down to exactly those downstream relationships.
  down_df = down_ti_df[is_downstream].copy()

  # If none of the overlapping partners were downstream, just return 0 scores.
  if down_df.empty:
    return pd.Series(0.0, index=infer_ids, name='max_downstream_coverage')

  # We define 'downstream structure fraction' as how much of the giant downstream 
  # entity's total area is occupied by our localized target basin's overlap.
  down_df['downstream_frac'] = down_df['overlap_area'] / down_df['area2']
  
  # Sort all candidate downstream relationships for a specific target basin, 
  # and only keep the single largest fraction score.
  max_down = down_df.groupby('basin1')['downstream_frac'].max()

  # Apply the calculated value series back onto original complete list of IDs to 
  # maintain shape consistency. Any IDs with no matches properly receive a padding of 0.0.
  return max_down.reindex(infer_ids, fill_value=0.0).rename(
      'max_downstream_coverage'
  )


def generate_fold_aware_ml_features(
    fold_gauges,
    all_train_gauges,
    train_overlap_df,
    basin_areas,
    calc_new_train_features,
):
  """Computes targeted sub-features for testing without cross-contamination.

  Iterates explicitly through the designated cross-validation fold parameters to 
  guarantee that generated features strictly respect data leakage boundaries. It 
  achieves this by temporarily masking out overlapping basins depending on which 
  fold the model is currently stepping through.

  Args:
      fold_gauges (dict): A dictionary explicitly labeling which string basin IDs are 
        'train' and which are 'test' for any given fold.
      all_train_gauges (list): The list of all unique string IDs present in the 
        active environment overall.
      train_overlap_df (pd.DataFrame): The massive dataset defining every geometric 
        overlap linking basins to other basins across the entire footprint.
      basin_areas (pd.Series): The precalculated dictionary mapping basin IDs to 
        their geographic area in square meters.
      calc_new_train_features (bool): Determines if the function needs to process 
        the overlaps strictly from scratch or if it is allowed to load past cached 
        features from disk storage.

  Returns:
      pd.DataFrame: A final, fully assembled DataFrame aggregating the extracted 
        coverage scores tied explicitly to fold identification boundaries.
  """
  print(
      f"\n[{time.strftime('%H:%M:%S')}] --- Starting Part 1: Step 5 (Fold-Aware"
      ' Feature Generation) ---'
  )

  # Check if we were commanded to rerun the calculation from square one.
  if calc_new_train_features:
    step_start = time.time()
    total_coverages_list = []
    
    # Store list of IDs inside a Python Set data structure to make looking up items blazing fast.
    all_train_gauges_set = set(all_train_gauges)

    # Loop carefully over every cross-validation test fold definition.
    for fold in tqdm(fold_gauges, desc='Processing Folds'):
      
      # Validate the list to absolutely guarantee we never attempt to target an ID we don't have.
      valid_train = [
          tid
          for tid in fold_gauges[fold]['train']
          if tid in all_train_gauges_set
      ]
      valid_test = [
          tid
          for tid in fold_gauges[fold]['test']
          if tid in all_train_gauges_set
      ]

      # We slice out the combinations mapping exactly our testing gauges back to our training gauges.
      ti_mask = train_overlap_df['basin1'].isin(valid_test) & train_overlap_df[
          'basin2'
      ].isin(valid_train)
      fold_ti_df = train_overlap_df[ti_mask].copy()

      # We similarly slice out combinations mapping our training gauges to other training gauges.
      tt_mask = train_overlap_df['basin1'].isin(valid_train) & train_overlap_df[
          'basin2'
      ].isin(valid_train)
      fold_tt_df = train_overlap_df[
          tt_mask & (train_overlap_df['basin1'] != train_overlap_df['basin2'])
      ].copy()

      # Run the isolated datasets through our specialized geographic feature estimators.
      fold_up_df = calculate_total_upstream_coverage(
          infer_ids=valid_test,
          tt_df=fold_tt_df,
          ti_df=fold_ti_df,
          basin_areas=basin_areas,
      )
      fold_down_series = calculate_max_downstream_coverage(
          infer_ids=valid_test, ti_df=fold_ti_df, basin_areas=basin_areas
      )

      # Combine the upstream and downstream datasets together seamlessly...
      fold_df = fold_up_df.copy()
      fold_df['max_downstream_coverage'] = fold_down_series
      fold_df = fold_df.reset_index().rename(columns={'index': 'gauge_id'})
      
      # Bind the explicit fold identifier back onto the row so the ML model remembers.
      fold_df['fold'] = fold
      total_coverages_list.append(fold_df)

    # Convert the array stack of individual fold datasets into one massive unified Pandas table.
    all_folds_coverage_df = pd.concat(total_coverages_list, ignore_index=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] all_folds_coverage_df shape:"
        f' {all_folds_coverage_df.shape}'
    )

    print(
        f"[{time.strftime('%H:%M:%S')}] Saving ML Features to"
        f' {file_paths.TRAIN_ML_FEATURES_PATH}'
    )
    
    # Save a complete CSV back to the Cloud network so we can skip this loop if doing it again later.
    with gfile.Open(file_paths.TRAIN_ML_FEATURES_PATH, 'w') as f:
      all_folds_coverage_df.to_csv(f, index=False)

    print(
        f"[{time.strftime('%H:%M:%S')}] Step 5 completed in"
        f' {time.time() - step_start:.2f} seconds.'
    )
  else:
    # Read the features natively out from the past cached file backup to save immense compute time.
    print(
        f"[{time.strftime('%H:%M:%S')}] Loading cached ML Features from"
        f' {file_paths.TRAIN_ML_FEATURES_PATH}'
    )
    t0_load_ml = time.time()
    with gfile.Open(file_paths.TRAIN_ML_FEATURES_PATH, 'r') as f:
      all_folds_coverage_df = pd.read_csv(f)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded cached ML Features in"
        f' {time.time() - t0_load_ml:.2f}s. Shape:'
        f' {all_folds_coverage_df.shape}'
    )

  return all_folds_coverage_df


def plot_ml_feature_distributions_training(all_folds_coverage_df):
  """Uses graph plots to visualize the distributions of the modeled features.

  Drawing graphical histograms for the calculated ML feature columns allows engineers 
  to verify if the generated coverage distributions make logical intuitive sense before 
  we begin passing it blindly to testing loops. Extreme outliers or completely flat 
  graphs can flag hidden logical bugs right away.

  Args:
      all_folds_coverage_df (pd.DataFrame): The completed geometric feature extraction 
        scores tied to cross-validation sets.

  Returns:
      None: Solely pushes visual information to the terminal or graphical notebook view.
  """
  print(
      f"\n[{time.strftime('%H:%M:%S')}] --- Starting Part 1: Step 6 (Plotting"
      ' Features) ---'
  )
  
  # Set up a grid representing four vertical stacked visual panels, which allows 
  # us to compare multiple metrics quickly in one glance.
  fig, axes = plt.subplots(4, 1, figsize=(10, 24))

  # Panel 1: Graphing how many basins simply have a 'Total Upstream Coverage'.
  sns.histplot(
      all_folds_coverage_df['total_upstream_coverage'],
      bins=25,
      kde=True,
      color='teal',
      ax=axes[0],
      alpha=0.6,
  )
  axes[0].set_title(
      'Total Upstream Coverage Distribution (Train)',
      fontsize=14,
      fontweight='bold',
  )
  axes[0].set_xlabel('Coverage Fraction', fontsize=12)
  axes[0].set_ylabel('Frequency', fontsize=12)


  # Panel 2: Graphing the maximum singular continuous contributor score distributions.
  sns.histplot(
      all_folds_coverage_df['max_upstream_coverage'],
      bins=25,
      kde=True,
      color='mediumseagreen',
      ax=axes[1],
      alpha=0.6,
  )
  axes[1].set_title(
      'Max Upstream Coverage Distribution (Train)',
      fontsize=14,
      fontweight='bold',
  )
  axes[1].set_xlabel('Coverage Fraction', fontsize=12)
  axes[1].set_ylabel('Frequency', fontsize=12)

  # Panel 3: Graphing how many upstream components form the coverage geometry. 
  # This uses a discrete layout (bar mode without curved lines) since basin count is whole integers.
  sns.histplot(
      all_folds_coverage_df['upstream_gauge_count'],
      bins=15,
      kde=False,
      color='purple',
      ax=axes[2],
      discrete=True,
  )
  axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
  axes[2].set_title(
      'Upstream Gauge Count Distribution (Train)',
      fontsize=14,
      fontweight='bold',
  )
  axes[2].set_xlabel('Number of Upstream Gauges', fontsize=12)
  axes[2].set_ylabel('Frequency', fontsize=12)

  # Panel 4: Graphing how much downstream structure is typically captured.
  sns.histplot(
      all_folds_coverage_df['max_downstream_coverage'],
      bins=25,
      kde=True,
      color='salmon',
      ax=axes[3],
      alpha=0.6,
  )
  axes[3].set_title(
      'Max Downstream Coverage Distribution (Train)',
      fontsize=14,
      fontweight='bold',
  )
  axes[3].set_xlabel('Coverage Fraction', fontsize=12)
  axes[3].set_ylabel('Frequency', fontsize=12)

  # Tighten empty layout borders and display it finally.
  plt.tight_layout()
  plt.show()


def process_hybas_continents(
    train_overlap_df, n_sample_basins, recalculate_hybas_overlaps
):
  """Computes overlaps and machine learning features for all global basins continent-by-continent.

  To support global rainfall predictions, we must map how every single target 
  basin anywhere on Earth (HyBAS) overlaps with our physical training dataset (Caravan). 
  Since the entire planet's geometric footprint is too large to fit in memory at 
  once, we loop through the target basins continent-by-continent and selectively 
  cache outputs to disk.

  Args:
      train_overlap_df (pd.DataFrame): Training-to-Training overlap pairs.
      n_sample_basins (int): Subsampling target for faster regional testing.
      recalculate_hybas_overlaps (bool): Force clearing cache and computing from scratch.
  """
  # Find all continent-level shapefiles in our defined folder.
  hybas_files = gfile.Glob(
      os.path.join(file_paths.HYBAS_POLYGONS_DIR, '*.geojson')
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Found {len(hybas_files)} HyBAS continent"
      ' GeoJSON files.'
  )

  # Check if we should forcibly execute recalculations of the underlying Caravan geometries.
  calc_new_caravan_overlaps = recalculate_hybas_overlaps or not gfile.Exists(
      file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH
  )

  # Keep lazy-load pointers to Caravan variables so we only load them into RAM if truly needed.
  caravan_gdf = None
  caravan_areas = None

  def ensure_caravan_loaded(cgdf, careas):
    """Lazy loader to prevent memory spiking unless absolutely necessary."""
    if cgdf is None:
      print(f"[{time.strftime('%H:%M:%S')}] Loading Caravan polygons...")
      t0_load_caravan = time.time()
      cgdf = load_gdf_from_cns(file_paths.CARAVAN_POLYGONS_FILE)
      print(
          f"[{time.strftime('%H:%M:%S')}] Loaded {len(cgdf)} Caravan polygons"
          f' in {time.time() - t0_load_caravan:.2f}s. Shape: {cgdf.shape}'
      )
      careas = cgdf.geometry.area
    return cgdf, careas

  caravan_tt_df = None
  
  # Calculate or Load Caravan-to-Caravan self-overlaps.
  if calc_new_caravan_overlaps:
    # We must load the caravan shapes to compute geometric intersections.
    caravan_gdf, caravan_areas = ensure_caravan_loaded(
        caravan_gdf, caravan_areas
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] Calculating Caravan-to-Caravan"
        ' overlaps...'
    )
    step_start = time.time()
    caravan_tt_df = compute_all_overlaps(caravan_gdf)
    print(
        f"[{time.strftime('%H:%M:%S')}] caravan_tt_df shape:"
        f' {caravan_tt_df.shape}'
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] Saving Caravan pairwise overlaps to"
        f' {file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH}'
    )
    
    # Save the output explicitly so we don't do this expensive task twice.
    with gfile.Open(file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH, 'w') as f:
      caravan_tt_df.to_csv(f, index=False)
    print(
        f"[{time.strftime('%H:%M:%S')}] Caravan overlaps completed in"
        f' {time.time() - step_start:.2f} seconds.'
    )
  else:
    # Check if a past run exists on the disk. If it exists, retrieve it natively.
    if gfile.Exists(file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH):
      print(
          f"[{time.strftime('%H:%M:%S')}] Loading cached Caravan pairwise"
          f' overlaps from {file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH}'
      )
      t0_load_caravan_over = time.time()
      with gfile.Open(file_paths.CARAVAN_PAIRWISE_OVERLAPS_PATH, 'r') as f:
        caravan_tt_df = pd.read_csv(f)
      print(
          f"[{time.strftime('%H:%M:%S')}] Loaded caravan_tt_df in"
          f' {time.time() - t0_load_caravan_over:.2f}s. Shape:'
          f' {caravan_tt_df.shape}'
      )

  # Prepare the training overlap dataframe explicitly stripped of self-loops (Basin A = Basin A).
  global_tt_df = train_overlap_df[
      train_overlap_df['basin1'] != train_overlap_df['basin2']
  ].copy()
  
  # Start the Continent-by-Continent loop to handle data at a manageable scale.
  for hybas_file in hybas_files:
    # Extract the continent string tag from the standardized filename structure.
    filename = os.path.basename(hybas_file)
    continent = filename.split('_')[1].upper()

    # Route the output save files explicitly according to the identified continent.
    pairwise_out_path = os.path.join(
        file_paths.HYBAS_COVERAGE_DIR,
        f'infer_pairwise_overlaps_{continent}.csv',
    )
    features_out_path = os.path.join(
        file_paths.HYBAS_COVERAGE_DIR, f'infer_ml_features_{continent}.csv'
    )

    # Validate individually per continent if its cached files already exist on disk.
    calc_new_pairwise = recalculate_hybas_overlaps or not gfile.Exists(
        pairwise_out_path
    )
    calc_new_features = recalculate_hybas_overlaps or not gfile.Exists(
        features_out_path
    )

    # If the user did not force recalculation and the cached data exists, fast-forward to the next continent.
    if not (calc_new_pairwise or calc_new_features):
      print(
          f"\n[{time.strftime('%H:%M:%S')}] Skipping Continent: {continent} -"
          ' Outputs already exist and are cached.'
      )
      continue

    print(f"\n[{time.strftime('%H:%M:%S')}] {'-'*50}")
    print(f"[{time.strftime('%H:%M:%S')}] PROCESSING CONTINENT: {continent}")
    print(f"[{time.strftime('%H:%M:%S')}] {'-'*50}")
    cont_start = time.time()

    # Step A: Attempt to load the raw geometric shapes representing the basins in this continent.
    print(
        f"[{time.strftime('%H:%M:%S')}] Step A: Loading HyBAS dataset for"
        f' {continent}...'
    )
    t0_load_hybas = time.time()
    hybas_gdf = load_gdf_from_cns(hybas_file)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded hybas_gdf for {continent} in"
        f' {time.time() - t0_load_hybas:.2f}s. Shape: {hybas_gdf.shape}'
    )

    # Note: file_paths.HYBAS_POLYGONS_DIR already contains the pre-built full
    # drainage polygons (hybas_*_lev12_v1c.geojson), so we do not need to
    # reconstruct them from subbasins here.

    # Sample a small fraction of basins if debugging to speed up pipeline validation.
    if (
        n_sample_basins
        and n_sample_basins > 0
        and n_sample_basins < len(hybas_gdf)
    ):
      print(
          f"[{time.strftime('%H:%M:%S')}] Notice: Subsampling {continent}"
          f' inference set to {n_sample_basins} basins for testing.'
      )
      sample_ids = random.sample(list(hybas_gdf.index), n_sample_basins)
      hybas_gdf = hybas_gdf.loc[sample_ids]

    # Pre-cache the square physical areas since area calculations are heavily used later.
    hybas_areas = hybas_gdf.geometry.area

    # Check if the global training bounds need to be formally pulled into RAM at this point.
    caravan_gdf, caravan_areas = ensure_caravan_loaded(
        caravan_gdf, caravan_areas
    )

    # Fuse target area dictionary lookup tables with training area dictionary tables seamlessly.
    combined_areas = pd.concat([caravan_areas, hybas_areas])

    # Strip duplicate entries strictly protecting the first occurrence it saw.
    combined_areas = combined_areas[
        ~combined_areas.index.duplicated(keep='first')
    ]

    # Step B: Overlaps Generation
    if calc_new_pairwise:
      print(
          f"[{time.strftime('%H:%M:%S')}] Step B: Calculating Pairwise Overlaps"
          ' against global Caravan basins...'
      )
      step_start = time.time()

      # Send the two geometry frames to our optimized overlap area calculator function.
      infer_overlap_df = compute_all_overlaps(gdf1=hybas_gdf, gdf2=caravan_gdf)
      print(
          f"[{time.strftime('%H:%M:%S')}] infer_overlap_df shape:"
          f' {infer_overlap_df.shape}'
      )

      # Store this continent's geometric intersections specifically tied to its continent flag.
      print(
          f"[{time.strftime('%H:%M:%S')}] Saving {continent} pairwise"
          ' overlaps...'
      )
      with gfile.Open(pairwise_out_path, 'w') as f:
        infer_overlap_df.to_csv(f, index=False)

      print(
          f"[{time.strftime('%H:%M:%S')}] Overlaps step completed in"
          f' {time.time() - step_start:.2f} seconds.'
      )
    else:
      # If previous runs already finished part B, simply read it from the network.
      print(
          f"[{time.strftime('%H:%M:%S')}] Step B: Loading cached pairwise"
          f' overlaps for {continent}...'
      )
      t0_load_infer_over = time.time()
      with gfile.Open(pairwise_out_path, 'r') as f:
        infer_overlap_df = pd.read_csv(f)
      print(
          f"[{time.strftime('%H:%M:%S')}] Loaded infer_overlap_df in"
          f' {time.time() - t0_load_infer_over:.2f}s. Shape:'
          f' {infer_overlap_df.shape}'
      )

    # Step C: Feature Generation
    if calc_new_features:
      print(
          f"[{time.strftime('%H:%M:%S')}] Step C: Calculating ML Features for"
          f' {continent}...'
      )
      step_start = time.time()

      # Derive exactly how much upstream territory is covered by gauged counterparts.
      infer_up_df = calculate_total_upstream_coverage(
          infer_ids=list(hybas_gdf.index),
          tt_df=caravan_tt_df,
          ti_df=infer_overlap_df,
          basin_areas=combined_areas,
      )

      # Derive downstream overlap scores determining what proportion is enveloped by known gauges.
      infer_down_series = calculate_max_downstream_coverage(
          infer_ids=list(hybas_gdf.index),
          ti_df=infer_overlap_df,
          basin_areas=combined_areas,
      )

      # Consolidate metrics and finalize a dataframe representing this isolated continent region.
      infer_df = infer_up_df.copy()
      infer_df['max_downstream_coverage'] = infer_down_series
      infer_df = infer_df.reset_index().rename(columns={'index': 'hybas_id'})
      infer_df['continent'] = continent
      print(
          f"[{time.strftime('%H:%M:%S')}] infer_df created. Shape:"
          f' {infer_df.shape}'
      )

      # Flush resulting scores back to cloud storage targeting our target folder path.
      print(f"[{time.strftime('%H:%M:%S')}] Saving {continent} ML features...")
      with gfile.Open(features_out_path, 'w') as f:
        infer_df.to_csv(f, index=True)

      print(
          f"[{time.strftime('%H:%M:%S')}] Feature step completed in"
          f' {time.time() - step_start:.2f} seconds.'
      )

    # Congratulate the program and close out the specific continent.
    print(
        f"[{time.strftime('%H:%M:%S')}] --- Successfully finished {continent}"
        f' processing in {time.time() - cont_start:.2f} seconds. ---'
    )

  # Notify completion for all geographic chunks in the global file. 
  print(f"\n[{time.strftime('%H:%M:%S')}] {'='*60}")
  print(
      f"[{time.strftime('%H:%M:%S')}] ALL CONTINENTS PROCESSED SUCCESSFULLY."
      ' PIPELINE COMPLETE.'
  )
  print(f"[{time.strftime('%H:%M:%S')}] {'='*60}")


def plot_ml_feature_distributions_inference():
  """Generates visual comparisons for features strictly tied to inference data.

  Reads all completed feature extraction results for ungauged spatial targets 
  (the inference domains) and graphs visual distribution charts. By visually comparing 
  these generated bounds and peaks to the training plots produced earlier, engineering 
  teams can double-check whether the inference landscape is vastly skewed from the 
  original mathematical training environments.

  Returns:
      None: Solely executes matplotlib logic writing pictures to standard output view.
  """
  # Rapidly ingest the feature CSV datasets covering the inference groups from storage.
  infer_feature_files = gfile.Glob(
      os.path.join(file_paths.HYBAS_COVERAGE_DIR, 'infer_ml_features_*.csv')
  )

  # Validate the pipeline output folder wasn't completely empty before continuing.
  if infer_feature_files:
    infer_df_list = []
    # Read each continent data structure...
    for fpath in infer_feature_files:
      with gfile.Open(fpath, 'r') as f:
        infer_df_list.append(pd.read_csv(f))

    # Compile the disparate continent arrays dynamically into one global array structure.
    all_infer_features_df = pd.concat(infer_df_list, ignore_index=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] all_infer_features_df created. Shape:"
        f' {all_infer_features_df.shape}'
    )

    # Scaffold the chart array layout holding four overlapping metric visualizations.
    fig, axes = plt.subplots(4, 1, figsize=(10, 24))

    # Panel 1: Graphing Total Upstream Coverage frequencies.
    sns.histplot(
        all_infer_features_df['total_upstream_coverage'],
        bins=25,
        kde=True,
        color='teal',
        ax=axes[0],
        alpha=0.6,
    )
    axes[0].set_title(
        'Total Upstream Coverage Distribution (HyBAS)',
        fontsize=14,
        fontweight='bold',
    )
    axes[0].set_xlabel('Coverage Fraction', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    # Panel 2: Graphing the maximum singular continuous contributor score distributions.
    sns.histplot(
        all_infer_features_df['max_upstream_coverage'],
        bins=25,
        kde=True,
        color='mediumseagreen',
        ax=axes[1],
        alpha=0.6,
    )
    axes[1].set_title(
        'Max Upstream Coverage Distribution (HyBAS)',
        fontsize=14,
        fontweight='bold',
    )
    axes[1].set_xlabel('Coverage Fraction', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    # Panel 3: Graphing upstream component connections specifically using whole-integer tick limits.
    sns.histplot(
        all_infer_features_df['upstream_gauge_count'],
        bins=15,
        kde=False,
        color='purple',
        ax=axes[2],
        discrete=True,
    )
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set_title(
        'Upstream Gauge Count Distribution (HyBAS)',
        fontsize=14,
        fontweight='bold',
    )
    axes[2].set_xlabel('Number of Upstream Gauges', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)

    # Panel 4: Graphing how heavily downstream basins integrate components.
    sns.histplot(
        all_infer_features_df['max_downstream_coverage'],
        bins=25,
        kde=True,
        color='salmon',
        ax=axes[3],
        alpha=0.6,
    )
    axes[3].set_title(
        'Max Downstream Coverage Distribution (HyBAS)',
        fontsize=14,
        fontweight='bold',
    )
    axes[3].set_xlabel('Coverage Fraction', fontsize=12)
    axes[3].set_ylabel('Frequency', fontsize=12)

    # Constrain margins explicitly and output visualization.
    plt.tight_layout()
    plt.show()
  else:
    # Safely gracefully alert standard out when graphs simply cannot generate.
    print(
        f"[{time.strftime('%H:%M:%S')}] No inference feature files found to"
        ' plot.'
    )


def main(argv):
  """Main execution function to run the spatial overlap fractions pipeline."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  N_SAMPLE_BASINS = FLAGS.n_sample_basins
  RECALCULATE_TRAIN_OVERLAPS = FLAGS.recalculate_train_overlaps
  RECALCULATE_HYBAS_OVERLAPS = FLAGS.recalculate_hybas_overlaps
  # Dynamic Caching Logic: Automatically calculate if the files don't exist OR if recalculation is forced
  CALC_NEW_TRAIN_OVERLAPS = RECALCULATE_TRAIN_OVERLAPS or not gfile.Exists(
      file_paths.TRAIN_PAIRWISE_OVERLAPS_PATH
  )
  CALC_NEW_TRAIN_FEATURES = RECALCULATE_TRAIN_OVERLAPS or not gfile.Exists(
      file_paths.TRAIN_ML_FEATURES_PATH
  )

  file_paths.ensure_directories_exist()
  print(f"[{time.strftime('%H:%M:%S')}] Configuration Loaded.")
  print(
      f"[{time.strftime('%H:%M:%S')}] Primary Output Directory:"
      f' {file_paths.COVERAGE_DIR}'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Inference Output Directory:"
      f' {file_paths.HYBAS_COVERAGE_DIR}'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Need to calculate new TRAIN overlaps?"
      f' {CALC_NEW_TRAIN_OVERLAPS}'
  )
  print(
      f"[{time.strftime('%H:%M:%S')}] Need to calculate new TRAIN ML features?"
      f' {CALC_NEW_TRAIN_FEATURES}'
  )

  fold_gauges, all_train_gauges = load_cross_validation_splits()

  all_train_gdf, all_train_gauges, basin_areas = (
      load_and_clean_training_geometries(all_train_gauges, N_SAMPLE_BASINS)
  )

  train_overlap_df = run_or_load_global_overlaps(
      all_train_gdf, CALC_NEW_TRAIN_OVERLAPS
  )

  verify_global_overlaps(train_overlap_df, basin_areas)

  all_folds_coverage_df = generate_fold_aware_ml_features(
      fold_gauges,
      all_train_gauges,
      train_overlap_df,
      basin_areas,
      CALC_NEW_TRAIN_FEATURES,
  )

  plot_ml_feature_distributions_training(all_folds_coverage_df)

  process_hybas_continents(
      train_overlap_df, N_SAMPLE_BASINS, RECALCULATE_HYBAS_OVERLAPS
  )

  plot_ml_feature_distributions_inference()


if __name__ == '__main__':
  app.run(main)
