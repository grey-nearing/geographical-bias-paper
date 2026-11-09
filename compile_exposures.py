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

"""Compiles sharded per-return-period exposure files into a single global CSV."""

import os
import time
from absl import app
from absl import flags
import pandas as pd
from utils import gfile_compat as gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_dir',
    os.path.join(os.environ.get('GEO_BIAS_DATA_DIR', os.path.join(os.getcwd(), 'data')), 'exposure_full_world'),
    'Directory containing the sharded exposure CSVs.',
)
flags.DEFINE_string(
    'output_file',
    os.path.join(os.environ.get('GEO_BIAS_DATA_DIR', os.path.join(os.getcwd(), 'data')), 'exposure', 'global_basin_exposure_by_return_period.csv'),
    'Output CSV file path.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  return_periods = [5, 10, 20, 50, 100]
  combined_df = None

  t0 = time.time()
  for T in return_periods:
    print(f"[{time.strftime('%H:%M:%S')}] Processing return period T = {T} years...")
    t_rp = time.time()
    
    # List all 10 partitions
    shards = []
    for i in range(10):
      shard_name = f"global_basin_exposure_undefended_{T}.csv-{i:05d}-of-00010"
      shard_path = os.path.join(FLAGS.input_dir, shard_name)
      shards.append(shard_path)
      
    # Read and concatenate shards
    dfs = []
    for shard in shards:
      if not gfile.Exists(shard):
        raise FileNotFoundError(f"Missing expected shard file: {shard}")
      with gfile.Open(shard, 'r') as f:
        df = pd.read_csv(f, usecols=['HYBAS_ID', 'exposed_population'])
        dfs.append(df)
    
    rp_df = pd.concat(dfs, ignore_index=True)
    rp_df.rename(columns={'exposed_population': f'exposed_pop_{T}', 'HYBAS_ID': 'hybas_id'}, inplace=True)
    
    # Merge with main dataframe
    if combined_df is None:
      combined_df = rp_df
    else:
      combined_df = pd.merge(combined_df, rp_df, on='hybas_id', how='outer')
      
    print(f"   -> Completed T={T} in {time.time() - t_rp:.2f}s. Shape: {combined_df.shape}")

  # Fill missing values with 0 (if any basin was missing in some return periods)
  combined_df.fillna(0.0, inplace=True)
  
  # Ensure the output directory exists
  out_dir = os.path.dirname(FLAGS.output_file)
  if not gfile.Exists(out_dir):
    gfile.MakeDirs(out_dir)

  print(f"[{time.strftime('%H:%M:%S')}] Saving consolidated exposures to {FLAGS.output_file}...")
  t_save = time.time()
  with gfile.Open(FLAGS.output_file, 'w') as f:
    combined_df.to_csv(f, index=False)
  print(f"   -> Saved in {time.time() - t_save:.2f}s. Total compiling time: {time.time() - t0:.2f}s")


if __name__ == '__main__':
  app.run(main)
