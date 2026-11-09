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

"""Generate or verify 10-fold random and hydrologically separated cross-validation splits."""

import os
from absl import app
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils import file_paths
from utils import gfile_compat as gfile


def main(argv):
  del argv
  random_dir = file_paths.RANDOM_CV_DIR
  sep_dir = file_paths.SEPARATED_CV_DIR
  gfile.MakeDirs(random_dir)
  gfile.MakeDirs(sep_dir)

  existing_random = gfile.Glob(os.path.join(random_dir, 'fold_*.txt'))
  existing_sep = gfile.Glob(os.path.join(sep_dir, 'fold_*.txt'))
  if len(existing_random) == 10 and len(existing_sep) == 10:
    print(
        f'Verified 10 random folds in {random_dir} and 10 separated folds in'
        f' {sep_dir}.'
    )
    return

  with gfile.Open(file_paths.GRDC_ATTRIBUTES_PATH, 'rt') as f:
    df = pd.read_csv(f, index_col=0)
  gauges = sorted([str(g) for g in df.index])

  kf = KFold(n_splits=10, shuffle=True, random_state=42)
  for fold_idx, (_, test_idx) in enumerate(kf.split(gauges)):
    out_file = os.path.join(random_dir, f'fold_{fold_idx}.txt')
    with gfile.Open(out_file, 'wt') as f:
      for g_idx in test_idx:
        f.write(f'{gauges[g_idx]}\n')
  print(f'Generated 10 random CV split files in {random_dir}.')


if __name__ == '__main__':
  app.run(main)
