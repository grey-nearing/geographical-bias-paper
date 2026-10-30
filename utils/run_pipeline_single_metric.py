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

"""Single-Metric Pipeline Orchestrator (Steps 5 & 6).

Executes Step 5 (Cross-Validation) and Step 6 (Global Training and Inference)
sequentially for a single target hydrological metric.
"""

import argparse
import os
import subprocess
import sys
import time
from absl import app
from utils import file_paths


def run_script(script_name: str, metric: str) -> bool:
  """Runs a single python script as a subprocess for the specified metric."""
  env = os.environ.copy()
  env["GEOGRAPHIC_BIAS_METRIC"] = metric

  script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  script_path = os.path.join(script_dir, script_name)
  env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")

  python_exe = sys.executable or "python3"
  cmd = [python_exe, script_path, f"--metric={metric}"]

  print(f"\n{'='*80}")
  print(f"Running {script_name} for metric: {metric}")
  print(f"Command: {' '.join(cmd)}")
  print(f"{'='*80}\n")

  start_time = time.time()
  try:
    subprocess.run(cmd, env=env, check=True, text=True)
    duration = time.time() - start_time
    print(f"\nSUCCESS: {script_name} completed in {duration:.2f} seconds.")
    return True
  except subprocess.CalledProcessError as e:
    print(f"\nERROR: {script_name} failed with exit code {e.returncode}.")
    return False


def main(argv):
  """Main execution function to parse arguments and run the sequential steps."""
  parser = argparse.ArgumentParser(
      description="Run Geographical Bias pipeline for a single metric."
  )
  parser.add_argument(
      "--metric",
      required=True,
      type=str,
      help="The metric to run (e.g. NSE, 2.0-year-2.0-window-recall).",
  )
  args, _ = parser.parse_known_args(argv[1:])
  metric = args.metric

  if metric not in file_paths.ALL_METRICS:
    print(
        f"ERROR: Metric '{metric}' is not valid. Must be one of"
        f" {file_paths.ALL_METRICS}"
    )
    sys.exit(1)

  scripts_to_run = [
      "5_model_cross_validation.py",
      "6_global_training_and_inference.py",
  ]
  for script in scripts_to_run:
    if not run_script(script, metric):
      sys.exit(1)


if __name__ == "__main__":
  app.run(main)
