# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading and folder creation utilities."""

import os
import pathlib


def create_remote_folder_if_necessary(folder: pathlib.Path):
  """Creates a directory if it doesn't already exist."""
  if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)
