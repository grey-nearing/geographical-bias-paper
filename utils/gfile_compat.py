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

"""Standard-library compatibility layer for file I/O operations."""

import glob
import os
import shutil


def Open(path: str, mode: str = 'r'):
  """Opens a file using standard Python open()."""
  encoding = 'utf-8' if ('b' not in mode) else None
  return open(path, mode, encoding=encoding)


GFile = Open


def Exists(path: str) -> bool:
  """Returns True if path exists."""
  return os.path.exists(path)


def MakeDirs(path: str) -> None:
  """Creates directories recursively."""
  os.makedirs(path, exist_ok=True)


def Glob(pattern: str) -> list[str]:
  """Returns sorted list of paths matching pattern."""
  return sorted(glob.glob(pattern))


def ListDir(path: str) -> list[str]:
  """Returns list of entries in directory."""
  return os.listdir(path)


def Copy(src: str, dst: str, overwrite: bool = False) -> None:
  """Copies a file from src to dst."""
  if not overwrite and os.path.exists(dst):
    raise FileExistsError(f'Destination already exists: {dst}')
  shutil.copy2(src, dst)


def Remove(path: str) -> None:
  """Removes a file."""
  os.remove(path)
