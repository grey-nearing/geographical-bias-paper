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

"""Common Alerting Protocol (CAP) operational flood warning tier utilities.

Implements country-level classifications from Justin Ginnetti's CAP gap
analysis, providing both:
1. A 4-tier operational taxonomy (for Extended Data tables and figures):
   - Tier 1: Comprehensive CAP Alerting
   - Tier 2: Proportionality Gap (Under-Alerting)
   - Tier 3: Non-CAP Domestic Architecture
   - Tier 4: Complete Systemic Gap (No CAP)
2. A binary operational grouping (for main-text headline results in Table 1
   and Figure 5):
   - With System (Tier 1 + Tier 3)
   - Without Effective System (Tier 2 + Tier 4)
"""

import pandas as pd

TIER_1_LABEL = 'Tier 1: Comprehensive CAP Alerting'
TIER_2_LABEL = 'Tier 2: Proportionality Gap (Under-Alerting)'
TIER_3_LABEL = 'Tier 3: Non-CAP Domestic Architecture'
TIER_4_LABEL = 'Tier 4: Complete Systemic Gap (No CAP)'

CAP_TIER_ORDER = [
    TIER_1_LABEL,
    TIER_3_LABEL,
    TIER_2_LABEL,
    TIER_4_LABEL,
]

BINARY_WITH_SYSTEM_LABEL = 'With Effective Warning System'
BINARY_WITHOUT_SYSTEM_LABEL = 'Without Effective Warning System'

CAP_BINARY_ORDER = [
    BINARY_WITH_SYSTEM_LABEL,
    BINARY_WITHOUT_SYSTEM_LABEL,
]

# Countries with established non-CAP domestic flood warning architectures
NON_CAP_DOMESTIC_COUNTRIES = frozenset({
    'Japan',
    'Peru',
    'Malaysia',
    'Nepal',
    'Vietnam',
    'Viet Nam',
    'Bangladesh',
    'Pakistan',
    'Thailand',
    'Bolivia',
    'Guatemala',
    'Uruguay',
    'Rwanda',
    'South Africa',
    'Sri Lanka',
    'Dominican Republic',
    'Dominican Rep.',
    'Chile',
    'Uganda',
    'Angola',
    'Colombia',
})

# Countries with comprehensive operational CAP flood alerting
TIER_1_COMPREHENSIVE_COUNTRIES = frozenset({
    'United States',
    'United States of America',
    'China',
    'India',
    'Brazil',
    'Germany',
    'Argentina',
    'Canada',
    'Spain',
    'Russia',
    'Russian Federation',
    'Norway',
    'United Kingdom',
    'France',
    'Italy',
    'Switzerland',
    'Austria',
    'Netherlands',
    'Belgium',
    'Sweden',
    'Finland',
    'Denmark',
    'Poland',
    'Czechia',
    'Czech Republic',
    'Slovakia',
    'Slovenia',
    'Croatia',
    'Hungary',
    'Romania',
    'Portugal',
    'Greece',
    'Estonia',
    'Israel',
    'New Zealand',
    'Australia',
    'Philippines',
    'Mexico',
    'Paraguay',
    'Belize',
    'Maldives',
    'Tanzania',
    'Ukraine',
    'Benin',
    'Chad',
})

# Countries with CAP adoption but severe proportionality / under-alerting gaps
TIER_2_PROPORTIONALITY_COUNTRIES = frozenset({
    'Nigeria',
    'Democratic Republic of the Congo',
    'Democratic Republic of Congo',
    'Dem. Rep. Congo',
    'Congo DR',
    'Mali',
    'Congo',
    'Republic of the Congo',
    'Liberia',
    'Guinea',
    'Senegal',
    'Ecuador',
    'South Sudan',
    'S. Sudan',
    'Yemen',
    'Indonesia',
    'Algeria',
    'Jordan',
    'Serbia',
    'Bosnia and Herzegovina',
    'Bosnia and Herz.',
    'North Macedonia',
    'Macedonia',
    'Montenegro',
    'Panama',
    'Costa Rica',
    'Trinidad and Tobago',
    'Solomon Islands',
    'Solomon Is.',
    'Vanuatu',
    'Lesotho',
    'Guinea-Bissau',
    'Togo',
    'Burundi',
    'Mongolia',
    'Eswatini',
    'eSwatini',
    'Bahrain',
})


def assign_cap_tier(country_name: str) -> str:
  """Maps a country name to its 4-tier operational flood warning status."""
  if pd.isna(country_name):
    return TIER_4_LABEL
  c_str = str(country_name).strip()
  if c_str in NON_CAP_DOMESTIC_COUNTRIES:
    return TIER_3_LABEL
  if c_str in TIER_1_COMPREHENSIVE_COUNTRIES:
    return TIER_1_LABEL
  if c_str in TIER_2_PROPORTIONALITY_COUNTRIES:
    return TIER_2_LABEL
  return TIER_4_LABEL


def has_effective_flood_warning_system(country_name: str) -> bool:
  """Returns True if the country is in Tier 1 (Comprehensive CAP) or Tier 3 (Non-CAP Domestic)."""
  tier = assign_cap_tier(country_name)
  return tier in (TIER_1_LABEL, TIER_3_LABEL)


def assign_cap_binary_group(country_name: str) -> str:
  """Maps a country name to its binary operational flood warning status."""
  if has_effective_flood_warning_system(country_name):
    return BINARY_WITH_SYSTEM_LABEL
  return BINARY_WITHOUT_SYSTEM_LABEL
