#!/usr/bin/env bash
# End-to-end non-Fathom reproduction script for:
#   "What Fraction of the World's Watersheds Are Ungauged?"
#
# Reproduces all non-Fathom analyses, figures (Figs. 1, 2, 3, 4, 6, S1–S5),
# tables (Table 1, Tables S1–S5), and runs automated quantitative verification
# purely from public code and public data (Zenodo 10.5281/zenodo.21416986 +
# public upstream hydrofabrics). Does NOT retrain the 300 OpenHydroNet LSTMs
# (starts from archived OHN CV evaluation CSVs) and does NOT require licensed
# Fathom v3 flood-hazard maps.
#
# Usage:
#   export GEO_BIAS_DATA_DIR=/path/to/data   # default: ./data
#   bash run_reproduce_non_fathom.sh [--download-zenodo] [--skip-stage1] [--skip-stage3]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEO_BIAS_REPO_DIR="${GEO_BIAS_REPO_DIR:-$SCRIPT_DIR}"
export GEO_BIAS_DATA_DIR="${GEO_BIAS_DATA_DIR:-$SCRIPT_DIR/data}"
export GEO_BIAS_THIRD_PARTY_DIR="${GEO_BIAS_THIRD_PARTY_DIR:-$GEO_BIAS_DATA_DIR}"
export GEO_BIAS_OUTPUT_DIR="${GEO_BIAS_OUTPUT_DIR:-$GEO_BIAS_DATA_DIR}"
export GLOBAL_UNGAUGEDNESS_DATA_DIR="${GLOBAL_UNGAUGEDNESS_DATA_DIR:-$SCRIPT_DIR/01_hydrofabric_ungaugedness/data}"
export GLOBAL_UNGAUGEDNESS_RESULTS_DIR="${GLOBAL_UNGAUGEDNESS_RESULTS_DIR:-$SCRIPT_DIR/01_hydrofabric_ungaugedness/results}"

DOWNLOAD_ZENODO=0
SKIP_STAGE1=0
SKIP_STAGE3=0
ZENODO_CONCEPT_ID="${ZENODO_CONCEPT_ID:-21416986}"

for arg in "$@"; do
  case "$arg" in
    --download-zenodo) DOWNLOAD_ZENODO=1 ;;
    --skip-stage1)     SKIP_STAGE1=1 ;;
    --skip-stage3)     SKIP_STAGE3=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

NCPU="$(nproc 2>/dev/null || echo 8)"
THREADS_PER_WORKER=4
META_WORKERS=$(( NCPU / THREADS_PER_WORKER ))
if (( META_WORKERS < 1 )); then META_WORKERS=1; fi
if (( META_WORKERS > 12 )); then META_WORKERS=12; fi

echo "========================================================================"
echo " Geographical Bias Paper — Full Non-Fathom Reproduction (v3.0.0)"
echo "========================================================================"
echo " Repo dir     : $GEO_BIAS_REPO_DIR"
echo " Data dir     : $GEO_BIAS_DATA_DIR"
echo " Hydro data   : $GLOBAL_UNGAUGEDNESS_DATA_DIR"
echo " Hydro results: $GLOBAL_UNGAUGEDNESS_RESULTS_DIR"
echo " CPUs         : $NCPU (meta-model workers=$META_WORKERS x $THREADS_PER_WORKER threads)"
echo "========================================================================"

mkdir -p "$GEO_BIAS_DATA_DIR"

# ------------------------------------------------------------------------------
# Optional Step 0: Download & extract public Zenodo v3.0.0 data archives
# ------------------------------------------------------------------------------
if (( DOWNLOAD_ZENODO == 1 )); then
  echo ""
  echo "[Step 0] Fetching latest public Zenodo archives for concept DOI 10.5281/zenodo.${ZENODO_CONCEPT_ID}..."
  python3 - "$ZENODO_CONCEPT_ID" "$GEO_BIAS_DATA_DIR" << 'PYEOF'
import json, subprocess, sys, urllib.request
from pathlib import Path

concept_id = sys.argv[1]
data_dir = Path(sys.argv[2])
api_url = f"https://zenodo.org/api/records/{concept_id}"
req = urllib.request.Request(api_url, headers={"Accept": "application/json", "User-Agent": "geo-bias-repro/3.0"})
with urllib.request.urlopen(req, timeout=60) as resp:
    meta = json.loads(resp.read().decode("utf-8"))

# Resolve latest published version if concept record redirects via links.latest
latest_url = meta.get("links", {}).get("latest")
if latest_url and latest_url != api_url:
    req2 = urllib.request.Request(latest_url, headers={"Accept": "application/json", "User-Agent": "geo-bias-repro/3.0"})
    with urllib.request.urlopen(req2, timeout=60) as resp2:
        meta = json.loads(resp2.read().decode("utf-8"))

print(f"Resolved Zenodo record id={meta.get('id')} version={meta.get('metadata', {}).get('version')}", flush=True)
wanted = {
    "01_global_gauge_catalog.tar.gz",
    "02_hydroatlas_worldpop_and_auxiliary.tar.gz",
    "03_ohn_cv_metrics_and_splits.tar.gz",
}
dl_dir = data_dir / "_zenodo_downloads"
dl_dir.mkdir(parents=True, exist_ok=True)
found = set()
for fobj in meta.get("files", []):
    key = fobj["key"]
    if key not in wanted:
        continue
    found.add(key)
    dst = dl_dir / key
    url = fobj["links"]["self"]
    exp_size = int(fobj.get("size", 0))
    if not (dst.exists() and (exp_size == 0 or dst.stat().st_size == exp_size)):
        print(f"  Downloading {key} ({exp_size / 1e6:.1f} MB)...", flush=True)
        subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-o", str(dst), url], check=True)
    else:
        print(f"  [skip] {key} already downloaded", flush=True)
    print(f"  Extracting {key} -> {data_dir}...", flush=True)
    subprocess.run(["tar", "-xzf", str(dst), "-C", str(data_dir)], check=True)

missing = wanted - found
if missing:
    raise RuntimeError(f"Missing expected Zenodo archives: {sorted(missing)}")
print("Zenodo download & extraction complete!", flush=True)
PYEOF
fi

# ------------------------------------------------------------------------------
# Unit Tests: Mathematical identities for Eq. 4, 1D river & bifurcating DAG
# ------------------------------------------------------------------------------
echo ""
echo "[Tests] Running hydrofabric mathematical identity unit tests..."
pytest -v "$GEO_BIAS_REPO_DIR/01_hydrofabric_ungaugedness/tests/test_mathematical_identities.py"

# ------------------------------------------------------------------------------
# Stage 1: Multi-Hydrofabric Global Ungaugedness (17 runs + 4-fabric OOD)
# ------------------------------------------------------------------------------
if (( SKIP_STAGE1 == 0 )); then
  echo ""
  echo "[Stage 1/4] Staging gauges & WorldPop, downloading public hydrofabrics, and running all 17 benchmarks..."
  H_SCRIPTS="$GEO_BIAS_REPO_DIR/01_hydrofabric_ungaugedness/scripts"
  python3 "$H_SCRIPTS/00_stage_gauges_and_population.py"
  python3 "$H_SCRIPTS/01_download_hydrosheds.py"
  python3 "$H_SCRIPTS/02_download_tdx_geoglows.py"
  python3 "$H_SCRIPTS/03_download_merit_and_grades.py"
  python3 "$H_SCRIPTS/04_download_grit.py"

  python3 "$H_SCRIPTS/11_run_hydrosheds.py"
  python3 "$H_SCRIPTS/12_run_tdx_geoglows.py"
  python3 "$H_SCRIPTS/13_run_merit_grades.py"
  python3 "$H_SCRIPTS/14_run_grit.py"
  python3 "$H_SCRIPTS/15_run_ohn_ood_4fabrics.py"
  python3 "$H_SCRIPTS/20_synthesize_and_plot.py"
else
  echo "[Stage 1/4] Skipped (--skip-stage1)."
fi

# ------------------------------------------------------------------------------
# Stage 3: 1,034,083-Basin Tier Features, 171 Meta-Models, Table S3 & SHAP
# ------------------------------------------------------------------------------
if (( SKIP_STAGE3 == 0 )); then
  echo ""
  echo "[Stage 3/4] Running 1,034,083-basin tier features, scale invariance, 171 meta-models, Table S3 & SHAP..."
  S3_DIR="$GEO_BIAS_REPO_DIR/03_skill_predictors_and_shap"

  python3 "$S3_DIR/stage_b1_tier_features_1034k.py" --core-tiers-only --threads "$NCPU"
  python3 "$S3_DIR/stage_b1a_scale_invariance_audit.py"
  python3 "$S3_DIR/stage_c_train_meta_models.py" --phase all --workers "$META_WORKERS" --threads "$THREADS_PER_WORKER"
  python3 "$S3_DIR/stage_d_counterfactuals_bootstrap_s16.py" --part table_s3 --threads "$NCPU"
  python3 "$S3_DIR/stage_e_global_kernelshap.py" --sample-size 1000 --k-bg 100 --pairs-per-bg 1
else
  echo "[Stage 3/4] Skipped (--skip-stage3)."
fi

# ------------------------------------------------------------------------------
# Stage 4: Main Figures, Supplementary Figures, Tables & Verification
# ------------------------------------------------------------------------------
echo ""
echo "[Stage 4/4] Generating public figures (Figs. 1–4, 6, S1–S5), tables (Table 1, S1–S5), and verifying..."
S4_DIR="$GEO_BIAS_REPO_DIR/04_figures_and_tables"
FIG_OUT="$GEO_BIAS_DATA_DIR/round2/figures"
TAB_OUT="$GEO_BIAS_DATA_DIR/round2/tables"
VER_OUT="$GEO_BIAS_DATA_DIR/round2/verification"

python3 "$S4_DIR/build_tables.py" --out-dir "$TAB_OUT"
python3 "$S4_DIR/generate_supp_figures.py" --out-dir "$FIG_OUT" --only figs1,figs2,figs3,figs4,figs5
python3 "$S4_DIR/generate_main_figures.py" --out-dir "$FIG_OUT" --only fig1,fig2,fig3,fig4,fig6
python3 "$S4_DIR/verify_reproduction.py" \
  --fig-dir "$FIG_OUT" \
  --table-dir "$TAB_OUT" \
  --shap-dir "$GEO_BIAS_DATA_DIR/round2/shap" \
  --out-dir "$VER_OUT"

echo ""
echo "========================================================================"
echo " SUCCESS: Full non-Fathom reproduction verified!"
echo " Figures     : $FIG_OUT"
echo " Tables      : $TAB_OUT"
echo " Verification: $VER_OUT/verification_report.md"
echo "========================================================================"
