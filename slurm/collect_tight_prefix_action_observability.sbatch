#!/usr/bin/env bash
#SBATCH --job-name=vlsa-action-observe
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_action_observability.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "action-observability requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "action-observability CPU allocation differs"
[[ -x "$AEGIS_PYTHON" ]] || die "action-observability Python is unavailable"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"

case_id=$(
  PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO "$AEGIS_PYTHON" - "$CONFIG" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from pathlib import Path
from main.multilink_ellipsoid.tight_prefix_action_observability import load_config
print(load_config(Path(sys.argv[1]))["cases"][int(sys.argv[2])]["case_id"])
PY
)
mkdir -p "$OUTPUT_DIR"
output="$OUTPUT_DIR/$case_id.json"
[[ ! -e "$output" ]] || die "immutable action-observability case exists"
runtime=$(mktemp -d "$OUTPUT_DIR/.runtime-$case_id.XXXXXX")
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ $exit_code -ne 0 && ! -e "$runtime/runtime-failure.json" ]]; then
    printf '{"exit_code":%d,"scientific_result":false,"status":"apparatus_failure"}\n' "$exit_code" >"$runtime/runtime-failure.json"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export QUERY_RISK_CPU_CANARY=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$runtime/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"

cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_tight_prefix_action_observability -v \
  >"$runtime/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/collect_tight_prefix_action_observability.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$output"
