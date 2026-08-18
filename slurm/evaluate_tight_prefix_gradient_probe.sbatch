#!/usr/bin/env bash
#SBATCH --job-name=vlsa-tight-grad-probe
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_gradient_probe.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "tight gradient-probe requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "tight gradient-probe CPU allocation differs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "tight gradient-probe memory differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "tight gradient-probe requires one H100"
[[ -x "$AEGIS_PYTHON" && -x "$OPENPI_PYTHON" ]] || die "tight gradient-probe Python is unavailable"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"

case_id=$(
  PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO "$AEGIS_PYTHON" - "$CONFIG" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from pathlib import Path
from main.multilink_ellipsoid.tight_prefix_gradient_probe import load_config
print(load_config(Path(sys.argv[1]))["cases"][int(sys.argv[2])]["case_id"])
PY
)
mkdir -p "$OUTPUT_DIR"
output="$OUTPUT_DIR/$case_id.json"
[[ ! -e "$output" ]] || die "immutable tight gradient-probe case exists"
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
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
"$AEGIS_PYTHON" -m unittest \
  tests.test_tight_prefix_gradient_probe \
  tests.test_tight_prefix_risk_dataset -v >"$runtime/preflight.txt" 2>&1
definitions="$runtime/gradient-definitions.json"
"$OPENPI_PYTHON" scripts/evaluate_tight_prefix_gradient_probe.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --case-index "$SLURM_ARRAY_TASK_ID" --mode generate \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$definitions"
"$AEGIS_PYTHON" scripts/evaluate_tight_prefix_gradient_probe.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --case-index "$SLURM_ARRAY_TASK_ID" --mode evaluate \
  --definitions "$definitions" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$output"
