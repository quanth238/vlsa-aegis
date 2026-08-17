#!/usr/bin/env bash
#SBATCH --job-name=vlsa-tight-ee
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
OUTPUT_DIR=${OUTPUT_DIR:?}
VISUALIZATION_ROOT=${VISUALIZATION_ROOT:-}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_tight_ee_geometry_audit.v1.json}

gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ -n "${SLURM_JOB_ID:-}" && $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
case_id=$(
  PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO "$AEGIS_PYTHON" - "$CONFIG_PATH" "$SLURM_ARRAY_TASK_ID" "$REMOTE_REPO" <<'PY'
import sys
from pathlib import Path
from main.multilink_ellipsoid.tight_ee_geometry import load_config
config = load_config(Path(sys.argv[1]), repo_root=Path(sys.argv[3]))
print(config["cohort"]["cases"][int(sys.argv[2])]["case_id"])
PY
)
output="$OUTPUT_DIR/$case_id.json"
[[ ! -e "$output" ]] || exit 2
mkdir -p "$OUTPUT_DIR"
run_root="$OUTPUT_DIR/.runtime-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$run_root" ]] || exit 2
mkdir "$run_root"
export LIBERO_CONFIG_PATH="$run_root/libero-config"
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_tight_ee_geometry -v
arguments=(
  --repo-root "$REMOTE_REPO"
  --config "$CONFIG_PATH"
  --expected-commit "$EXPECTED_GIT_COMMIT"
  --case-index "$SLURM_ARRAY_TASK_ID"
  --output "$output"
)
if [[ -n "$VISUALIZATION_ROOT" ]]; then
  arguments+=(--visualization-root "$VISUALIZATION_ROOT")
fi
exec "$AEGIS_PYTHON" scripts/audit_tight_ee_geometry_case.py "${arguments[@]}"
