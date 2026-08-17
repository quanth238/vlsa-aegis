#!/usr/bin/env bash
#SBATCH --job-name=vlsa-top5-verify
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
OUTPUT_DIR=${OUTPUT_DIR:?}; TABLE1_ROOT=${TABLE1_ROOT:?}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_distal_top5_exact_verified_pilot.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ -n "${SLURM_JOB_ID:-}" && $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
case_id=$(
  PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO "$AEGIS_PYTHON" - "$CONFIG_PATH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["cases"][int(sys.argv[2])]["case_id"])
PY
)
output="$OUTPUT_DIR/$case_id.json"; test ! -e "$output" || exit 2
run_root="$OUTPUT_DIR/.runtime-$case_id"; test ! -e "$run_root" || exit 2
mkdir -p "$OUTPUT_DIR" "$run_root"
export LIBERO_CONFIG_PATH="$run_root/libero-config"; mkdir "$LIBERO_CONFIG_PATH"
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_top5_exact_verified_pilot.py' -v
"$AEGIS_PYTHON" -m scripts.evaluate_distal_top5_exact_verified_pilot \
  --repo-root "$REMOTE_REPO" --table1-root "$TABLE1_ROOT" \
  --config "$CONFIG_PATH" --case-index "$SLURM_ARRAY_TASK_ID" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --run-root "$run_root" \
  --output "$output"
