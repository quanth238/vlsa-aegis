#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-active-search
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --array=0-3%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_PRODUCER_COMMIT=${EXPECTED_PRODUCER_COMMIT:?EXPECTED_PRODUCER_COMMIT is required}
EXPECTED_VALIDATOR_COMMIT=${EXPECTED_VALIDATOR_COMMIT:?EXPECTED_VALIDATOR_COMMIT is required}
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
COVERAGE_ROOT=${COVERAGE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-query-boundary-coverage/query-coverage-population-20260814a}
TABLE1_ROOT=${TABLE1_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "active search validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_VALIDATOR_COMMIT" ]] || { echo "validator source differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "validator source is dirty" >&2; exit 2; }
JOB_ROOT=$PRODUCER_ROOT/job-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
OUTPUT=$JOB_ROOT/validation.json
test -f "$JOB_ROOT/result.json" || { echo "producer result missing" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "validation exists" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$JOB_ROOT/validator-libero-config
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_active_boundary_search tests.test_distal_query_action_risk -v >"$JOB_ROOT/validation-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_distal_active_boundary_search.py \
  --repo-root "$REMOTE_REPO" --producer-root "$PRODUCER_ROOT" \
  --population-manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --coverage-root "$COVERAGE_ROOT" \
  --search-config "$REMOTE_REPO/configs/vlsa_distal_active_boundary_search.v1.json" \
  --base-config "$REMOTE_REPO/configs/vlsa_distal_query_action_risk_e05.v1.json" \
  --geometry-config "$REMOTE_REPO/configs/vlsa_distal_clean_action_risk_geometry.v4.json" \
  --table1-root "$TABLE1_ROOT" --array-index "$SLURM_ARRAY_TASK_ID" \
  --expected-producer-commit "$EXPECTED_PRODUCER_COMMIT" \
  --expected-validator-commit "$EXPECTED_VALIDATOR_COMMIT" --output "$OUTPUT"
