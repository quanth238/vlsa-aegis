#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-pncbf-value
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-pncbf-value}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
SOURCE_GIT_COMMIT=${SOURCE_GIT_COMMIT:-$EXPECTED_GIT_COMMIT}
SOURCE_RUN_ID=${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}
VALIDATION_RUN_ID=${VALIDATION_RUN_ID:?VALIDATION_RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-pncbf-policy-value-e05}
SOURCE_RESULT=$EXPERIMENT_ROOT/$SOURCE_RUN_ID/result.json
VALIDATION_ROOT=$EXPERIMENT_ROOT/$VALIDATION_RUN_ID
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || { echo "source commit differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 2; }
[[ -f "$SOURCE_RESULT" ]] || { echo "source result missing" >&2; exit 2; }
mkdir "$VALIDATION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$VALIDATION_ROOT/libero-config
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
"$AEGIS_PYTHON" scripts/validate_distal_pncbf_backup_oracle_e05.py \
  --repo-root "$REMOTE_REPO" --result "$SOURCE_RESULT" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --archived /mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json \
  --geometry-config "$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json" \
  --experiment-config "$REMOTE_REPO/configs/vlsa_distal_pncbf_policy_value_e05.v1.json" \
  --expected-commit "$SOURCE_GIT_COMMIT" \
  --validator-commit "$EXPECTED_GIT_COMMIT" \
  --output "$VALIDATION_ROOT/validation.json"
