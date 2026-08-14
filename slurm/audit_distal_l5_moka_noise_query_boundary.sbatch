#!/usr/bin/env bash
#SBATCH --job-name=vlsa-l5-moka-boundary
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --array=0-3%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
RUN_ID=${RUN_ID:?}
DISCOVERY_ROOT=${DISCOVERY_ROOT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-l5-moka-noise-boundary}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  echo "requires exactly one H100" >&2; exit 2;
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]]
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID/case-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$EXPERIMENT_ROOT/$RUN_ID"
mkdir "$RUN_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_l5_moka_noise_boundary -v \
  >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/audit_distal_l5_moka_noise_query_boundary.py \
  --repo-root "$REMOTE_REPO" \
  --population-manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --boundary-config "$REMOTE_REPO/configs/vlsa_distal_l5_moka_noise_boundary.v1.json" \
  --boundary-manifest "$REMOTE_REPO/manifests/vlsa_distal_l5_moka_noise_boundary.v1.jsonl" \
  --discovery-root "$DISCOVERY_ROOT" --case-index "$SLURM_ARRAY_TASK_ID" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/result.json"
