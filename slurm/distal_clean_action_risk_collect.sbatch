#!/usr/bin/env bash
#SBATCH --job-name=vlsa-clean-risk
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --array=0-17%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-clean-action-risk}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-clean-action-risk}
TABLE1_ROOT=${TABLE1_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
CANARY_STATE_LIMIT=${CANARY_STATE_LIMIT:-}
CANARY_PROPOSAL_LIMIT=${CANARY_PROPOSAL_LIMIT:-}
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "simulation requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID/case-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$EXPERIMENT_ROOT/$RUN_ID"
mkdir "$RUN_ROOT" || die "immutable case root exists"
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_clean_action_risk tests.test_distal_pncbf_pure_backup -v >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
extra_args=()
if [[ -n "$CANARY_STATE_LIMIT" || -n "$CANARY_PROPOSAL_LIMIT" ]]; then
  [[ "$CANARY_STATE_LIMIT" =~ ^[1-4]$ ]] || die "invalid canary state limit"
  [[ "$CANARY_PROPOSAL_LIMIT" =~ ^([1-9]|1[0-9]|2[0-6])$ ]] || die "invalid canary proposal limit"
  extra_args+=(--state-limit "$CANARY_STATE_LIMIT" --proposal-limit "$CANARY_PROPOSAL_LIMIT")
fi
"$AEGIS_PYTHON" scripts/collect_distal_clean_action_risk.py \
  --repo-root "$REMOTE_REPO" \
  --population-manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --selection-manifest "$REMOTE_REPO/manifests/vlsa_distal_clean_action_risk.v1.jsonl" \
  --experiment-config "$REMOTE_REPO/configs/vlsa_distal_clean_action_risk.v1.json" \
  --geometry-config "$REMOTE_REPO/configs/vlsa_distal_clean_action_risk_geometry.v4.json" \
  --table1-root "$TABLE1_ROOT" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/result.json" \
  "${extra_args[@]}"
