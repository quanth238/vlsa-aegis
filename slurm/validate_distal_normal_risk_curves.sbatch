#!/usr/bin/env bash
#SBATCH --job-name=vlsa-validate-normal-curve
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-l5-aegis-consistent}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
RUN_ID=${RUN_ID:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-normal-risk-curve}
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
case "${SLURMD_NODENAME:-}" in login*|login-restricted*) exit 2 ;; esac
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]]
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]]
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
for index in 00 01 02; do [[ -s "$RUN_ROOT/case-$index/result.json" ]]; done
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_distal_normal_risk_curves.py \
  --config "$REMOTE_REPO/configs/vlsa_distal_normal_risk_curve.v1.json" \
  --selection-manifest "$REMOTE_REPO/manifests/vlsa_distal_analytical_repulsion_generalization.v1.jsonl" \
  --result "$RUN_ROOT/case-00/result.json" \
  --result "$RUN_ROOT/case-01/result.json" \
  --result "$RUN_ROOT/case-02/result.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --summary "$RUN_ROOT/summary.json" \
  --validation "$RUN_ROOT/validation.json"
