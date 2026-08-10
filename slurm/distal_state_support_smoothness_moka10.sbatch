#!/usr/bin/env bash
#SBATCH --job-name=vlsa-support-audit
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-state-support-smoothness}
DATASET_ROOT=${DATASET_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-affine-coefficient-dataset/affine-coeff-data-20260810b}
MULTI_ROOT=${MULTI_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-multi-region-affine-oracle/multi-region-affine-20260810a}
LEARNED_ROOT=${LEARNED_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-region-aware-mlp/region-aware-mlp-20260810a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_state_support_smoothness_moka10.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "support audit requires exactly one H100 allocation"
[[ -x "$AEGIS_PYTHON" ]] || die "evaluation Python is missing"
for path in \
  "$DATASET_ROOT/dataset.json" \
  "$MULTI_ROOT/result.json" "$MULTI_ROOT/validation.json" \
  "$LEARNED_ROOT/learned-result.json" "$LEARNED_ROOT/learned-validation.json"
do
  [[ -f "$path" && ! -L "$path" ]] || die "immutable support-audit input missing: $path"
done
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 5dacdc8e5eb843db3dcb27ecd6747059fcd3cd785c10a54404d162a46fc340ec ]] || die "support-audit config hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() {
  code=$?
  trap - EXIT INT TERM
  if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then
    temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE"
  fi
  return "$code"
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy; print({"numpy":numpy.__version__})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_state_support_smoothness -v >>"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_state_support_smoothness_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --dataset "$DATASET_ROOT/dataset.json" \
  --multi-region-result "$MULTI_ROOT/result.json" \
  --multi-region-validation "$MULTI_ROOT/validation.json" \
  --learned-result "$LEARNED_ROOT/learned-result.json" \
  --learned-validation "$LEARNED_ROOT/learned-validation.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_state_support_smoothness_moka10.py \
  --result "$RUN_ROOT/result.json" --config "$CONFIG" \
  --dataset "$DATASET_ROOT/dataset.json" \
  --multi-region-result "$MULTI_ROOT/result.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"
