#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-smooth-attr
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
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
VALIDATOR_GIT_COMMIT=${VALIDATOR_GIT_COMMIT:?VALIDATOR_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
SOURCE_RUN_ID=${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-smooth-field-attribution-e05}

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "validation requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || die "allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_GIT_COMMIT" ]] || die "validator source differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"
SOURCE_ROOT=$EXPERIMENT_ROOT/$SOURCE_RUN_ID
[[ -f "$SOURCE_ROOT/result.json" ]] || die "source result missing"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable validation root exists"
"$AEGIS_PYTHON" "$REMOTE_REPO/scripts/validate_distal_smooth_field_attribution_e05.py" \
  --result "$SOURCE_ROOT/result.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --validator-commit "$VALIDATOR_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"

