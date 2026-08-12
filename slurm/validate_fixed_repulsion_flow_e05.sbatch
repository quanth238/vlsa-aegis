#!/usr/bin/env bash
#SBATCH --job-name=verify-repulse-flow
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_VALIDATOR_COMMIT=${EXPECTED_VALIDATOR_COMMIT:?EXPECTED_VALIDATOR_COMMIT is required}
EXPECTED_PRODUCER_COMMIT=${EXPECTED_PRODUCER_COMMIT:?EXPECTED_PRODUCER_COMMIT is required}
RESULT=${RESULT:?RESULT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "validation requires Slurm"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_VALIDATOR_COMMIT" ]] || die "validator source differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "validator source is dirty"
[[ -f "$RESULT" && ! -L "$RESULT" ]] || die "result is missing"
[[ ! -e "$OUTPUT" ]] || die "validation output exists"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_fixed_repulsion_flow_e05.py \
  --result "$RESULT" \
  --expected-producer-commit "$EXPECTED_PRODUCER_COMMIT" \
  --output "$OUTPUT"
