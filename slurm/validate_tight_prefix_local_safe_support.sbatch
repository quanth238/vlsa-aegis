#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-tight-local-support
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
ACCEPTED_RESULT_COMMIT=${ACCEPTED_RESULT_COMMIT:?ACCEPTED_RESULT_COMMIT is required}
PRODUCER_RESULT=${PRODUCER_RESULT:?PRODUCER_RESULT is required}
REPLAY_RESULT=${REPLAY_RESULT:?REPLAY_RESULT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_local_safe_support.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
[[ ! -e "$OUTPUT" ]] || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_tight_prefix_local_safe_support -v \
  >"$(dirname "$OUTPUT")/validation-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_tight_prefix_local_safe_support.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --producer-result "$PRODUCER_RESULT" --replay-result "$REPLAY_RESULT" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --accepted-result-commit "$ACCEPTED_RESULT_COMMIT" --output "$OUTPUT"
