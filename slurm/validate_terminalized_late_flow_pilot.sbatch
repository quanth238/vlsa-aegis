#!/usr/bin/env bash
#SBATCH --job-name=vlsa-terminal-flow-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-terminal-flow-pilot}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-terminalized-late-flow-pilot}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "validation requires Slurm" >&2; exit 1; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || { echo "source commit differs" >&2; exit 1; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 1; }
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
PRODUCER_RESULT=${PRODUCER_RESULT:-$RUN_ROOT/producer/result.json}
REPLAY_RESULT=${REPLAY_RESULT:-$RUN_ROOT/replay/result.json}
[[ -f "$PRODUCER_RESULT" && -f "$REPLAY_RESULT" ]] || { echo "pilot result missing" >&2; exit 1; }
[[ ! -e "$RUN_ROOT/validation.json" ]] || { echo "immutable validation exists" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_terminalized_late_flow_pilot.py \
  --producer "$PRODUCER_RESULT" \
  --replay "$REPLAY_RESULT" \
  --output "$RUN_ROOT/validation.json"
