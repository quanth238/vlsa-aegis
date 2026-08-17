#!/usr/bin/env bash
#SBATCH --job-name=vlsa-terminal-task-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-terminal-task-success}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-terminalized-late-flow-task-success}
PRODUCER_RESULT=${PRODUCER_RESULT:-$EXPERIMENT_ROOT/$RUN_ID/producer/result.json}
REPLAY_RESULT=${REPLAY_RESULT:-$EXPERIMENT_ROOT/$RUN_ID/replay/result.json}

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "validation requires Slurm" >&2; exit 1; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || { echo "source commit differs" >&2; exit 1; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "source tree is dirty" >&2; exit 1; }
[[ -f "$PRODUCER_RESULT" && -f "$REPLAY_RESULT" ]] || { echo "task-success result missing" >&2; exit 1; }
[[ ! -e "$EXPERIMENT_ROOT/$RUN_ID/validation.json" ]] || { echo "immutable validation exists" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_terminalized_late_flow_task_success.py \
  --producer "$PRODUCER_RESULT" \
  --replay "$REPLAY_RESULT" \
  --output "$EXPERIMENT_ROOT/$RUN_ID/validation.json"
