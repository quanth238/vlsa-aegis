#!/usr/bin/env bash
#SBATCH --job-name=vlsa-late-waypoint-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-late-flow-waypoint-search}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]]
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
ROOT=$EXPERIMENT_ROOT/$RUN_ID
[[ -f "$ROOT/producer/result.json" && -f "$ROOT/replica/result.json" ]]
mkdir "$ROOT/validation"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_late_flow_waypoint_search -v >"$ROOT/validation/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_late_flow_waypoint_search.py \
  --producer "$ROOT/producer/result.json" \
  --replica "$ROOT/replica/result.json" \
  --output "$ROOT/validation/validation.json"
