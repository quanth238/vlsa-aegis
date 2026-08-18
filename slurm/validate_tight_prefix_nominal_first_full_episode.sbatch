#!/usr/bin/env bash
#SBATCH --job-name=vlsa-tight-nominal-val
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
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-tight-prefix-nominal-first-full-episode}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "nominal-first validation requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 ]] || die "expected exactly 2 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 8192 ]] || die "expected 8 GiB"
[[ -x "$AEGIS_PYTHON" ]] || die "Python environment missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree dirty"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
PRODUCER_RESULT=$RUN_ROOT/producer/result.json
REPLAY_RESULT=$RUN_ROOT/replay/result.json
VALIDATION_ROOT=$RUN_ROOT/validation
[[ -f "$PRODUCER_RESULT" && -f "$REPLAY_RESULT" ]] || die "validated inputs missing"
mkdir "$VALIDATION_ROOT" || die "immutable validation root exists"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_tight_prefix_nominal_first_full_episode -v >"$VALIDATION_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_tight_prefix_nominal_first_full_episode.py \
  --producer "$PRODUCER_RESULT" --replay "$REPLAY_RESULT" \
  --output "$VALIDATION_ROOT/validation.json"
