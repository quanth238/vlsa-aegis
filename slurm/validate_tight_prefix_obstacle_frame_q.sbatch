#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-obstacle-frame-q
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
ACCEPTED_TRAINING_COMMIT=${ACCEPTED_TRAINING_COMMIT:?ACCEPTED_TRAINING_COMMIT is required}
PRODUCER=${PRODUCER:?PRODUCER is required}
REPLAY=${REPLAY:?REPLAY is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_obstacle_frame_q.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
[[ -f "$PRODUCER" && -f "$REPLAY" && ! -e "$OUTPUT" ]] || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests \
  -p 'test_tight_prefix_obstacle_frame_q.py' -v
"$AEGIS_PYTHON" scripts/validate_tight_prefix_obstacle_frame_q.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --producer "$PRODUCER" --replay "$REPLAY" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --accepted-training-commit "$ACCEPTED_TRAINING_COMMIT" --output "$OUTPUT"
