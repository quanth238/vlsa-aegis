#!/usr/bin/env bash
#SBATCH --job-name=vlsa-candidate-worker-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
SEQUENTIAL_RESULT=${SEQUENTIAL_RESULT:?}; PARALLEL_RESULT=${PARALLEL_RESULT:?}
OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_candidate_parallel.py' -v
"$AEGIS_PYTHON" -m scripts.validate_distal_candidate_worker_equivalence \
  --sequential "$SEQUENTIAL_RESULT" --parallel "$PARALLEL_RESULT" \
  --output "$OUTPUT"
