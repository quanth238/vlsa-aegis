#!/usr/bin/env bash
#SBATCH --job-name=vlsa-freeze-palm-l6
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
CONFIG=${CONFIG:?}; MANIFEST_OUTPUT=${MANIFEST_OUTPUT:?}; OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test ! -e "$MANIFEST_OUTPUT"; test ! -e "$OUTPUT" || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_pi05_palm_l6_cohort.py' -v
"$AEGIS_PYTHON" scripts/freeze_distal_pi05_palm_l6_cohort.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --manifest-output "$MANIFEST_OUTPUT" --output "$OUTPUT"
