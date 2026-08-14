#!/usr/bin/env bash
#SBATCH --job-name=vlsa-row-contact
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
CONFIG=${CONFIG:?CONFIG is required}
POPULATION_SUMMARY=${POPULATION_SUMMARY:?POPULATION_SUMMARY is required}
TARGETED_SUMMARY=${TARGETED_SUMMARY:?TARGETED_SUMMARY is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
PYTHON=${PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "row-contact audit requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test ! -e "$OUTPUT" || { echo "row-contact audit output exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$PYTHON" -m unittest discover -s tests -p 'test_distal_row_contact_alignment.py' -v
"$PYTHON" scripts/audit_distal_row_contact_alignment.py \
  --repo-root "$REMOTE_REPO" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --config "$CONFIG" --summary "$POPULATION_SUMMARY" \
  --summary "$TARGETED_SUMMARY" --output "$OUTPUT"
