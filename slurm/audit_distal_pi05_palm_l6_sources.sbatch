#!/usr/bin/env bash
#SBATCH --job-name=vlsa-pi05-palm-l6-src
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}; OUTPUT=${OUTPUT:?}
TABLE1_ROOT=${TABLE1_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_distal_pi05_palm_l6_source_audit.v1.json}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test -d "$TABLE1_ROOT"; test ! -e "$OUTPUT" || exit 2; mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_pi05_palm_l6_source_selection.py' -v
"$AEGIS_PYTHON" scripts/audit_distal_pi05_palm_l6_sources.py \
  --repo-root "$REMOTE_REPO" --table1-root "$TABLE1_ROOT" --config "$CONFIG_PATH" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
