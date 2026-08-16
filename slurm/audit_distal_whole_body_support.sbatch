#!/usr/bin/env bash
#SBATCH --job-name=vlsa-audit-wholebody-support
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
EXPECTED_ARTIFACT_COMMIT=${EXPECTED_ARTIFACT_COMMIT:?}
PRODUCER_DIR=${PRODUCER_DIR:?}; REPLAY_DIR=${REPLAY_DIR:?}
SOURCE_VALIDATION=${SOURCE_VALIDATION:?}; OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
AUDIT_CONFIG=${AUDIT_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_whole_body_support_audit.v1.json}
COHORT_CONFIG=${COHORT_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_whole_body_prospective_population.v1.json}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test -d "$PRODUCER_DIR"; test -d "$REPLAY_DIR"; test -f "$SOURCE_VALIDATION"
test ! -e "$OUTPUT" || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_whole_body_support_audit.py' -v
"$AEGIS_PYTHON" scripts/audit_distal_whole_body_support.py \
  --repo-root "$REMOTE_REPO" --audit-config "$AUDIT_CONFIG" \
  --cohort-config "$COHORT_CONFIG" --producer-dir "$PRODUCER_DIR" \
  --replay-dir "$REPLAY_DIR" --source-validation "$SOURCE_VALIDATION" \
  --expected-audit-commit "$EXPECTED_GIT_COMMIT" \
  --expected-artifact-commit "$EXPECTED_ARTIFACT_COMMIT" --output "$OUTPUT"
