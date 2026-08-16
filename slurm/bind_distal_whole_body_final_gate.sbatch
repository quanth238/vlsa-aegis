#!/usr/bin/env bash
#SBATCH --job-name=vlsa-bind-wb-final
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
MODE=${MODE:?}; OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test ! -e "$OUTPUT" || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_whole_body_final_bindings.py' -v
if [[ "$MODE" == combined-audit ]]; then
  BASE=${BASE:-$REMOTE_REPO/configs/vlsa_distal_whole_body_combined_audit.v1.json}
  TARGETED_COHORT=${TARGETED_COHORT:?}; TARGETED_ARTIFACT_ROOT=${TARGETED_ARTIFACT_ROOT:?}
  TARGETED_ARTIFACT_COMMIT=${TARGETED_ARTIFACT_COMMIT:?}; TARGETED_VALIDATION=${TARGETED_VALIDATION:?}
  "$AEGIS_PYTHON" scripts/bind_distal_whole_body_final_gate.py combined-audit \
    --base "$BASE" --targeted-cohort "$TARGETED_COHORT" \
    --targeted-artifact-root "$TARGETED_ARTIFACT_ROOT" \
    --targeted-artifact-commit "$TARGETED_ARTIFACT_COMMIT" \
    --targeted-validation "$TARGETED_VALIDATION" --output "$OUTPUT"
elif [[ "$MODE" == q-prediction ]]; then
  PROTOCOL=${PROTOCOL:-$REMOTE_REPO/configs/vlsa_distal_whole_body_q_only_prediction_protocol.v1.json}
  COVERAGE_AUDIT=${COVERAGE_AUDIT:?}
  "$AEGIS_PYTHON" scripts/bind_distal_whole_body_final_gate.py q-prediction \
    --protocol "$PROTOCOL" --coverage-audit "$COVERAGE_AUDIT" --output "$OUTPUT"
else
  exit 2
fi
