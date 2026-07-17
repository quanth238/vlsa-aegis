#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_aegis_capture.sh canary|population}
case "$MODE" in
  canary)
    EXPECTED_STAGE=capture-canary
    EXPECTED_ARRAY_TASKS=1
    EXPECTED_THROTTLE=1
    ;;
  population)
    EXPECTED_STAGE=capture-population
    EXPECTED_ARRAY_TASKS=32
    EXPECTED_THROTTLE=2
    ;;
  *)
    echo "unsupported capture mode: $MODE" >&2
    exit 2
    ;;
esac

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
# shellcheck source=slurm/aegis_runtime_common.sh
source "$REMOTE_REPO/slurm/aegis_runtime_common.sh"
trap aegis_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

aegis_assert_allocation 4 32768
aegis_assert_array_contract "$EXPECTED_ARRAY_TASKS" \
  "$((EXPECTED_ARRAY_TASKS - 1))" "$EXPECTED_THROTTLE"
aegis_validate_identity "$EXPECTED_STAGE"
if [[ "$MODE" == canary ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" == 0 ]] || aegis_die "capture canary is task zero only"
  CASE_ORDINAL=$CONTRACT_CASE_ORDINAL
  CASE_SELECTOR=case-ordinal=$CASE_ORDINAL
  CASE_ARGUMENTS=(--case-ordinal "$CASE_ORDINAL")
else
  (( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 32 )) || {
    aegis_die "capture population task must be in [0, 31]"
  }
  CASE_SELECTOR=group-index=$SLURM_ARRAY_TASK_ID
  GROUP_START=$((SLURM_ARRAY_TASK_ID * 50))
  GROUP_STOP=$((GROUP_START + 49))
  GROUP_ORDINALS=$(seq -s, "$GROUP_START" "$GROUP_STOP")
  CASE_ARGUMENTS=(--case-ordinal "$GROUP_ORDINALS")
fi
aegis_create_task_root "$SLURM_ARRAY_TASK_ID"
aegis_prepare_environment

# These exported guards make the capture-only contract explicit to the runner.
# No policy server is started, and no model/perception/QP argument is supplied.
export AEGIS_STAGE=capture
export POLICY_MODEL_EXECUTION_ALLOWED=false
export GROUNDING_EXECUTION_ALLOWED=false
export SAFETY_QP_EXECUTION_ALLOWED=false
export EXPECTED_ARRAY_TASKS EXPECTED_THROTTLE CASE_SELECTOR

AEGIS_FAILURE_STAGE=source_and_capture_asset_preflight
aegis_run_preflight capture

CAPTURE_RUNNER=${CAPTURE_RUNNER:-$REMOTE_REPO/main/capture_safelibero_labels.py}
[[ -f "$CAPTURE_RUNNER" && ! -L "$CAPTURE_RUNNER" ]] || {
  aegis_die "capture runner is missing or symlinked: $CAPTURE_RUNNER"
}
CAPTURE_OUTPUT_ROOT=$TASK_ROOT/captures
mkdir "$CAPTURE_OUTPUT_ROOT"

AEGIS_FAILURE_STAGE=settled_image_capture
"$AEGIS_PYTHON" "$CAPTURE_RUNNER" \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$CAPTURE_OUTPUT_ROOT" \
  --repo-root "$REMOTE_REPO" \
  "${CASE_ARGUMENTS[@]}"

AEGIS_FAILURE_STAGE=complete
