#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_aegis_evaluation.sh canary|population}
case "$MODE" in
  canary)
    EXPECTED_STAGE=paired-canary
    EXPECTED_ARRAY_TASKS=1
    EXPECTED_THROTTLE=1
    ;;
  population)
    EXPECTED_STAGE=population
    EXPECTED_ARRAY_TASKS=32
    EXPECTED_THROTTLE=2
    ;;
  *) echo "unsupported evaluation mode: $MODE" >&2; exit 2 ;;
esac

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
# shellcheck source=slurm/aegis_runtime_common.sh
source "$REMOTE_REPO/slurm/aegis_runtime_common.sh"
trap aegis_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

aegis_assert_allocation 8 65536
aegis_assert_array_contract "$EXPECTED_ARRAY_TASKS" \
  "$((EXPECTED_ARRAY_TASKS - 1))" "$EXPECTED_THROTTLE"
aegis_validate_identity "$EXPECTED_STAGE"
if [[ "$MODE" == canary ]]; then
  [[ "$SLURM_ARRAY_TASK_ID" == 0 ]] || aegis_die "paired canary is task zero only"
  CASE_ORDINAL=$CONTRACT_CASE_ORDINAL
  CASE_SELECTOR=case-ordinal=$CASE_ORDINAL
  CASE_ARGUMENTS=(--case-ordinal "$CASE_ORDINAL")
else
  (( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 32 )) || {
    aegis_die "population task must be in [0, 31]"
  }
  CASE_SELECTOR=group-index=$SLURM_ARRAY_TASK_ID
  GROUP_START=$((SLURM_ARRAY_TASK_ID * 50))
  GROUP_STOP=$((GROUP_START + 49))
  GROUP_ORDINALS=$(seq -s, "$GROUP_START" "$GROUP_STOP")
  CASE_ARGUMENTS=(--case-ordinal "$GROUP_ORDINALS")
fi
aegis_create_task_root "$SLURM_ARRAY_TASK_ID"
aegis_prepare_environment
export AEGIS_STAGE=evaluation
export CASE_SELECTOR

AEGIS_FAILURE_STAGE=source_and_evaluation_asset_preflight
aegis_run_preflight evaluation

EVALUATOR=${EVALUATOR:-$REMOTE_REPO/main/evaluate_safelibero_aegis.py}
[[ -f "$EVALUATOR" && ! -L "$EVALUATOR" ]] || {
  aegis_die "paired evaluator is missing or symlinked: $EVALUATOR"
}

AEGIS_FAILURE_STAGE=pi05_server_startup
aegis_start_pi05_server

# Both arms consume the exact same manifest row(s), selector, server, and
# per-request query-indexed noise schedule.  Running them serially prevents
# two policy services from competing for the same allocation.
OUTPUT_ROOT=$TASK_ROOT/results
mkdir "$OUTPUT_ROOT"
for ARM in pi05_translational pi05_plus_aegis_translational; do
  export ARM
  AEGIS_FAILURE_STAGE=evaluator_$ARM
  if [[ "$ARM" == pi05_translational ]]; then
    EVALUATOR_MODE=pi05
    ARM_ARGUMENTS=(--labels "$LABEL_MANIFEST_PATH")
  else
    EVALUATOR_MODE=aegis
    ARM_ARGUMENTS=(
      --labels "$LABEL_MANIFEST_PATH"
      --groundingdino-config "$DINO_CONFIG"
      --groundingdino-checkpoint "$DINO_CHECKPOINT"
      --groundingdino-device "$GROUNDINGDINO_DEVICE"
    )
  fi
  "$AEGIS_PYTHON" "$EVALUATOR" \
    --manifest "$MANIFEST_PATH" \
    --mode "$EVALUATOR_MODE" \
    --output-dir "$OUTPUT_ROOT" \
    --repo-root "$REMOTE_REPO" \
    --host 127.0.0.1 \
    --port "$POLICY_PORT" \
    "${CASE_ARGUMENTS[@]}" \
    "${ARM_ARGUMENTS[@]}"
done

if [[ "$MODE" == canary ]]; then
  AEGIS_FAILURE_STAGE=paired_canary_validation
  CANARY_VALIDATOR=$REMOTE_REPO/scripts/validate_aegis_run_artifacts.py
  [[ -f "$CANARY_VALIDATOR" && ! -L "$CANARY_VALIDATOR" ]] || {
    aegis_die "paired-canary validator is missing or symlinked"
  }
  "$AEGIS_PYTHON" "$CANARY_VALIDATOR" paired-canary \
    --run-root "$RUN_ROOT" \
    --expected-commit "$EXPECTED_GIT_COMMIT" \
    --config "$CONFIG_PATH" \
    --manifest "$MANIFEST_PATH" \
    --manifest-receipt "$MANIFEST_RECEIPT_PATH" \
    --labels "$LABEL_MANIFEST_PATH" \
    --pi05-hash-receipt "$PI05_HASH_RECEIPT_PATH" \
    --case-ordinal "$CASE_ORDINAL" \
    --output "$RUN_ROOT/paired-canary-validation.json"
fi

AEGIS_FAILURE_STAGE=complete
