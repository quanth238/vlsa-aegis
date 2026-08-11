#!/usr/bin/env bash
#SBATCH --job-name=vlsa-explicit-j
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-explicit-execution-jacobian}
RESERVED_RUN=${RESERVED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-reserved/direct-horizon-reserved-20260811c}
RESERVED_VALIDATION=${RESERVED_VALIDATION:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-reserved-validation/direct-horizon-reserved-20260811c-validation-i/collection-validation.json}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
SOURCE_VALIDATION=${SOURCE_VALIDATION:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution-validation/factorized-execution-20260811d-validation-a}
SURFACE_RUN=${SURFACE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-surface-error/factorized-surface-error-20260811a}
ADAPTATION_RUN=${ADAPTATION_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811a}
ONE_SIDED_RUN=${ONE_SIDED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811b}
UNSUPPORTED_RUN=${UNSUPPORTED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-unsupported-state-audit/unsupported-state-audit-20260811b}
TIME_RUN=${TIME_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-time-conditioned-decoder/time-conditioned-decoder-20260811a}
ORIENTATION_RUN=${ORIENTATION_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-orientation-joint-mask/orientation-joint-mask-20260811a}
STRUCTURED_RUN=${STRUCTURED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-structured-orientation/structured-orientation-20260811b}
CUMULATIVE_RUN=${CUMULATIVE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-cumulative-delta-q/cumulative-delta-q-20260811b}
DIRECT_RUN=${DIRECT_RUN:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-displacement/direct-horizon-displacement-20260811a}
DIRECT_VALIDATION=${DIRECT_VALIDATION:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-displacement/direct-horizon-displacement-20260811a-validation-a/validation.json}
ROOT_CAUSE_RUN=${ROOT_CAUSE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-root-cause-audit/direct-horizon-root-cause-audit-20260811c}
NORMALIZED_RUN=${NORMALIZED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-normalized-secant/direct-horizon-normalized-secant-20260811b}
NORMALIZED_VALIDATION=${NORMALIZED_VALIDATION:-$NORMALIZED_RUN/validation.json}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
SOURCE_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_one_sided_geometry_moka10.v2.json
TIME_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_time_conditioned_decoder_moka10.v1.json
STRUCTURED_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_structured_orientation_moka10.v1.json
CUMULATIVE_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_cumulative_delta_q_moka10.v1.json
EXPLICIT_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_explicit_execution_jacobian_moka10.v1.json
NORMALIZED_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_direct_horizon_normalized_secant_moka10.v1.json
ORIENTATION_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_orientation_joint_mask_moka10.v1.json
FACTOR_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_execution_moka10.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl
SAME_TASK=$REMOTE_REPO/manifests/vlsa_distal_same_task_boundary_expansion_moka10.v1.jsonl
TARGETED=$REMOTE_REPO/manifests/vlsa_distal_targeted_same_task_screen_moka10.v1.jsonl
LEGACY_RESERVED=$REMOTE_REPO/manifests/vlsa_distal_delta_q_reserved_evaluation_moka10.v1.jsonl
DIRECT_RESERVED=$REMOTE_REPO/manifests/vlsa_distal_direct_horizon_reserved_prediction_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 98304 ]] || die "registered allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "explicit-J experiment requires exactly one H100"
for source_dir in "$SOURCE_RUN" "$STRUCTURED_RUN" "$CUMULATIVE_RUN" "$DIRECT_RUN" "$ROOT_CAUSE_RUN" "$NORMALIZED_RUN"; do
  [[ -d "$source_dir" && ! -L "$source_dir" ]] || die "immutable source missing or symlinked"
done
[[ -x "$AEGIS_PYTHON" ]] || die "evaluation runtime missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
[[ "$(sha256sum "$EXPLICIT_CONFIG" | awk '{print $1}')" == 8aae74181f24db35c7b86c5fdd17ec84d62bb9fd05ab034d5b3b72a688eb79bf ]] || die "explicit-J config differs"
[[ "$(sha256sum "$NORMALIZED_CONFIG" | awk '{print $1}')" == a9661a6ab8725374f509e61ffc11861216ba4809692f448cc9f44d0b731ab4df ]] || die "normalized-secant config differs"
[[ "$(sha256sum "$NORMALIZED_RUN/model.npz" | awk '{print $1}')" == 60791198b705cd36bc6ad7765d72089483f62fa68a7873c02d99d6e77ef90d2b ]] || die "normalized model differs"
[[ "$(sha256sum "$NORMALIZED_RUN/predictions.npz" | awk '{print $1}')" == 1a5da8391be988980c4078e09dd943db85fadaa97a56ba5a5f8aa6b5d9571c29 ]] || die "normalized predictions differ"
[[ "$(sha256sum "$NORMALIZED_RUN/result.json" | awk '{print $1}')" == 265e2a16d5dfd49877d48b8eab9961f4fa9b65d3680a9d8c9a1279fcf3abf868 ]] || die "normalized result differs"
[[ "$(sha256sum "$NORMALIZED_VALIDATION" | awk '{print $1}')" == 5661919f60f8f250a7afa8939feeeeedc217780e95fe08c8b72397cf4a4b8312 ]] || die "normalized validation differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\ndatasets: %s\nassets: %s\n' "$safelibero_root" "$safelibero_root/bddl_files" "$safelibero_root/init_files" "$REMOTE_REPO/safelibero/libero/datasets" "$safelibero_root/assets" >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_explicit_execution_jacobian tests.test_distal_factorized_direct_horizon_displacement tests.test_distal_factorized_direct_horizon_root_cause_audit tests.test_distal_factorized_cumulative_delta_q tests.test_distal_factorized_structured_orientation -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$SOURCE_CONFIG" --time-config "$TIME_CONFIG" --structured-config "$STRUCTURED_CONFIG" --cumulative-config "$CUMULATIVE_CONFIG" --direct-horizon-config "$EXPLICIT_CONFIG" --orientation-config "$ORIENTATION_CONFIG" --factorized-config "$FACTOR_CONFIG"
  --complete-dataset "$COMPLETE_ROOT/dataset.json" --complete-collection "$COMPLETE_ROOT/collection-result.json"
  --trajectory-metadata "$SOURCE_RUN/trajectory-metadata.json" --trajectory-array "$SOURCE_RUN/trajectory-dataset.npz" --trajectory-collection "$SOURCE_RUN/collection-result.json"
  --baseline-model "$SOURCE_RUN/factorized-execution-model.npz" --baseline-predictions "$SOURCE_RUN/predictions.npz" --factorized-result "$SOURCE_RUN/result.json" --factorized-validation "$SOURCE_VALIDATION/validation.json"
  --surface-result "$SURFACE_RUN/result.json" --adaptation-result "$ADAPTATION_RUN/result.json" --adaptation-validation "$ADAPTATION_RUN/validation.json"
  --source-one-sided-model "$ONE_SIDED_RUN/experimental-model.npz" --source-one-sided-predictions "$ONE_SIDED_RUN/predictions.npz" --source-one-sided-result "$ONE_SIDED_RUN/result.json" --source-one-sided-validation "$ONE_SIDED_RUN/validation.json"
  --unsupported-audit-result "$UNSUPPORTED_RUN/result.json" --unsupported-audit-validation "$UNSUPPORTED_RUN/validation.json"
  --source-time-model "$TIME_RUN/experimental-model.npz" --source-time-result "$TIME_RUN/result.json" --source-time-validation "$TIME_RUN/validation.json"
  --orientation-result "$ORIENTATION_RUN/result.json" --orientation-validation "$ORIENTATION_RUN/validation.json"
  --source-structured-flat-model "$STRUCTURED_RUN/flat-model.npz" --source-structured-time-model "$STRUCTURED_RUN/time-model.npz" --source-structured-predictions "$STRUCTURED_RUN/predictions.npz" --source-structured-result "$STRUCTURED_RUN/result.json" --source-structured-validation "$STRUCTURED_RUN/validation.json"
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED" --reserved-manifest "$LEGACY_RESERVED" --direct-reserved-manifest "$DIRECT_RESERVED"
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT"
  --reserved-collection "$RESERVED_RUN/collection-result.json" --reserved-metadata "$RESERVED_RUN/trajectory-metadata.json" --reserved-array "$RESERVED_RUN/trajectory-dataset.npz" --reserved-validation "$RESERVED_VALIDATION"
  --previous-factorized-model "$ONE_SIDED_RUN/experimental-model.npz" --previous-cumulative-model "$CUMULATIVE_RUN/model.npz"
  --matched-direct-run "$DIRECT_RUN" --matched-direct-validation "$DIRECT_VALIDATION" --root-cause-run "$ROOT_CAUSE_RUN"
  --matched-normalized-config "$NORMALIZED_CONFIG" --matched-normalized-run "$NORMALIZED_RUN" --matched-normalized-validation "$NORMALIZED_VALIDATION"
)
"$AEGIS_PYTHON" scripts/evaluate_distal_explicit_execution_jacobian_moka10.py "${common[@]}" --experimental-model "$RUN_ROOT/model.npz" --predictions "$RUN_ROOT/predictions.npz" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_explicit_execution_jacobian_moka10.py "${common[@]}" --experimental-model "$RUN_ROOT/model.npz" --predictions "$RUN_ROOT/predictions.npz" --result "$RUN_ROOT/result.json" --expected-result-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/validation.json"
