#!/usr/bin/env bash
#SBATCH --job-name=vlsa-struct-orient
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-structured-orientation}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
SOURCE_VALIDATION=${SOURCE_VALIDATION:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution-validation/factorized-execution-20260811d-validation-a}
SURFACE_RUN=${SURFACE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-surface-error/factorized-surface-error-20260811a}
ADAPTATION_RUN=${ADAPTATION_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811a}
ONE_SIDED_RUN=${ONE_SIDED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811b}
UNSUPPORTED_RUN=${UNSUPPORTED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-unsupported-state-audit/unsupported-state-audit-20260811b}
TIME_RUN=${TIME_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-time-conditioned-decoder/time-conditioned-decoder-20260811a}
ORIENTATION_RUN=${ORIENTATION_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-orientation-joint-mask/orientation-joint-mask-20260811a}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
SOURCE_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_one_sided_geometry_moka10.v2.json
TIME_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_time_conditioned_decoder_moka10.v1.json
STRUCTURED_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_structured_orientation_moka10.v1.json
ORIENTATION_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_orientation_joint_mask_moka10.v1.json
FACTOR_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_execution_moka10.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl
SAME_TASK=$REMOTE_REPO/manifests/vlsa_distal_same_task_boundary_expansion_moka10.v1.jsonl
TARGETED=$REMOTE_REPO/manifests/vlsa_distal_targeted_same_task_screen_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "structured-orientation ablation requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$SOURCE_RUN" && -d "$SOURCE_VALIDATION" && -d "$SURFACE_RUN" && -d "$ADAPTATION_RUN" && -d "$ONE_SIDED_RUN" && -d "$UNSUPPORTED_RUN" && -d "$TIME_RUN" && -d "$ORIENTATION_RUN" ]] || die "runtime or immutable source missing"
[[ ! -L "$ONE_SIDED_RUN" && ! -L "$TIME_RUN" && ! -L "$ORIENTATION_RUN" ]] || die "immutable source symlink forbidden"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "structured-orientation source identity differs"
[[ "$(sha256sum "$STRUCTURED_CONFIG" | awk '{print $1}')" == adf6a42bdf01d1eb4910c1d50275065b34a6df37583865922d2a186e14df241c ]] || die "structured-orientation config differs"
[[ "$(sha256sum "$SOURCE_CONFIG" | awk '{print $1}')" == c2d4d6b69294531100a0b361b703c156b276238a9581e6b414132c2564995ec2 ]] || die "source one-sided config differs"
[[ "$(sha256sum "$TIME_CONFIG" | awk '{print $1}')" == 452fe00d53f3b6e9c278962116c5cf760821f14b18b06a6d4079a577bd5a2d17 ]] || die "time config differs"
[[ "$(sha256sum "$ORIENTATION_RUN/result.json" | awk '{print $1}')" == 1cd507f5b00ad4a5a243f957ae3466348680e821a9fab973caf133f73023cfc1 ]] || die "orientation confirmation result differs"
[[ "$(sha256sum "$ORIENTATION_RUN/validation.json" | awk '{print $1}')" == b5b3931bc85021c849a4916e5e260b90c389b7a267931e9b9d8f0adb1ebfaa66 ]] || die "orientation confirmation validation differs"
[[ "$(sha256sum "$TIME_RUN/result.json" | awk '{print $1}')" == 4b98951a387aa1f7f21ece321e611c085274ede19b7b1a2825558c7d5a56c30f ]] || die "source time result differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable structured-orientation root exists"
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_structured_orientation tests.test_distal_factorized_time_conditioned_decoder tests.test_distal_factorized_one_sided_geometry -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$SOURCE_CONFIG" --time-config "$TIME_CONFIG" --structured-config "$STRUCTURED_CONFIG" --orientation-config "$ORIENTATION_CONFIG" --factorized-config "$FACTOR_CONFIG"
  --complete-dataset "$COMPLETE_ROOT/dataset.json" --complete-collection "$COMPLETE_ROOT/collection-result.json"
  --trajectory-metadata "$SOURCE_RUN/trajectory-metadata.json" --trajectory-array "$SOURCE_RUN/trajectory-dataset.npz" --trajectory-collection "$SOURCE_RUN/collection-result.json"
  --baseline-model "$SOURCE_RUN/factorized-execution-model.npz" --baseline-predictions "$SOURCE_RUN/predictions.npz"
  --factorized-result "$SOURCE_RUN/result.json" --factorized-validation "$SOURCE_VALIDATION/validation.json"
  --surface-result "$SURFACE_RUN/result.json" --adaptation-result "$ADAPTATION_RUN/result.json" --adaptation-validation "$ADAPTATION_RUN/validation.json"
  --source-one-sided-model "$ONE_SIDED_RUN/experimental-model.npz" --source-one-sided-predictions "$ONE_SIDED_RUN/predictions.npz"
  --source-one-sided-result "$ONE_SIDED_RUN/result.json" --source-one-sided-validation "$ONE_SIDED_RUN/validation.json"
  --unsupported-audit-result "$UNSUPPORTED_RUN/result.json" --unsupported-audit-validation "$UNSUPPORTED_RUN/validation.json"
  --source-time-model "$TIME_RUN/experimental-model.npz" --source-time-result "$TIME_RUN/result.json" --source-time-validation "$TIME_RUN/validation.json"
  --orientation-result "$ORIENTATION_RUN/result.json" --orientation-validation "$ORIENTATION_RUN/validation.json"
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED"
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/evaluate_distal_factorized_structured_orientation_moka10.py "${common[@]}" --flat-model "$RUN_ROOT/flat-model.npz" --time-model "$RUN_ROOT/time-model.npz" --predictions "$RUN_ROOT/predictions.npz" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_structured_orientation_moka10.py "${common[@]}" --flat-model "$RUN_ROOT/flat-model.npz" --time-model "$RUN_ROOT/time-model.npz" --predictions "$RUN_ROOT/predictions.npz" --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
