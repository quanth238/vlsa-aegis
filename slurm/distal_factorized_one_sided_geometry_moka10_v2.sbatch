#!/usr/bin/env bash
#SBATCH --job-name=vlsa-one-sided-v2
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
SOURCE_VALIDATION=${SOURCE_VALIDATION:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution-validation/factorized-execution-20260811d-validation-a}
SURFACE_RUN=${SURFACE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-surface-error/factorized-surface-error-20260811a}
ADAPTATION_RUN=${ADAPTATION_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811a}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_one_sided_geometry_moka10.v2.json
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
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "one-sided geometry v2 requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$SOURCE_RUN" && -d "$SOURCE_VALIDATION" && -d "$SURFACE_RUN" && -d "$ADAPTATION_RUN" ]] || die "runtime or immutable source missing"
[[ ! -L "$SOURCE_RUN" && ! -L "$SOURCE_VALIDATION" && ! -L "$SURFACE_RUN" && ! -L "$ADAPTATION_RUN" ]] || die "immutable source symlink forbidden"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "one-sided geometry v2 source identity differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == c2d4d6b69294531100a0b361b703c156b276238a9581e6b414132c2564995ec2 ]] || die "one-sided geometry v2 config differs"
[[ "$(sha256sum "$SOURCE_RUN/result.json" | awk '{print $1}')" == 70c4dd81e7b20d03541d85ed392e0f552cd1e4b8b3e8f81d0c148f2fbe36b271 ]] || die "factorized result differs"
[[ "$(sha256sum "$SOURCE_VALIDATION/validation.json" | awk '{print $1}')" == ed78d2f528010095f1bc4db85f73e160996db62e764f30122196d8fb8821ef1e ]] || die "factorized validation differs"
[[ "$(sha256sum "$SURFACE_RUN/result.json" | awk '{print $1}')" == 263b51cf93cea9aee6b894962562229f8ad79ebd8ce1aa308bde873495e38b30 ]] || die "surface result differs"
[[ "$(sha256sum "$ADAPTATION_RUN/result.json" | awk '{print $1}')" == bad80f43ce205246515ba5aaffcb582b167d211bf7cb6862aea0781edd392afd ]] || die "v1 adaptation result differs"
[[ "$(sha256sum "$ADAPTATION_RUN/validation.json" | awk '{print $1}')" == 5cb612cdf35b7b891b820d5430c445e780f7af307832fc7eb6a2af2a2ac28bf4 ]] || die "v1 adaptation validation differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable one-sided geometry v2 root exists"
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_one_sided_geometry tests.test_distal_factorized_execution_pilot -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$CONFIG" --factorized-config "$FACTOR_CONFIG"
  --complete-dataset "$COMPLETE_ROOT/dataset.json" --complete-collection "$COMPLETE_ROOT/collection-result.json"
  --trajectory-metadata "$SOURCE_RUN/trajectory-metadata.json" --trajectory-array "$SOURCE_RUN/trajectory-dataset.npz" --trajectory-collection "$SOURCE_RUN/collection-result.json"
  --baseline-model "$SOURCE_RUN/factorized-execution-model.npz" --baseline-predictions "$SOURCE_RUN/predictions.npz"
  --factorized-result "$SOURCE_RUN/result.json" --factorized-validation "$SOURCE_VALIDATION/validation.json"
  --surface-result "$SURFACE_RUN/result.json" --adaptation-result "$ADAPTATION_RUN/result.json" --adaptation-validation "$ADAPTATION_RUN/validation.json"
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED"
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/evaluate_distal_factorized_one_sided_geometry_moka10.py "${common[@]}" --experimental-model "$RUN_ROOT/experimental-model.npz" --predictions "$RUN_ROOT/predictions.npz" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_one_sided_geometry_moka10.py "${common[@]}" --experimental-model "$RUN_ROOT/experimental-model.npz" --predictions "$RUN_ROOT/predictions.npz" --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
