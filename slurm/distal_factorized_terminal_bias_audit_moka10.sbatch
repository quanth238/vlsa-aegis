#!/usr/bin/env bash
#SBATCH --job-name=vlsa-terminal-bias
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-terminal-bias-audit}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
SOURCE_VALIDATION=${SOURCE_VALIDATION:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution-validation/factorized-execution-20260811d-validation-a}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_terminal_bias_audit_moka10.v1.json
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
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "terminal-bias audit requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$SOURCE_RUN" && -d "$SOURCE_VALIDATION" && ! -L "$SOURCE_RUN" && ! -L "$SOURCE_VALIDATION" ]] || die "runtime or immutable sources missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "terminal-bias source identity differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 85a9f0cf83dae6c97d47f6adde013a1145f5ebaec6e8af6e4e1b48c106b112a0 ]] || die "terminal-bias config differs"
[[ "$(sha256sum "$SOURCE_RUN/factorized-execution-model.npz" | awk '{print $1}')" == 262ee006523b76a09f120c5c4e8986f501ae49de50a194c16c697f4ad67c995f ]] || die "factorized model differs"
[[ "$(sha256sum "$SOURCE_RUN/predictions.npz" | awk '{print $1}')" == d888d3fafbf44cb74caf675b438e70af986200ce6027a2fee49712f0a2f82ab3 ]] || die "factorized predictions differ"
[[ "$(sha256sum "$SOURCE_RUN/result.json" | awk '{print $1}')" == 70c4dd81e7b20d03541d85ed392e0f552cd1e4b8b3e8f81d0c148f2fbe36b271 ]] || die "factorized result differs"
[[ "$(sha256sum "$SOURCE_VALIDATION/validation.json" | awk '{print $1}')" == ed78d2f528010095f1bc4db85f73e160996db62e764f30122196d8fb8821ef1e ]] || die "factorized validation differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable terminal-bias root exists"
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_terminal_bias_audit tests.test_distal_factorized_execution_pilot -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$CONFIG" --factorized-config "$FACTOR_CONFIG"
  --complete-dataset "$COMPLETE_ROOT/dataset.json" --complete-collection "$COMPLETE_ROOT/collection-result.json"
  --trajectory-metadata "$SOURCE_RUN/trajectory-metadata.json" --trajectory-array "$SOURCE_RUN/trajectory-dataset.npz" --trajectory-collection "$SOURCE_RUN/collection-result.json"
  --factorized-model "$SOURCE_RUN/factorized-execution-model.npz" --predictions "$SOURCE_RUN/predictions.npz"
  --factorized-result "$SOURCE_RUN/result.json" --factorized-validation "$SOURCE_VALIDATION/validation.json"
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED"
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/audit_distal_factorized_terminal_bias_moka10.py \
  "${common[@]}" --records "$RUN_ROOT/records.npz" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_terminal_bias_moka10.py \
  "${common[@]}" --records "$RUN_ROOT/records.npz" \
  --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
