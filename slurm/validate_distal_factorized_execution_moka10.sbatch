#!/usr/bin/env bash
#SBATCH --job-name=vlsa-factor-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_VALIDATOR_COMMIT=${EXPECTED_VALIDATOR_COMMIT:?EXPECTED_VALIDATOR_COMMIT is required}
RESULT_COMMIT=${RESULT_COMMIT:-7062d70a5cc90f9689def2bfc4bf2bdd347b1c8e}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution-validation}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_execution_moka10.v1.json
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
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "factorized-execution validation requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$SOURCE_RUN" && ! -L "$SOURCE_RUN" ]] || die "runtime or immutable source run missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_VALIDATOR_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "validator source identity differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == dfe0afbcf6439e7ef317ec73d913b0784892f1deb4cedce4908b766c2aecdfc8 ]] || die "factorized-execution config differs"
[[ "$(sha256sum "$SOURCE_RUN/result.json" | awk '{print $1}')" == 70c4dd81e7b20d03541d85ed392e0f552cd1e4b8b3e8f81d0c148f2fbe36b271 ]] || die "factorized-execution result differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable validation root exists"
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
printf '{"result_commit":"%s","validator_commit":"%s","source_result_sha256":"%s"}\n' "$RESULT_COMMIT" "$EXPECTED_VALIDATOR_COMMIT" "70c4dd81e7b20d03541d85ed392e0f552cd1e4b8b3e8f81d0c148f2fbe36b271" >"$RUN_ROOT/preflight.json"
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_execution_pilot -v >"$RUN_ROOT/tests.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_distal_factorized_execution_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --complete-dataset "$COMPLETE_ROOT/dataset.json" \
  --complete-collection "$COMPLETE_ROOT/collection-result.json" \
  --trajectory-metadata "$SOURCE_RUN/trajectory-metadata.json" \
  --trajectory-array "$SOURCE_RUN/trajectory-dataset.npz" \
  --trajectory-collection "$SOURCE_RUN/collection-result.json" \
  --direct-model "$SOURCE_RUN/direct-margin-model.npz" \
  --factorized-model "$SOURCE_RUN/factorized-execution-model.npz" \
  --predictions "$SOURCE_RUN/predictions.npz" \
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" \
  --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED" \
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" \
  --exact-box-config "$EXACT_BOX" --expected-commit "$RESULT_COMMIT" \
  --result "$SOURCE_RUN/result.json" --output "$RUN_ROOT/validation.json"
