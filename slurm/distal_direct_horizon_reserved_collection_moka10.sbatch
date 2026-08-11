#!/usr/bin/env bash
#SBATCH --job-name=vlsa-direct-reserved
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-reserved}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_direct_horizon_displacement_moka10.v1.json
FACTOR_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_execution_moka10.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
RESERVED=$REMOTE_REPO/manifests/vlsa_distal_direct_horizon_reserved_prediction_moka10.v1.jsonl
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl
SAME_TASK=$REMOTE_REPO/manifests/vlsa_distal_same_task_boundary_expansion_moka10.v1.jsonl
TARGETED=$REMOTE_REPO/manifests/vlsa_distal_targeted_same_task_screen_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "reserved collection requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$COMPLETE_ROOT" && -d "$ARCHIVED_ROOT" ]] || die "runtime or immutable source missing"
[[ ! -L "$COMPLETE_ROOT" && ! -L "$ARCHIVED_ROOT" ]] || die "immutable source symlink forbidden"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 13181315beeca4dfbe98b7dbfa024f8f2f00728a770623e1f98d2e7aeed3c37f ]] || die "direct-horizon config differs"
[[ "$(sha256sum "$RESERVED" | awk '{print $1}')" == f99ce2b38b5c2b9ceb4073611cc16356efc36e1820193926a5770f396e93ffcd ]] || die "reserved manifest differs"
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_direct_horizon_displacement tests.test_distal_factorized_execution_pilot -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$CONFIG" --factorized-config "$FACTOR_CONFIG"
  --population-manifest "$POPULATION" --reserved-manifest "$RESERVED"
  --selected-manifest "$SELECTED" --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED"
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX"
  --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/collect_distal_direct_horizon_reserved_moka10.py "${common[@]}" \
  --complete-dataset "$COMPLETE_ROOT/dataset.json" --complete-collection "$COMPLETE_ROOT/collection-result.json" \
  --array-dataset "$RUN_ROOT/trajectory-dataset.npz" --metadata "$RUN_ROOT/trajectory-metadata.json" --output "$RUN_ROOT/collection-result.json"
"$AEGIS_PYTHON" scripts/validate_distal_direct_horizon_reserved_moka10.py "${common[@]}" \
  --collection "$RUN_ROOT/collection-result.json" --metadata "$RUN_ROOT/trajectory-metadata.json" --array-dataset "$RUN_ROOT/trajectory-dataset.npz" --output "$RUN_ROOT/collection-validation.json"
