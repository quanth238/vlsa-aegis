#!/usr/bin/env bash
#SBATCH --job-name=vlsa-direct-reserved-val
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
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-reserved-validation}
SOURCE_RUN=${SOURCE_RUN:-/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-reserved/direct-horizon-reserved-20260811c}
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
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "reserved validation requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$SOURCE_RUN" && ! -L "$SOURCE_RUN" ]] || die "runtime or immutable source missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
[[ "$(sha256sum "$SOURCE_RUN/collection-result.json" | awk '{print $1}')" == bd2e24e82554f51d1a331a7b2e20dcdaa7e256bacdfa1aaf286ce72ef9d31689 ]] || die "collection result differs"
[[ "$(sha256sum "$SOURCE_RUN/trajectory-metadata.json" | awk '{print $1}')" == 96df2ca783d06e002bd8b268ee4e4a6b4d63eeb31db3f76bd4ae57109961c0b1 ]] || die "collection metadata differs"
[[ "$(sha256sum "$SOURCE_RUN/trajectory-dataset.npz" | awk '{print $1}')" == edf0f3351d195a7751a49be1d30259829de9e3e92d7def3c4583cc33b1754ec1 ]] || die "collection array differs"
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_direct_horizon_displacement -v >"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/validate_distal_direct_horizon_reserved_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" --factorized-config "$FACTOR_CONFIG" \
  --population-manifest "$POPULATION" --reserved-manifest "$RESERVED" \
  --selected-manifest "$SELECTED" --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED" \
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" \
  --collection "$SOURCE_RUN/collection-result.json" --metadata "$SOURCE_RUN/trajectory-metadata.json" --array-dataset "$SOURCE_RUN/trajectory-dataset.npz" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/collection-validation.json"
