#!/usr/bin/env bash
#SBATCH --job-name=vlsa-contact-ranker-e05
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-contact-ranker-e05}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
SUCCESSFUL_SITL_RESULT=${SUCCESSFUL_SITL_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-sitl-minimal-replay-e05/sitl-minimal-replay-20260808h/result.json}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_distal_contact_ranker_e05.v1.json}
GEOMETRY_CONFIG=${GEOMETRY_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json}
EXACT_BOX_CONFIG=${EXACT_BOX_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "contact-ranker experiment requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected exactly 64 GiB"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"
[[ "$(sha256sum "$ARCHIVED_TABLE1_RESULT" | awk '{print $1}')" == 273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b ]] || die "Table-1 hash differs"
[[ "$(sha256sum "$SUCCESSFUL_SITL_RESULT" | awk '{print $1}')" == e23320ebad7c17f8d07d52f204b0fc32c2c3d81ab8007d0a1f1b6ba365c7bf08 ]] || die "successful SITL hash differs"
for path in "$CONFIG" "$GEOMETRY_CONFIG" "$EXACT_BOX_CONFIG"; do [[ -f "$path" && ! -L "$path" ]] || die "required input missing"; done

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root already exists"
FAILURE_PATH=$RUN_ROOT/runtime-failure.json
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ $exit_code -ne 0 && ! -e "$FAILURE_PATH" ]]; then
    temporary=$(mktemp "$RUN_ROOT/.runtime-failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$exit_code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE_PATH"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"

cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy, torch; assert torch.cuda.is_available(); print({"numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0)})' >"$RUN_ROOT/runtime-preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_contact_ranker tests.test_distal_oracle_affine tests.test_distal_execution_margin_nn -v >"$RUN_ROOT/test-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_contact_ranker_e05.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --successful-sitl "$SUCCESSFUL_SITL_RESULT" \
  --geometry-config "$GEOMETRY_CONFIG" \
  --exact-box-config "$EXACT_BOX_CONFIG" \
  --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --dataset "$RUN_ROOT/dataset.json" \
  --model "$RUN_ROOT/model.pt" \
  --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_contact_ranker_e05.py \
  --dataset "$RUN_ROOT/dataset.json" \
  --model "$RUN_ROOT/model.pt" \
  --result "$RUN_ROOT/result.json" \
  --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"
