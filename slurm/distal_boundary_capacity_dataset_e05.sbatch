#!/usr/bin/env bash
#SBATCH --job-name=vlsa-boundary-data-e05
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
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-boundary-capacity-dataset-e05}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
FALSE_SAFE_RESULT=${FALSE_SAFE_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-sitl-minimal-replay-e05/sitl-minimal-replay-20260808b/result.json}
OBSTACLE_DISCOVERY_RESULT=${OBSTACLE_DISCOVERY_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-oracle-mesh-obstacle-e05/mesh-obstacle-20260809a/result.json}
BASELINE_NN_RESULT=${BASELINE_NN_RESULT:?BASELINE_NN_RESULT is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_distal_boundary_capacity_e05.v1.json}
GEOMETRY_CONFIG=${GEOMETRY_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json}
EXACT_BOX_CONFIG=${EXACT_BOX_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "boundary dataset requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected exactly 64 GiB"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "boundary dataset requires exactly one H100"
[[ -x "$AEGIS_PYTHON" ]] || die "evaluation Python is missing"
for path in "$ARCHIVED_TABLE1_RESULT" "$FALSE_SAFE_RESULT" "$OBSTACLE_DISCOVERY_RESULT" "$BASELINE_NN_RESULT" "$CONFIG" "$GEOMETRY_CONFIG" "$EXACT_BOX_CONFIG"; do
  [[ -f "$path" && ! -L "$path" ]] || die "required immutable input is missing"
done
[[ "$(sha256sum "$ARCHIVED_TABLE1_RESULT" | awk '{print $1}')" == 273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b ]] || die "archived Table-1 hash differs"
[[ "$(sha256sum "$FALSE_SAFE_RESULT" | awk '{print $1}')" == d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603 ]] || die "false-safe ledger hash differs"
[[ "$(sha256sum "$OBSTACLE_DISCOVERY_RESULT" | awk '{print $1}')" == 3db37092b2c5573e70cfc604bbbf01f1361be822ec87c8733d2c80b84685b0fd ]] || die "exact-box discovery hash differs"
[[ "$(sha256sum "$BASELINE_NN_RESULT" | awk '{print $1}')" == 7fed872f2e189c9cdc07180f7ef3950c0a1a605065650e9650c8a9cee167fcca ]] || die "baseline neural result hash differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 95bd3b39b24d5e5e3f905e3af115ebe7b5e9717927cf52d426d7c82a231849a1 ]] || die "boundary config hash differs"
[[ "$(sha256sum "$GEOMETRY_CONFIG" | awk '{print $1}')" == fd9042ebc7605c68e71d1ab51bfdc8b4412fcb9dda11f1879d85271641436da1 ]] || die "distal geometry config hash differs"
[[ "$(sha256sum "$EXACT_BOX_CONFIG" | awk '{print $1}')" == cc568a85c2acf215beda1cef4abcc31a92b3f6d3772d6c36147410afd471bf9f ]] || die "exact-box config hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"

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
"$AEGIS_PYTHON" -c 'import numpy, osqp, scipy; print({"numpy": numpy.__version__, "osqp": osqp.__version__, "scipy": scipy.__version__})' >"$RUN_ROOT/runtime-preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_boundary_capacity -v >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/collect_distal_boundary_capacity_e05.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --false-safe-result "$FALSE_SAFE_RESULT" \
  --obstacle-discovery-result "$OBSTACLE_DISCOVERY_RESULT" \
  --baseline-nn-result "$BASELINE_NN_RESULT" \
  --geometry-config "$GEOMETRY_CONFIG" \
  --exact-box-config "$EXACT_BOX_CONFIG" \
  --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --dataset "$RUN_ROOT/dataset.json" \
  --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_boundary_capacity_dataset_e05.py \
  --result "$RUN_ROOT/result.json" \
  --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"
