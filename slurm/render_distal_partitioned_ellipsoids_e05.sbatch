#!/usr/bin/env bash
#SBATCH --job-name=vlsa-distal-partitioned-geometry
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
EVALUATION_PYTHON=${EVALUATION_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-partitioned-ellipsoid-geometry}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_distal_partitioned_ellipsoid_shadow_e05.v3.json}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

case "$RUN_ID" in
  ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "SafeLIBERO rendering requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected exactly 64 GiB"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  die "rendering requires exactly one visible H100"
}
[[ -x "$EVALUATION_PYTHON" ]] || die "evaluation Python is missing"
[[ -f "$CONFIG_PATH" && ! -L "$CONFIG_PATH" ]] || die "geometry config is missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || {
  die "source commit differs"
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  die "source tree is dirty"
}

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root already exists"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
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
"$EVALUATION_PYTHON" -m unittest tests.test_multilink_ellipsoid -v \
  >"$RUN_ROOT/numeric-preflight.txt" 2>&1
"$EVALUATION_PYTHON" scripts/render_multilink_ellipsoid_overlay.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --config "$CONFIG_PATH" \
  --output-dir "$RUN_ROOT/visualization"
