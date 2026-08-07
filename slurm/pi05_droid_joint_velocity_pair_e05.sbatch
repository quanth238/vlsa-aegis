#!/usr/bin/env bash
#SBATCH --job-name=vlsa-droid-jv-pair
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
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
PI05_DROID_CHECKPOINT=${PI05_DROID_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_droid}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-pi05-droid-joint-velocity-pair}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

case "$RUN_ID" in
  ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "joint-velocity simulation requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected exactly 64 GiB"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  die "joint-velocity simulation requires exactly one visible H100"
}
[[ -x "$OPENPI_PYTHON" && -x "$AEGIS_PYTHON" ]] || die "registered Python is missing"
[[ -d "$PI05_DROID_CHECKPOINT" && ! -L "$PI05_DROID_CHECKPOINT" ]] || {
  die "π0.5-DROID checkpoint is missing"
}
[[ -f "$PI05_DROID_CHECKPOINT/assets/droid/norm_stats.json" ]] || {
  die "π0.5-DROID normalization statistics are missing"
}
[[ -f "$ARCHIVED_TABLE1_RESULT" && ! -L "$ARCHIVED_TABLE1_RESULT" ]] || {
  die "archived Table-1 geometry source is missing"
}
[[ "$(sha256sum "$ARCHIVED_TABLE1_RESULT" | awk '{print $1}')" == \
  273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b ]] || {
  die "archived Table-1 geometry source hash differs"
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || {
  die "source commit differs"
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  die "source tree is dirty"
}

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root already exists"
SERVER_LOG=$RUN_ROOT/pi05-droid-server.log
FAILURE_PATH=$RUN_ROOT/runtime-failure.json
slurm_numeric_id=${SLURM_JOB_ID%%_*}
POLICY_PORT=$((20000 + 10#$slurm_numeric_id % 30000))
SERVER_PID=

cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ $exit_code -ne 0 && ! -e "$FAILURE_PATH" ]]; then
    temporary=$(mktemp "$RUN_ROOT/.runtime-failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' \
      "$exit_code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE_PATH"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export PIP_CACHE_DIR=/mnt/data/quanth/cache/pip
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
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
"$AEGIS_PYTHON" -m unittest \
  tests.test_pi05_droid_joint_velocity_pair \
  tests.test_multilink_ellipsoid -v \
  >"$RUN_ROOT/numeric-preflight.txt" 2>&1

(
  cd "$REMOTE_REPO"
  exec "$OPENPI_PYTHON" scripts/serve_policy.py \
    --port "$POLICY_PORT" \
    policy:checkpoint \
    --policy.config pi05_droid \
    --policy.dir "$PI05_DROID_CHECKPOINT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  kill -0 "$SERVER_PID" 2>/dev/null || {
    tail -n 200 "$SERVER_LOG" >&2 || true
    die "π0.5-DROID server exited before healthz"
  }
  if curl --fail --silent --show-error \
    "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error \
  "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null || {
  die "π0.5-DROID policy healthz failed"
}

exec "$AEGIS_PYTHON" scripts/evaluate_pi05_droid_joint_velocity_pair.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --config "$REMOTE_REPO/configs/vlsa_pi05_droid_joint_velocity_pair_e05.v1.json" \
  --checkpoint "$PI05_DROID_CHECKPOINT" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --host 127.0.0.1 \
  --port "$POLICY_PORT" \
  --output "$RUN_ROOT/result.json"
