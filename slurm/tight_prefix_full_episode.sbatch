#!/usr/bin/env bash
#SBATCH --job-name=vlsa-tight-full
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
PI05_CHECKPOINT=${PI05_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-tight-prefix-full-episode}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_full_episode.v1.json}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "tight full-episode producer requires Slurm"
POLICY_PORT=${POLICY_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
[[ "$POLICY_PORT" =~ ^[0-9]+$ && "$POLICY_PORT" -ge 20000 && "$POLICY_PORT" -le 39999 ]] || die "invalid policy port"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected 64 GiB"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -x "$OPENPI_PYTHON" ]] || die "Python environment missing"
[[ -d "$PI05_CHECKPOINT" && ! -L "$PI05_CHECKPOINT" ]] || die "checkpoint missing"
[[ -f "$EXPERIMENT_CONFIG" ]] || die "config missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree dirty"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID/producer
mkdir -p "$EXPERIMENT_ROOT/$RUN_ID"
mkdir "$RUN_ROOT" || die "immutable producer root exists"
SERVER_LOG=$RUN_ROOT/policy-server.log
FAILURE_PATH=$RUN_ROOT/runtime-failure.json
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
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$exit_code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE_PATH"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
export HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
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
"$AEGIS_PYTHON" -m unittest tests.test_tight_prefix_full_episode tests.test_terminalized_full_episode tests.test_terminal_branching -v >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
"$OPENPI_PYTHON" -m py_compile openpi/src/openpi/models/pi0.py openpi/src/openpi/policies/policy.py openpi/src/openpi/serving/websocket_policy_server.py >"$RUN_ROOT/openpi-preflight.txt" 2>&1
(
  cd "$REMOTE_REPO"
  exec "$OPENPI_PYTHON" scripts/serve_policy.py --port "$POLICY_PORT" policy:checkpoint --policy.config pi05_libero --policy.dir "$PI05_CHECKPOINT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 200 "$SERVER_LOG" >&2 || true; die "policy server exited"; }
  if curl --fail --silent --show-error "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl --fail --silent --show-error "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null || die "policy healthz failed"

"$AEGIS_PYTHON" scripts/evaluate_tight_prefix_full_episode.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --config "$EXPERIMENT_CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --replica producer --host 127.0.0.1 --port "$POLICY_PORT" \
  --output "$RUN_ROOT/result.json"
