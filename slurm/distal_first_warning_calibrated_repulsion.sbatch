#!/usr/bin/env bash
#SBATCH --job-name=vlsa-calibrated-repulse
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --array=0-2%2
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-l5-aegis-consistent}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
PI05_CHECKPOINT=${PI05_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-first-warning-calibrated-repulsion}
TABLE1_ROOT=${TABLE1_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "simulation requires Slurm array"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" == 3 && "${SLURM_ARRAY_TASK_MAX:-}" == 2 ]] || die "array differs"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*) die "login node forbidden" ;; esac
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ -x "$OPENPI_PYTHON" && -x "$AEGIS_PYTHON" ]] || die "registered Python is missing"
[[ -d "$PI05_CHECKPOINT" && ! -L "$PI05_CHECKPOINT" ]] || die "pi0.5 checkpoint differs"
[[ -d "$TABLE1_ROOT" && ! -L "$TABLE1_ROOT" ]] || die "Table1 root differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID/case-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$EXPERIMENT_ROOT/$RUN_ID"
mkdir "$RUN_ROOT" || die "immutable case root exists"
SERVER_LOG=$RUN_ROOT/pi05-libero-server.log
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
    printf '{"exit_code":%d,"scientific_result":false,"slurm_array_task_id":%d,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$exit_code" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE_PATH"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi HF_HOME=/mnt/data/quanth/cache/huggingface XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_first_warning_calibrated_repulsion tests.test_distal_analytical_repulsion_generalization -v >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
"$AEGIS_PYTHON" -c 'from scripts.evaluate_distal_first_warning_calibrated_repulsion import main' >>"$RUN_ROOT/evaluation-preflight.txt" 2>&1
(
  cd "$REMOTE_REPO"
  exec "$OPENPI_PYTHON" scripts/serve_policy.py --port "$POLICY_PORT" policy:checkpoint --policy.config pi05_libero --policy.dir "$PI05_CHECKPOINT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 200 "$SERVER_LOG" >&2 || true; die "pi0.5 server exited before healthz"; }
  if curl --fail --silent --show-error "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl --fail --silent --show-error "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null || die "pi0.5 healthz failed"

"$AEGIS_PYTHON" scripts/evaluate_distal_first_warning_calibrated_repulsion.py \
  --repo-root "$REMOTE_REPO" \
  --population-manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --selection-manifest "$REMOTE_REPO/manifests/vlsa_distal_analytical_repulsion_generalization.v1.jsonl" \
  --table1-root "$TABLE1_ROOT" \
  --geometry-config "$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_generalization_pilot.v4.json" \
  --base-experiment-config "$REMOTE_REPO/configs/vlsa_distal_analytical_repulsion_generalization.v1.json" \
  --calibrated-config "$REMOTE_REPO/configs/vlsa_distal_first_warning_calibrated_repulsion.v1.json" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --host 127.0.0.1 \
  --port "$POLICY_PORT" \
  --output "$RUN_ROOT/result.json"
