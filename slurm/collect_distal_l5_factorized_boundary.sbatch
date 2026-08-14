#!/usr/bin/env bash
#SBATCH --job-name=vlsa-l5-factor-data
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --array=0-11%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-l5-aegis-consistent}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-l5-factorized-boundary}
MANIFEST=${MANIFEST:-$REMOTE_REPO/manifests/vlsa_distal_l5_factorized_boundary.v1.jsonl}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_distal_l5_factorized_boundary.v1.json}
LABELS=${LABELS:-/mnt/data/quanth/experiments/vlsa-aegis-table1/frozen-inputs/vlsa-table1-codex-labels-20260718a/labels.jsonl}
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
PI05_CHECKPOINT=${PI05_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}
DINO_CONFIG=${DINO_CONFIG:-/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/groundingdino/config/GroundingDINO_SwinT_OGC.py}
DINO_CHECKPOINT=${DINO_CHECKPOINT:-/mnt/data/quanth/cache/aegis/groundingdino/groundingdino_swint_ogc.pth}

die() { printf '%s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "Slurm array required"
[[ "${SLURM_ARRAY_TASK_COUNT:-}" == 12 && "${SLURM_ARRAY_TASK_MAX:-}" == 11 ]] || die "array contract differs"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*|worker-0|worker-3) die "forbidden node" ;; esac
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"
[[ -f "$MANIFEST" && -f "$CONFIG" && -f "$LABELS" ]] || die "registered input missing"
[[ "$(sha256sum "$LABELS" | awk '{print $1}')" == f9a862f28f168f02de4e0987e37d297de24b167ae50fb96c7f8243a76916880e ]] || die "frozen labels differ"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID/case-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$EXPERIMENT_ROOT/$RUN_ID"
mkdir "$RUN_ROOT" || die "immutable case root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
SERVER_PID=
cleanup() {
  code=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then
    tmp=$(mktemp "$RUN_ROOT/.runtime-failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_array_task_id":%d,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" >"$tmp"
    mv "$tmp" "$FAILURE"
  fi
  return "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi HF_HOME=/mnt/data/quanth/cache/huggingface
export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export IMAGEIO_FFMPEG_EXE=/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2
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

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
case_id=$(printf '%s\n' "$line" | sed -n 's/.*"case_id":"\([^"]*\)".*/\1/p')
[[ -n "$case_id" ]] || die "cannot read case id"
port=$((24000 + (10#$SLURM_ARRAY_JOB_ID + 10#$SLURM_ARRAY_TASK_ID) % 20000))
(
  cd "$REMOTE_REPO"
  exec "$OPENPI_PYTHON" scripts/serve_policy.py --port "$port" policy:checkpoint \
    --policy.config pi05_libero --policy.dir "$PI05_CHECKPOINT"
) >"$RUN_ROOT/pi05-server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 120 "$RUN_ROOT/pi05-server.log" >&2; die "policy server exited"; }
  curl --fail --silent "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && break
  sleep 1
done
curl --fail --silent "http://127.0.0.1:$port/healthz" >/dev/null || die "policy server unhealthy"

mkdir "$RUN_ROOT/results"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_l5_factorized_boundary.py' -v >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
"$AEGIS_PYTHON" main/evaluate_safelibero_aegis.py \
  --manifest "$MANIFEST" --labels "$LABELS" --mode aegis \
  --output-dir "$RUN_ROOT/results" --case-id "$case_id" --repo-root "$REMOTE_REPO" \
  --host 127.0.0.1 --port "$port" --groundingdino-config "$DINO_CONFIG" \
  --groundingdino-checkpoint "$DINO_CHECKPOINT" --groundingdino-device cpu \
  --failure-diagnostics
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=

mkdir "$RUN_ROOT/coverage"
"$AEGIS_PYTHON" scripts/audit_distal_l5_factorized_boundary.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" --manifest "$MANIFEST" \
  --case-index "$SLURM_ARRAY_TASK_ID" --case-root "$RUN_ROOT" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --classification-output "$RUN_ROOT/classification.json" \
  --coverage-output "$RUN_ROOT/coverage/result.json"
if [[ -f "$RUN_ROOT/coverage/result.json" ]] && \
   [[ "$(jq -r '.retained_state_count' "$RUN_ROOT/coverage/result.json")" == 1 ]]; then
  mkdir "$RUN_ROOT/risk"
  "$AEGIS_PYTHON" scripts/collect_distal_l5_factorized_risk.py \
    --repo-root "$REMOTE_REPO" --config "$CONFIG" --manifest "$MANIFEST" \
    --coverage-result "$RUN_ROOT/coverage/result.json" --table1-root "$RUN_ROOT" \
    --case-index "$SLURM_ARRAY_TASK_ID" --expected-commit "$EXPECTED_GIT_COMMIT" \
    --output "$RUN_ROOT/risk/result.json"
fi
