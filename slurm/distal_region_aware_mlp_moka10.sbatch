#!/usr/bin/env bash
#SBATCH --job-name=vlsa-region-mlp
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
PI05_CHECKPOINT=${PI05_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-region-aware-mlp}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
DATASET_ROOT=${DATASET_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-affine-coefficient-dataset/affine-coeff-data-20260810b}
OFF_GRID_ROOT=${OFF_GRID_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-affine-oracle-comparison/affine-oracle-cmp-20260810a}
MULTI_ROOT=${MULTI_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-multi-region-affine-oracle/multi-region-affine-20260810a}
DECISION_ROOT=${DECISION_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-multi-region-decision-stability/region-decision-stability-20260810a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_region_aware_mlp_moka10.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
E05_ARCHIVED=$ARCHIVED_ROOT/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "region-aware experiment requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -x "$OPENPI_PYTHON" ]] || die "registered Python is missing"
[[ -d "$PI05_CHECKPOINT" && ! -L "$PI05_CHECKPOINT" ]] || die "pi0.5 checkpoint is missing"
for path in "$DATASET_ROOT/dataset.json" "$OFF_GRID_ROOT/result.json" "$MULTI_ROOT/result.json" "$MULTI_ROOT/validation.json" "$DECISION_ROOT/result.json" "$DECISION_ROOT/validation.json" "$E05_ARCHIVED"; do
  [[ -f "$path" && ! -L "$path" ]] || die "immutable region-aware input missing: $path"
done
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 976c8305b0b5d1e28803fa48ee936252c00a98c096ead0ae52e77d3239adc867 ]] || die "region-aware config hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
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
    temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE"
  fi
  return "$code"
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi HF_HOME=/mnt/data/quanth/cache/huggingface XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
export PIP_CACHE_DIR=/mnt/data/quanth/cache/pip UV_CACHE_DIR=/mnt/data/quanth/cache/uv WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\ndatasets: %s\nassets: %s\n' "$safelibero_root" "$safelibero_root/bddl_files" "$safelibero_root/init_files" "$REMOTE_REPO/safelibero/libero/datasets" "$safelibero_root/assets" >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy, osqp, scipy, torch; print({"numpy":numpy.__version__,"osqp":osqp.__version__,"scipy":scipy.__version__,"torch":torch.__version__})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_region_aware_mlp tests.test_distal_multi_region_decision_stability tests.test_distal_multi_region_affine_oracle -v >>"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_region_aware_mlp_moka10.py \
  --repo-root "$REMOTE_REPO" --population-manifest "$POPULATION" \
  --selected-manifest "$SELECTED" --archived-root "$ARCHIVED_ROOT" \
  --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" \
  --config "$CONFIG" --dataset "$DATASET_ROOT/dataset.json" \
  --off-grid-result "$OFF_GRID_ROOT/result.json" \
  --multi-region-result "$MULTI_ROOT/result.json" \
  --multi-region-validation "$MULTI_ROOT/validation.json" \
  --decision-stability-result "$DECISION_ROOT/result.json" \
  --decision-stability-validation "$DECISION_ROOT/validation.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --model "$RUN_ROOT/model.npz" --output "$RUN_ROOT/learned-result.json"
"$AEGIS_PYTHON" scripts/validate_distal_region_aware_mlp_moka10.py \
  --result "$RUN_ROOT/learned-result.json" --model "$RUN_ROOT/model.npz" \
  --config "$CONFIG" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/learned-validation.json"
if jq -e '.closed_loop_e05_authorized == true' "$RUN_ROOT/learned-validation.json" >/dev/null; then
  SERVER_LOG=$RUN_ROOT/pi05-libero-server.log
  slurm_numeric_id=${SLURM_JOB_ID%%_*}
  POLICY_PORT=$((20000 + 10#$slurm_numeric_id % 30000))
  (
    cd "$REMOTE_REPO"
    exec "$OPENPI_PYTHON" scripts/serve_policy.py --port "$POLICY_PORT" \
      policy:checkpoint --policy.config pi05_libero --policy.dir "$PI05_CHECKPOINT"
  ) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 200 "$SERVER_LOG" >&2 || true; die "pi0.5 server exited before healthz"; }
    if curl --fail --silent --show-error "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl --fail --silent --show-error "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null || die "pi0.5 healthz failed"
  "$AEGIS_PYTHON" scripts/evaluate_distal_region_aware_mlp_closed_loop_e05.py \
    --repo-root "$REMOTE_REPO" --manifest "$POPULATION" \
    --archived "$E05_ARCHIVED" --geometry-config "$GEOMETRY" \
    --exact-box-config "$EXACT_BOX" --config "$CONFIG" \
    --learned-result "$RUN_ROOT/learned-result.json" \
    --learned-validation "$RUN_ROOT/learned-validation.json" \
    --model "$RUN_ROOT/model.npz" --expected-commit "$EXPECTED_GIT_COMMIT" \
    --host 127.0.0.1 --port "$POLICY_PORT" \
    --output "$RUN_ROOT/closed-loop-result.json"
  "$AEGIS_PYTHON" scripts/validate_distal_region_aware_mlp_closed_loop_e05.py \
    --result "$RUN_ROOT/closed-loop-result.json" --config "$CONFIG" \
    --expected-commit "$EXPECTED_GIT_COMMIT" \
    --output "$RUN_ROOT/closed-loop-validation.json"
fi
