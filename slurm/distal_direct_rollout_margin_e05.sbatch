#!/usr/bin/env bash
#SBATCH --job-name=vlsa-direct-margin-e05
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?required}
RUN_ID=${RUN_ID:?required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-direct-rollout-margin-e05}
DATA_ROOT=${DATA_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-two-step-margin-dataset/two-step-data-20260809a}
ARCHIVED=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_direct_rollout_margin_e05.v1.json
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
die(){ printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID";; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $(printf '%s\n' "$gpu" | wc -l) -eq 1 && "$gpu" == *H100* ]] || die "requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -x "$OPENPI_PYTHON" ]] || die "registered Python runtime missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 4d10473c3e98d8f196f49bb55c7909f8e123a6df1705d80f160a200bdfee4239 ]] || die "direct-margin config hash differs"
declare -A expected
expected["$DATA_ROOT/dataset.json"]=475913f1f42f26ed5cfc2a3424632ea7c8409986b9db0467e60157c85bb35fe3
expected["$DATA_ROOT/result.json"]=304383acd57d3ba8108007677c7c46ee3a8b65fe0981e4df69e7a9a75d9b4268
expected["$DATA_ROOT/validation.json"]=7cb255ed5ac1239ac6ac9129cc30bd4e8e26e7479825c0913523744d6f9b51c8
expected["$ARCHIVED"]=273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b
for path in "${!expected[@]}"; do
  [[ -f "$path" && ! -L "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "${expected[$path]}" ]] || die "immutable source differs: $path"
done
RUN_ROOT=$ROOT/$RUN_ID
mkdir -p "$ROOT"
mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup(){ code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
libero=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$libero"
  printf 'bddl_files: %s\n' "$libero/bddl_files"
  printf 'init_states: %s\n' "$libero/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$libero/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy, osqp, scipy, torch; print({"runtime":"simulation","torch":torch.__version__})' >"$RUN_ROOT/runtime-preflight.txt"
"$OPENPI_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print({"runtime":"training","cuda":torch.cuda.get_device_name(0)})' >>"$RUN_ROOT/runtime-preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_direct_rollout_margin tests.test_distal_two_step_margin tests.test_distal_gradient_random_control -v >"$RUN_ROOT/test-preflight.txt" 2>&1
"$OPENPI_PYTHON" scripts/train_distal_direct_rollout_margin_e05.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --dataset "$DATA_ROOT/dataset.json" --dataset-result "$DATA_ROOT/result.json" \
  --dataset-validation "$DATA_ROOT/validation.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --model "$RUN_ROOT/model.npz" \
  --output "$RUN_ROOT/training.json"
"$AEGIS_PYTHON" scripts/evaluate_distal_direct_rollout_margin_e05.py \
  --repo-root "$REMOTE_REPO" --population-manifest "$POPULATION" \
  --archived "$ARCHIVED" --geometry-config "$GEOMETRY" \
  --exact-box-config "$EXACT_BOX" --dataset "$DATA_ROOT/dataset.json" \
  --dataset-result "$DATA_ROOT/result.json" \
  --dataset-validation "$DATA_ROOT/validation.json" \
  --training "$RUN_ROOT/training.json" --model "$RUN_ROOT/model.npz" \
  --config "$CONFIG" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_direct_rollout_margin_e05.py \
  --result "$RUN_ROOT/result.json" --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"
