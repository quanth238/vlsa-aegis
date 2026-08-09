#!/usr/bin/env bash
#SBATCH --job-name=vlsa-ellipsoid-support-e05
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?required}
RUN_ID=${RUN_ID:?required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-ellipsoid-candidate-support-e05}
ARCHIVED=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
SUCCESS=${SUCCESSFUL_SITL_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-sitl-minimal-replay-e05/sitl-minimal-replay-20260808h/result.json}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_distal_ellipsoid_candidate_support_e05.v1.json}
GEOMETRY=${GEOMETRY_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json}
die(){ printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID";; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader); [[ $(printf '%s\n' "$gpu"|wc -l) -eq 1 && "$gpu" == *H100* ]] || die "requires one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source differs"
[[ "$(sha256sum "$ARCHIVED"|awk '{print $1}')" == 273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b ]] || die "Table-1 differs"
[[ "$(sha256sum "$SUCCESS"|awk '{print $1}')" == e23320ebad7c17f8d07d52f204b0fc32c2c3d81ab8007d0a1f1b6ba365c7bf08 ]] || die "SITL differs"
RUN_ROOT=$ROOT/$RUN_ID; mkdir -p "$ROOT"; mkdir "$RUN_ROOT" || die "run exists"
trap 'code=$?; if [[ $code -ne 0 && ! -e "$RUN_ROOT/runtime-failure.json" ]]; then printf "{\"exit_code\":%d,\"scientific_result\":false,\"slurm_job_id\":\"%s\",\"status\":\"apparatus_failure\"}\n" "$code" "$SLURM_JOB_ID" >"$RUN_ROOT/runtime-failure.json"; fi' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config; mkdir "$LIBERO_CONFIG_PATH"; libero=$REMOTE_REPO/safelibero/libero/libero
{ printf 'benchmark_root: %s\n' "$libero"; printf 'bddl_files: %s\n' "$libero/bddl_files"; printf 'init_states: %s\n' "$libero/init_files"; printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"; printf 'assets: %s\n' "$libero/assets"; } >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_oracle_affine tests.test_distal_execution_margin_nn -v >"$RUN_ROOT/test-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_ellipsoid_candidate_support_e05.py --repo-root "$REMOTE_REPO" --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" --archived "$ARCHIVED" --successful-sitl "$SUCCESS" --geometry-config "$GEOMETRY" --config "$CONFIG" --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_ellipsoid_candidate_support_e05.py --result "$RUN_ROOT/result.json" --config "$CONFIG" --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/validation.json"
