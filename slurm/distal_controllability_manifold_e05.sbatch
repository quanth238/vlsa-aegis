#!/usr/bin/env bash
#SBATCH --job-name=vlsa-e05-manifold
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=normal
#SBATCH --exclude=worker-0
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-e05-manifold-%j.out

set -euo pipefail
repo_root=${REPO_ROOT:-/home/quanth/working_space/vlsa-aegis-moka-audit}
run_root=${RUN_ROOT:?RUN_ROOT is required}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
mkdir -p "${run_root}"
cd "${repo_root}"
test "$(git rev-parse HEAD)" = "${expected_commit}"
test -z "$(git status --porcelain --untracked-files=all)"
nvidia-smi -L > "${run_root}/allocation-preflight.txt"
test -n "${CUDA_VISIBLE_DEVICES:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=${repo_root}/safelibero:${repo_root}/main:${repo_root}
export LIBERO_CONFIG_PATH=${run_root}/libero-config
mkdir -p "${LIBERO_CONFIG_PATH}"
safelibero_root=${repo_root}/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "${safelibero_root}"
  printf 'bddl_files: %s\n' "${safelibero_root}/bddl_files"
  printf 'init_states: %s\n' "${safelibero_root}/init_files"
  printf 'datasets: %s\n' "${repo_root}/safelibero/libero/datasets"
  printf 'assets: %s\n' "${safelibero_root}/assets"
} > "${LIBERO_CONFIG_PATH}/config.yaml"
/mnt/data/quanth/venvs/safety_vla/main/bin/python -m unittest tests.test_distal_detour_manifold -v
/mnt/data/quanth/venvs/safety_vla/main/bin/python \
  scripts/evaluate_distal_controllability_manifold_e05.py \
  --repo-root "${repo_root}" \
  --manifest "${repo_root}/manifests/vlsa_table1_population.jsonl" \
  --archived /mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json \
  --geometry-config "${repo_root}/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json" \
  --experiment-config "${repo_root}/configs/vlsa_distal_controllability_manifold_e05.v1.json" \
  --expected-commit "${expected_commit}" \
  --output "${run_root}/result.json"
