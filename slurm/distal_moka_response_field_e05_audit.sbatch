#!/usr/bin/env bash
#SBATCH --job-name=vlsa-moka-field-audit
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=normal
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-moka-field-audit-%j.out

set -euo pipefail

repo_root=${REPO_ROOT:-/home/quanth/working_space/vlsa-aegis-table-repro}
run_root=${RUN_ROOT:?RUN_ROOT is required}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
frozen_result=${FROZEN_RESULT:?FROZEN_RESULT is required}
mkdir -p "${run_root}"
cd "${repo_root}"

test "$(git rev-parse HEAD)" = "${expected_commit}"
test -z "$(git status --porcelain --untracked-files=all)"
nvidia-smi -L > "${run_root}/allocation-preflight.txt"
test -n "${CUDA_VISIBLE_DEVICES:-}"

/mnt/data/quanth/venvs/safety_vla/main/bin/python \
  scripts/audit_distal_moka_response_field_e05.py \
  --repo-root "${repo_root}" \
  --manifest "${repo_root}/manifests/vlsa_table1_population.jsonl" \
  --archived /mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json \
  --geometry-config "${repo_root}/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json" \
  --experiment-config "${repo_root}/configs/vlsa_distal_moka_response_field_e05.v1.json" \
  --audit-config "${repo_root}/configs/vlsa_distal_moka_response_field_e05_audit.v1.json" \
  --frozen-result "${frozen_result}" \
  --expected-commit "${expected_commit}" \
  --output "${run_root}/result.json"
