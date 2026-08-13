#!/usr/bin/env bash
#SBATCH --job-name=vlsa-moka-field-audit-val
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-moka-field-audit-val-%j.out

set -euo pipefail

repo_root=${REPO_ROOT:-/home/quanth/working_space/vlsa-aegis-table-repro}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
result=${RESULT:?RESULT is required}
frozen_result=${FROZEN_RESULT:?FROZEN_RESULT is required}
output=${OUTPUT:?OUTPUT is required}
cd "${repo_root}"

test "$(git rev-parse HEAD)" = "${expected_commit}"
test -z "$(git status --porcelain --untracked-files=all)"
test -n "${CUDA_VISIBLE_DEVICES:-}"

/mnt/data/quanth/venvs/safety_vla/main/bin/python \
  scripts/validate_distal_moka_response_field_e05_audit.py \
  --result "${result}" \
  --frozen-result "${frozen_result}" \
  --expected-commit "${expected_commit}" \
  --output "${output}"
