#!/usr/bin/env bash
#SBATCH --job-name=vlsa-moka-radius-val
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --qos=normal
#SBATCH --exclude=worker-0
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-moka-radius-val-%j.out

set -euo pipefail

repo_root=${REPO_ROOT:-/home/quanth/working_space/vlsa-aegis-moka-audit}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
result=${RESULT:?RESULT is required}
reference_result=${REFERENCE_RESULT:?REFERENCE_RESULT is required}
output=${OUTPUT:?OUTPUT is required}
cd "${repo_root}"

test "$(git rev-parse HEAD)" = "${expected_commit}"
test -z "$(git status --porcelain --untracked-files=all)"
test -n "${CUDA_VISIBLE_DEVICES:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=${repo_root}/safelibero:${repo_root}/main:${repo_root}

/mnt/data/quanth/venvs/safety_vla/main/bin/python \
  scripts/validate_distal_moka_secant_radius_e05.py \
  --result "${result}" \
  --reference-result "${reference_result}" \
  --expected-commit "${expected_commit}" \
  --output "${output}"
