#!/usr/bin/env bash
#SBATCH --job-name=vlsa-moka-ranker-val
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --qos=normal
#SBATCH --exclude=worker-0
#SBATCH --output=/mnt/data/quanth/slurm_logs/vlsa-moka-ranker-validate-%j.out

set -euo pipefail
repo_root=${REPO_ROOT:-/home/quanth/working_space/vlsa-aegis-moka-audit}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}
result=${RESULT:?RESULT is required}
frozen_result=${FROZEN_RESULT:?FROZEN_RESULT is required}
output=${OUTPUT:?OUTPUT is required}
cd "${repo_root}"
test "$(git rev-parse HEAD)" = "${expected_commit}"
test -z "$(git status --porcelain --untracked-files=all)"
nvidia-smi -L >/dev/null
/mnt/data/quanth/venvs/safety_vla/main/bin/python \
  scripts/validate_distal_moka_frozen_ranker_audit_e05.py \
  --result "${result}" \
  --frozen-result "${frozen_result}" \
  --expected-commit "${expected_commit}" \
  --output "${output}"
