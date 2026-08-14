#!/usr/bin/env bash
#SBATCH --job-name=vlsa-audit-active-search
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_PRODUCER_COMMIT=${EXPECTED_PRODUCER_COMMIT:?EXPECTED_PRODUCER_COMMIT is required}
EXPECTED_VALIDATOR_COMMIT=${EXPECTED_VALIDATOR_COMMIT:?EXPECTED_VALIDATOR_COMMIT is required}
EXPECTED_AUDITOR_COMMIT=${EXPECTED_AUDITOR_COMMIT:?EXPECTED_AUDITOR_COMMIT is required}
PRODUCER_ROOT=${PRODUCER_ROOT:?PRODUCER_ROOT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "active search audit requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || { echo "requires exactly one H100" >&2; exit 2; }
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_AUDITOR_COMMIT" ]] || { echo "auditor source differs" >&2; exit 2; }
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || { echo "auditor source is dirty" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "audit output exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_active_boundary_search -v
"$AEGIS_PYTHON" scripts/audit_distal_active_boundary_search.py \
  --repo-root "$REMOTE_REPO" --producer-root "$PRODUCER_ROOT" \
  --search-config "$REMOTE_REPO/configs/vlsa_distal_active_boundary_search.v1.json" \
  --expected-producer-commit "$EXPECTED_PRODUCER_COMMIT" \
  --expected-validator-commit "$EXPECTED_VALIDATOR_COMMIT" \
  --expected-auditor-commit "$EXPECTED_AUDITOR_COMMIT" --output "$OUTPUT"
