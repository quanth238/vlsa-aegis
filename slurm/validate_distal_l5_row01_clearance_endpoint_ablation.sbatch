#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-clear-end
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
REMOTE_REPO=${REMOTE_REPO:?}; PRODUCER_COMMIT=${PRODUCER_COMMIT:?}; VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?}; RESULT=${RESULT:?}; OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ -n "${SLURM_JOB_ID:-}" && $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test -f "$RESULT" && test ! -e "$OUTPUT" || exit 2; mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_l5_row01_clearance_endpoint_ablation.py' -v
"$AEGIS_PYTHON" scripts/validate_distal_l5_row01_clearance_endpoint_ablation.py --repo-root "$REMOTE_REPO" --result "$RESULT" --config "$REMOTE_REPO/configs/vlsa_distal_l5_row01_clearance_endpoint_ablation.v1.json" --producer-commit "$PRODUCER_COMMIT" --validator-commit "$VALIDATOR_COMMIT" --output "$OUTPUT"
