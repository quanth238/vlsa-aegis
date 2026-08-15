#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-l5-wd
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
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
RESULT=${RESULT:?RESULT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "weight-decay validation requires Slurm" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  echo "requires exactly one H100" >&2; exit 2;
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || {
  echo "weight-decay validator source differs" >&2; exit 2;
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "weight-decay validator source is dirty" >&2; exit 2;
}
test -f "$RESULT" || { echo "weight-decay result is absent" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "weight-decay validation exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover \
  -s tests -p 'test_distal_l5_row01_weight_decay_ablation.py' -v
"$AEGIS_PYTHON" scripts/validate_distal_l5_row01_weight_decay_ablation.py \
  --repo-root "$REMOTE_REPO" --result "$RESULT" \
  --config "$REMOTE_REPO/configs/vlsa_distal_l5_row01_weight_decay_ablation.v1.json" \
  --producer-commit "$PRODUCER_COMMIT" --validator-commit "$VALIDATOR_COMMIT" \
  --output "$OUTPUT"
