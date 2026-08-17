#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-tight-ee
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?}
PRODUCER_DIR=${PRODUCER_DIR:?}
REPLAY_DIR=${REPLAY_DIR:?}
OUTPUT=${OUTPUT:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_tight_ee_geometry_audit.v1.json}

gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ -n "${SLURM_JOB_ID:-}" && $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || exit 2
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
[[ -d "$PRODUCER_DIR" && -d "$REPLAY_DIR" && ! -e "$OUTPUT" ]] || exit 2
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_tight_ee_geometry -v
exec "$AEGIS_PYTHON" scripts/validate_tight_ee_geometry.py \
  --repo-root "$REMOTE_REPO" \
  --config "$CONFIG_PATH" \
  --producer-dir "$PRODUCER_DIR" \
  --replay-dir "$REPLAY_DIR" \
  --producer-commit "$PRODUCER_COMMIT" \
  --validator-commit "$VALIDATOR_COMMIT" \
  --output "$OUTPUT"
