#!/usr/bin/env bash
#SBATCH --job-name=vlsa-aegis-jv-verify
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --exclude=worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 1
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || exit 1
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || exit 1
gpu_line=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 1
[[ -x "$AEGIS_PYTHON" ]] || exit 1
[[ -f "$RUN_ROOT/result.json" && ! -L "$RUN_ROOT/result.json" ]] || exit 1
[[ ! -e "$RUN_ROOT/validation.json" ]] || exit 1
[[ -f "$ARCHIVED_TABLE1_RESULT" && ! -L "$ARCHIVED_TABLE1_RESULT" ]] || exit 1
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || exit 1
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO

cd "$REMOTE_REPO"
exec "$AEGIS_PYTHON" scripts/validate_aegis_cartesian_joint_velocity_bridge_pair.py \
  --repo-root "$REMOTE_REPO" \
  --candidate "$RUN_ROOT/result.json" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --config "$REMOTE_REPO/configs/vlsa_aegis_cartesian_joint_velocity_bridge_pair_e05.v1.json" \
  --producer-commit "$PRODUCER_COMMIT" \
  --validator-commit "$VALIDATOR_COMMIT" \
  --output "$RUN_ROOT/validation.json"
