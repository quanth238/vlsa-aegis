#!/usr/bin/env bash
#SBATCH --job-name=validate-five-detour-e05
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
DETOUR_ONLY_ROOT=${DETOUR_ONLY_ROOT:?DETOUR_ONLY_ROOT is required}
RECEDING_ROOT=${RECEDING_ROOT:?RECEDING_ROOT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
[[ -n "${SLURM_JOB_ID:-}" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]]
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]]
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]]
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]]
[[ ! -e "$OUTPUT" ]]
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/validate_distal_five_action_detour_e05.py \
  --detour-only-root "$DETOUR_ONLY_ROOT" \
  --receding-root "$RECEDING_ROOT" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --detour-only-commit f4d3ad313f3930b567ac2d7d242b341b07bc1348 \
  --receding-commit ca84597d09f425660e914471483581f20de4d54e \
  --validator-commit "$EXPECTED_GIT_COMMIT" \
  --output "$OUTPUT"
