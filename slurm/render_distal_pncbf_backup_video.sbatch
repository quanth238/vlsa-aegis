#!/usr/bin/env bash
#SBATCH --job-name=vlsa-pncbf-video
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
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-moka-audit}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
RESULT_PATH=${RESULT_PATH:?RESULT_PATH is required}
EXPECTED_RESULT_SHA256=${EXPECTED_RESULT_SHA256:?EXPECTED_RESULT_SHA256 is required}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-pncbf-backup-video}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "render requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || die "allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires one H100"
[[ "$(sha256sum "$RESULT_PATH" | awk '{print $1}')" == "$EXPECTED_RESULT_SHA256" ]] || die "result hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source dirty"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
[[ ! -e "$RUN_ROOT" ]] || die "immutable render root exists"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$EXPERIMENT_ROOT/.libero-$RUN_ID
mkdir "$LIBERO_CONFIG_PATH"
trap 'rmdir "$LIBERO_CONFIG_PATH" 2>/dev/null || true' EXIT
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"

cd "$REMOTE_REPO"
"$AEGIS_PYTHON" scripts/render_distal_pncbf_backup_video.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --result "$RESULT_PATH" \
  --expected-result-sha256 "$EXPECTED_RESULT_SHA256" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output-root "$RUN_ROOT"
