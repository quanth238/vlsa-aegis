#!/usr/bin/env bash
#SBATCH --job-name=vlsa-predictive-video
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

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-predictive-flow-guidance-video}
ACCEPTED_RESULT=${ACCEPTED_RESULT:-/mnt/data/quanth/experiments/vlsa-predictive-flow-guidance-e05/vlsa-predictive-flow-e05-20260808d/result.json}
EXPECTED_RESULT_SHA256=${EXPECTED_RESULT_SHA256:-739adbca0287193bac477b737a7131e6fa9ed55c452ba9e7eea1113e5e4649c1}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_predictive_flow_guidance_e05.v1.json}
VIDEO_BASENAME=${VIDEO_BASENAME:-predictive-flow-e05-ellipsoids}
IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

case "$RUN_ID" in
  ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;;
esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "predictive video replay requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected exactly 64 GiB"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || {
  die "predictive video replay requires exactly one visible H100"
}
[[ -x "$AEGIS_PYTHON" && -x "$IMAGEIO_FFMPEG_EXE" ]] || die "runtime is missing"
[[ -f "$ACCEPTED_RESULT" && ! -L "$ACCEPTED_RESULT" ]] || die "accepted result is missing"
[[ -f "$CONFIG_PATH" && ! -L "$CONFIG_PATH" ]] || die "video config is missing"
case "$VIDEO_BASENAME" in
  ""|*[!A-Za-z0-9._-]*) die "unsafe VIDEO_BASENAME" ;;
esac
[[ "$(sha256sum "$ACCEPTED_RESULT" | awk '{print $1}')" == "$EXPECTED_RESULT_SHA256" ]] || {
  die "accepted result hash differs"
}
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || {
  die "source commit differs"
}
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  die "source tree is dirty"
}

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root already exists"
FAILURE_PATH=$RUN_ROOT/runtime-failure.json

cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ $exit_code -ne 0 && ! -e "$FAILURE_PATH" ]]; then
    temporary=$(mktemp "$RUN_ROOT/.runtime-failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' \
      "$exit_code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE_PATH"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export IMAGEIO_FFMPEG_EXE
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"

cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest \
  tests.test_multilink_ellipsoid_h100_contract \
  tests.test_multilink_ellipsoid -v \
  >"$RUN_ROOT/numeric-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/render_predictive_flow_guidance_video.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --config "$CONFIG_PATH" \
  --accepted-result "$ACCEPTED_RESULT" \
  --expected-result-sha256 "$EXPECTED_RESULT_SHA256" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --video "$RUN_ROOT/$VIDEO_BASENAME.mp4" \
  --preview "$RUN_ROOT/$VIDEO_BASENAME.jpg" \
  --receipt "$RUN_ROOT/video-receipt.json"
