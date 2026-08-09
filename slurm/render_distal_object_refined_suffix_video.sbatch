#!/usr/bin/env bash
#SBATCH --job-name=vlsa-object-suffix-video
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
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-object-refined-suffix-video}
ACCEPTED_RESULT=${ACCEPTED_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-object-refined-suffix-e05/object-refined-suffix-20260809c/result.json}
PREFIX_RESULT=${PREFIX_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-waypoint-closed-loop-e05/waypoint-closed-loop-20260809a/result.json}
EXPECTED_RESULT_SHA256=ada5dab210675b3d8ebcdb6e3f7f1eb978899fb6f4770679db902b4d898b7c52
EXPECTED_PREFIX_SHA256=ae040e13b06390eedf152817e9ab2ecd9300b0c9248764c2b6424694b14a45ce
MANIFEST=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "visual replay requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -f "$ACCEPTED_RESULT" && ! -L "$ACCEPTED_RESULT" && -f "$PREFIX_RESULT" && ! -L "$PREFIX_RESULT" ]] || die "visual replay input differs"
[[ "$(sha256sum "$ACCEPTED_RESULT" | awk '{print $1}')" == "$EXPECTED_RESULT_SHA256" ]] || die "accepted suffix hash differs"
[[ "$(sha256sum "$PREFIX_RESULT" | awk '{print $1}')" == "$EXPECTED_PREFIX_SHA256" ]] || die "accepted prefix hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() {
  code=$?
  trap - EXIT INT TERM
  if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then
    temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE"
  fi
  return "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\ndatasets: %s\nassets: %s\n' "$safelibero_root" "$safelibero_root/bddl_files" "$safelibero_root/init_files" "$REMOTE_REPO/safelibero/libero/datasets" "$safelibero_root/assets" >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import imageio, numpy, torch; print({"runtime":"visual_replay","torch":torch.__version__,"imageio":imageio.__version__})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" scripts/render_distal_affine_oracle_closed_loop_video.py \
  --repo-root "$REMOTE_REPO" --manifest "$MANIFEST" \
  --accepted-result "$ACCEPTED_RESULT" --prefix-result "$PREFIX_RESULT" \
  --expected-result-sha256 "$EXPECTED_RESULT_SHA256" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --video-fps 20 \
  --video "$RUN_ROOT/episode.mp4" --final-jpg "$RUN_ROOT/final.jpg" \
  --receipt "$RUN_ROOT/receipt.json"
