#!/usr/bin/env bash
#SBATCH --job-name=vlsa-early-grad-video
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
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
ACCEPTED_RESULT_COMMIT=${ACCEPTED_RESULT_COMMIT:?ACCEPTED_RESULT_COMMIT is required}
PREP=${PREP:?PREP is required}
RESULT=${RESULT:?RESULT is required}
VALIDATION=${VALIDATION:?VALIDATION is required}
EXPECTED_PREP_SHA256=${EXPECTED_PREP_SHA256:?EXPECTED_PREP_SHA256 is required}
EXPECTED_RESULT_SHA256=${EXPECTED_RESULT_SHA256:?EXPECTED_RESULT_SHA256 is required}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?EXPECTED_VALIDATION_SHA256 is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_tight_prefix_early_gradient_probe.v1.json}
MANIFEST=${MANIFEST:-$REMOTE_REPO/manifests/vlsa_table1_population.jsonl}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "gradient video requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || die "gradient video allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "gradient video requires one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "gradient video source differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "gradient video source dirty"
[[ ! -e "$OUTPUT_ROOT" ]] || die "immutable gradient video root exists"
[[ "$(sha256sum "$PREP" | awk '{print $1}')" == "$EXPECTED_PREP_SHA256" ]] || die "gradient video prep differs"
[[ "$(sha256sum "$RESULT" | awk '{print $1}')" == "$EXPECTED_RESULT_SHA256" ]] || die "gradient video result differs"
[[ "$(sha256sum "$VALIDATION" | awk '{print $1}')" == "$EXPECTED_VALIDATION_SHA256" ]] || die "gradient video validation differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
runtime=$(mktemp -d "$(dirname "$OUTPUT_ROOT")/.video-runtime.XXXXXX")
export LIBERO_CONFIG_PATH=$runtime/libero-config
mkdir "$LIBERO_CONFIG_PATH"
trap 'exit_code=$?; if [[ $exit_code -ne 0 ]]; then printf "{\"exit_code\":%d,\"status\":\"apparatus_failure\"}\n" "$exit_code" >"$runtime/runtime-failure.json"; fi; exit "$exit_code"' EXIT
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"

cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_tight_prefix_early_gradient_probe -v >"$runtime/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/render_tight_prefix_early_gradient_comparisons.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" --manifest "$MANIFEST" \
  --prep "$PREP" --result "$RESULT" --validation "$VALIDATION" \
  --expected-prep-sha256 "$EXPECTED_PREP_SHA256" \
  --expected-result-sha256 "$EXPECTED_RESULT_SHA256" \
  --expected-validation-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --accepted-result-commit "$ACCEPTED_RESULT_COMMIT" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output-root "$OUTPUT_ROOT"
