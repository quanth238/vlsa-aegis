#!/usr/bin/env bash
#SBATCH --job-name=vlsa-affine-oracle-closed-loop
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-affine-oracle-closed-loop-e05}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
REPRESENTATION_ROOT=${REPRESENTATION_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-two-step-oracle-affine-safe-set/two-step-affine-oracle-20260809c}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_affine_oracle_closed_loop_e05.v1.json
SOURCE_CONFIG=$REMOTE_REPO/configs/vlsa_distal_two_step_margin_moka10.v1.json
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
REPRESENTATION_RESULT=$REPRESENTATION_ROOT/result.json
REPRESENTATION_VALIDATION=$REPRESENTATION_ROOT/validation.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "closed-loop oracle requires exactly one H100"
for path in "$REPRESENTATION_RESULT" "$REPRESENTATION_VALIDATION"; do
  [[ -f "$path" && ! -L "$path" ]] || die "closed-loop prerequisite missing"
done
[[ -x "$AEGIS_PYTHON" ]] || die "evaluation Python missing"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 0a82ee6c6250996e10d2a4c30a0bb48927d08240b298a0fb5b21a438c855186b ]] || die "closed-loop config hash differs"
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
"$AEGIS_PYTHON" -c 'import imageio, numpy, osqp, scipy, torch; print({"runtime":"simulation","torch":torch.__version__,"imageio":imageio.__version__})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_affine_oracle_closed_loop_e05 -v >>"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_affine_oracle_closed_loop_e05.py \
  --repo-root "$REMOTE_REPO" --population-manifest "$POPULATION" \
  --selected-manifest "$SELECTED" --archived-root "$ARCHIVED_ROOT" \
  --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" \
  --source-config "$SOURCE_CONFIG" --config "$CONFIG" \
  --representation-result "$REPRESENTATION_RESULT" \
  --representation-validation "$REPRESENTATION_VALIDATION" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/result.json" \
  --video "$RUN_ROOT/episode.mp4" --final-jpg "$RUN_ROOT/final.jpg"
"$AEGIS_PYTHON" scripts/validate_distal_affine_oracle_closed_loop_e05.py \
  --result "$RUN_ROOT/result.json" --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/validation.json"
