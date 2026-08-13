#!/usr/bin/env bash
#SBATCH --job-name=vlsa-pncbf-value-e05
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-pncbf-value}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-pncbf-policy-value-e05}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
CONTROLLABILITY_RESULT=${CONTROLLABILITY_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-controllability-manifold-e05-20260813a/result.json}
PROPOSAL_RESULT=${PROPOSAL_RESULT:-/mnt/data/quanth/experiments/vlsa-distal-soft-prefix-live-replan-e05/soft-prefix-live-replan-e05-20260813c/result.json}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_pncbf_policy_value_e05.v1.json}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "simulation requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ -x "$AEGIS_PYTHON" ]] || die "evaluation Python is missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"
[[ "$(sha256sum "$ARCHIVED_TABLE1_RESULT" | awk '{print $1}')" == 273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b ]] || die "Table-1 hash differs"
[[ "$(sha256sum "$CONTROLLABILITY_RESULT" | awk '{print $1}')" == ddc676f775b19cdf507e88359d086b019f8aea2e3191e6a4c0023cb1109c6d21 ]] || die "controllability result differs"
[[ "$(sha256sum "$PROPOSAL_RESULT" | awk '{print $1}')" == 70494008c6a873803ff9fdf9a9c3925a5d7c2d8d3586fa8124a3eadea65e34a7 ]] || die "proposal result differs"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root exists"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}
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
"$AEGIS_PYTHON" -m unittest tests.test_distal_pncbf_backup tests.test_distal_pncbf_policy_value -v >"$RUN_ROOT/evaluation-preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_pncbf_backup_oracle_e05.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --controllability-result "$CONTROLLABILITY_RESULT" \
  --geometry-config "$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json" \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --host 127.0.0.1 --port 1 \
  --proposal-result "$PROPOSAL_RESULT" \
  --output "$RUN_ROOT/result.json"
