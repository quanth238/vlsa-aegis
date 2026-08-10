#!/usr/bin/env bash
#SBATCH --job-name=vlsa-input-ablate
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-matched-input-ablation}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_matched_input_ablation_moka10.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "matched-input ablation requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]] || die "runtime or immutable source root missing"
[[ -f "$SOURCE_ROOT/dataset.json" && -f "$SOURCE_ROOT/collection-result.json" ]] || die "immutable fresh source artifacts missing"
[[ "$(sha256sum "$SOURCE_ROOT/dataset.json" | awk '{print $1}')" == 585f696eedef9bd9a4c9d0ba576d6d86e5660638696862ec2d5d1af434fb426b ]] || die "fresh dataset hash differs"
[[ "$(sha256sum "$SOURCE_ROOT/collection-result.json" | awk '{print $1}')" == bd85f3175c790a153833a7c903a4e119c71f771271b2ecb92cc1c579269ce966 ]] || die "fresh collection hash differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 9a62b52423386e34ccbc947700ff5437f3fbd3791b4faf11a61154c0483060fe ]] || die "matched-input config hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy, scipy, torch; print({"numpy":numpy.__version__,"scipy":scipy.__version__,"torch":torch.__version__})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_matched_input_ablation tests.test_distal_complete_osc_margin -v >>"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_matched_input_ablation_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --dataset "$SOURCE_ROOT/dataset.json" \
  --collection-result "$SOURCE_ROOT/collection-result.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --old56-model "$RUN_ROOT/old56-model.npz" \
  --complete-model "$RUN_ROOT/completeOSC-model.npz" \
  --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_matched_input_ablation_moka10.py \
  --config "$CONFIG" --dataset "$SOURCE_ROOT/dataset.json" \
  --result "$RUN_ROOT/result.json" \
  --old56-model "$RUN_ROOT/old56-model.npz" \
  --complete-model "$RUN_ROOT/completeOSC-model.npz" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"
