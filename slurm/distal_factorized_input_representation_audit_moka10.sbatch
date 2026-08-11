#!/usr/bin/env bash
#SBATCH --job-name=vlsa-input-audit
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-input-representation-audit}
TRAJECTORY_ROOT=${TRAJECTORY_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
FLAT_ROOT=${FLAT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811b}
TIME_ROOT=${TIME_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-time-conditioned-decoder/time-conditioned-decoder-20260811a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_input_representation_audit_moka10.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "input-representation audit requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$TRAJECTORY_ROOT" && -d "$COMPLETE_ROOT" && -d "$FLAT_ROOT" && -d "$TIME_ROOT" ]] || die "runtime or immutable source missing"
[[ ! -L "$TRAJECTORY_ROOT" && ! -L "$COMPLETE_ROOT" && ! -L "$FLAT_ROOT" && ! -L "$TIME_ROOT" ]] || die "immutable source symlink forbidden"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "input-representation source identity differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == e9e1b4d954696a704a28e1982270235ce1ff65d25166199fbf3eef4fa3805043 ]] || die "input-representation config differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable input-representation audit root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_input_representation_audit tests.test_distal_factorized_time_conditioned_decoder tests.test_distal_factorized_one_sided_geometry -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$CONFIG"
  --complete-dataset "$COMPLETE_ROOT/dataset.json"
  --trajectory-metadata "$TRAJECTORY_ROOT/trajectory-metadata.json"
  --trajectory-array "$TRAJECTORY_ROOT/trajectory-dataset.npz"
  --flat-model "$FLAT_ROOT/experimental-model.npz"
  --flat-predictions "$FLAT_ROOT/predictions.npz"
  --flat-result "$FLAT_ROOT/result.json"
  --flat-validation "$FLAT_ROOT/validation.json"
  --time-model "$TIME_ROOT/experimental-model.npz"
  --time-predictions "$TIME_ROOT/predictions.npz"
  --time-result "$TIME_ROOT/result.json"
  --time-validation "$TIME_ROOT/validation.json"
  --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/audit_distal_factorized_input_representation_moka10.py "${common[@]}" --records "$RUN_ROOT/records.npz" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_input_representation_moka10.py "${common[@]}" --records "$RUN_ROOT/records.npz" --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
