#!/usr/bin/env bash
#SBATCH --job-name=vlsa-orient-mask
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-orientation-joint-mask}
TRAJECTORY_ROOT=${TRAJECTORY_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
COMPLETE_ROOT=${COMPLETE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
FLAT_ROOT=${FLAT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-one-sided-geometry/one-sided-geometry-20260811b}
TIME_ROOT=${TIME_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-time-conditioned-decoder/time-conditioned-decoder-20260811a}
AUDIT_ROOT=${AUDIT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-input-representation-audit/input-representation-audit-20260811a}
AUDIT_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_input_representation_audit_moka10.v1.json
JOINT_CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_orientation_joint_mask_moka10.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "orientation joint-mask confirmation requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$TRAJECTORY_ROOT" && -d "$COMPLETE_ROOT" && -d "$FLAT_ROOT" && -d "$TIME_ROOT" && -d "$AUDIT_ROOT" ]] || die "runtime or immutable source missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "orientation joint-mask source identity differs"
[[ "$(sha256sum "$JOINT_CONFIG" | awk '{print $1}')" == bc11384a0aeb35ab8ad609d47e460d0b3eea0efec991581e04f8ddf5bc616fce ]] || die "orientation joint-mask config differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable orientation joint-mask root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_orientation_joint_mask tests.test_distal_factorized_input_representation_audit -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$AUDIT_CONFIG" --joint-config "$JOINT_CONFIG"
  --complete-dataset "$COMPLETE_ROOT/dataset.json"
  --trajectory-metadata "$TRAJECTORY_ROOT/trajectory-metadata.json"
  --trajectory-array "$TRAJECTORY_ROOT/trajectory-dataset.npz"
  --flat-model "$FLAT_ROOT/experimental-model.npz" --flat-predictions "$FLAT_ROOT/predictions.npz"
  --flat-result "$FLAT_ROOT/result.json" --flat-validation "$FLAT_ROOT/validation.json"
  --time-model "$TIME_ROOT/experimental-model.npz" --time-predictions "$TIME_ROOT/predictions.npz"
  --time-result "$TIME_ROOT/result.json" --time-validation "$TIME_ROOT/validation.json"
  --audit-records "$AUDIT_ROOT/records.npz" --audit-result "$AUDIT_ROOT/result.json" --audit-validation "$AUDIT_ROOT/validation.json"
  --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/audit_distal_factorized_orientation_joint_mask_moka10.py "${common[@]}" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_orientation_joint_mask_moka10.py "${common[@]}" --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
