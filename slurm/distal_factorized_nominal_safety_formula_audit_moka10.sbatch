#!/usr/bin/env bash
#SBATCH --job-name=vlsa-q-fk-phi
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-nominal-safety-formula-audit}
TRAJECTORY_RUN=${TRAJECTORY_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution/factorized-execution-20260811d}
TERMINAL_RUN=${TERMINAL_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-terminal-bias-audit/terminal-bias-20260811b}
STRUCTURED_RUN=${STRUCTURED_RUN:-/mnt/data/quanth/experiments/vlsa-distal-factorized-structured-orientation/structured-orientation-20260811b}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_nominal_safety_formula_audit_moka10.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 32768 ]] || die "registered allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "formula audit requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$TRAJECTORY_RUN" && -d "$TERMINAL_RUN" && -d "$STRUCTURED_RUN" ]] || die "runtime or immutable source missing"
[[ ! -L "$TRAJECTORY_RUN" && ! -L "$TERMINAL_RUN" && ! -L "$STRUCTURED_RUN" ]] || die "immutable source symlink forbidden"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 43c0ca12bf2c83de294383d2952586713b9f1644fc0e0cb8637ebb10c17b181a ]] || die "formula config differs"
RUN_ROOT=$OUTPUT_ROOT/$RUN_ID
mkdir -p "$OUTPUT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_nominal_safety_audit -v >"$RUN_ROOT/preflight.txt" 2>&1
common=(
  --repo-root "$REMOTE_REPO" --config "$CONFIG"
  --trajectory-metadata "$TRAJECTORY_RUN/trajectory-metadata.json" --trajectory-array "$TRAJECTORY_RUN/trajectory-dataset.npz"
  --terminal-records "$TERMINAL_RUN/records.npz" --terminal-result "$TERMINAL_RUN/result.json" --terminal-validation "$TERMINAL_RUN/validation.json"
  --structured-flat-model "$STRUCTURED_RUN/flat-model.npz" --structured-time-model "$STRUCTURED_RUN/time-model.npz"
  --structured-predictions "$STRUCTURED_RUN/predictions.npz" --structured-result "$STRUCTURED_RUN/result.json" --structured-validation "$STRUCTURED_RUN/validation.json"
  --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/audit_distal_factorized_nominal_safety_formula_moka10.py "${common[@]}" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_nominal_safety_formula_moka10.py "${common[@]}" --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
