#!/usr/bin/env bash
#SBATCH --job-name=vlsa-l6-proxy-scale
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
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-l5-aegis-consistent}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
RUN_ID=${RUN_ID:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-l6-proxy-scale-audit}
die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "L6 proxy-scale audit requires Slurm"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*) die "login node forbidden" ;; esac
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"
mkdir "$RUN_ROOT" || die "immutable run root exists"
trap 'code=$?; if [[ $code -ne 0 && ! -e "$RUN_ROOT/runtime-failure.json" ]]; then printf "{\"exit_code\":%d,\"scientific_result\":false,\"status\":\"apparatus_failure\"}\n" "$code" >"$RUN_ROOT/runtime-failure.json"; fi; exit $code' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_l6_proxy_scale_audit -v >"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/audit_distal_l6_proxy_scale.py \
  --repo-root "$REMOTE_REPO" \
  --config "$REMOTE_REPO/configs/vlsa_distal_l6_proxy_scale_audit.v1.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/result.json"
