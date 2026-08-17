#!/usr/bin/env bash
#SBATCH --job-name=vlsa-wb-select-audit
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
OUTPUT=${OUTPUT:?}; PREDICTION=${PREDICTION:?}
PREDICTION_FILE_SHA256=${PREDICTION_FILE_SHA256:?}
PREDICTION_PAYLOAD_SHA256=${PREDICTION_PAYLOAD_SHA256:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
die() { printf '%s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || die "CPU allocation required"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*|worker-0|worker-3) die "forbidden node" ;; esac
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "dirty source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_whole_body_action_selection_audit -v
tmp=$(mktemp "$(dirname "$OUTPUT")/.selection.XXXXXX")
"$AEGIS_PYTHON" -m scripts.audit_distal_whole_body_action_selection \
  --repo-root "$REMOTE_REPO" --prediction "$PREDICTION" \
  --prediction-file-sha256 "$PREDICTION_FILE_SHA256" \
  --prediction-payload-sha256 "$PREDICTION_PAYLOAD_SHA256" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$tmp"
mv "$tmp" "$OUTPUT"
