#!/usr/bin/env bash
#SBATCH --job-name=vlsa-compact-select
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
OUTPUT=${OUTPUT:?}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_distal_compact_selector_ablation.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
die() { printf '%s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || die "CPU allocation required"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*|worker-0|worker-3) die "forbidden node" ;; esac
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "dirty source"
test ! -e "$OUTPUT" || die "output already exists"
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_compact_selector_ablation -v
tmp=$(mktemp "$(dirname "$OUTPUT")/.compact-selector.XXXXXX")
"$AEGIS_PYTHON" -m scripts.audit_distal_compact_selector_ablation \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$tmp"
mv "$tmp" "$OUTPUT"
