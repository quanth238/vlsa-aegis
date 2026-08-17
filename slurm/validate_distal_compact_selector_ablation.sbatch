#!/usr/bin/env bash
#SBATCH --job-name=vlsa-val-compact-select
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}
PRODUCER=${PRODUCER:?}; REPLAY=${REPLAY:?}; OUTPUT=${OUTPUT:?}
CONFIG=${CONFIG:-$REMOTE_REPO/configs/vlsa_distal_compact_selector_ablation.v1.json}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
die() { printf '%s\n' "$*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "allocation required"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "one H100 required"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*|worker-0|worker-3) die "forbidden node" ;; esac
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "dirty source"
test -f "$PRODUCER"; test -f "$REPLAY"; test ! -e "$OUTPUT" || die "validator input/output differs"
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests \
  -p 'test_distal_compact_selector_ablation.py' -v
tmp=$(mktemp "$(dirname "$OUTPUT")/.compact-selector-validation.XXXXXX")
"$AEGIS_PYTHON" -m scripts.validate_distal_compact_selector_ablation \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --producer "$PRODUCER" --replay "$REPLAY" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$tmp"
mv "$tmp" "$OUTPUT"
