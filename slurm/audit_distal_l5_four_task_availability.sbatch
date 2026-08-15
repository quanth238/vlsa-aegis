#!/usr/bin/env bash
#SBATCH --job-name=vlsa-l5-task-avail
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}; OUTPUT=${OUTPUT:?}
TABLE1_ROOT=${TABLE1_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ -n "${SLURM_JOB_ID:-}" && $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
test -d "$TABLE1_ROOT"; test ! -e "$OUTPUT" || exit 2; mkdir -p "$(dirname "$OUTPUT")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_l5_four_task_availability.py' -v
"$AEGIS_PYTHON" scripts/audit_distal_l5_four_task_availability.py --repo-root "$REMOTE_REPO" --table1-root "$TABLE1_ROOT" --config "$REMOTE_REPO/configs/vlsa_distal_l5_four_task_availability_audit.v1.json" --expected-commit "$EXPECTED_GIT_COMMIT" --output "$OUTPUT"
