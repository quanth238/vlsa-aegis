#!/usr/bin/env bash
#SBATCH --job-name=vlsa-palm-prim
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%A_%a.out
set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?}; EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?}; OUTPUT_DIR=${OUTPUT_DIR:?}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_distal_palm_primitive_audit.v1.json}
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ -n "${SLURM_JOB_ID:-}" && $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || exit 2
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
case_id=$(
  "$AEGIS_PYTHON" - "$CONFIG_PATH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))["cohort"]
ids=c["contact_case_ids"]+c["control_case_ids"]
print(ids[int(sys.argv[2])])
PY
)
output="$OUTPUT_DIR/$case_id.json"
test ! -e "$output" || exit 2
mkdir -p "$OUTPUT_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO MUJOCO_GL=egl
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest discover -s tests -p 'test_distal_palm_primitive_audit.py' -v
"$AEGIS_PYTHON" scripts/audit_distal_palm_primitive_case.py \
  --repo-root "$REMOTE_REPO" \
  --config "$CONFIG_PATH" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --output "$output"
