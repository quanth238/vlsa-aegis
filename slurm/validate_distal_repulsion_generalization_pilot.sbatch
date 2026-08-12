#!/usr/bin/env bash
#SBATCH --job-name=vlsa-repulse-gen-val
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-repulsion-generalization-pilot}
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "validation requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "allocation differs"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*) die "login node forbidden" ;; esac
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"

paths=(
  "$RUN_ROOT/goal_e00_l6_moka/result.json"
  "$RUN_ROOT/goal_e24_l6_milk/result.json"
  "$RUN_ROOT/spatial_e12_l5_moka/result.json"
)
for path in "${paths[@]}"; do [[ -f "$path" && ! -L "$path" ]] || die "producer result missing"; done
SUMMARY=$RUN_ROOT/summary.json
VALIDATION=$RUN_ROOT/validation.json
[[ ! -e "$SUMMARY" && ! -e "$VALIDATION" ]] || die "immutable aggregate exists"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_repulsion_generalization -v >"$RUN_ROOT/validation-preflight.txt" 2>&1
common=(
  --config "$REMOTE_REPO/configs/vlsa_distal_repulsion_generalization_pilot.v1.json"
  --selection-manifest "$REMOTE_REPO/manifests/vlsa_distal_repulsion_generalization_pilot.v1.jsonl"
  --result "${paths[0]}"
  --result "${paths[1]}"
  --result "${paths[2]}"
  --expected-commit "$EXPECTED_GIT_COMMIT"
)
"$AEGIS_PYTHON" scripts/aggregate_distal_repulsion_generalization_pilot.py "${common[@]}" --output "$SUMMARY"
"$AEGIS_PYTHON" scripts/validate_distal_repulsion_generalization_pilot.py --summary "$SUMMARY" "${common[@]}" --output "$VALIDATION"
