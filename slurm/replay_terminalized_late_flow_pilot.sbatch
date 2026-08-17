#!/usr/bin/env bash
#SBATCH --job-name=vlsa-terminal-flow-replay
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-terminal-flow-pilot}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
ACCEPTED_PRODUCER_COMMIT=${ACCEPTED_PRODUCER_COMMIT:?ACCEPTED_PRODUCER_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-terminalized-late-flow-pilot}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$REMOTE_REPO/configs/vlsa_distal_terminalized_late_flow.v1.json}
TABLE1_ROOT=${TABLE1_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
ARCHIVED_TABLE1_RESULT=${ARCHIVED_TABLE1_RESULT:-$TABLE1_ROOT/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json}
PRODUCER_RESULT=${PRODUCER_RESULT:?PRODUCER_RESULT is required}

die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die "pilot replay requires Slurm"
[[ "${SLURM_CPUS_PER_TASK:-}" == 8 ]] || die "expected exactly 8 CPUs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "expected 64 GiB"
[[ -x "$AEGIS_PYTHON" ]] || die "Python environment is missing"
[[ -f "$EXPERIMENT_CONFIG" && ! -L "$EXPERIMENT_CONFIG" ]] || die "experiment config is missing"
[[ -f "$ARCHIVED_TABLE1_RESULT" && ! -L "$ARCHIVED_TABLE1_RESULT" ]] || die "Table-1 result is missing"
[[ -f "$PRODUCER_RESULT" && ! -L "$PRODUCER_RESULT" ]] || die "producer result is missing"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" ]] || die "source commit differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source tree is dirty"

RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID/replay
mkdir -p "$EXPERIMENT_ROOT/$RUN_ID"
mkdir "$RUN_ROOT" || die "immutable replay root already exists"
FAILURE_PATH=$RUN_ROOT/runtime-failure.json
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ $exit_code -ne 0 && ! -e "$FAILURE_PATH" ]]; then
    temporary=$(mktemp "$RUN_ROOT/.runtime-failure.XXXXXX")
    printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$exit_code" "$SLURM_JOB_ID" >"$temporary"
    mv "$temporary" "$FAILURE_PATH"
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export QUERY_RISK_CPU_CANARY=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
{
  printf 'benchmark_root: %s\n' "$safelibero_root"
  printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
  printf 'init_states: %s\n' "$safelibero_root/init_files"
  printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
  printf 'assets: %s\n' "$safelibero_root/assets"
} >"$LIBERO_CONFIG_PATH/config.yaml"

cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest \
  tests.test_terminal_branching \
  tests.test_distal_exact_group_boundary -v \
  >"$RUN_ROOT/evaluation-preflight.txt" 2>&1

"$AEGIS_PYTHON" scripts/evaluate_terminalized_late_flow_pilot.py \
  --repo-root "$REMOTE_REPO" \
  --manifest "$REMOTE_REPO/manifests/vlsa_table1_population.jsonl" \
  --table1-root "$TABLE1_ROOT" \
  --archived "$ARCHIVED_TABLE1_RESULT" \
  --config "$EXPERIMENT_CONFIG" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --replica replay --host unused --port 1 \
  --run-root "$RUN_ROOT/runtime" --output "$RUN_ROOT/result.json" \
  --frozen-producer-result "$PRODUCER_RESULT" \
  --accepted-producer-commit "$ACCEPTED_PRODUCER_COMMIT"
