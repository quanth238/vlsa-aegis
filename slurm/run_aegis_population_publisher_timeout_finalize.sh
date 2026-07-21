#!/usr/bin/env bash
set -euo pipefail

# Finalize-only recovery for publisher 28940. This runner never performs
# inference, simulation, rendering, training, aggregation, failure-report
# construction, or gallery construction. It independently revalidates the
# immutable completed attempt and atomically writes the one run-level receipt.

readonly RUN_ID=vlsa-table1-contact-authority-population-20260718a
readonly POPULATION_ARRAY_JOB_ID=28609
readonly TIMED_OUT_PUBLISHER_JOB_ID=28940
readonly POPULATION_SOURCE_GIT_COMMIT=1592aa59361f431ba96c6ddcbebcb596f6c20853
readonly TIMED_OUT_PUBLISHER_SOURCE_GIT_COMMIT=5fcb15015a6d24b5e86c685d5cdfaa748778fb99

: "${EXPECTED_RECOVERY_GIT_COMMIT:?set the exact reviewed recovery release commit}"

POPULATION_RUNTIME_REPO=${POPULATION_RUNTIME_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
TIMED_OUT_PUBLISHER_REPO=${TIMED_OUT_PUBLISHER_REPO:-/home/quanth/working_space/vlsa-aegis-publication-retry-v3}
RECOVERY_RELEASE_REPO=${RECOVERY_RELEASE_REPO:-/home/quanth/working_space/vlsa-aegis-publication-timeout-finalize}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

for name in SLURM_JOB_ID SLURM_JOB_DEPENDENCY SLURMD_NODENAME \
  SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE; do
  [[ -n "${!name:-}" ]] || {
    echo "missing required allocation variable: $name" >&2
    exit 2
  }
done
[[ "$SLURM_JOB_DEPENDENCY" == "afterany:$TIMED_OUT_PUBLISHER_JOB_ID" ]] || {
  echo "finalize recovery dependency must be afterany:$TIMED_OUT_PUBLISHER_JOB_ID" >&2
  exit 2
}
case "$SLURMD_NODENAME" in
  worker-3|login*|login-restricted*)
    echo "finalize recovery cannot execute on $SLURMD_NODENAME" >&2
    exit 2
    ;;
esac
[[ "$SLURM_CPUS_PER_TASK" == 4 && "$SLURM_MEM_PER_NODE" == 32768 ]] || {
  echo "finalize recovery allocation changed from 4 CPUs / 32768 MiB" >&2
  exit 2
}
[[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES:-}" == NoDevFiles ]] || {
  echo "finalize recovery must not consume a GPU" >&2
  exit 2
}

for source_root in "$POPULATION_RUNTIME_REPO" "$TIMED_OUT_PUBLISHER_REPO" \
  "$RECOVERY_RELEASE_REPO"; do
  case "$source_root" in
    /home/quanth/working_space/*) ;;
    *) echo "source escaped /home/quanth/working_space: $source_root" >&2; exit 2 ;;
  esac
  [[ -d "$source_root" && ! -L "$source_root" ]] || {
    echo "source is missing or symlinked: $source_root" >&2
    exit 2
  }
done
[[ "$(git -C "$POPULATION_RUNTIME_REPO" rev-parse HEAD)" == \
  "$POPULATION_SOURCE_GIT_COMMIT" ]] || {
  echo "population source commit changed" >&2
  exit 2
}
[[ "$(git -C "$TIMED_OUT_PUBLISHER_REPO" rev-parse HEAD)" == \
  "$TIMED_OUT_PUBLISHER_SOURCE_GIT_COMMIT" ]] || {
  echo "timed-out publisher source commit changed" >&2
  exit 2
}
[[ "$(git -C "$RECOVERY_RELEASE_REPO" rev-parse HEAD)" == \
  "$EXPECTED_RECOVERY_GIT_COMMIT" ]] || {
  echo "recovery source commit changed" >&2
  exit 2
}
for source_root in "$POPULATION_RUNTIME_REPO" "$TIMED_OUT_PUBLISHER_REPO" \
  "$RECOVERY_RELEASE_REPO"; do
  [[ -z "$(git -C "$source_root" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "source tree is dirty: $source_root" >&2
    exit 2
  }
done

readonly RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
readonly TIMED_OUT_ATTEMPT_ROOT=$RUN_ROOT/publication-attempts/job-$TIMED_OUT_PUBLISHER_JOB_ID
readonly RECOVERY_ATTEMPT_ROOT=$RUN_ROOT/publication-attempts/job-$SLURM_JOB_ID
mkdir "$RECOVERY_ATTEMPT_ROOT" || {
  echo "recovery attempt already exists: $RECOVERY_ATTEMPT_ROOT" >&2
  exit 2
}

RECOVERY_STAGE=timeout_accounting
recovery_cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ $exit_code -ne 0 && ! -e "$RECOVERY_ATTEMPT_ROOT/publisher-failure.json" ]]; then
    local temporary
    temporary=$(mktemp "$RECOVERY_ATTEMPT_ROOT/.publisher-failure.XXXXXX")
    {
      printf '{\n'
      printf '  "schema_version": "vlsa_table1_population_publisher_failure.v1",\n'
      printf '  "status": "apparatus_failure",\n'
      printf '  "scientific_result": false,\n'
      printf '  "run_id": "%s",\n' "$RUN_ID"
      printf '  "population_array_job_id": "%s",\n' "$POPULATION_ARRAY_JOB_ID"
      printf '  "publisher_job_id": "%s",\n' "$SLURM_JOB_ID"
      printf '  "host": "%s",\n' "$SLURMD_NODENAME"
      printf '  "failure_stage": "%s",\n' "$RECOVERY_STAGE"
      printf '  "exit_code": %d\n' "$exit_code"
      printf '}\n'
    } >"$temporary"
    chmod 0444 "$temporary"
    mv "$temporary" "$RECOVERY_ATTEMPT_ROOT/publisher-failure.json"
  fi
  return "$exit_code"
}
trap recovery_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

readonly TIMEOUT_ACCOUNTING=$RECOVERY_ATTEMPT_ROOT/timed-out-publisher-sacct.txt
sacct -X -n -P -j "$TIMED_OUT_PUBLISHER_JOB_ID" \
  --format=JobID,JobIDRaw,State,ExitCode,NodeList,Elapsed,Start,End \
  >"$TIMEOUT_ACCOUNTING"
chmod 0444 "$TIMEOUT_ACCOUNTING"

readonly RECOVERY_AUTHORITY=$RECOVERY_ATTEMPT_ROOT/publisher-timeout-recovery-authority.json
RECOVERY_STAGE=timeout_recovery_authority
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
"$AEGIS_PYTHON" \
  "$RECOVERY_RELEASE_REPO/scripts/build_aegis_publisher_timeout_recovery_authority.py" \
  --run-root "$RUN_ROOT" \
  --population-source-repo "$POPULATION_RUNTIME_REPO" \
  --timed-out-publisher-source-repo "$TIMED_OUT_PUBLISHER_REPO" \
  --recovery-source-repo "$RECOVERY_RELEASE_REPO" \
  --recovery-source-commit "$EXPECTED_RECOVERY_GIT_COMMIT" \
  --timed-out-attempt-root "$TIMED_OUT_ATTEMPT_ROOT" \
  --timed-out-publisher-log \
    "/mnt/data/quanth/slurm_logs/vlsa-aegis-publisher-retry-28940.out" \
  --timed-out-publisher-accounting "$TIMEOUT_ACCOUNTING" \
  --validator "$RECOVERY_RELEASE_REPO/scripts/validate_aegis_run_artifacts.py" \
  --recovery-authority-builder \
    "$RECOVERY_RELEASE_REPO/scripts/build_aegis_publisher_timeout_recovery_authority.py" \
  --recovery-runner \
    "$RECOVERY_RELEASE_REPO/slurm/run_aegis_population_publisher_timeout_finalize.sh" \
  --recovery-sbatch \
    "$RECOVERY_RELEASE_REPO/slurm/aegis_population_publisher_timeout_finalize.sbatch" \
  --output "$RECOVERY_AUTHORITY"

RECOVERY_STAGE=population_finalize
"$AEGIS_PYTHON" \
  "$RECOVERY_RELEASE_REPO/scripts/validate_aegis_run_artifacts.py" \
  population-finalize \
  --run-root "$RUN_ROOT" \
  --prepublish-receipt "$TIMED_OUT_ATTEMPT_ROOT/prepublish-validation.json" \
  --config "$POPULATION_RUNTIME_REPO/configs/vlsa_table1_translational.json" \
  --manifest-receipt \
    "$POPULATION_RUNTIME_REPO/manifests/vlsa_table1_population.receipt.json" \
  --manifest "$POPULATION_RUNTIME_REPO/manifests/vlsa_table1_population.jsonl" \
  --results "$RUN_ROOT/tasks" \
  --summary "$TIMED_OUT_ATTEMPT_ROOT/population-summary.json" \
  --gallery "$TIMED_OUT_ATTEMPT_ROOT/gallery/index.html" \
  --failure-cases "$TIMED_OUT_ATTEMPT_ROOT/failure-analysis/cases.jsonl" \
  --failure-report "$TIMED_OUT_ATTEMPT_ROOT/failure-analysis/report.json" \
  --failure-markdown "$TIMED_OUT_ATTEMPT_ROOT/failure-analysis/report.md" \
  --publisher-timeout-recovery-authority "$RECOVERY_AUTHORITY" \
  --output "$RUN_ROOT/population-publication-receipt.json"

RECOVERY_STAGE=complete
