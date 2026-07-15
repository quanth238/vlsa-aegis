#!/usr/bin/env bash

# Read one exact singleton array task from Slurm accounting.  JobIDRaw is the
# numeric parent allocation on this cluster, so it cannot identify task 0.
# Query the exact task and compare the display JobID instead.

crfs_wait_for_exact_completed_array_task() {
  local array_job_id=${1:-}
  local max_attempts=${2:-30}
  local delay_seconds=${3:-1}
  local exact_task_id attempt job_id observed_state observed_exit rest matches
  local accounting_output accounting_status
  local state exit_code
  local last_state= last_exit=

  case "$array_job_id" in *[!0-9]*|'') return 2 ;; esac
  case "$max_attempts" in *[!0-9]*|'') return 2 ;; esac
  case "$delay_seconds" in *[!0-9]*|'') return 2 ;; esac
  [ "$max_attempts" -gt 0 ] || return 2
  exact_task_id=${array_job_id}_0
  attempt=0

  while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    matches=0
    state=
    exit_code=
    if accounting_output=$(sacct -X -n -P -j "$exact_task_id" \
      --format=JobID,State,ExitCode 2>/dev/null); then
      accounting_status=0
    else
      accounting_status=$?
      accounting_output=
    fi
    if [ "$accounting_status" -eq 0 ]; then
      while IFS='|' read -r job_id observed_state observed_exit rest; do
        [ "$job_id" = "$exact_task_id" ] || continue
        matches=$((matches + 1))
        [ "$matches" -eq 1 ] || return 2
        state=$observed_state
        exit_code=$observed_exit
      done <<<"$accounting_output"
    fi

    if [ "$matches" -eq 1 ]; then
      last_state=$state
      last_exit=$exit_code
      if [ "$state" = COMPLETED ] && [ "$exit_code" = 0:0 ]; then
        printf '%s|%s\n' "$state" "$exit_code"
        return 0
      fi
      case "$state" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
          printf '%s|%s\n' "${state:-missing}" "${exit_code:-missing}"
          return 3
          ;;
      esac
    fi

    if [ "$attempt" -lt "$max_attempts" ] && [ "$delay_seconds" -gt 0 ]; then
      sleep "$delay_seconds"
    fi
  done

  printf '%s|%s\n' "${last_state:-missing}" "${last_exit:-missing}"
  return 1
}
