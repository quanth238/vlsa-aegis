#!/usr/bin/env bash

# Parse one exact `scontrol show job <array>_<task> -o` record without assuming
# whether this Slurm build displays JobId as the parent or child spelling.
crfs_slurm_record_field() {
  local record=$1 key=$2
  printf '%s\n' "$record" | awk -v key="$key" '
    BEGIN {
      count = 0
      value = ""
      prefix = key "="
    }
    {
      for (field = 1; field <= NF; field += 1) {
        if (index($field, prefix) == 1) {
          count += 1
          value = substr($field, length(prefix) + 1)
        }
      }
    }
    END {
      if (count != 1 || value == "") {
        exit 2
      }
      print value
    }
  '
}

crfs_validate_exact_array_task_identity() {
  local record=$1 expected_array=$2 expected_task=$3
  local displayed_job array_job array_task
  case "$expected_array" in *[!0-9]*|'') return 2 ;; esac
  case "$expected_task" in *[!0-9]*|'') return 2 ;; esac
  displayed_job=$(crfs_slurm_record_field "$record" JobId) || return 2
  array_job=$(crfs_slurm_record_field "$record" ArrayJobId) || return 2
  array_task=$(crfs_slurm_record_field "$record" ArrayTaskId) || return 2
  case "$displayed_job" in
    "$expected_array"|"${expected_array}_${expected_task}") ;;
    *) return 2 ;;
  esac
  test "$array_job" = "$expected_array" || return 2
  test "$array_task" = "$expected_task" || return 2
}
