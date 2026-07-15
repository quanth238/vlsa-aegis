#!/usr/bin/env bash

# Full-lifetime, job-scoped cgroup-v2 memory.current monitor for R05A.
#
# This file is a sourceable Bash library.  It contains no model/runtime code
# and never enumerates a cgroup tree.  The monitor resolves the caller's one
# cgroup-v2 membership through the unique longest-prefix cgroup2 mount, walks
# only that membership's ancestors, and stops at the exact job_<id> boundary.
#
# Public entry point:
#
#   crfs_monitor_cgroup_v2_full_lifetime \
#     PROC_CGROUP MOUNTINFO OSRELEASE TRACE READY STOP \
#     POLICY_LAUNCH POLICY_CLEANUP GPU_CLEANUP WORKLOAD_CLEANUP \
#     EXPECTED_JOB_ID 0.1 500000000
#
# READY is atomically published only after the before snapshots and first
# positive sample are durable in the hidden trace candidate.  The caller then
# writes the four lifecycle marker files and finally STOP.  After observing
# STOP, the monitor records its stop timestamp, takes a final sample, records
# after snapshots, and atomically seals TRACE.  A completed but unsupported
# diagnostic is still sealed and returned with status 3; unsafe mapping or an
# inability to construct an atomic artifact returns 2 without publishing TRACE.

_crfs_fl_decode_mount_path() {
  printf '%s' "$1" | sed \
    -e 's/\\040/ /g' \
    -e 's/\\011/\	/g' \
    -e 's/\\012/\
/g' \
    -e 's/\\134/\\/g'
}

_crfs_fl_normalize_absolute() {
  local value=$1
  case "$value" in
    /*) ;;
    *) return 1 ;;
  esac
  case "$value" in
    *'/../'*|*'/./'*|*'//'|*'/..'|*'/.') return 1 ;;
  esac
  while [ "$value" != / ] && [ "${value%/}" != "$value" ]; do
    value=${value%/}
  done
  printf '%s\n' "$value"
}

_crfs_fl_suffix() {
  local membership=$1
  local mount_root=$2
  if [ "$mount_root" = / ]; then
    [ "$membership" = / ] && printf '\n' || printf '%s\n' "$membership"
    return 0
  fi
  if [ "$membership" = "$mount_root" ]; then
    printf '\n'
    return 0
  fi
  case "$membership" in
    "$mount_root"/*) printf '/%s\n' "${membership#"$mount_root"/}" ;;
    *) return 1 ;;
  esac
}

_crfs_fl_escape() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//$'\t'/\\t}
  value=${value//$'\n'/\\n}
  printf '%s' "$value"
}

_crfs_fl_is_canonical_uint() {
  local value=$1
  case "$value" in
    ''|*[!0-9]*) return 1 ;;
    0|[1-9]*) ;;
    *) return 1 ;;
  esac
  # Every arithmetic operation in this helper is signed 64-bit integer
  # arithmetic.  Reject values outside that domain before comparing them.
  if [ "${#value}" -gt 19 ]; then
    return 1
  fi
  if [ "${#value}" -eq 19 ] && [[ "$value" > 9223372036854775807 ]]; then
    return 1
  fi
  return 0
}

_crfs_fl_monotonic_ns() {
  local raw rest seconds fraction value
  [ -r /proc/uptime ] || return 1
  IFS=' ' read -r raw rest </proc/uptime || return 1
  case "$raw" in
    *.*)
      seconds=${raw%%.*}
      fraction=${raw#*.}
      ;;
    *)
      seconds=$raw
      fraction=0
      ;;
  esac
  _crfs_fl_is_canonical_uint "$seconds" || return 1
  case "$fraction" in ''|*[!0-9]*) return 1 ;; esac
  fraction=${fraction}000000000
  fraction=${fraction:0:9}
  # 10# prevents a padded fraction from being interpreted as octal.
  value=$((10#$seconds * 1000000000 + 10#$fraction)) || return 1
  [ "$value" -gt 0 ] || return 1
  printf '%s\n' "$value"
}

_crfs_fl_read_line() {
  local path=$1
  local value=
  [ -f "$path" ] && [ -r "$path" ] || return 1
  IFS= read -r value <"$path" || return 1
  # Marker and scalar files must contain exactly one line.
  [ "$(wc -l <"$path" | tr -d '[:space:]')" = 1 ] || return 1
  printf '%s\n' "$value"
}

_crfs_fl_scalar_record() {
  local path=$1
  local allow_max=$2
  local value=
  if [ ! -e "$path" ]; then
    printf 'missing\t-\n'
    return 0
  fi
  if [ ! -r "$path" ]; then
    printf 'unreadable\t-\n'
    return 0
  fi
  IFS= read -r value <"$path" || {
    printf 'read_error\t-\n'
    return 0
  }
  if _crfs_fl_is_canonical_uint "$value"; then
    printf 'readable_integer\t%s\n' "$value"
  elif [ "$allow_max" = true ] && [ "$value" = max ]; then
    printf 'readable_unlimited\tmax\n'
  else
    printf 'readable_invalid\t%s\n' "$(_crfs_fl_escape "$value")"
  fi
}

_crfs_fl_events_state() {
  local path=$1
  local key value extra valid=true
  local count=0
  if [ ! -e "$path" ]; then
    printf 'missing\n'
    return 0
  fi
  if [ ! -r "$path" ]; then
    printf 'unreadable\n'
    return 0
  fi
  awk '
    NF != 2 || $1 !~ /^[a-z][a-z0-9_]*$/ || $2 !~ /^(0|[1-9][0-9]*)$/ { exit 1 }
    seen[$1]++ { if (seen[$1] > 1) exit 1 }
    END { if (NR == 0) exit 1 }
  ' "$path" || { printf 'readable_invalid\n'; return 0; }
  while IFS=' ' read -r key value extra; do
    _crfs_fl_is_canonical_uint "$value" || valid=false
    count=$((count + 1))
  done <"$path"
  [ "$count" -gt 0 ] && [ "$valid" = true ] \
    && printf 'readable_flat_keys\n' || printf 'readable_invalid\n'
}

_crfs_fl_append_events() {
  local path=$1
  local phase=$2
  local row_type=$3
  local destination=$4
  local key value extra
  CRFS_FL_SELECTED_MAX=
  CRFS_FL_SELECTED_OOM=
  CRFS_FL_SELECTED_OOM_KILL=
  [ "$(_crfs_fl_events_state "$path")" = readable_flat_keys ] || return 1
  while IFS=' ' read -r key value extra; do
    printf '%s\t%s\t%s\t%s\n' "$row_type" "$phase" "$key" "$value" >>"$destination" || return 1
    case "$key" in
      max) CRFS_FL_SELECTED_MAX=$value ;;
      oom) CRFS_FL_SELECTED_OOM=$value ;;
      oom_kill) CRFS_FL_SELECTED_OOM_KILL=$value ;;
    esac
  done <"$path"
  if [ "$row_type" = event ]; then
    [ -n "$CRFS_FL_SELECTED_MAX" ] \
      && [ -n "$CRFS_FL_SELECTED_OOM" ] \
      && [ -n "$CRFS_FL_SELECTED_OOM_KILL" ] || return 1
  fi
}

_crfs_fl_append_scope_counters() {
  local index=$1
  local source=$2
  local path=$3
  local destination=$4
  local key value extra
  [ "$(_crfs_fl_events_state "$path")" = readable_flat_keys ] || return 1
  while IFS=' ' read -r key value extra; do
    printf 'counter\t%s\t%s\t%s\t%s\n' "$index" "$source" "$key" "$value" >>"$destination" || return 1
  done <"$path"
  return 0
}

_crfs_fl_atomic_marker() {
  local destination=$1
  local value=$2
  local temporary
  [ ! -e "$destination" ] || return 1
  temporary=$(mktemp "${destination}.tmp.XXXXXX") || return 1
  printf '%s\n' "$value" >"$temporary" || { rm -f "$temporary"; return 1; }
  mv "$temporary" "$destination"
}

crfs_write_cgroup_v2_full_lifetime_marker() {
  local destination=${1:?marker path is required}
  local timestamp
  timestamp=$(_crfs_fl_monotonic_ns) || return 1
  _crfs_fl_atomic_marker "$destination" "$timestamp"
}

_crfs_fl_read_marker() {
  local path=$1
  local value
  value=$(_crfs_fl_read_line "$path") || return 1
  _crfs_fl_is_canonical_uint "$value" || return 1
  [ "$value" -gt 0 ] || return 1
  printf '%s\n' "$value"
}

crfs_monitor_cgroup_v2_full_lifetime() {
  local proc_cgroup_file=${1:?proc cgroup file is required}
  local mountinfo_file=${2:?mountinfo file is required}
  local osrelease_file=${3:?osrelease file is required}
  local trace=${4:?sealed trace path is required}
  local ready=${5:?ready marker path is required}
  local stop=${6:?stop marker path is required}
  local policy_launch=${7:?policy launch marker path is required}
  local policy_cleanup=${8:?policy cleanup marker path is required}
  local gpu_cleanup=${9:?GPU cleanup marker path is required}
  local workload_cleanup=${10:?workload cleanup marker path is required}
  local expected_job_id=${11:?expected job id is required}
  local sample_delay=${12:?sample delay is required}
  local registered_max_gap=${13:?registered maximum gap is required}
  local v2_count raw_cgroup_line membership
  local best_length=-1 best_root= best_point= best_raw= best_suffix=
  local tie=false encoded_root encoded_point raw_line root point suffix
  local hierarchy_state job_component job_boundary job_path logical mapped kind index
  local scope_state current_record max_record events_state local_events_state state value
  local osrelease temporary trace_parent marker_parent path
  local before_max before_oom before_oom_kill after_max after_oom after_oom_kill
  local before_limit after_limit delta_max= delta_oom= delta_oom_kill=
  local sample_index=0 sample_count=0 sample_high=0 sample_max_gap=0
  local first_sample_ns= last_sample_ns= previous_sample_ns= now_ns= gap= current_value=
  local monitor_ready_ns= policy_launch_ns= policy_cleanup_ns= gpu_cleanup_ns=
  local workload_cleanup_ns= monitor_stop_ns= native_record native_state native_value
  local runtime_valid=true reason=sampled_current_contract_supported
  local max_samples=100000 final_attempt

  case "$expected_job_id" in ''|*[!0-9]*) return 2 ;; esac
  [ "$expected_job_id" = 0 ] || [ "${expected_job_id#0}" = "$expected_job_id" ] || return 2
  _crfs_fl_is_canonical_uint "$registered_max_gap" || return 2
  [ "$registered_max_gap" -gt 0 ] || return 2
  [ "$sample_delay" = 0.1 ] || return 2
  [ -r "$proc_cgroup_file" ] && [ -r "$mountinfo_file" ] && [ -r "$osrelease_file" ] || return 2
  for path in "$trace" "$ready" "$stop" "$policy_launch" "$policy_cleanup" "$gpu_cleanup" "$workload_cleanup"; do
    case "$path" in /*) ;; *) return 2 ;; esac
    [ ! -e "$path" ] || return 2
  done
  trace_parent=${trace%/*}; [ -n "$trace_parent" ] || trace_parent=/
  [ -d "$trace_parent" ] && [ -w "$trace_parent" ] || return 2
  for path in "$ready" "$stop" "$policy_launch" "$policy_cleanup" "$gpu_cleanup" "$workload_cleanup"; do
    marker_parent=${path%/*}; [ -n "$marker_parent" ] || marker_parent=/
    [ "$marker_parent" = "$trace_parent" ] || return 2
  done

  v2_count=$(awk -F: '$1 == "0" && $2 == "" { count++ } END { print count + 0 }' "$proc_cgroup_file") || return 2
  [ "$v2_count" = 1 ] || return 2
  raw_cgroup_line=$(awk -F: '$1 == "0" && $2 == "" { print; exit }' "$proc_cgroup_file") || return 2
  membership=$(awk -F: '
    $1 == "0" && $2 == "" {
      path=$3
      for (i=4; i<=NF; i++) path=path ":" $i
      print path
      exit
    }
  ' "$proc_cgroup_file") || return 2
  membership=$(_crfs_fl_normalize_absolute "$membership") || return 2

  while IFS=$'\t' read -r encoded_root encoded_point raw_line; do
    [ -n "$encoded_root" ] || continue
    root=$(_crfs_fl_decode_mount_path "$encoded_root")
    point=$(_crfs_fl_decode_mount_path "$encoded_point")
    root=$(_crfs_fl_normalize_absolute "$root") || continue
    point=$(_crfs_fl_normalize_absolute "$point") || continue
    suffix=$(_crfs_fl_suffix "$membership" "$root") || continue
    if [ "${#root}" -gt "$best_length" ]; then
      best_length=${#root}
      best_root=$root
      best_point=$point
      best_raw=$raw_line
      best_suffix=$suffix
      tie=false
    elif [ "${#root}" -eq "$best_length" ]; then
      # Even byte-identical mappings are ambiguous when mountinfo exposes more
      # than one longest-prefix cgroup2 entry.  The contract requires one
      # selected mount, not merely one selected path string.
      tie=true
    fi
  done < <(
    awk '
      {
        separator=0
        for (i=1; i<=NF; i++) if ($i == "-") { separator=i; break }
        if (separator && $(separator + 1) == "cgroup2") print $4 "\t" $5 "\t" $0
      }
    ' "$mountinfo_file"
  )
  [ "$best_length" -ge 0 ] && [ "$tie" = false ] || return 2

  hierarchy_state=$(printf '%s\n' "$best_raw" | awk '
    {
      separator=0
      for (i=1; i<=NF; i++) if ($i == "-") { separator=i; break }
      if (!separator || separator + 3 > NF) exit 1
      local_only=0
      count=split($6, mount_options, ",")
      for (i=1; i<=count; i++) if (mount_options[i] == "memory_localevents") local_only=1
      count=split($(separator + 3), super_options, ",")
      for (i=1; i<=count; i++) if (super_options[i] == "memory_localevents") local_only=1
      print local_only ? "local_only_memory_localevents" : "hierarchical"
    }
  ') || return 2
  [ "$hierarchy_state" = hierarchical ] || return 2

  job_component=job_$expected_job_id
  job_boundary=$(printf '%s\n' "$membership" | awk -F/ -v component="$job_component" '
    {
      prefix=""; found=0
      for (i=2; i<=NF; i++) {
        prefix=prefix "/" $i
        if ($i == component) { boundary=prefix; found++ }
      }
      if (found == 1) print boundary
    }
  ') || return 2
  [ -n "$job_boundary" ] || return 2
  suffix=$(_crfs_fl_suffix "$job_boundary" "$best_root") || return 2
  job_path="${best_point%/}${suffix}"
  [ -n "$job_path" ] || job_path=/
  [ -d "$job_path" ] && [ -x "$job_path" ] || return 2

  IFS= read -r osrelease <"$osrelease_file" || return 2
  [ -n "$osrelease" ] || return 2
  temporary=$(mktemp "${trace}.tmp.XXXXXX") || return 2
  {
    printf 'record\tschema_version\t1.0\n'
    printf 'record\tartifact_role\tr05a_full_lifetime_cgroup_v2_current_trace\n'
    printf 'record\tslurm_array_job_id\t%s\n' "$expected_job_id"
    printf 'record\tslurm_array_task_id\t0\n'
    printf 'record\tproc_cgroup_line\t%s\n' "$(_crfs_fl_escape "$raw_cgroup_line")"
    printf 'record\tmembership_path\t%s\n' "$(_crfs_fl_escape "$membership")"
    printf 'record\tselected_mountinfo_line\t%s\n' "$(_crfs_fl_escape "$best_raw")"
    printf 'record\tmount_root\t%s\n' "$(_crfs_fl_escape "$best_root")"
    printf 'record\tmount_point\t%s\n' "$(_crfs_fl_escape "$best_point")"
    printf 'record\tmemory_events_hierarchy_state\thierarchical\n'
    printf 'record\tmembership_relative_to_mount_root\t%s\n' "$(_crfs_fl_escape "$best_suffix")"
    printf 'record\tjob_boundary_path\t%s\n' "$(_crfs_fl_escape "$job_boundary")"
    printf 'record\tkernel_osrelease\t%s\n' "$(_crfs_fl_escape "$osrelease")"
  } >"$temporary" || { rm -f "$temporary"; return 2; }

  # Record only the constructed membership chain from task leaf to exact job.
  logical=$membership
  index=0
  while :; do
    suffix=$(_crfs_fl_suffix "$logical" "$best_root") || { rm -f "$temporary"; return 2; }
    mapped="${best_point%/}${suffix}"; [ -n "$mapped" ] || mapped=/
    if [ "$logical" = "$membership" ]; then kind=task
    elif [ "$logical" = "$job_boundary" ]; then kind=job
    elif [ "${logical##*/}" = step_batch ]; then kind=step
    else kind=intermediate
    fi
    if [ ! -e "$mapped" ]; then scope_state=missing
    elif [ ! -d "$mapped" ]; then scope_state=not_directory
    elif [ ! -x "$mapped" ]; then scope_state=unsearchable
    else scope_state=searchable
    fi
    printf 'scope\t%s\t%s\t%s\t%s\t%s\n' \
      "$index" "$kind" "$(_crfs_fl_escape "$logical")" \
      "$(_crfs_fl_escape "$mapped")" "$scope_state" >>"$temporary" || { rm -f "$temporary"; return 2; }
    current_record=$(_crfs_fl_scalar_record "$mapped/memory.current" false)
    max_record=$(_crfs_fl_scalar_record "$mapped/memory.max" true)
    events_state=$(_crfs_fl_events_state "$mapped/memory.events")
    local_events_state=$(_crfs_fl_events_state "$mapped/memory.events.local")
    printf 'metric\t%s\tmemory.current\t%s\n' "$index" "$current_record" >>"$temporary"
    printf 'metric\t%s\tmemory.max\t%s\n' "$index" "$max_record" >>"$temporary"
    printf 'metric\t%s\tmemory.events\t%s\t-\n' "$index" "$events_state" >>"$temporary"
    printf 'metric\t%s\tmemory.events.local\t%s\t-\n' "$index" "$local_events_state" >>"$temporary"
    if [ "$events_state" = readable_flat_keys ]; then
      _crfs_fl_append_scope_counters "$index" memory.events "$mapped/memory.events" "$temporary" \
        || { rm -f "$temporary"; return 2; }
    fi
    if [ "$local_events_state" = readable_flat_keys ]; then
      _crfs_fl_append_scope_counters "$index" memory.events.local "$mapped/memory.events.local" "$temporary" \
        || { rm -f "$temporary"; return 2; }
    fi
    [ "$scope_state" = searchable ] || runtime_valid=false
    if [ "$logical" = "$job_boundary" ]; then break; fi
    logical=${logical%/*}
    [ -n "$logical" ] && [ "${#logical}" -ge "${#job_boundary}" ] \
      || { rm -f "$temporary"; return 2; }
    index=$((index + 1))
  done

  max_record=$(_crfs_fl_scalar_record "$job_path/memory.max" true)
  IFS=$'\t' read -r state before_limit <<<"$max_record"
  printf 'snapshot\tbefore\tmemory.max\t%s\t%s\n' "$state" "$before_limit" >>"$temporary"
  if [ "$state" != readable_integer ] || ! _crfs_fl_is_canonical_uint "$before_limit" \
    || [ "$before_limit" -le 0 ]; then
    runtime_valid=false
    reason=invalid_before_memory_max
  fi
  events_state=$(_crfs_fl_events_state "$job_path/memory.events")
  local_events_state=$(_crfs_fl_events_state "$job_path/memory.events.local")
  if [ "$events_state" = readable_flat_keys ]; then
    _crfs_fl_append_events "$job_path/memory.events" before event "$temporary" || runtime_valid=false
    before_max=$CRFS_FL_SELECTED_MAX
    before_oom=$CRFS_FL_SELECTED_OOM
    before_oom_kill=$CRFS_FL_SELECTED_OOM_KILL
  else
    runtime_valid=false
    reason=invalid_before_memory_events
    before_max= before_oom= before_oom_kill=
  fi
  if [ "$local_events_state" = readable_flat_keys ]; then
    _crfs_fl_append_events "$job_path/memory.events.local" before local_event "$temporary" || runtime_valid=false
  else
    runtime_valid=false
    reason=invalid_before_memory_events_local
  fi

  # A supported monitor never signals READY unless its initial contract is
  # usable.  This prevents model/setup work from racing ahead of the baseline.
  [ "$runtime_valid" = true ] || { rm -f "$temporary"; return 2; }
  now_ns=$(_crfs_fl_monotonic_ns) || { rm -f "$temporary"; return 2; }
  current_record=$(_crfs_fl_scalar_record "$job_path/memory.current" false)
  IFS=$'\t' read -r state current_value <<<"$current_record"
  [ "$state" = readable_integer ] && _crfs_fl_is_canonical_uint "$current_value" \
    && [ "$current_value" -gt 0 ] || { rm -f "$temporary"; return 2; }
  printf 'sample\t0\t%s\t%s\n' "$now_ns" "$current_value" >>"$temporary" \
    || { rm -f "$temporary"; return 2; }
  sample_count=1
  sample_index=1
  sample_high=$current_value
  first_sample_ns=$now_ns
  last_sample_ns=$now_ns
  previous_sample_ns=$now_ns
  monitor_ready_ns=$(_crfs_fl_monotonic_ns) || { rm -f "$temporary"; return 2; }
  [ "$monitor_ready_ns" -ge "$first_sample_ns" ] || { rm -f "$temporary"; return 2; }
  _crfs_fl_atomic_marker "$ready" "$monitor_ready_ns" || { rm -f "$temporary"; return 2; }

  while [ ! -e "$stop" ]; do
    sleep "$sample_delay" || { runtime_valid=false; reason=sample_sleep_failed; break; }
    now_ns=$(_crfs_fl_monotonic_ns) || { runtime_valid=false; reason=sample_clock_failed; break; }
    current_record=$(_crfs_fl_scalar_record "$job_path/memory.current" false)
    IFS=$'\t' read -r state current_value <<<"$current_record"
    if [ "$state" != readable_integer ] || ! _crfs_fl_is_canonical_uint "$current_value" \
      || [ "$current_value" -le 0 ]; then
      runtime_valid=false; reason=sample_current_invalid; break
    fi
    gap=$((now_ns - previous_sample_ns))
    if [ "$gap" -le 0 ]; then runtime_valid=false; reason=sample_timestamp_not_increasing; break; fi
    [ "$gap" -le "$sample_max_gap" ] || sample_max_gap=$gap
    if [ "$gap" -gt "$registered_max_gap" ]; then runtime_valid=false; reason=sample_gap_exceeded; fi
    printf 'sample\t%s\t%s\t%s\n' "$sample_index" "$now_ns" "$current_value" >>"$temporary" \
      || { runtime_valid=false; reason=sample_write_failed; break; }
    sample_count=$((sample_count + 1))
    sample_index=$((sample_index + 1))
    [ "$current_value" -le "$sample_high" ] || sample_high=$current_value
    previous_sample_ns=$now_ns
    last_sample_ns=$now_ns
    if [ "$sample_count" -ge "$max_samples" ]; then
      runtime_valid=false; reason=sample_count_limit_exceeded; break
    fi
  done

  if [ ! -e "$stop" ]; then
    runtime_valid=false
    reason=${reason:-monitor_stopped_without_stop_marker}
  fi
  monitor_stop_ns=$(_crfs_fl_monotonic_ns) || { runtime_valid=false; monitor_stop_ns=-; reason=stop_clock_invalid; }

  # Force the final sample to occur after the stop observation and after the
  # preceding sample.  /proc/uptime is commonly quantized to 10 ms.
  if [ "$monitor_stop_ns" != - ]; then
    final_attempt=0
    while [ "$final_attempt" -lt 20 ]; do
      now_ns=$(_crfs_fl_monotonic_ns) || { now_ns=-; break; }
      if [ "$now_ns" != - ] && [ "$now_ns" -gt "$previous_sample_ns" ] \
        && [ "$now_ns" -ge "$monitor_stop_ns" ]; then
        break
      fi
      sleep 0.01 || break
      final_attempt=$((final_attempt + 1))
    done
  else
    now_ns=-
  fi
  if [ "$now_ns" = - ] || ! _crfs_fl_is_canonical_uint "$now_ns" \
    || [ "$now_ns" -le "$previous_sample_ns" ]; then
    runtime_valid=false
    reason=final_sample_timestamp_invalid
  else
    current_record=$(_crfs_fl_scalar_record "$job_path/memory.current" false)
    IFS=$'\t' read -r state current_value <<<"$current_record"
    if [ "$state" != readable_integer ] || ! _crfs_fl_is_canonical_uint "$current_value" \
      || [ "$current_value" -le 0 ]; then
      runtime_valid=false
      reason=final_sample_current_invalid
    else
      gap=$((now_ns - previous_sample_ns))
      [ "$gap" -le "$sample_max_gap" ] || sample_max_gap=$gap
      [ "$gap" -le "$registered_max_gap" ] || { runtime_valid=false; reason=sample_gap_exceeded; }
      printf 'sample\t%s\t%s\t%s\n' "$sample_index" "$now_ns" "$current_value" >>"$temporary" \
        || { runtime_valid=false; reason=final_sample_write_failed; }
      sample_count=$((sample_count + 1))
      [ "$current_value" -le "$sample_high" ] || sample_high=$current_value
      previous_sample_ns=$now_ns
      last_sample_ns=$now_ns
    fi
  fi

  policy_launch_ns=$(_crfs_fl_read_marker "$policy_launch") || policy_launch_ns=-
  policy_cleanup_ns=$(_crfs_fl_read_marker "$policy_cleanup") || policy_cleanup_ns=-
  gpu_cleanup_ns=$(_crfs_fl_read_marker "$gpu_cleanup") || gpu_cleanup_ns=-
  workload_cleanup_ns=$(_crfs_fl_read_marker "$workload_cleanup") || workload_cleanup_ns=-
  printf 'lifecycle\t0\tmonitor_ready\t%s\n' "$monitor_ready_ns" >>"$temporary"
  printf 'lifecycle\t1\tpolicy_launch\t%s\n' "$policy_launch_ns" >>"$temporary"
  printf 'lifecycle\t2\tpolicy_cleanup_complete\t%s\n' "$policy_cleanup_ns" >>"$temporary"
  printf 'lifecycle\t3\tgpu_monitor_cleanup_complete\t%s\n' "$gpu_cleanup_ns" >>"$temporary"
  printf 'lifecycle\t4\tworkload_cleanup_complete\t%s\n' "$workload_cleanup_ns" >>"$temporary"
  printf 'lifecycle\t5\tmonitor_stop_observed\t%s\n' "$monitor_stop_ns" >>"$temporary"
  for value in "$policy_launch_ns" "$policy_cleanup_ns" "$gpu_cleanup_ns" "$workload_cleanup_ns" "$monitor_stop_ns"; do
    if ! _crfs_fl_is_canonical_uint "$value" || [ "$value" -le 0 ]; then
      runtime_valid=false
      reason=lifecycle_marker_invalid
    fi
  done
  if [ "$runtime_valid" = true ]; then
    if [ "$first_sample_ns" -gt "$monitor_ready_ns" ] \
      || [ "$monitor_ready_ns" -gt "$policy_launch_ns" ] \
      || [ "$policy_launch_ns" -gt "$policy_cleanup_ns" ] \
      || [ "$policy_cleanup_ns" -gt "$gpu_cleanup_ns" ] \
      || [ "$gpu_cleanup_ns" -gt "$workload_cleanup_ns" ] \
      || [ "$workload_cleanup_ns" -gt "$monitor_stop_ns" ] \
      || [ "$monitor_stop_ns" -gt "$last_sample_ns" ]; then
      runtime_valid=false
      reason=lifecycle_order_invalid
    fi
  fi

  max_record=$(_crfs_fl_scalar_record "$job_path/memory.max" true)
  IFS=$'\t' read -r state after_limit <<<"$max_record"
  printf 'snapshot\tafter\tmemory.max\t%s\t%s\n' "$state" "$after_limit" >>"$temporary"
  if [ "$state" != readable_integer ] || ! _crfs_fl_is_canonical_uint "$after_limit" \
    || [ "$after_limit" -le 0 ] || [ "$after_limit" != "$before_limit" ]; then
    runtime_valid=false
    reason=memory_max_changed_or_invalid
  fi

  events_state=$(_crfs_fl_events_state "$job_path/memory.events")
  local_events_state=$(_crfs_fl_events_state "$job_path/memory.events.local")
  if [ "$events_state" = readable_flat_keys ]; then
    _crfs_fl_append_events "$job_path/memory.events" after event "$temporary" || runtime_valid=false
    after_max=$CRFS_FL_SELECTED_MAX
    after_oom=$CRFS_FL_SELECTED_OOM
    after_oom_kill=$CRFS_FL_SELECTED_OOM_KILL
  else
    runtime_valid=false
    reason=invalid_after_memory_events
    after_max= after_oom= after_oom_kill=
  fi
  if [ "$local_events_state" = readable_flat_keys ]; then
    _crfs_fl_append_events "$job_path/memory.events.local" after local_event "$temporary" || runtime_valid=false
  else
    runtime_valid=false
    reason=invalid_after_memory_events_local
  fi
  if [ -n "$before_max" ] && [ -n "$after_max" ] \
    && [ -n "$before_oom" ] && [ -n "$after_oom" ] \
    && [ -n "$before_oom_kill" ] && [ -n "$after_oom_kill" ]; then
    delta_max=$((after_max - before_max))
    delta_oom=$((after_oom - before_oom))
    delta_oom_kill=$((after_oom_kill - before_oom_kill))
    if [ "$delta_max" -lt 0 ] || [ "$delta_oom" -lt 0 ] || [ "$delta_oom_kill" -lt 0 ]; then
      runtime_valid=false; reason=memory_event_counter_decreased
    elif [ "$delta_max" -ne 0 ] || [ "$delta_oom" -ne 0 ] || [ "$delta_oom_kill" -ne 0 ]; then
      runtime_valid=false; reason=memory_limit_or_oom_event_observed
    fi
  else
    runtime_valid=false
    reason=memory_event_counter_missing
    delta_max=-; delta_oom=-; delta_oom_kill=-
  fi

  native_record=$(_crfs_fl_scalar_record "$job_path/memory.peak" false)
  IFS=$'\t' read -r native_state native_value <<<"$native_record"
  if [ "$native_state" = readable_integer ]; then
    if ! _crfs_fl_is_canonical_uint "$native_value" || [ "$native_value" -le 0 ]; then
      runtime_valid=false; reason=native_memory_peak_invalid
    fi
  elif [ "$native_state" = missing ]; then
    native_value=-
  else
    runtime_valid=false
    reason=native_memory_peak_invalid
  fi
  printf 'capability\tjob_memory_peak\t%s\t%s\n' "$native_state" "$native_value" >>"$temporary"

  if [ "$sample_high" -le 0 ] || [ "$sample_high" -gt "$before_limit" ]; then
    runtime_valid=false
    reason=sampled_high_water_outside_job_limit
  fi
  if [ "$sample_max_gap" -gt "$registered_max_gap" ]; then
    runtime_valid=false
    reason=sample_gap_exceeded
  fi
  [ "$runtime_valid" = true ] || [ "$reason" != sampled_current_contract_supported ] \
    || reason=sampled_current_contract_unsupported
  {
    printf 'summary\tjob_scope_path\t%s\n' "$(_crfs_fl_escape "$job_path")"
    printf 'summary\tjob_memory_max_before_bytes\t%s\n' "$before_limit"
    printf 'summary\tjob_memory_max_after_bytes\t%s\n' "$after_limit"
    printf 'summary\tsample_count_observed\t%s\n' "$sample_count"
    printf 'summary\tsample_interval_requested_seconds\t%s\n' "$sample_delay"
    printf 'summary\tfirst_sample_monotonic_ns\t%s\n' "$first_sample_ns"
    printf 'summary\tlast_sample_monotonic_ns\t%s\n' "$last_sample_ns"
    printf 'summary\tsample_max_gap_ns\t%s\n' "$sample_max_gap"
    printf 'summary\tsampled_memory_current_high_water_bytes\t%s\n' "$sample_high"
    printf 'summary\tmemory_events_max_delta\t%s\n' "$delta_max"
    printf 'summary\tmemory_events_oom_delta\t%s\n' "$delta_oom"
    printf 'summary\tmemory_events_oom_kill_delta\t%s\n' "$delta_oom_kill"
    printf 'summary\tnative_memory_peak_state\t%s\n' "$native_state"
    printf 'summary\tnative_memory_peak_value\t%s\n' "$native_value"
    printf 'summary\tfirst_sample_no_later_than_policy_launch\t%s\n' "$runtime_valid"
    printf 'summary\tlast_sample_no_earlier_than_workload_cleanup\t%s\n' "$runtime_valid"
    printf 'summary\thost_cgroup_sampled_current_is_lower_bound_not_peak\ttrue\n'
    if [ "$runtime_valid" = true ]; then
      printf 'summary\toutcome\tfull_lifetime_sampled_current_supported\n'
    else
      printf 'summary\toutcome\t%s\n' "$reason"
    fi
    printf 'record\tstatus\tcompleted\n'
  } >>"$temporary" || { rm -f "$temporary"; return 2; }
  mv "$temporary" "$trace" || { rm -f "$temporary"; return 2; }
  [ "$runtime_valid" = true ] && return 0
  return 3
}
