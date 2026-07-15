#!/usr/bin/env bash

# Bounded cgroup-v2 capability probe. The caller supplies procfs files so the
# same shell path can be exercised against dependency-free fixtures.

_crfs_cap_decode_mount_path() {
  printf '%s' "$1" | sed \
    -e 's/\\040/ /g' \
    -e 's/\\011/\	/g' \
    -e 's/\\012/\
/g' \
    -e 's/\\134/\\/g'
}

_crfs_cap_normalize_absolute() {
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

_crfs_cap_suffix() {
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

_crfs_cap_escape() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//$'\t'/\\t}
  value=${value//$'\n'/\\n}
  printf '%s' "$value"
}

_crfs_cap_scalar() {
  local path=$1
  local allow_max=$2
  local value=
  if [ ! -e "$path" ]; then
    printf 'missing\t\n'
    return 0
  fi
  if [ ! -r "$path" ]; then
    printf 'unreadable\t\n'
    return 0
  fi
  IFS= read -r value <"$path" || {
    printf 'read_error\t\n'
    return 0
  }
  case "$value" in
    *[!0-9]*|'')
      if [ "$allow_max" = true ] && [ "$value" = max ]; then
        printf 'readable_unlimited\tmax\n'
      else
        printf 'readable_invalid\t%s\n' "$(_crfs_cap_escape "$value")"
      fi
      ;;
    *) printf 'readable_integer\t%s\n' "$value" ;;
  esac
}

_crfs_cap_events_state() {
  local path=$1
  if [ ! -e "$path" ]; then
    printf 'missing\n'
    return 0
  fi
  if [ ! -r "$path" ]; then
    printf 'unreadable\n'
    return 0
  fi
  if awk '
    NF != 2 || $1 !~ /^[a-z0-9_]+$/ || $2 !~ /^[0-9]+$/ { exit 1 }
    seen[$1]++ { if (seen[$1] > 1) exit 1 }
    END { if (NR == 0) exit 1 }
  ' "$path"; then
    printf 'readable_flat_keys\n'
  else
    printf 'readable_invalid\n'
  fi
}

_crfs_cap_event_value() {
  local path=$1
  local key=$2
  awk -v wanted="$key" '$1 == wanted { print $2; found++ } END { if (found != 1) exit 1 }' "$path"
}

_crfs_cap_monotonic_ns() {
  local uptime rest value
  [ -r /proc/uptime ] || return 1
  IFS=' ' read -r uptime rest </proc/uptime || return 1
  value=$(awk -v raw="$uptime" '
    BEGIN {
      if (raw !~ /^[0-9]+([.][0-9]+)?$/) exit 1
      printf "%.0f\n", raw * 1000000000
    }
  ') || return 1
  case "$value" in *[!0-9]*|'') return 1 ;; esac
  printf '%s\n' "$value"
}

crfs_probe_cgroup_v2_current_capability() {
  local proc_cgroup_file=${1:?proc cgroup file is required}
  local mountinfo_file=${2:?mountinfo file is required}
  local osrelease_file=${3:?osrelease file is required}
  local destination=${4:?destination is required}
  local expected_job_id=${5:?expected allocation job id is required}
  local sample_count=${6:-20}
  local sample_delay=${7:-0.1}
  local v2_count raw_cgroup_line membership
  local best_length=-1 best_root= best_point= best_raw= best_suffix=
  local tie=false encoded_root encoded_point raw_line root point suffix
  local job_component job_boundary logical mapped kind index=0
  local temporary current_record max_record events_state local_events_state
  local state value job_path= job_current_state= job_current_value=
  local job_max_state= job_max_value= job_events_state= job_local_events_state=
  local peak_record peak_state peak_value osrelease
  local memory_events_hierarchy_state=
  local before_max= before_oom= before_oom_kill=
  local after_max= after_oom= after_oom_kill=
  local delta_max= delta_oom= delta_oom_kill=
  local events_complete=false counter=
  local sampled=0 sample_failed=false sample_high=0 max_gap=0 previous_ns= now_ns= gap= sample_value=
  local outcome supported=false

  case "$expected_job_id" in *[!0-9]*|'') return 2 ;; esac
  case "$sample_count" in *[!0-9]*|'') return 2 ;; esac
  [ "$sample_count" -ge 1 ] && [ "$sample_count" -le 100 ] || return 2
  # Production is frozen to 100 ms; zero is reserved for dependency-free fixtures.
  case "$sample_delay" in 0|0.1) ;; *) return 2 ;; esac
  [ -r "$proc_cgroup_file" ] && [ -r "$mountinfo_file" ] && [ -r "$osrelease_file" ] || return 2
  [ ! -e "$destination" ] || return 2

  v2_count=$(awk -F: '$1 == "0" && $2 == "" { count++ } END { print count + 0 }' "$proc_cgroup_file")
  [ "$v2_count" = 1 ] || return 2
  raw_cgroup_line=$(awk -F: '$1 == "0" && $2 == "" { print; exit }' "$proc_cgroup_file")
  membership=$(awk -F: '
    $1 == "0" && $2 == "" {
      path=$3
      for (i=4; i<=NF; i++) path=path ":" $i
      print path
      exit
    }
  ' "$proc_cgroup_file")
  membership=$(_crfs_cap_normalize_absolute "$membership") || return 2

  while IFS=$'\t' read -r encoded_root encoded_point raw_line; do
    [ -n "$encoded_root" ] || continue
    root=$(_crfs_cap_decode_mount_path "$encoded_root")
    point=$(_crfs_cap_decode_mount_path "$encoded_point")
    root=$(_crfs_cap_normalize_absolute "$root") || continue
    point=$(_crfs_cap_normalize_absolute "$point") || continue
    suffix=$(_crfs_cap_suffix "$membership" "$root") || continue
    if [ "${#root}" -gt "$best_length" ]; then
      best_length=${#root}
      best_root=$root
      best_point=$point
      best_raw=$raw_line
      best_suffix=$suffix
      tie=false
    elif [ "${#root}" -eq "$best_length" ] && { [ "$root" != "$best_root" ] || [ "$point" != "$best_point" ]; }; then
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

  memory_events_hierarchy_state=$(printf '%s\n' "$best_raw" | awk '
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
  case "$memory_events_hierarchy_state" in
    hierarchical|local_only_memory_localevents) ;;
    *) return 2 ;;
  esac

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
  ')
  [ -n "$job_boundary" ] || return 2

  temporary=$(mktemp "${destination}.tmp.XXXXXX") || return 2
  {
    printf 'record\tschema_version\t1.0\n'
    printf 'record\tartifact_role\tr05a_cgroup_v2_current_capability\n'
    printf 'record\tproc_cgroup_line\t%s\n' "$(_crfs_cap_escape "$raw_cgroup_line")"
    printf 'record\tmembership_path\t%s\n' "$(_crfs_cap_escape "$membership")"
    printf 'record\tselected_mountinfo_line\t%s\n' "$(_crfs_cap_escape "$best_raw")"
    printf 'record\tmount_root\t%s\n' "$(_crfs_cap_escape "$best_root")"
    printf 'record\tmount_point\t%s\n' "$(_crfs_cap_escape "$best_point")"
    printf 'record\tmemory_events_hierarchy_state\t%s\n' "$memory_events_hierarchy_state"
    printf 'record\tmembership_relative_to_mount_root\t%s\n' "$(_crfs_cap_escape "$best_suffix")"
    printf 'record\tjob_boundary_path\t%s\n' "$(_crfs_cap_escape "$job_boundary")"
  } >"$temporary" || { rm -f "$temporary"; return 2; }

  logical=$membership
  while :; do
    suffix=$(_crfs_cap_suffix "$logical" "$best_root") || { rm -f "$temporary"; return 2; }
    mapped="${best_point%/}${suffix}"
    [ -n "$mapped" ] || mapped=/
    if [ "$logical" = "$membership" ]; then
      kind=task
    elif [ "$logical" = "$job_boundary" ]; then
      kind=job
    elif [ "${logical##*/}" = step_batch ]; then
      kind=step
    else
      kind=intermediate
    fi
    if [ ! -e "$mapped" ]; then state=missing
    elif [ ! -d "$mapped" ]; then state=not_directory
    elif [ ! -x "$mapped" ]; then state=unsearchable
    else state=searchable
    fi
    printf 'scope\t%s\t%s\t%s\t%s\t%s\n' \
      "$index" "$kind" "$(_crfs_cap_escape "$logical")" "$(_crfs_cap_escape "$mapped")" "$state" >>"$temporary"
    current_record=$(_crfs_cap_scalar "$mapped/memory.current" false)
    max_record=$(_crfs_cap_scalar "$mapped/memory.max" true)
    events_state=$(_crfs_cap_events_state "$mapped/memory.events")
    local_events_state=$(_crfs_cap_events_state "$mapped/memory.events.local")
    printf 'metric\t%s\tmemory.current\t%s\n' "$index" "$current_record" >>"$temporary"
    printf 'metric\t%s\tmemory.max\t%s\n' "$index" "$max_record" >>"$temporary"
    printf 'metric\t%s\tmemory.events\t%s\t\n' "$index" "$events_state" >>"$temporary"
    printf 'metric\t%s\tmemory.events.local\t%s\t\n' "$index" "$local_events_state" >>"$temporary"
    if [ "$events_state" = readable_flat_keys ]; then
      for counter in max oom oom_kill; do
        value=$(_crfs_cap_event_value "$mapped/memory.events" "$counter") || value=
        printf 'counter\t%s\tmemory.events\t%s\t%s\n' "$index" "$counter" "$value" >>"$temporary"
      done
    fi
    if [ "$local_events_state" = readable_flat_keys ]; then
      for counter in max oom oom_kill; do
        value=$(_crfs_cap_event_value "$mapped/memory.events.local" "$counter") || value=
        printf 'counter\t%s\tmemory.events.local\t%s\t%s\n' "$index" "$counter" "$value" >>"$temporary"
      done
    fi
    if [ "$logical" = "$job_boundary" ]; then
      job_path=$mapped
      IFS=$'\t' read -r job_current_state job_current_value <<<"$current_record"
      IFS=$'\t' read -r job_max_state job_max_value <<<"$max_record"
      job_events_state=$events_state
      job_local_events_state=$local_events_state
      break
    fi
    logical=${logical%/*}
    [ -n "$logical" ] && [ "${#logical}" -ge "${#job_boundary}" ] || { rm -f "$temporary"; return 2; }
    index=$((index + 1))
  done

  IFS= read -r osrelease <"$osrelease_file" || osrelease=
  printf 'record\tkernel_osrelease\t%s\n' "$(_crfs_cap_escape "$osrelease")" >>"$temporary"
  peak_record=$(_crfs_cap_scalar "$job_path/memory.peak" false)
  IFS=$'\t' read -r peak_state peak_value <<<"$peak_record"
  printf 'capability\tjob_memory_peak\t%s\t%s\n' "$peak_state" "$peak_value" >>"$temporary"

  if [ "$job_events_state" = readable_flat_keys ]; then
    before_max=$(_crfs_cap_event_value "$job_path/memory.events" max) || before_max=
    before_oom=$(_crfs_cap_event_value "$job_path/memory.events" oom) || before_oom=
    before_oom_kill=$(_crfs_cap_event_value "$job_path/memory.events" oom_kill) || before_oom_kill=
  fi
  printf 'event\tbefore\tmax\t%s\n' "$before_max" >>"$temporary"
  printf 'event\tbefore\toom\t%s\n' "$before_oom" >>"$temporary"
  printf 'event\tbefore\toom_kill\t%s\n' "$before_oom_kill" >>"$temporary"

  if [ "$job_current_state" = readable_integer ]; then
    index=0
    while [ "$index" -lt "$sample_count" ]; do
      now_ns=$(_crfs_cap_monotonic_ns) || { sample_failed=true; break; }
      current_record=$(_crfs_cap_scalar "$job_path/memory.current" false)
      IFS=$'\t' read -r state sample_value <<<"$current_record"
      if [ "$state" != readable_integer ]; then
        sample_failed=true
        break
      fi
      printf 'sample\t%s\t%s\t%s\n' "$index" "$now_ns" "$sample_value" >>"$temporary"
      sampled=$((sampled + 1))
      [ "$sample_value" -le "$sample_high" ] || sample_high=$sample_value
      if [ -n "$previous_ns" ]; then
        gap=$((now_ns - previous_ns))
        if [ "$gap" -lt 0 ] || { [ "$sample_delay" != 0 ] && [ "$gap" -eq 0 ]; }; then
          sample_failed=true
          break
        fi
        [ "$gap" -le "$max_gap" ] || max_gap=$gap
      fi
      previous_ns=$now_ns
      index=$((index + 1))
      [ "$index" -ge "$sample_count" ] || sleep "$sample_delay"
    done
  else
    sample_failed=true
  fi

  job_events_state=$(_crfs_cap_events_state "$job_path/memory.events")
  if [ "$job_events_state" = readable_flat_keys ]; then
    after_max=$(_crfs_cap_event_value "$job_path/memory.events" max) || after_max=
    after_oom=$(_crfs_cap_event_value "$job_path/memory.events" oom) || after_oom=
    after_oom_kill=$(_crfs_cap_event_value "$job_path/memory.events" oom_kill) || after_oom_kill=
  fi
  printf 'event\tafter\tmax\t%s\n' "$after_max" >>"$temporary"
  printf 'event\tafter\toom\t%s\n' "$after_oom" >>"$temporary"
  printf 'event\tafter\toom_kill\t%s\n' "$after_oom_kill" >>"$temporary"
  events_complete=true
  for value in "$before_max" "$after_max" "$before_oom" "$after_oom" "$before_oom_kill" "$after_oom_kill"; do
    case "$value" in *[!0-9]*|'') events_complete=false ;; esac
  done
  if [ "$events_complete" = true ]; then
    delta_max=$((after_max - before_max))
    delta_oom=$((after_oom - before_oom))
    delta_oom_kill=$((after_oom_kill - before_oom_kill))
  fi

  if [ "$peak_state" = readable_integer ] && [ "$peak_value" -gt 0 ]; then
    outcome=native_memory_peak_available
  elif [ "$peak_state" = missing ] \
    && [ "$job_current_state" = readable_integer ] \
    && [ "$job_max_state" = readable_integer ] \
    && [ "$job_max_value" -gt 0 ] \
    && [ "$memory_events_hierarchy_state" = hierarchical ] \
    && [ "$job_events_state" = readable_flat_keys ] \
    && [ "$events_complete" = true ] \
    && [ "$sample_failed" = false ] \
    && [ "$sampled" -eq "$sample_count" ] \
    && [ "$sample_high" -gt 0 ] \
    && [ "$max_gap" -le 500000000 ] \
    && [ "$delta_max" = 0 ] && [ "$delta_oom" = 0 ] && [ "$delta_oom_kill" = 0 ]; then
    outcome=sampled_current_contract_supported
    supported=true
  else
    outcome=sampled_current_contract_unsupported
  fi
  {
    printf 'summary\tjob_scope_path\t%s\n' "$(_crfs_cap_escape "$job_path")"
    printf 'summary\tjob_memory_current_state\t%s\n' "$job_current_state"
    printf 'summary\tjob_memory_max_state\t%s\n' "$job_max_state"
    printf 'summary\tjob_memory_max_value\t%s\n' "$job_max_value"
    printf 'summary\tjob_memory_events_state\t%s\n' "$job_events_state"
    printf 'summary\tjob_memory_events_local_state\t%s\n' "$job_local_events_state"
    printf 'summary\tmemory_events_hierarchy_state\t%s\n' "$memory_events_hierarchy_state"
    printf 'summary\tsample_count_requested\t%s\n' "$sample_count"
    printf 'summary\tsample_count_observed\t%s\n' "$sampled"
    printf 'summary\tsample_interval_requested_seconds\t%s\n' "$sample_delay"
    printf 'summary\tsample_max_gap_ns\t%s\n' "$max_gap"
    printf 'summary\tsampled_memory_current_high_water_bytes\t%s\n' "$sample_high"
    printf 'summary\tmemory_events_max_delta\t%s\n' "$delta_max"
    printf 'summary\tmemory_events_oom_delta\t%s\n' "$delta_oom"
    printf 'summary\tmemory_events_oom_kill_delta\t%s\n' "$delta_oom_kill"
    printf 'summary\tsampled_current_contract_capability_supported\t%s\n' "$supported"
    printf 'summary\toutcome\t%s\n' "$outcome"
    printf 'record\tstatus\tcompleted\n'
  } >>"$temporary" || { rm -f "$temporary"; return 2; }
  mv "$temporary" "$destination"
}
