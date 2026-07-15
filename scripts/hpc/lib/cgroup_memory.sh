#!/usr/bin/env bash

# Resolve and read the live peak of the cgroup that contains this Slurm step.
#
# Linux reports two independent paths:
#   * /proc/self/cgroup gives the process membership in a cgroup hierarchy;
#   * /proc/self/mountinfo gives that hierarchy's mounted root and mountpoint.
#
# A bind/subtree mount can have a root other than "/", so concatenating a
# fixed /sys/fs/cgroup prefix with the membership path is not portable.  This
# helper maps the membership relative to the actual mount root and supports
# both cgroup v2 memory.peak and v1 memory.max_usage_in_bytes.

_crfs_decode_mountinfo_path() {
  # mountinfo escapes space, tab, newline, and backslash as octal sequences.
  # Decode backslash last so a literal "\040" name is not decoded twice.
  printf '%s' "$1" | sed \
    -e 's/\\040/ /g' \
    -e 's/\\011/\	/g' \
    -e 's/\\012/\
/g' \
    -e 's/\\134/\\/g'
}

_crfs_normalize_absolute_path() {
  local value=$1
  case "$value" in
    /*) ;;
    *) return 1 ;;
  esac
  while [ "$value" != / ] && [ "${value%/}" != "$value" ]; do
    value=${value%/}
  done
  printf '%s\n' "$value"
}

_crfs_membership_suffix() {
  local membership=$1
  local mount_root=$2
  if [ "$mount_root" = / ]; then
    if [ "$membership" = / ]; then
      printf '\n'
    else
      printf '%s\n' "$membership"
    fi
    return 0
  fi
  if [ "$membership" = "$mount_root" ]; then
    printf '\n'
    return 0
  fi
  case "$membership" in
    "$mount_root"/*)
      printf '/%s\n' "${membership#"$mount_root"/}"
      return 0
      ;;
  esac
  return 1
}

_crfs_diag_escape() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//$'\t'/\\t}
  value=${value//$'\n'/\\n}
  printf '%s' "$value"
}

_crfs_write_cgroup_memory_diagnostic() {
  local destination=$1
  local status=$2
  local reason=$3
  local version=$4
  local membership=$5
  local mount_root=$6
  local mount_point=$7
  local relative_path=$8
  local peak_file=$9
  local peak_bytes=${10}
  local proc_cgroup_file=${11}
  local mountinfo_file=${12}
  local temporary

  temporary=$(mktemp "${destination}.tmp.XXXXXX") || return 1
  {
    printf 'schema_version\t1.0\n'
    printf 'artifact_role\tr05a_live_slurm_cgroup_memory_peak_diagnostic\n'
    printf 'status\t%s\n' "$(_crfs_diag_escape "$status")"
    printf 'reason\t%s\n' "$(_crfs_diag_escape "$reason")"
    printf 'cgroup_version\t%s\n' "$(_crfs_diag_escape "$version")"
    printf 'membership_path\t%s\n' "$(_crfs_diag_escape "$membership")"
    printf 'mount_root\t%s\n' "$(_crfs_diag_escape "$mount_root")"
    printf 'mount_point\t%s\n' "$(_crfs_diag_escape "$mount_point")"
    printf 'membership_relative_to_mount_root\t%s\n' "$(_crfs_diag_escape "$relative_path")"
    printf 'peak_file\t%s\n' "$(_crfs_diag_escape "$peak_file")"
    printf 'peak_bytes\t%s\n' "$(_crfs_diag_escape "$peak_bytes")"
    printf 'proc_cgroup_file\t%s\n' "$(_crfs_diag_escape "$proc_cgroup_file")"
    printf 'mountinfo_file\t%s\n' "$(_crfs_diag_escape "$mountinfo_file")"
  } >"$temporary" || {
    rm -f "$temporary"
    return 1
  }
  mv "$temporary" "$destination"
}

crfs_read_live_cgroup_memory_peak() {
  local proc_cgroup_file=${1:-/proc/self/cgroup}
  local mountinfo_file=${2:-/proc/self/mountinfo}
  local diagnostic_file=${3:?diagnostic output path is required}
  local memberships
  local v1_membership=
  local v2_membership=
  local selected_version=
  local selected_membership=
  local best_length=-1
  local best_version=
  local best_membership=
  local best_root=
  local best_point=
  local best_suffix=
  local best_peak_file=
  local fallback_length=-1
  local fallback_version=
  local fallback_membership=
  local fallback_root=
  local fallback_point=
  local fallback_suffix=
  local fallback_peak_file=
  local version encoded_root encoded_point mount_root mount_point suffix peak_file metric
  local peak_bytes=

  if [ ! -r "$proc_cgroup_file" ]; then
    _crfs_write_cgroup_memory_diagnostic \
      "$diagnostic_file" unavailable proc_cgroup_unreadable '' '' '' '' '' '' '' \
      "$proc_cgroup_file" "$mountinfo_file" || true
    return 1
  fi
  if [ ! -r "$mountinfo_file" ]; then
    _crfs_write_cgroup_memory_diagnostic \
      "$diagnostic_file" unavailable mountinfo_unreadable '' '' '' '' '' '' '' \
      "$proc_cgroup_file" "$mountinfo_file" || true
    return 1
  fi

  memberships=$(awk -F: '
    {
      controllers=$2
      path=$3
      if (NF > 3) {
        for (field_index=4; field_index<=NF; field_index++) path=path ":" $field_index
      }
      if ($1 == "0" && controllers == "") print "2\t" path
      count=split(controllers, names, ",")
      for (controller_index=1; controller_index<=count; controller_index++) {
        if (names[controller_index] == "memory") print "1\t" path
      }
    }
  ' "$proc_cgroup_file")
  while IFS=$'\t' read -r version selected_membership; do
    case "$version" in
      1) [ -n "$v1_membership" ] || v1_membership=$selected_membership ;;
      2) [ -n "$v2_membership" ] || v2_membership=$selected_membership ;;
    esac
  done <<<"$memberships"

  # In a hybrid hierarchy, an explicit v1 memory membership is authoritative.
  if [ -n "$v1_membership" ]; then
    selected_version=1
    selected_membership=$v1_membership
  elif [ -n "$v2_membership" ]; then
    selected_version=2
    selected_membership=$v2_membership
  else
    _crfs_write_cgroup_memory_diagnostic \
      "$diagnostic_file" unavailable memory_membership_missing '' '' '' '' '' '' '' \
      "$proc_cgroup_file" "$mountinfo_file" || true
    return 1
  fi
  selected_membership=$(_crfs_normalize_absolute_path "$selected_membership") || {
    _crfs_write_cgroup_memory_diagnostic \
      "$diagnostic_file" unavailable membership_not_absolute "$selected_version" \
      "$selected_membership" '' '' '' '' '' "$proc_cgroup_file" "$mountinfo_file" || true
    return 1
  }
  # Preserve the selected hierarchy even when no advertised mount root covers
  # its membership.  The failure sidecar must retain the evidence needed to
  # diagnose that namespace/layout mismatch.
  fallback_version=$selected_version
  fallback_membership=$selected_membership

  while IFS=$'\t' read -r version encoded_root encoded_point; do
    [ "$version" = "$selected_version" ] || continue
    mount_root=$(_crfs_decode_mountinfo_path "$encoded_root")
    mount_point=$(_crfs_decode_mountinfo_path "$encoded_point")
    mount_root=$(_crfs_normalize_absolute_path "$mount_root") || continue
    mount_point=$(_crfs_normalize_absolute_path "$mount_point") || continue
    suffix=$(_crfs_membership_suffix "$selected_membership" "$mount_root") || continue
    if [ "$selected_version" = 2 ]; then
      metric=memory.peak
    else
      metric=memory.max_usage_in_bytes
    fi
    # mount_point is normalized but "/" must not produce a leading "//".
    peak_file="${mount_point%/}${suffix}/$metric"
    if [ "${#mount_root}" -gt "$fallback_length" ]; then
      fallback_length=${#mount_root}
      fallback_version=$selected_version
      fallback_membership=$selected_membership
      fallback_root=$mount_root
      fallback_point=$mount_point
      fallback_suffix=$suffix
      fallback_peak_file=$peak_file
    fi
    if [ -r "$peak_file" ] && [ "${#mount_root}" -gt "$best_length" ]; then
      best_length=${#mount_root}
      best_version=$selected_version
      best_membership=$selected_membership
      best_root=$mount_root
      best_point=$mount_point
      best_suffix=$suffix
      best_peak_file=$peak_file
    fi
  done < <(
    awk '
      {
        separator=0
        for (field_index=1; field_index<=NF; field_index++) {
          if ($field_index == "-") {separator=field_index; break}
        }
        if (!separator || separator + 3 > NF) next
        filesystem=$(separator + 1)
        super_options=$(separator + 3)
        if (filesystem == "cgroup2") {
          print "2\t" $4 "\t" $5
        } else if (filesystem == "cgroup") {
          count=split(super_options, names, ",")
          for (name_index=1; name_index<=count; name_index++) {
            if (names[name_index] == "memory") {
              print "1\t" $4 "\t" $5
              break
            }
          }
        }
      }
    ' "$mountinfo_file"
  )

  if [ "$best_length" -lt 0 ]; then
    _crfs_write_cgroup_memory_diagnostic \
      "$diagnostic_file" unavailable peak_file_unreadable \
      "$fallback_version" "$fallback_membership" "$fallback_root" "$fallback_point" \
      "$fallback_suffix" "$fallback_peak_file" '' \
      "$proc_cgroup_file" "$mountinfo_file" || true
    return 1
  fi

  IFS= read -r peak_bytes <"$best_peak_file" || peak_bytes=
  case "$peak_bytes" in
    *[!0-9]*|'')
      _crfs_write_cgroup_memory_diagnostic \
        "$diagnostic_file" invalid peak_not_positive_integer \
        "$best_version" "$best_membership" "$best_root" "$best_point" \
        "$best_suffix" "$best_peak_file" "$peak_bytes" \
        "$proc_cgroup_file" "$mountinfo_file" || true
      return 1
      ;;
  esac
  if [ "$peak_bytes" -le 0 ]; then
    _crfs_write_cgroup_memory_diagnostic \
      "$diagnostic_file" invalid peak_not_positive_integer \
      "$best_version" "$best_membership" "$best_root" "$best_point" \
      "$best_suffix" "$best_peak_file" "$peak_bytes" \
      "$proc_cgroup_file" "$mountinfo_file" || true
    return 1
  fi
  if ! _crfs_write_cgroup_memory_diagnostic \
    "$diagnostic_file" measured live_positive_peak \
    "$best_version" "$best_membership" "$best_root" "$best_point" \
    "$best_suffix" "$best_peak_file" "$peak_bytes" \
    "$proc_cgroup_file" "$mountinfo_file"; then
    return 1
  fi
  printf '%s\n' "$peak_bytes"
}
