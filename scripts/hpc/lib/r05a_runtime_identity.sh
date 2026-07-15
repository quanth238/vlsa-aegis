#!/usr/bin/env bash

# Frozen interpreter identities for the R05A constrained-flow apparatus.
# Validation is shell-only: it must never invoke either interpreter.
CRFS_R05A_OPENPI_PYTHON=/mnt/data/quanth/venvs/openpi/bin/python
CRFS_R05A_OPENPI_PYTHON_LINK_TARGET=/mnt/data/quanth/anaconda3/bin/python
CRFS_R05A_OPENPI_PYTHON_RESOLVED=/mnt/data/quanth/anaconda3/bin/python3.11
CRFS_R05A_OPENPI_PYTHON_SHA256=c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9
CRFS_R05A_LIBERO_PYTHON=/mnt/data/quanth/venvs/openpi-libero-client/bin/python
CRFS_R05A_LIBERO_PYTHON_LINK_TARGET=/home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8
CRFS_R05A_LIBERO_PYTHON_RESOLVED=/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8
CRFS_R05A_LIBERO_PYTHON_SHA256=c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62
readonly \
  CRFS_R05A_OPENPI_PYTHON \
  CRFS_R05A_OPENPI_PYTHON_LINK_TARGET \
  CRFS_R05A_OPENPI_PYTHON_RESOLVED \
  CRFS_R05A_OPENPI_PYTHON_SHA256 \
  CRFS_R05A_LIBERO_PYTHON \
  CRFS_R05A_LIBERO_PYTHON_LINK_TARGET \
  CRFS_R05A_LIBERO_PYTHON_RESOLVED \
  CRFS_R05A_LIBERO_PYTHON_SHA256

crfs_require_exact_interpreter_identity() {
  local label=${1:?interpreter label is required}
  local observed_public=${2:?interpreter public path is required}
  local expected_public=${3:?expected public path is required}
  local expected_link_target=${4:?expected direct link target is required}
  local expected_resolved=${5:?expected resolved executable is required}
  local expected_sha256=${6:?expected executable SHA-256 is required}
  local observed_link_target
  local observed_resolved
  local digest_record
  local observed_sha256

  case "$expected_sha256" in
    *[!0-9a-f]*|'') echo "$label interpreter SHA-256 contract is invalid" >&2; return 2 ;;
  esac
  if [ "${#expected_sha256}" -ne 64 ]; then
    echo "$label interpreter SHA-256 contract has the wrong length" >&2
    return 2
  fi
  if [ "$observed_public" != "$expected_public" ]; then
    echo "$label interpreter public path changed: $observed_public" >&2
    return 2
  fi
  if [ ! -L "$observed_public" ]; then
    echo "$label interpreter public path is not the reviewed symlink: $observed_public" >&2
    return 2
  fi
  if ! observed_link_target=$(readlink -- "$observed_public"); then
    echo "$label interpreter direct link target is unreadable" >&2
    return 2
  fi
  if [ "$observed_link_target" != "$expected_link_target" ]; then
    echo "$label interpreter direct link target changed: $observed_link_target" >&2
    return 2
  fi
  if ! observed_resolved=$(readlink -f -- "$observed_public"); then
    echo "$label interpreter link chain does not resolve" >&2
    return 2
  fi
  if [ "$observed_resolved" != "$expected_resolved" ]; then
    echo "$label interpreter resolved executable changed: $observed_resolved" >&2
    return 2
  fi
  if [ ! -f "$observed_resolved" ] || [ -L "$observed_resolved" ] || [ ! -x "$observed_resolved" ]; then
    echo "$label interpreter resolved target is not a regular non-symlink executable" >&2
    return 2
  fi
  if ! digest_record=$(sha256sum -- "$observed_resolved"); then
    echo "$label interpreter resolved executable cannot be hashed" >&2
    return 2
  fi
  observed_sha256=${digest_record%% *}
  if [ "$observed_sha256" != "$expected_sha256" ]; then
    echo "$label interpreter resolved executable SHA-256 changed: $observed_sha256" >&2
    return 2
  fi
}

crfs_validate_r05a_openpi_python() {
  crfs_require_exact_interpreter_identity \
    OpenPI \
    "${1:?OpenPI interpreter path is required}" \
    "$CRFS_R05A_OPENPI_PYTHON" \
    "$CRFS_R05A_OPENPI_PYTHON_LINK_TARGET" \
    "$CRFS_R05A_OPENPI_PYTHON_RESOLVED" \
    "$CRFS_R05A_OPENPI_PYTHON_SHA256"
}

crfs_validate_r05a_libero_python() {
  crfs_require_exact_interpreter_identity \
    LIBERO \
    "${1:?LIBERO interpreter path is required}" \
    "$CRFS_R05A_LIBERO_PYTHON" \
    "$CRFS_R05A_LIBERO_PYTHON_LINK_TARGET" \
    "$CRFS_R05A_LIBERO_PYTHON_RESOLVED" \
    "$CRFS_R05A_LIBERO_PYTHON_SHA256"
}
