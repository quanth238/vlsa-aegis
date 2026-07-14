#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?JSON Schema overlay preparation must execute inside Slurm}"
: "${LIBERO_PYTHON:=/mnt/data/quanth/venvs/openpi-libero-client/bin/python}"
: "${JSONSCHEMA_SOURCE_SITE:=/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages}"
: "${JSONSCHEMA_OVERLAY:=/mnt/data/quanth/cache/crfs/jsonschema-4.23.0-py38}"

EXPECTED_SOURCE_BUNDLE_SHA256=72ccff502fcffe6ab4515cff5f9e3de8fa70e0a2b54c235389df003d25880a6c
PACKAGE_PATHS=(
  attr
  attrs
  attrs-25.3.0.dist-info
  importlib_resources
  importlib_resources-6.4.5.dist-info
  jsonschema
  jsonschema-4.23.0.dist-info
  jsonschema_specifications
  jsonschema_specifications-2023.12.1.dist-info
  pkgutil_resolve_name.py
  pkgutil_resolve_name-1.3.10.dist-info
  referencing
  referencing-0.35.1.dist-info
  rpds
  rpds_py-0.20.1.dist-info
  typing_extensions.py
  typing_extensions-4.12.2.dist-info
  zipp
  zipp-3.20.2.dist-info
)

test -x "$LIBERO_PYTHON" || {
  echo "missing LIBERO Python for JSON Schema overlay: $LIBERO_PYTHON" >&2
  exit 2
}
"$LIBERO_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version'
for package_path in "${PACKAGE_PATHS[@]}"; do
  test -e "$JSONSCHEMA_SOURCE_SITE/$package_path" || {
    echo "missing frozen JSON Schema source package: $package_path" >&2
    exit 2
  }
done

bundle_hash() {
  root=$1
  (
    cd "$root"
    find "${PACKAGE_PATHS[@]}" -type f \
      ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

validate_overlay() {
  root=$1
  test -d "$root" || return 1
  test -f "$root/.source-bundle-sha256" || return 1
  test ! -L "$root/.source-bundle-sha256" || return 1
  marker=$(tr -d '[:space:]' < "$root/.source-bundle-sha256")
  test "$marker" = "$EXPECTED_SOURCE_BUNDLE_SHA256" || return 1
  expected_top_level=$(printf '%s\n' \
    "${PACKAGE_PATHS[@]}" .source-bundle-sha256 | sort)
  observed_top_level=$(find "$root" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
  test "$observed_top_level" = "$expected_top_level" || return 1
  test -z "$(find "$root" \( -name __pycache__ -o -name '*.pyc' \) -print -quit)" \
    || return 1
  test -z "$(
    cd "$root"
    find "${PACKAGE_PATHS[@]}" ! -type f ! -type d -print -quit
  )" || return 1
  observed_bundle_sha256=$(bundle_hash "$root")
  test "$observed_bundle_sha256" = "$EXPECTED_SOURCE_BUNDLE_SHA256"
}

SOURCE_BUNDLE_SHA256=$(bundle_hash "$JSONSCHEMA_SOURCE_SITE")
test "$SOURCE_BUNDLE_SHA256" = "$EXPECTED_SOURCE_BUNDLE_SHA256" || {
  echo "JSON Schema source bundle differs from the reviewed dependency" >&2
  exit 3
}

command -v flock >/dev/null || {
  echo "flock is required for atomic JSON Schema overlay publication" >&2
  exit 3
}
mkdir -p "$(dirname "$JSONSCHEMA_OVERLAY")"
exec 9>"$JSONSCHEMA_OVERLAY.lock"
flock -x 9

if [ -e "$JSONSCHEMA_OVERLAY" ]; then
  validate_overlay "$JSONSCHEMA_OVERLAY" || {
    echo "existing JSON Schema overlay failed content verification" >&2
    exit 3
  }
  echo "$JSONSCHEMA_OVERLAY"
  exit 0
fi

TMP=$JSONSCHEMA_OVERLAY.incomplete-$SLURM_JOB_ID
test ! -e "$TMP" || {
  echo "temporary JSON Schema overlay already exists: $TMP" >&2
  exit 3
}
mkdir -p "$TMP"
(
  cd "$JSONSCHEMA_SOURCE_SITE"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf - "${PACKAGE_PATHS[@]}"
) | (
  cd "$TMP"
  tar -xf -
)
BUILT_BUNDLE_SHA256=$(bundle_hash "$TMP")
test "$BUILT_BUNDLE_SHA256" = "$EXPECTED_SOURCE_BUNDLE_SHA256" || {
  echo "built JSON Schema overlay failed content verification" >&2
  exit 3
}
printf '%s\n' "$EXPECTED_SOURCE_BUNDLE_SHA256" > "$TMP/.source-bundle-sha256"
validate_overlay "$TMP" || {
  echo "built JSON Schema overlay failed exact-tree verification" >&2
  exit 3
}
mv -T "$TMP" "$JSONSCHEMA_OVERLAY"
validate_overlay "$JSONSCHEMA_OVERLAY" || {
  echo "published JSON Schema overlay failed exact-tree verification" >&2
  exit 3
}
echo "$JSONSCHEMA_OVERLAY"
