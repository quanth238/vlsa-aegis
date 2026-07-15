#!/usr/bin/env bash
set -euo pipefail

# These digests were reviewed from the installed Transformers 4.53.2 source,
# the five repository replacements, and the resulting patched tree on VinUni.
# The bundle format is the SHA-256 of the sorted, relative ``sha256sum``
# manifest, excluding bytecode caches.  The cache path is content-addressed so
# the older marker-only overlay remains untouched and can never be reused here.
EXPECTED_SOURCE_BUNDLE_SHA256=430b00a688e12ff457cdd65929bd164fd001ffa1716dc83589d5388806d2bb33
EXPECTED_REPLACEMENT_BUNDLE_SHA256=2e1b546bdf42e9872c84734b2d5baf52bd411664685922458e734732c8434098
EXPECTED_OVERLAY_BUNDLE_SHA256=24be8ac6749a4cf7e19c261b14b39e951a499ec61b0602d0badcc4354171d261
DEFAULT_TRANSFORMERS_SITE_PACKAGES=/mnt/data/quanth/venvs/openpi/lib/python3.11/site-packages
DEFAULT_TRANSFORMERS_OVERLAY=/mnt/data/quanth/cache/crfs/transformers-openpi-4.53.2-exact-24be8ac6749a

bundle_hash() {
  local root=${1:?bundle root is required}
  local subtree=${2:?bundle subtree is required}
  (
    cd "$root"
    find "$subtree" -type f \
      ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

validate_plain_source_tree() {
  local root=${1:?source root is required}
  local subtree=${2:?source subtree is required}
  test -d "$root/$subtree" || return 1
  # The installed source may contain caches, but no symlink or other special
  # file may contribute an unreviewed import target.
  test -z "$(find "$root/$subtree" ! -type f ! -type d -print -quit)"
}

validate_cache_free_tree() {
  local root=${1:?tree root is required}
  local subtree=${2:?tree subtree is required}
  test -d "$root/$subtree" || return 1
  test -z "$(find "$root/$subtree" \( -type d -name __pycache__ -o -type f -name '*.pyc' \) -print -quit)" \
    || return 1
  test -z "$(find "$root/$subtree" ! -type f ! -type d -print -quit)"
}

validate_overlay() {
  local root=${1:?overlay root is required}
  local source_sha=${2:?source digest is required}
  local replacement_sha=${3:?replacement digest is required}
  local overlay_sha=${4:?overlay digest is required}
  local expected_top_level observed_top_level marker

  test -d "$root" && test ! -L "$root" || return 1
  expected_top_level=$(printf '%s\n' \
    .overlay-bundle-sha256 \
    .replacement-bundle-sha256 \
    .source-bundle-sha256 \
    transformers | LC_ALL=C sort)
  observed_top_level=$(find "$root" -mindepth 1 -maxdepth 1 -exec basename {} \; | LC_ALL=C sort)
  test "$observed_top_level" = "$expected_top_level" || return 1
  validate_cache_free_tree "$root" transformers || return 1
  for marker in .source-bundle-sha256 .replacement-bundle-sha256 .overlay-bundle-sha256; do
    test -f "$root/$marker" && test ! -L "$root/$marker" || return 1
  done
  test "$(tr -d '[:space:]' <"$root/.source-bundle-sha256")" = "$source_sha" || return 1
  test "$(tr -d '[:space:]' <"$root/.replacement-bundle-sha256")" = "$replacement_sha" || return 1
  test "$(tr -d '[:space:]' <"$root/.overlay-bundle-sha256")" = "$overlay_sha" || return 1
  test "$(bundle_hash "$root" transformers)" = "$overlay_sha"
}

# A dependency-free regression seam lets the contract test exercise the exact
# validator against mutated bytes, bytecode caches, and symlinks.  It does not
# alter or construct a production overlay.
if [ "${1:-}" = --validate-overlay-fixture ]; then
  test "$#" -eq 5 || exit 64
  validate_overlay "$2" "$3" "$4" "$5"
  exit
fi

: "${SLURM_JOB_ID:?Transformers overlay preparation must execute inside Slurm}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${TRANSFORMERS_SITE_PACKAGES:=$DEFAULT_TRANSFORMERS_SITE_PACKAGES}"
: "${TRANSFORMERS_OVERLAY:=$DEFAULT_TRANSFORMERS_OVERLAY}"

SOURCE=$TRANSFORMERS_SITE_PACKAGES/transformers
REPLACEMENTS=$REMOTE_REPO/openpi/src/openpi/models_pytorch/transformers_replace
test -f "$SOURCE/__init__.py" || { echo "installed Transformers package not found at $SOURCE" >&2; exit 2; }
test -d "$REPLACEMENTS" || { echo "OpenPI replacement files not found at $REPLACEMENTS" >&2; exit 2; }
validate_plain_source_tree "$TRANSFORMERS_SITE_PACKAGES" transformers || {
  echo "installed Transformers source contains an unsupported file type" >&2
  exit 3
}
validate_cache_free_tree "$REPLACEMENTS" . || {
  echo "OpenPI replacement tree contains cache or unsupported file bytes" >&2
  exit 3
}
SOURCE_BUNDLE_SHA256=$(bundle_hash "$TRANSFORMERS_SITE_PACKAGES" transformers)
REPLACEMENT_BUNDLE_SHA256=$(bundle_hash "$REPLACEMENTS" .)
test "$SOURCE_BUNDLE_SHA256" = "$EXPECTED_SOURCE_BUNDLE_SHA256" || {
  echo "installed Transformers source differs from the reviewed 4.53.2 bundle" >&2
  exit 3
}
test "$REPLACEMENT_BUNDLE_SHA256" = "$EXPECTED_REPLACEMENT_BUNDLE_SHA256" || {
  echo "OpenPI replacement bytes differ from the reviewed bundle" >&2
  exit 3
}

command -v flock >/dev/null || { echo "flock is required for atomic Transformers overlay publication" >&2; exit 3; }
mkdir -p "$(dirname "$TRANSFORMERS_OVERLAY")"
exec 9>"$TRANSFORMERS_OVERLAY.lock"
flock -x 9

if [ -e "$TRANSFORMERS_OVERLAY" ]; then
  validate_overlay \
    "$TRANSFORMERS_OVERLAY" \
    "$EXPECTED_SOURCE_BUNDLE_SHA256" \
    "$EXPECTED_REPLACEMENT_BUNDLE_SHA256" \
    "$EXPECTED_OVERLAY_BUNDLE_SHA256" || {
      echo "existing Transformers overlay failed exact-tree verification: $TRANSFORMERS_OVERLAY" >&2
      exit 3
    }
  echo "$TRANSFORMERS_OVERLAY"
  exit 0
fi

TMP=$TRANSFORMERS_OVERLAY.incomplete-$SLURM_JOB_ID
test ! -e "$TMP" || { echo "temporary overlay already exists: $TMP" >&2; exit 3; }
mkdir -p "$TMP/transformers"
(
  cd "$SOURCE"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf - .
) | (
  cd "$TMP/transformers"
  tar -xf -
)
(
  cd "$REPLACEMENTS"
  tar --exclude='__pycache__' --exclude='*.pyc' -cf - .
) | (
  cd "$TMP/transformers"
  tar -xf -
)
printf '%s\n' "$EXPECTED_SOURCE_BUNDLE_SHA256" >"$TMP/.source-bundle-sha256"
printf '%s\n' "$EXPECTED_REPLACEMENT_BUNDLE_SHA256" >"$TMP/.replacement-bundle-sha256"
printf '%s\n' "$EXPECTED_OVERLAY_BUNDLE_SHA256" >"$TMP/.overlay-bundle-sha256"
# Older harnesses also consume this shared helper.  Make the published tree
# read-only so their interpreters cannot add bytecode even if they do not yet
# export PYTHONDONTWRITEBYTECODE.
chmod -R a-w "$TMP"
validate_overlay \
  "$TMP" \
  "$EXPECTED_SOURCE_BUNDLE_SHA256" \
  "$EXPECTED_REPLACEMENT_BUNDLE_SHA256" \
  "$EXPECTED_OVERLAY_BUNDLE_SHA256" || {
    echo "built Transformers overlay failed exact-tree verification" >&2
    exit 3
  }
mv -T "$TMP" "$TRANSFORMERS_OVERLAY"
validate_overlay \
  "$TRANSFORMERS_OVERLAY" \
  "$EXPECTED_SOURCE_BUNDLE_SHA256" \
  "$EXPECTED_REPLACEMENT_BUNDLE_SHA256" \
  "$EXPECTED_OVERLAY_BUNDLE_SHA256" || {
    echo "published Transformers overlay failed exact-tree verification" >&2
    exit 3
  }
echo "$TRANSFORMERS_OVERLAY"
