#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Transformers overlay preparation must execute inside Slurm}"
: "${REMOTE_REPO:=/home/quanth/working_space/vlsa-aegis-crfs}"
: "${TRANSFORMERS_SITE_PACKAGES:=/mnt/data/quanth/venvs/openpi/lib/python3.11/site-packages}"
: "${TRANSFORMERS_OVERLAY:=/mnt/data/quanth/cache/crfs/transformers-openpi-4.53.2}"

SOURCE=$TRANSFORMERS_SITE_PACKAGES/transformers
REPLACEMENTS=$REMOTE_REPO/openpi/src/openpi/models_pytorch/transformers_replace
test -f "$SOURCE/__init__.py" || { echo "installed Transformers package not found at $SOURCE" >&2; exit 2; }
test -d "$REPLACEMENTS" || { echo "OpenPI replacement files not found at $REPLACEMENTS" >&2; exit 2; }
REPLACEMENT_HASH=$(find "$REPLACEMENTS" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')

if [ -f "$TRANSFORMERS_OVERLAY/.openpi-replacement-sha256" ]; then
  EXISTING_HASH=$(tr -d '[:space:]' < "$TRANSFORMERS_OVERLAY/.openpi-replacement-sha256")
  test "$EXISTING_HASH" = "$REPLACEMENT_HASH" || {
    echo "existing Transformers overlay was built from different replacement files: $TRANSFORMERS_OVERLAY" >&2
    exit 3
  }
  echo "$TRANSFORMERS_OVERLAY"
  exit 0
fi
test ! -e "$TRANSFORMERS_OVERLAY" || {
  echo "partial Transformers overlay exists; inspect before cleanup: $TRANSFORMERS_OVERLAY" >&2
  exit 3
}

TMP=$TRANSFORMERS_OVERLAY.incomplete-$SLURM_JOB_ID
test ! -e "$TMP" || { echo "temporary overlay already exists: $TMP" >&2; exit 3; }
mkdir -p "$TMP"
cp -a "$SOURCE" "$TMP/transformers"
cp -a "$REPLACEMENTS/." "$TMP/transformers/"
printf '%s\n' "$REPLACEMENT_HASH" > "$TMP/.openpi-replacement-sha256"
mv "$TMP" "$TRANSFORMERS_OVERLAY"
echo "$TRANSFORMERS_OVERLAY"
