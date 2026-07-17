#!/usr/bin/env bash
set -euo pipefail

DEST=/mnt/data/quanth/cache/aegis/groundingdino
NAME=groundingdino_swint_ogc.pth
URL=https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

mkdir -p "$DEST"
if [[ -f "$DEST/$NAME" ]]; then
    sha256sum "$DEST/$NAME"
    exit 0
fi

TMP="$DEST/$NAME.partial.${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
trap 'rm -f "$TMP" "$TMP.sha256"' EXIT
curl -fL --retry 3 --retry-delay 5 -o "$TMP" "$URL"
sha256sum "$TMP" > "$TMP.sha256"
mv "$TMP" "$DEST/$NAME"
mv "$TMP.sha256" "$DEST/$NAME.sha256"
trap - EXIT
sha256sum "$DEST/$NAME"
