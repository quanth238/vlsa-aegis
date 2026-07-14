#!/usr/bin/env bash
set -euo pipefail

HOST=${VINUNI_HOST:-vinuni}
ROOT=${1:-/mnt/data/quanth/experiments/crfs-oracle}
echo "Print-only cleanup candidates under $ROOT; this script never deletes."
ssh "$HOST" "find '$ROOT' -maxdepth 4 -type f \( -name 'failure.json' -o -name '*.tmp' \) -mtime +7 -print 2>/dev/null | head -n 200"
