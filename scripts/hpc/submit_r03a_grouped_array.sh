#!/usr/bin/env bash
set -euo pipefail

# Preserve the historical fail-closed concurrency contract even though the
# population itself is retired.  This check is local and occurs before any
# preflight or remote access.
if [[ $# -ge 6 && "$6" != "2" ]]; then
  echo "R03A source-node grouped full run has fixed total concurrency two" >&2
  exit 2
fi

if [[ "${R03A_HISTORICAL_REPLAY_ACK:-}" != "ADR-0028-R03A-REPLAY-ONLY" ]]; then
  echo "R03A grouped population was retired unrun by ADR-0028; no new submission is authorized" >&2
  echo "set R03A_HISTORICAL_REPLAY_ACK=ADR-0028-R03A-REPLAY-ONLY only for an explicitly reviewed historical reproducibility exercise" >&2
  exit 2
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
R03A_SUBMISSION_MODE=grouped_array exec "$ROOT/scripts/hpc/submit_r03a_job.sh" "$@"
