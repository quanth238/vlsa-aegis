#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
R03A_SUBMISSION_MODE=smoke exec "$ROOT/scripts/hpc/submit_r03a_job.sh" "$@"
