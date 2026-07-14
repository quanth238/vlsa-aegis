#!/usr/bin/env bash
set -euo pipefail

JOB_ID=${1:?usage: monitor_job.sh JOB_ID}
HOST=${VINUNI_HOST:-vinuni}
ssh "$HOST" "squeue -j '$JOB_ID' -o '%.18i %.32j %.2t %.10M %.12l %.6D %R'; scontrol show job '$JOB_ID' | head -n 80; sacct -j '$JOB_ID' --format=JobID,State,Elapsed,ExitCode,AllocTRES,MaxRSS -X 2>/dev/null | head -n 80 || true"
