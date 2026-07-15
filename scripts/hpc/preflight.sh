#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
HOST=${VINUNI_HOST:-vinuni}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${PREFLIGHT_OUT:-$ROOT/.harness/live/preflight-$STAMP.txt}
mkdir -p "$(dirname "$OUT")"

ssh "$HOST" 'bash -s' >"$OUT" <<'REMOTE'
set -euo pipefail
echo "captured_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "user=$(whoami)"
echo "[jobs]"
squeue -u "$(whoami)" -o "%.18i %.32j %.2t %.10M %.12l %.6D %R"
echo "[partitions]"
sinfo -o "%20P %8a %10l %8D %20G %N"
echo "[nodes]"
sinfo -N -o "%20N %10t %8c %10m %20G %E"
echo "[partition-main-detail]"
scontrol show partition main -o
echo "[worker-1-detail]"
scontrol show node worker-1 -o
echo "[qos]"
sacctmgr -n -P show qos normal format=Name,MaxWall,MaxTRESPerJob,MaxTRESPU,GrpTRES,MaxJobsPU,MaxSubmitJobsPU 2>&1
echo "[association]"
sacctmgr -n -P show assoc user="$(whoami)" format=Cluster,Account,User,Partition,QOS,DefaultQOS,MaxWall,MaxTRESPerJob,MaxTRESPU,GrpTRES,MaxJobs,MaxSubmitJobs 2>&1
echo "[storage]"
df -h /mnt/data
echo "[paths]"
for path in \
  /home/quanth/working_space \
  /mnt/data/quanth/experiments \
  /mnt/data/quanth/slurm_logs \
  /mnt/data/quanth/cache \
  /mnt/data/quanth/venvs/openpi/bin/python \
  /mnt/data/quanth/venvs/openpi-libero-client/bin/python; do
  if [ -e "$path" ]; then echo "present $path"; else echo "MISSING $path"; fi
done
echo "[login-process-audit]"
pgrep -au "$(whoami)" "tmux|screen|python|train\.py|render\.py|metrics\.py|torch|cuda|serve|run_crfs_oracle|uvicorn|gradio|streamlit|jupyter|vllm|ollama" || true
REMOTE

if ! grep -Eq '^host=login-restricted-[0-9]+$' "$OUT"; then
  echo "WHAT: SSH did not report an expected VinUni login host" >&2
  echo "WHY: compute must never be sent to an unknown machine" >&2
  echo "FIX: inspect $OUT and VINUNI_HOST" >&2
  exit 1
fi
if grep -q '^MISSING ' "$OUT"; then
  echo "WHAT: required remote paths are missing" >&2
  echo "WHY: an incomplete environment cannot produce reproducible evidence" >&2
  echo "FIX: inspect $OUT and provision paths inside a Slurm setup job" >&2
  exit 1
fi
storage_percent=$(awk '
  /^\[storage\]$/ {storage=1; next}
  storage && $0 !~ /^Filesystem/ && NF >= 5 {value=$5; gsub(/%/, "", value); print value; exit}
' "$OUT")
case "$storage_percent" in
  *[!0-9]*|'')
    echo "WHAT: /mnt/data utilization could not be parsed" >&2
    echo "WHY: the guide requires a bounded shared-storage check before submission" >&2
    echo "FIX: inspect $OUT and rerun preflight" >&2
    exit 1
    ;;
esac
if [ "$storage_percent" -ge 90 ]; then
  echo "WHAT: /mnt/data is ${storage_percent}% full" >&2
  echo "WHY: the VinUni guide requires storage pressure to be resolved before a new job" >&2
  echo "FIX: stop; inspect only narrow project paths and coordinate any cleanup explicitly" >&2
  exit 1
fi
if awk '/^\[login-process-audit\]$/{audit=1; next} audit && NF{found=1} END{exit !found}' "$OUT"; then
  echo "WHAT: possible experiment or service process exists on the login node" >&2
  echo "WHY: all CRFS compute and serving must remain inside Slurm allocations" >&2
  echo "FIX: inspect the exact process in $OUT; do not kill it without confirming ownership and purpose" >&2
  exit 1
fi
echo "preflight saved to $OUT"
