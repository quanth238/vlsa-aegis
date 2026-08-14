#!/usr/bin/env bash
#SBATCH --job-name=vlsa-prep-validator
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
VALIDATOR_BRANCH=${VALIDATOR_BRANCH:?VALIDATOR_BRANCH is required}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
SOURCE_URL=${SOURCE_URL:-}

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "source preparation requires Slurm" >&2
  exit 2
}
fresh_clone=0
if [[ ! -d "$REMOTE_REPO" ]]; then
  [[ -n "$SOURCE_URL" ]] || {
    echo "worker-local source is missing and SOURCE_URL is unset: $REMOTE_REPO" >&2
    exit 2
  }
  mkdir -p "$(dirname "$REMOTE_REPO")"
  git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$REMOTE_REPO"
  fresh_clone=1
fi
if [[ "$fresh_clone" == 0 ]]; then
  [[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "worker-local source is dirty" >&2
    exit 2
  }
fi
git -C "$REMOTE_REPO" fetch origin "$VALIDATOR_BRANCH"
git -C "$REMOTE_REPO" checkout --detach "$VALIDATOR_COMMIT"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || {
  echo "worker-local source preparation failed" >&2
  exit 2
}

echo "prepared=$REMOTE_REPO"
echo "commit=$VALIDATOR_COMMIT"
