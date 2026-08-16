#!/usr/bin/env bash
#SBATCH --job-name=vlsa-prep-perconstraint
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

BASE_REPO=${BASE_REPO:?BASE_REPO is required}
TARGET_REPO=${TARGET_REPO:?TARGET_REPO is required}
TARGET_BRANCH=${TARGET_BRANCH:?TARGET_BRANCH is required}
TARGET_COMMIT=${TARGET_COMMIT:?TARGET_COMMIT is required}

[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ ! -e "$TARGET_REPO" ]] || {
  echo "target source already exists: $TARGET_REPO" >&2
  exit 2
}
[[ -z "$(git -C "$BASE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "base source is dirty: $BASE_REPO" >&2
  exit 2
}

git -C "$BASE_REPO" fetch origin "$TARGET_BRANCH"
git -C "$BASE_REPO" worktree add --detach "$TARGET_REPO" "$TARGET_COMMIT"
[[ "$(git -C "$TARGET_REPO" rev-parse HEAD)" == "$TARGET_COMMIT" ]]
[[ -z "$(git -C "$TARGET_REPO" status --porcelain=v1 --untracked-files=all)" ]]

echo "prepared=$TARGET_REPO"
echo "commit=$TARGET_COMMIT"
