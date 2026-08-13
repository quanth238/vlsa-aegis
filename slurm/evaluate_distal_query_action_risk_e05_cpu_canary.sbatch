#!/usr/bin/env bash
#SBATCH --job-name=vlsa-query-risk-cpu
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --nodelist=worker-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:?REMOTE_REPO is required}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
CANDIDATE_LIMIT=${CANDIDATE_LIMIT:-2}
export QUERY_RISK_CPU_CANARY=1
exec bash "$REMOTE_REPO/slurm/evaluate_distal_query_action_risk_e05.sbatch"
