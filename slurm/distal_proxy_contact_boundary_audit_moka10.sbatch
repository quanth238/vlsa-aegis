#!/usr/bin/env bash
#SBATCH --job-name=vlsa-proxy-contact
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-proxy-contact-boundary-audit}
SUPPORT_ROOT=${SUPPORT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-targeted-boundary-expansion/targeted-boundary-expansion-20260810a}
SUPPORTED_MLP_ROOT=${SUPPORTED_MLP_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-supported-region-aware-mlp/supported-region-aware-mlp-20260810a}
TARGET_ROOT=${TARGET_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-margin-target-authority/margin-target-authority-20260810a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_proxy_contact_boundary_audit_moka10.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_MEM_PER_NODE:-}" == 16384 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "proxy-contact audit requires exactly one H100 allocation"
for path in \
  "$SUPPORT_ROOT/expanded-dataset.json" "$SUPPORT_ROOT/expanded-oracle.json" \
  "$SUPPORTED_MLP_ROOT/validation-fresh.json" \
  "$TARGET_ROOT/result.json" "$TARGET_ROOT/validation.json"
do
  [[ -f "$path" && ! -L "$path" ]] || die "immutable proxy-contact input missing: $path"
done
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 79070aead126870039a17154903089efc9ec075b10dda38834ee47d5ffcaf680 ]] || die "proxy-contact config hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
{
  "$AEGIS_PYTHON" -c 'import numpy; print({"numpy":numpy.__version__})'
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits
  "$AEGIS_PYTHON" -m unittest tests.test_distal_proxy_contact_boundary_audit -v
} >"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_proxy_contact_boundary_audit_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --expanded-dataset "$SUPPORT_ROOT/expanded-dataset.json" \
  --expanded-oracle "$SUPPORT_ROOT/expanded-oracle.json" \
  --validation-fresh "$SUPPORTED_MLP_ROOT/validation-fresh.json" \
  --target-result "$TARGET_ROOT/result.json" \
  --target-validation "$TARGET_ROOT/validation.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_proxy_contact_boundary_audit_moka10.py \
  --result "$RUN_ROOT/result.json" --config "$CONFIG" \
  --expanded-dataset "$SUPPORT_ROOT/expanded-dataset.json" \
  --expanded-oracle "$SUPPORT_ROOT/expanded-oracle.json" \
  --validation-fresh "$SUPPORTED_MLP_ROOT/validation-fresh.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" --output "$RUN_ROOT/validation.json"
