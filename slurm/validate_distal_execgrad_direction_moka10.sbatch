#!/usr/bin/env bash
#SBATCH --job-name=vlsa-execgrad-verify
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
VALIDATOR_COMMIT=${VALIDATOR_COMMIT:?VALIDATOR_COMMIT is required}
PRODUCER_COMMIT=${PRODUCER_COMMIT:?PRODUCER_COMMIT is required}
SOURCE_RUN=${SOURCE_RUN:?SOURCE_RUN is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXECGRAD_CONFIG=$REMOTE_REPO/configs/vlsa_distal_execgrad_direction_moka10.v1.json

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "validation requires Slurm"
gpu_line=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "validation requires exactly one H100"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$VALIDATOR_COMMIT" ]] || die "validator source differs"
[[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "validator source is dirty"
[[ -d "$SOURCE_RUN" && ! -L "$SOURCE_RUN" ]] || die "immutable source run missing or symlinked"
[[ ! -e "$OUTPUT" ]] || die "validation output already exists"
[[ -x "$AEGIS_PYTHON" ]] || die "evaluation runtime missing"
[[ "$(sha256sum "$EXECGRAD_CONFIG" | awk '{print $1}')" == 6f0a6306fbf56a78e5ec01c3df0cf729640b34ff5436a3f3817d1906e5b3ca8a ]] || die "ExecGrad config differs"
[[ "$(sha256sum "$SOURCE_RUN/preprocess.npz" | awk '{print $1}')" == b90327ab5eee63694e8a72c8930d8eee34cdd53d4902f37baa78ad74175af9f7 ]] || die "preprocess artifact differs"
[[ "$(sha256sum "$SOURCE_RUN/model.npz" | awk '{print $1}')" == b58b17ebfb26c719ac597443ab1a52338798a6f420a3f13fe3341b0afe657b54 ]] || die "model artifact differs"
[[ "$(sha256sum "$SOURCE_RUN/predictions.npz" | awk '{print $1}')" == 84255c824b28cb65f3bc2e23af79410a1b6e46a46b210536fe412aef92d44dec ]] || die "predictions artifact differs"
[[ "$(sha256sum "$SOURCE_RUN/result.json" | awk '{print $1}')" == 14af299f02932f14d57f9597970efebdaf189ba3d1fa2b109c0a70472b2d42b9 ]] || die "scientific result differs"

VALIDATION_ROOT=$(dirname "$OUTPUT")
mkdir -p "$VALIDATION_ROOT"
[[ ! -e "$VALIDATION_ROOT/preflight.txt" ]] || die "validation preflight already exists"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -m unittest tests.test_distal_execgrad -v >"$VALIDATION_ROOT/preflight.txt" 2>&1
exec "$AEGIS_PYTHON" scripts/validate_distal_execgrad_direction_moka10.py \
  --repo-root "$REMOTE_REPO" \
  --config "$EXECGRAD_CONFIG" \
  --result "$SOURCE_RUN/result.json" \
  --preprocess "$SOURCE_RUN/preprocess.npz" \
  --model "$SOURCE_RUN/model.npz" \
  --predictions "$SOURCE_RUN/predictions.npz" \
  --expected-commit "$PRODUCER_COMMIT" \
  --validator-commit "$VALIDATOR_COMMIT" \
  --output "$OUTPUT"
