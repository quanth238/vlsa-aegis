#!/usr/bin/env bash
#SBATCH --job-name=vlsa-targeted-cover
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-targeted-boundary-expansion}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
BASE_ROOT=${BASE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-same-task-boundary-expansion/same-task-boundary-expansion-20260810a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_targeted_boundary_expansion_moka10.v1.json
AFFINE_CONFIG=$REMOTE_REPO/configs/vlsa_distal_affine_coefficient_moka10.v1.json
MULTI_CONFIG=$REMOTE_REPO/configs/vlsa_distal_multi_region_affine_oracle_moka10.v1.json
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_targeted_same_task_screen_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "targeted boundary expansion requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$ARCHIVED_ROOT" && ! -L "$ARCHIVED_ROOT" ]] || die "runtime or immutable Table-1 root missing"
for path in \
  "$BASE_ROOT/expanded-dataset.json" "$BASE_ROOT/expanded-oracle.json" \
  "$BASE_ROOT/result.json" "$BASE_ROOT/validation.json"
do
  [[ -f "$path" && ! -L "$path" ]] || die "immutable targeted-expansion input missing: $path"
done
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 9c90aa1047cff37efa2091657a9c0c1d0d7e26719c29ac331602e76045d3813e ]] || die "targeted expansion config hash differs"
[[ "$(sha256sum "$SELECTED" | awk '{print $1}')" == 32a88c6e973031bb553c056abb244c39a8f5952f411e3d7b16c2a9c5b3717c3d ]] || die "targeted expansion selected manifest hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config; mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\ndatasets: %s\nassets: %s\n' "$safelibero_root" "$safelibero_root/bddl_files" "$safelibero_root/init_files" "$REMOTE_REPO/safelibero/libero/datasets" "$safelibero_root/assets" >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy, osqp, scipy, torch; print({"runtime":"simulation","torch":torch.__version__})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_targeted_boundary_expansion tests.test_distal_state_support_smoothness -v >>"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/evaluate_distal_targeted_boundary_expansion_moka10.py \
  --repo-root "$REMOTE_REPO" --population-manifest "$POPULATION" \
  --selected-manifest "$SELECTED" --archived-root "$ARCHIVED_ROOT" \
  --geometry-config "$GEOMETRY" --exact-box-config "$EXACT_BOX" \
  --affine-config "$AFFINE_CONFIG" --multi-region-config "$MULTI_CONFIG" \
  --config "$CONFIG" --base-expanded-dataset "$BASE_ROOT/expanded-dataset.json" \
  --base-expanded-oracle "$BASE_ROOT/expanded-oracle.json" \
  --prior-result "$BASE_ROOT/result.json" \
  --prior-validation "$BASE_ROOT/validation.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --scan-records "$RUN_ROOT/scan-records.json" \
  --selection "$RUN_ROOT/selection.json" \
  --additional-dataset "$RUN_ROOT/additional-dataset.json" \
  --collection-result "$RUN_ROOT/collection-result.json" \
  --expanded-dataset "$RUN_ROOT/expanded-dataset.json" \
  --expanded-oracle "$RUN_ROOT/expanded-oracle.json" \
  --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_targeted_boundary_expansion_moka10.py \
  --result "$RUN_ROOT/result.json" --config "$CONFIG" \
  --base-expanded-dataset "$BASE_ROOT/expanded-dataset.json" \
  --base-expanded-oracle "$BASE_ROOT/expanded-oracle.json" \
  --scan-records "$RUN_ROOT/scan-records.json" \
  --selection "$RUN_ROOT/selection.json" \
  --additional-dataset "$RUN_ROOT/additional-dataset.json" \
  --collection-result "$RUN_ROOT/collection-result.json" \
  --expanded-dataset "$RUN_ROOT/expanded-dataset.json" \
  --expanded-oracle "$RUN_ROOT/expanded-oracle.json" \
  --expected-commit "$EXPECTED_GIT_COMMIT" \
  --output "$RUN_ROOT/validation.json"
