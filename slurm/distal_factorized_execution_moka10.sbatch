#!/usr/bin/env bash
#SBATCH --job-name=vlsa-factor-exec
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail
REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
EXPECTED_GIT_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
RUN_ID=${RUN_ID:?RUN_ID is required}
AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-factorized-execution}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d}
MATCHED_ROOT=${MATCHED_ROOT:-/mnt/data/quanth/experiments/vlsa-distal-matched-input-ablation/matched-input-ablation-20260810a}
ARCHIVED_ROOT=${ARCHIVED_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a}
CONFIG=$REMOTE_REPO/configs/vlsa_distal_factorized_execution_moka10.v1.json
POPULATION=$REMOTE_REPO/manifests/vlsa_table1_population.jsonl
SELECTED=$REMOTE_REPO/manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl
SAME_TASK=$REMOTE_REPO/manifests/vlsa_distal_same_task_boundary_expansion_moka10.v1.jsonl
TARGETED=$REMOTE_REPO/manifests/vlsa_distal_targeted_same_task_screen_moka10.v1.jsonl
GEOMETRY=$REMOTE_REPO/configs/vlsa_distal_slabbed_ellipsoid_shadow_e05.v4.json
EXACT_BOX=$REMOTE_REPO/configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json
die() { printf '%s\n' "$*" >&2; exit 1; }
case "$RUN_ID" in ""|*[!A-Za-z0-9._-]*) die "unsafe RUN_ID" ;; esac
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || die "registered Slurm allocation differs"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "factorized-execution pilot requires exactly one H100"
[[ -x "$AEGIS_PYTHON" && -d "$ARCHIVED_ROOT" && ! -L "$ARCHIVED_ROOT" ]] || die "runtime or immutable archive missing"
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == 1fde25dc94f10bb4c4af91c35782df067172cdad5a0e6b6b42ae8b72b0ac4088 ]] || die "factorized-execution config hash differs"
[[ "$(sha256sum "$SOURCE_ROOT/dataset.json" | awk '{print $1}')" == 585f696eedef9bd9a4c9d0ba576d6d86e5660638696862ec2d5d1af434fb426b ]] || die "complete dataset hash differs"
[[ "$(sha256sum "$SOURCE_ROOT/collection-result.json" | awk '{print $1}')" == bd85f3175c790a153833a7c903a4e119c71f771271b2ecb92cc1c579269ce966 ]] || die "complete collection hash differs"
[[ "$(sha256sum "$MATCHED_ROOT/result.json" | awk '{print $1}')" == dc76f3b6d4f8f78d08861bc13b9cec2a396c931bb5479f5646b729fe0e76698a ]] || die "matched direct result hash differs"
[[ "$(git -C "$REMOTE_REPO" rev-parse HEAD)" == "$EXPECTED_GIT_COMMIT" && -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die "source identity differs"
RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
mkdir -p "$EXPERIMENT_ROOT"; mkdir "$RUN_ROOT" || die "immutable run root exists"
FAILURE=$RUN_ROOT/runtime-failure.json
cleanup() { code=$?; trap - EXIT INT TERM; if [[ $code -ne 0 && ! -e "$FAILURE" ]]; then temporary=$(mktemp "$RUN_ROOT/.failure.XXXXXX"); printf '{"exit_code":%d,"scientific_result":false,"slurm_job_id":"%s","status":"apparatus_failure"}\n' "$code" "$SLURM_JOB_ID" >"$temporary"; mv "$temporary" "$FAILURE"; fi; return "$code"; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main:$REMOTE_REPO
export LIBERO_CONFIG_PATH=$RUN_ROOT/libero-config
mkdir "$LIBERO_CONFIG_PATH"
safelibero_root=$REMOTE_REPO/safelibero/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\ndatasets: %s\nassets: %s\n' "$safelibero_root" "$safelibero_root/bddl_files" "$safelibero_root/init_files" "$REMOTE_REPO/safelibero/libero/datasets" "$safelibero_root/assets" >"$LIBERO_CONFIG_PATH/config.yaml"
cd "$REMOTE_REPO"
"$AEGIS_PYTHON" -c 'import numpy, scipy, torch; assert torch.cuda.is_available(); print({"numpy":numpy.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"cuda":torch.cuda.get_device_name(0)})' >"$RUN_ROOT/preflight.txt"
"$AEGIS_PYTHON" -m unittest tests.test_distal_factorized_execution_pilot tests.test_distal_complete_osc_margin -v >>"$RUN_ROOT/preflight.txt" 2>&1
"$AEGIS_PYTHON" scripts/collect_distal_factorized_execution_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --complete-dataset "$SOURCE_ROOT/dataset.json" \
  --complete-collection "$SOURCE_ROOT/collection-result.json" \
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" \
  --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED" \
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" \
  --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --array-dataset "$RUN_ROOT/trajectory-dataset.npz" \
  --metadata "$RUN_ROOT/trajectory-metadata.json" \
  --output "$RUN_ROOT/collection-result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_execution_dataset_moka10.py \
  --config "$CONFIG" --metadata "$RUN_ROOT/trajectory-metadata.json" \
  --array-dataset "$RUN_ROOT/trajectory-dataset.npz" \
  --collection "$RUN_ROOT/collection-result.json" \
  --output "$RUN_ROOT/dataset-validation.json"
"$AEGIS_PYTHON" scripts/evaluate_distal_factorized_execution_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --complete-dataset "$SOURCE_ROOT/dataset.json" \
  --complete-collection "$SOURCE_ROOT/collection-result.json" \
  --trajectory-metadata "$RUN_ROOT/trajectory-metadata.json" \
  --trajectory-array "$RUN_ROOT/trajectory-dataset.npz" \
  --trajectory-collection "$RUN_ROOT/collection-result.json" \
  --trajectory-validation "$RUN_ROOT/dataset-validation.json" \
  --matched-direct-result "$MATCHED_ROOT/result.json" \
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" \
  --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED" \
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" \
  --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --direct-model "$RUN_ROOT/direct-margin-model.npz" \
  --factorized-model "$RUN_ROOT/factorized-execution-model.npz" \
  --predictions "$RUN_ROOT/predictions.npz" --output "$RUN_ROOT/result.json"
"$AEGIS_PYTHON" scripts/validate_distal_factorized_execution_moka10.py \
  --repo-root "$REMOTE_REPO" --config "$CONFIG" \
  --complete-dataset "$SOURCE_ROOT/dataset.json" \
  --complete-collection "$SOURCE_ROOT/collection-result.json" \
  --trajectory-metadata "$RUN_ROOT/trajectory-metadata.json" \
  --trajectory-array "$RUN_ROOT/trajectory-dataset.npz" \
  --trajectory-collection "$RUN_ROOT/collection-result.json" \
  --direct-model "$RUN_ROOT/direct-margin-model.npz" \
  --factorized-model "$RUN_ROOT/factorized-execution-model.npz" \
  --predictions "$RUN_ROOT/predictions.npz" \
  --population-manifest "$POPULATION" --selected-manifest "$SELECTED" \
  --same-task-manifest "$SAME_TASK" --targeted-manifest "$TARGETED" \
  --archived-root "$ARCHIVED_ROOT" --geometry-config "$GEOMETRY" \
  --exact-box-config "$EXACT_BOX" --expected-commit "$EXPECTED_GIT_COMMIT" \
  --result "$RUN_ROOT/result.json" --output "$RUN_ROOT/validation.json"
