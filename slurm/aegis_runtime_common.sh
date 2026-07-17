#!/usr/bin/env bash

# Shared allocation-side functions.  Callers enable `set -euo pipefail`.

AEGIS_SERVER_PID=
AEGIS_FAILURE_STAGE=allocation_contract

aegis_die() {
  echo "AEGIS runtime error: $*" >&2
  return 2
}

aegis_require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || aegis_die "required environment variable is unset: $name"
}

aegis_contract_value() {
  local key=$1
  awk -F $'\t' -v wanted="$key" '
    $1 == wanted {
      if (seen++) exit 3
      value=$2
    }
    END {
      if (seen != 1) exit 4
      print value
    }
  ' "$RUN_CONTRACT"
}

aegis_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

aegis_assert_allocation() {
  local expected_cpus=$1
  local expected_mem_mib=$2
  aegis_require_env SLURM_JOB_ID
  aegis_require_env SLURM_ARRAY_JOB_ID
  aegis_require_env SLURM_ARRAY_TASK_ID
  aegis_require_env SLURMD_NODENAME
  [[ "$SLURMD_NODENAME" != worker-3 ]] || aegis_die "worker-3 is excluded"
  case "$SLURMD_NODENAME" in
    login*|login-restricted*) aegis_die "compute cannot run on a login node" ;;
  esac
  [[ "${SLURM_CPUS_PER_TASK:-}" == "$expected_cpus" ]] || {
    aegis_die "expected $expected_cpus CPUs, found ${SLURM_CPUS_PER_TASK:-unset}"
  }
  [[ "${SLURM_MEM_PER_NODE:-}" == "$expected_mem_mib" ]] || {
    aegis_die "expected ${expected_mem_mib} MiB, found ${SLURM_MEM_PER_NODE:-unset}"
  }
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != NoDevFiles ]] || {
    aegis_die "one allocation-visible GPU is required"
  }
  case "$CUDA_VISIBLE_DEVICES" in
    *,*) aegis_die "exactly one allocation-visible GPU is required" ;;
  esac
}

aegis_assert_array_contract() {
  local expected_count=$1
  local expected_max=$2
  local expected_throttle=$3
  [[ "${SLURM_ARRAY_TASK_MIN:-}" == 0 ]] || {
    aegis_die "array minimum changed: ${SLURM_ARRAY_TASK_MIN:-unset}"
  }
  [[ "${SLURM_ARRAY_TASK_MAX:-}" == "$expected_max" ]] || {
    aegis_die "array maximum changed: ${SLURM_ARRAY_TASK_MAX:-unset}"
  }
  [[ "${SLURM_ARRAY_TASK_STEP:-}" == 1 ]] || {
    aegis_die "array step changed: ${SLURM_ARRAY_TASK_STEP:-unset}"
  }
  [[ "${SLURM_ARRAY_TASK_COUNT:-}" == "$expected_count" ]] || {
    aegis_die "array task count changed: ${SLURM_ARRAY_TASK_COUNT:-unset}"
  }
  (( expected_count <= 100 )) || aegis_die "array exceeds 100 submitted tasks"
  local job_record
  job_record=$(scontrol show job "$SLURM_ARRAY_JOB_ID" -o)
  case " $job_record " in
    *" ArrayTaskThrottle=$expected_throttle "*) ;;
    *) aegis_die "array throttle is not the preregistered %$expected_throttle" ;;
  esac
}

aegis_validate_identity() {
  local expected_stage=$1
  aegis_require_env RUN_ID
  aegis_require_env EXPECTED_GIT_COMMIT
  case "$RUN_ID" in
    ""|*[!A-Za-z0-9._-]*) aegis_die "unsafe RUN_ID: $RUN_ID" ;;
  esac
  case "$EXPECTED_GIT_COMMIT" in
    *[!0-9a-f]*|"") aegis_die "EXPECTED_GIT_COMMIT is not lowercase hexadecimal" ;;
  esac
  [[ ${#EXPECTED_GIT_COMMIT} -eq 40 ]] || aegis_die "EXPECTED_GIT_COMMIT length changed"

  REMOTE_REPO=${REMOTE_REPO:-/home/quanth/working_space/vlsa-aegis-table-repro}
  EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/mnt/data/quanth/experiments/vlsa-aegis-table1}
  CONFIG_PATH=${CONFIG_PATH:-$REMOTE_REPO/configs/vlsa_table1_translational.json}
  MANIFEST_PATH=${MANIFEST_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.jsonl}
  MANIFEST_RECEIPT_PATH=${MANIFEST_RECEIPT_PATH:-$REMOTE_REPO/manifests/vlsa_table1_population.receipt.json}
  LABEL_MANIFEST_PATH=${LABEL_MANIFEST_PATH:-}
  PI05_CHECKPOINT=${PI05_CHECKPOINT:-/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero}
  DINO_CONFIG=${DINO_CONFIG:-/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/groundingdino/config/GroundingDINO_SwinT_OGC.py}
  DINO_CHECKPOINT=${DINO_CHECKPOINT:-/mnt/data/quanth/cache/aegis/groundingdino/groundingdino_swint_ogc.pth}
  GROUNDINGDINO_DEVICE=${GROUNDINGDINO_DEVICE:-}
  OPENPI_PYTHON=${OPENPI_PYTHON:-/mnt/data/quanth/venvs/openpi/bin/python}
  AEGIS_PYTHON=${AEGIS_PYTHON:-/mnt/data/quanth/venvs/safety_vla/main/bin/python}

  case "$EXPERIMENT_ROOT" in
    /mnt/data/quanth/experiments/*) ;;
    *) aegis_die "EXPERIMENT_ROOT escaped /mnt/data/quanth/experiments" ;;
  esac
  RUN_ROOT=$EXPERIMENT_ROOT/$RUN_ID
  RUN_CONTRACT=$RUN_ROOT/run-contract.tsv
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || aegis_die "reserved immutable run root is missing"
  [[ -f "$RUN_CONTRACT" && ! -L "$RUN_CONTRACT" ]] || aegis_die "run contract is missing or symlinked"
  [[ "$(aegis_contract_value schema_version)" == vlsa_table1_run_contract.v1 ]]
  [[ "$(aegis_contract_value run_id)" == "$RUN_ID" ]]
  [[ "$(aegis_contract_value run_stage)" == "$expected_stage" ]]
  CONTRACT_CASE_ORDINAL=$(aegis_contract_value case_ordinal)
  if [[ "$expected_stage" == capture-canary || "$expected_stage" == paired-canary ]]; then
    [[ "$CONTRACT_CASE_ORDINAL" =~ ^[0-9]+$ ]] || {
      aegis_die "run contract has an invalid canary case ordinal"
    }
    (( CONTRACT_CASE_ORDINAL >= 0 && CONTRACT_CASE_ORDINAL < 1600 )) || {
      aegis_die "run contract canary case ordinal is outside [0, 1599]"
    }
    if [[ -n "${CASE_ORDINAL:-}" && "$CASE_ORDINAL" != "$CONTRACT_CASE_ORDINAL" ]]; then
      aegis_die "CASE_ORDINAL differs from the reserved canary case"
    fi
  else
    [[ "$CONTRACT_CASE_ORDINAL" == all ]] || {
      aegis_die "population run contract must bind the full population"
    }
  fi
  [[ "$(aegis_contract_value git_commit)" == "$EXPECTED_GIT_COMMIT" ]]
  [[ "$(aegis_contract_value config_sha256)" == "$(aegis_sha256 "$CONFIG_PATH")" ]]
  [[ "$(aegis_contract_value manifest_sha256)" == "$(aegis_sha256 "$MANIFEST_PATH")" ]]
  [[ "$(aegis_contract_value manifest_receipt_sha256)" == "$(aegis_sha256 "$MANIFEST_RECEIPT_PATH")" ]]

  local expected_label_sha
  expected_label_sha=$(aegis_contract_value label_manifest_sha256)
  if [[ "$expected_stage" == paired-canary || "$expected_stage" == population ]]; then
    [[ -n "$LABEL_MANIFEST_PATH" && -f "$LABEL_MANIFEST_PATH" && ! -L "$LABEL_MANIFEST_PATH" ]] || {
      aegis_die "evaluation label manifest is missing or symlinked"
    }
    [[ "$expected_label_sha" == "$(aegis_sha256 "$LABEL_MANIFEST_PATH")" ]] || {
      aegis_die "frozen label manifest differs from the run contract"
    }
    [[ "$GROUNDINGDINO_DEVICE" == "$(aegis_contract_value groundingdino_device)" ]] || {
      aegis_die "GroundingDINO device differs from the run contract"
    }
    local contract_pi05_tree
    contract_pi05_tree=$(aegis_contract_value pi05_tree_sha256)
    if [[ "$contract_pi05_tree" == structural-and-metadata-only ]]; then
      [[ -z "${EXPECTED_PI05_TREE_SHA256:-}" ]] || {
        aegis_die "unexpected full pi0.5 hash after structural-only reservation"
      }
    else
      [[ "${EXPECTED_PI05_TREE_SHA256:-}" == "$contract_pi05_tree" ]] || {
        aegis_die "pi0.5 full tree hash differs from the run contract"
      }
    fi
  else
    [[ "$expected_label_sha" == none ]] || {
      aegis_die "capture run unexpectedly binds an outcome-stage label manifest"
    }
    [[ "$(aegis_contract_value groundingdino_device)" == none ]] || {
      aegis_die "capture run unexpectedly binds a GroundingDINO device"
    }
    [[ "$(aegis_contract_value pi05_tree_sha256)" == none ]] || {
      aegis_die "capture run unexpectedly binds a pi0.5 checkpoint"
    }
  fi

  local observed_commit
  observed_commit=$(git -C "$REMOTE_REPO" rev-parse HEAD)
  [[ "$observed_commit" == "$EXPECTED_GIT_COMMIT" ]] || aegis_die "source commit changed"
  [[ -z "$(git -C "$REMOTE_REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
    aegis_die "source tree is dirty"
  }
  [[ -x "$OPENPI_PYTHON" ]] || aegis_die "OpenPI interpreter is missing"
  [[ -x "$AEGIS_PYTHON" ]] || aegis_die "AEGIS interpreter is missing"
}

aegis_create_task_root() {
  local logical_task=$1
  TASK_ROOT=$RUN_ROOT/tasks/task-$logical_task
  mkdir -p "$RUN_ROOT/tasks"
  mkdir "$TASK_ROOT" || aegis_die "immutable task root already exists: $TASK_ROOT"
  SERVER_LOG=$TASK_ROOT/pi05-server.log
  PREFLIGHT_RECEIPT=$TASK_ROOT/allocation-preflight.json
  RUNTIME_FAILURE=$TASK_ROOT/runtime-failure.json
}

aegis_prepare_environment() {
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONUNBUFFERED=1
  export OPENPI_DATA_HOME=/mnt/data/quanth/cache/openpi
  export HF_HOME=/mnt/data/quanth/cache/huggingface
  export XDG_CACHE_HOME=/mnt/data/quanth/cache/xdg
  export PIP_CACHE_DIR=/mnt/data/quanth/cache/pip
  export UV_CACHE_DIR=/mnt/data/quanth/cache/uv
  export WANDB_MODE=disabled
  export TOKENIZERS_PARALLELISM=false
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export PYTHONPATH=$REMOTE_REPO/openpi/src:$REMOTE_REPO/openpi/packages/openpi-client/src:$REMOTE_REPO/safelibero:$REMOTE_REPO/main

  local safelibero_root=$REMOTE_REPO/safelibero/libero/libero
  [[ -d "$safelibero_root/bddl_files" && -d "$safelibero_root/init_files" ]] || {
    aegis_die "SafeLIBERO runtime tree is incomplete"
  }
  export LIBERO_CONFIG_PATH=$TASK_ROOT/libero-config
  mkdir "$LIBERO_CONFIG_PATH"
  {
    printf 'benchmark_root: %s\n' "$safelibero_root"
    printf 'bddl_files: %s\n' "$safelibero_root/bddl_files"
    printf 'init_states: %s\n' "$safelibero_root/init_files"
    printf 'datasets: %s\n' "$REMOTE_REPO/safelibero/libero/datasets"
    printf 'assets: %s\n' "$safelibero_root/assets"
  } >"$LIBERO_CONFIG_PATH/config.yaml"
}

aegis_run_preflight() {
  local profile=$1
  local args=(
    "$REMOTE_REPO/scripts/validate_aegis_assets.py"
    --profile "$profile"
    --repo-root "$REMOTE_REPO"
    --expected-commit "$EXPECTED_GIT_COMMIT"
    --config "$CONFIG_PATH"
    --manifest "$MANIFEST_PATH"
    --manifest-receipt "$MANIFEST_RECEIPT_PATH"
    --output "$PREFLIGHT_RECEIPT"
  )
  if [[ "$profile" == evaluation ]]; then
    args+=(
      --pi05-checkpoint "$PI05_CHECKPOINT"
      --dino-config "$DINO_CONFIG"
      --dino-checkpoint "$DINO_CHECKPOINT"
      --label-manifest "$LABEL_MANIFEST_PATH"
      --expected-label-manifest-sha256 "$(aegis_contract_value label_manifest_sha256)"
    )
    if [[ -n "${EXPECTED_PI05_TREE_SHA256:-}" ]]; then
      args+=(--expected-pi05-tree-sha256 "$EXPECTED_PI05_TREE_SHA256")
    fi
    if [[ "${EXPECTED_STAGE:-}" == population ]]; then
      args+=(--require-complete-label-population)
    else
      args+=(--required-case-ordinal "${CASE_ORDINAL:?}")
    fi
  fi
  "$AEGIS_PYTHON" "${args[@]}"
}

aegis_start_pi05_server() {
  local job_number=$((10#$SLURM_ARRAY_JOB_ID))
  local task_number=$((10#$SLURM_ARRAY_TASK_ID))
  POLICY_PORT=$((20000 + (job_number + task_number) % 30000))
  (
    cd "$REMOTE_REPO"
    exec "$OPENPI_PYTHON" scripts/serve_policy.py \
      --port "$POLICY_PORT" \
      policy:checkpoint \
      --policy.config pi05_libero \
      --policy.dir "$PI05_CHECKPOINT"
  ) >"$SERVER_LOG" 2>&1 &
  AEGIS_SERVER_PID=$!

  local attempt
  for attempt in $(seq 1 600); do
    if ! kill -0 "$AEGIS_SERVER_PID" 2>/dev/null; then
      tail -n 200 "$SERVER_LOG" >&2 || true
      aegis_die "JAX pi0.5 server exited before healthz"
      return
    fi
    if curl --fail --silent --show-error \
      "http://127.0.0.1:$POLICY_PORT/healthz" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  tail -n 200 "$SERVER_LOG" >&2 || true
  aegis_die "JAX pi0.5 server did not become healthy"
}

aegis_stop_pi05_server() {
  if [[ -n "${AEGIS_SERVER_PID:-}" ]] && kill -0 "$AEGIS_SERVER_PID" 2>/dev/null; then
    kill "$AEGIS_SERVER_PID" 2>/dev/null || true
    wait "$AEGIS_SERVER_PID" 2>/dev/null || true
  fi
  AEGIS_SERVER_PID=
}

aegis_write_runtime_failure() {
  local exit_code=$1
  [[ -n "${RUNTIME_FAILURE:-}" && -d "${TASK_ROOT:-}" ]] || return 0
  [[ ! -e "$RUNTIME_FAILURE" ]] || return 0
  local temporary
  temporary=$(mktemp "$TASK_ROOT/.runtime-failure.XXXXXX")
  {
    printf '{\n'
    printf '  "schema_version": "vlsa_table1_runtime_failure.v1",\n'
    printf '  "status": "apparatus_failure",\n'
    printf '  "scientific_result": false,\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "run_stage": "%s",\n' "$(aegis_contract_value run_stage)"
    printf '  "failure_stage": "%s",\n' "$AEGIS_FAILURE_STAGE"
    printf '  "exit_code": %d,\n' "$exit_code"
    printf '  "slurm_job_id": "%s",\n' "$SLURM_JOB_ID"
    printf '  "slurm_array_job_id": "%s",\n' "$SLURM_ARRAY_JOB_ID"
    printf '  "slurm_array_task_id": "%s",\n' "$SLURM_ARRAY_TASK_ID"
    printf '  "host": "%s",\n' "$SLURMD_NODENAME"
    printf '  "timestamp_utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '}\n'
  } >"$temporary"
  mv "$temporary" "$RUNTIME_FAILURE"
}

aegis_cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  aegis_stop_pi05_server
  if [[ $exit_code -ne 0 ]]; then
    aegis_write_runtime_failure "$exit_code" || true
  fi
  return "$exit_code"
}
