#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

python3 -m json.tool feature_list.json >/dev/null

python_files=(
  openpi/src/openpi/policies/policy.py
  openpi/src/openpi/serving/websocket_policy_server.py
)

for optional_file in \
  main/evaluate_safelibero_aegis.py \
  main/capture_safelibero_labels.py \
  analysis/aggregate_safelibero_aegis.py \
  analysis/build_aegis_failure_report.py \
  analysis/build_safelibero_video_gallery.py \
  analysis/validate_aegis_failure_diagnostics.py \
  manifests/build_vlsa_table1_population.py \
  scripts/build_poisson_link56_manifest.py \
  scripts/run_poisson_active_canary.py \
  scripts/run_poisson_fast_feasibility.py \
  scripts/run_poisson_numeric_validation.py \
  scripts/run_poisson_one_step_counterfactual.py \
  scripts/run_poisson_shadow_identification.py \
  scripts/run_poisson_shadow_parity.py \
  scripts/validate_aegis_assets.py \
  scripts/validate_aegis_run_artifacts.py \
  scripts/validate_poisson_one_step_counterfactual_artifact.py \
  scripts/validate_poisson_shadow_identification_artifact.py \
  scripts/validate_poisson_run_artifacts.py \
  scripts/validate_poisson_runtime_prerequisites.py
do
  if [[ -f "${optional_file}" ]]; then
    python_files+=("${optional_file}")
  fi
done

for optional_file in main/poisson_fullbody/*.py; do
  if [[ -f "${optional_file}" ]]; then
    python_files+=("${optional_file}")
  fi
done

python3 -m py_compile "${python_files[@]}"

for slurm_file in slurm/poisson_*.sbatch; do
  if [[ -f "${slurm_file}" ]]; then
    bash -n "${slurm_file}"
  fi
done

python3 -m unittest discover -s tests -p 'test_*.py'

active_count="$(
  python3 -c 'import json; d=json.load(open("feature_list.json")); print(sum(x["status"] == "active" for x in d["features"]))'
)"
if [[ "${active_count}" != "1" ]]; then
  echo "expected exactly one active feature, found ${active_count}" >&2
  exit 1
fi

echo "Local structural gate passed."
