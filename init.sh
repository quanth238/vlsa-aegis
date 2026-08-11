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
  scripts/validate_aegis_assets.py \
  scripts/validate_aegis_run_artifacts.py \
  scripts/evaluate_pi05_droid_joint_velocity_pair.py \
  scripts/validate_pi05_droid_joint_velocity_pair.py \
  scripts/replay_aegis_cartesian_joint_velocity_bridge_pair.py \
  scripts/validate_aegis_cartesian_joint_velocity_bridge_pair.py \
  scripts/evaluate_distal_two_step_oracle_affine_safe_set.py \
  scripts/validate_distal_two_step_oracle_affine_safe_set.py \
  scripts/collect_distal_affine_coefficient_moka10.py \
  scripts/validate_distal_affine_coefficient_dataset.py \
  scripts/analyze_distal_affine_coefficient_targets.py \
  scripts/train_distal_affine_coefficient_moka10.py \
  scripts/evaluate_distal_affine_coefficient_moka10.py \
  scripts/validate_distal_affine_coefficient_moka10.py \
  scripts/evaluate_distal_affine_oracle_comparison_moka10.py \
  scripts/validate_distal_affine_oracle_comparison_moka10.py \
  scripts/evaluate_distal_ridge_huber_oracle_moka10.py \
  scripts/validate_distal_ridge_huber_oracle_moka10.py \
  scripts/evaluate_distal_multi_region_affine_oracle_moka10.py \
  scripts/validate_distal_multi_region_affine_oracle_moka10.py \
  scripts/evaluate_distal_multi_region_decision_stability_moka10.py \
  scripts/validate_distal_multi_region_decision_stability_moka10.py \
  scripts/evaluate_distal_region_aware_mlp_moka10.py \
  scripts/validate_distal_region_aware_mlp_moka10.py \
  scripts/evaluate_distal_region_aware_mlp_closed_loop_e05.py \
  scripts/validate_distal_region_aware_mlp_closed_loop_e05.py \
  scripts/evaluate_distal_state_support_smoothness_moka10.py \
  scripts/validate_distal_state_support_smoothness_moka10.py \
  scripts/evaluate_distal_targeted_boundary_expansion_moka10.py \
  scripts/validate_distal_targeted_boundary_expansion_moka10.py \
  scripts/evaluate_distal_supported_region_aware_mlp_moka10.py \
  scripts/validate_distal_supported_region_aware_mlp_moka10.py \
  scripts/evaluate_distal_local_residual_bound_moka10.py \
  scripts/validate_distal_local_residual_bound_moka10.py \
  scripts/collect_distal_complete_osc_margin_moka10.py \
  scripts/evaluate_distal_complete_osc_margin_moka10.py \
  scripts/validate_distal_complete_osc_margin_moka10.py \
  scripts/evaluate_distal_matched_input_ablation_moka10.py \
  scripts/validate_distal_matched_input_ablation_moka10.py \
  scripts/evaluate_distal_omitted_variable_audit_moka10.py \
  scripts/validate_distal_omitted_variable_audit_moka10.py \
  scripts/collect_distal_factorized_execution_moka10.py \
  scripts/validate_distal_factorized_execution_dataset_moka10.py \
  scripts/evaluate_distal_factorized_execution_moka10.py \
  scripts/validate_distal_factorized_execution_moka10.py \
  scripts/evaluate_distal_proxy_contact_boundary_audit_moka10.py \
  scripts/validate_distal_proxy_contact_boundary_audit_moka10.py \
  scripts/evaluate_distal_initial_contact_audit_moka10.py \
  scripts/validate_distal_initial_contact_audit_moka10.py \
  scripts/audit_distal_factorized_nominal_safety_formula_moka10.py \
  scripts/validate_distal_factorized_nominal_safety_formula_moka10.py
do
  if [[ -f "${optional_file}" ]]; then
    python_files+=("${optional_file}")
  fi
done

python3 -m py_compile "${python_files[@]}"
python3 -m unittest discover -s tests -p 'test_*.py'

active_count="$(
  python3 -c 'import json; d=json.load(open("feature_list.json")); print(sum(x["status"] == "active" for x in d["features"]))'
)"
if [[ "${active_count}" != "1" ]]; then
  echo "expected exactly one active feature, found ${active_count}" >&2
  exit 1
fi

echo "Local structural gate passed."
