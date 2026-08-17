#!/usr/bin/env python3
"""Audit frozen compact/33D selector rules without simulation or retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def _prediction_map(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[float]],
) -> dict[tuple[str, str], list[float]]:
    _require(len(samples) == len(predictions),
             "compact selector specialist sample count differs")
    output = {}
    for sample, prediction in zip(samples, predictions):
        key = (str(sample["state_id"]), str(sample["candidate_name"]))
        _require(key not in output, "compact selector specialist identity repeats")
        output[key] = [float(value) for value in prediction]
    return output


def _validation_choice(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze one next-test rule using validation metrics, never test metrics."""
    ordered = sorted(arms.values(), key=lambda row: (
        int(row["known_all_physical_unsafe_selection_count"]),
        int(row["unknown_selection_count"]),
        -int(row["known_all_physical_safe_selection_count"]),
        int(row["abstention_count"]),
        float("inf") if row["mean_selected_correction"] is None
        else float(row["mean_selected_correction"]),
        str(row["arm"]),
    ))
    winner = ordered[0]
    return {
        "rule": (
            "minimize known physical unsafe, then UNKNOWN, maximize known safe, "
            "minimize abstention, intervention, and finally lexical arm name"
        ),
        "selected_arm": str(winner["arm"]),
        "selected_validation_summary": {
            key: winner[key] for key in (
                "selected_state_count", "abstention_count",
                "known_all_physical_safe_selection_count",
                "known_all_physical_unsafe_selection_count",
                "unknown_selection_count", "mean_selected_correction",
            )
        },
        "test_metrics_used": False,
    }


def run(*, repo_root: Path, config_path: Path,
        expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        candidate_records, load_config as load_compact_config,
        payload_sha256 as compact_payload_sha256,
    )
    from main.multilink_ellipsoid.compact_selector_ablation import (
        RESULT_SCHEMA, enrich_records, evaluate_arm, exact_float_lists_close,
        file_sha256, load_config, payload_sha256, validation_margins,
    )
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.whole_body_action_selection_audit import (
        predict_serialized_mlp,
    )
    from main.multilink_ellipsoid.whole_body_q_only_prediction import (
        combined_training_config, load_binding, load_protocol,
    )
    from scripts.train_distal_compact_safety_coordinate_q import (
        _load_all_candidate_samples,
    )
    from scripts.train_distal_whole_body_q_only_diagnostic import load_samples

    config = load_config(config_path)
    compact_config_path = repo_root / config["compact_config"]["path"]
    _require(
        file_sha256(compact_config_path)
        == config["compact_config"]["file_sha256"],
        "compact selector compact config differs",
    )
    compact_config = load_compact_config(compact_config_path)
    compact_result_path = Path(config["compact_result"]["path"])
    specialist_result_path = Path(config["L5_specialist_result"]["path"])
    _require(
        file_sha256(compact_result_path)
        == config["compact_result"]["file_sha256"]
        and file_sha256(specialist_result_path)
        == config["L5_specialist_result"]["file_sha256"],
        "compact selector frozen result file differs",
    )
    compact_result = _load(compact_result_path)
    specialist_result = _load(specialist_result_path)
    compact_model = compact_result["compact_shared_7D"]["model"]
    specialist_arm = specialist_result["arms"][
        config["L5_specialist_result"]["arm"]
    ]
    _require(
        compact_result.get("result_payload_sha256")
        == config["compact_result"]["payload_sha256"]
        == compact_payload_sha256(compact_result, "result_payload_sha256")
        and compact_model["model_sha256"]
        == config["compact_result"]["model_sha256"]
        and specialist_result.get("result_payload_sha256")
        == config["L5_specialist_result"]["payload_sha256"]
        and specialist_arm["model"]["model_sha256"]
        == config["L5_specialist_result"]["model_sha256"],
        "compact selector frozen payload/model differs",
    )

    protocol_path = repo_root / compact_config["base_protocol"]["path"]
    protocol = load_protocol(protocol_path)
    binding = load_binding(Path(compact_config["binding"]["path"]), protocol)
    training_config = combined_training_config(protocol, binding)
    established, established_source = load_samples(
        repo_root=repo_root, config=training_config,
    )
    samples, source = _load_all_candidate_samples(
        repo_root=repo_root, training_config=training_config,
        compact_config=compact_config,
    )

    tolerance = float(config["evaluation"]["serialized_prediction_tolerance"])
    compact_replay = {}
    specialist_replay = {}
    records = {}
    for split in ("validation", "test"):
        compact_frozen = compact_result["compact_shared_7D"][
            "all_candidate_predictions"
        ][split]
        compact_fresh = predict_serialized_mlp(
            [sample["feature"] for sample in samples[split]],
            compact_model["state_payload"],
        )
        compact_equal, compact_error = exact_float_lists_close(
            compact_fresh, compact_frozen, tolerance=tolerance,
        )
        _require(
            compact_equal,
            "compact selector compact replay differs: "
            f"maximum_absolute_error={compact_error:.17g}, "
            f"tolerance={tolerance:.17g}",
        )
        compact_replay[split] = {
            "sample_count": len(compact_fresh),
            "maximum_absolute_error": compact_error,
            "within_tolerance": compact_equal,
        }

        known_specialist_samples = established[split]["l5"]
        known_specialist_fresh = predict_serialized_mlp(
            [sample["feature_33d"] for sample in known_specialist_samples],
            specialist_arm["model"]["state_payload"],
        )
        specialist_equal, specialist_error = exact_float_lists_close(
            known_specialist_fresh, specialist_arm["predictions"][split],
            tolerance=tolerance,
        )
        _require(
            specialist_equal,
            "compact selector 33D known replay differs: "
            f"maximum_absolute_error={specialist_error:.17g}, "
            f"tolerance={tolerance:.17g}",
        )
        unique_candidates = [
            sample for sample in samples[split]
            if int(sample["row_index"]) == 0
        ]
        all_specialist = predict_serialized_mlp(
            [sample["feature_33d"] for sample in unique_candidates],
            specialist_arm["model"]["state_payload"],
        )
        specialist_replay[split] = {
            "known_candidate_count": len(known_specialist_samples),
            "all_candidate_count": len(unique_candidates),
            "known_maximum_absolute_error": specialist_error,
            "known_within_tolerance": specialist_equal,
        }
        compact_records = candidate_records(
            samples[split], compact_frozen,
            primary_rows=compact_config["primary_inference_rows"],
            physical_rows=compact_config["all_physical_audit_rows"],
        )
        records[split] = enrich_records(
            compact_records,
            _prediction_map(unique_candidates, all_specialist),
        )

    # Freeze all parameters from validation before any test arm is evaluated.
    compact_margins = validation_margins(
        records["validation"], prediction="compact",
    )
    hybrid_margins = validation_margins(
        records["validation"], prediction="compact_33D_L5_hybrid",
    )
    zeros = {group: 0.0 for group in compact_margins}
    l5_only = dict(zeros)
    l5_only["L5"] = compact_margins["L5"]
    hybrid_l5_only = dict(zeros)
    hybrid_l5_only["L5"] = hybrid_margins["L5"]
    frozen_parameters = {
        "compact_all_group_margin": compact_margins,
        "compact_L5_margin": l5_only,
        "hybrid_L5_margin": hybrid_l5_only,
        "fit_split": "validation",
        "known_candidate_labels_only": True,
        "UNKNOWN_censored": True,
        "test_metrics_used": False,
    }
    frozen_parameters["payload_sha256"] = payload_sha256(
        frozen_parameters, "payload_sha256",
    )

    arm_specs = {
        "compact_minimum_risk": ("compact", zeros, "minimum_risk"),
        "compact_zero_margin_safe_or_abstain": (
            "compact", zeros, "safe_or_abstain",
        ),
        "compact_all_group_margin_safe_or_abstain": (
            "compact", compact_margins, "safe_or_abstain",
        ),
        "compact_L5_margin_safe_or_abstain": (
            "compact", l5_only, "safe_or_abstain",
        ),
        "compact_33D_L5_hybrid_safe_or_abstain": (
            "compact_33D_L5_hybrid", hybrid_l5_only, "safe_or_abstain",
        ),
    }
    arms = {split: {} for split in config["evaluation"]["splits"]}
    for name in config["arms"]:
        prediction, margins, selection = arm_specs[name]
        arms["validation"][name] = evaluate_arm(
            records["validation"], name=name, prediction=prediction,
            margins=margins, selection=selection,
        )
    validation_choice = _validation_choice(arms["validation"])
    # Test is evaluated once after both the margins and arm choice are frozen.
    for name in config["arms"]:
        prediction, margins, selection = arm_specs[name]
        arms["test"][name] = evaluate_arm(
            records["test"], name=name, prediction=prediction,
            margins=margins, selection=selection,
        )

    focus = {}
    for state_id in config["evaluation"]["report_states"]:
        focus[state_id] = {
            name: next((
                row for row in arms["test"][name]["states"]
                if row["state_id"] == state_id
            ), None)
            for name in config["arms"]
        }
    chosen_name = validation_choice["selected_arm"]
    chosen_test = arms["test"][chosen_name]
    current_test = arms["test"]["compact_minimum_risk"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_posthoc_inference_only_diagnostic",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "source_artifacts": source,
        "established_source_artifacts": established_source,
        "frozen_models": {
            "compact_7D": config["compact_result"],
            "direct_L5_OSC_33D": config["L5_specialist_result"],
        },
        "serialized_prediction_replay": {
            "compact_7D": compact_replay,
            "direct_L5_OSC_33D": specialist_replay,
            "tolerance": tolerance,
        },
        "frozen_validation_parameters": frozen_parameters,
        "arms": arms,
        "validation_only_arm_choice": validation_choice,
        "chosen_arm_opened_test_diagnostic": chosen_test,
        "focus_states": focus,
        "diagnostic_comparison": {
            "current_compact_test_safe": current_test[
                "known_all_physical_safe_selection_count"
            ],
            "validation_chosen_test_safe": chosen_test[
                "known_all_physical_safe_selection_count"
            ],
            "validation_chosen_test_unsafe": chosen_test[
                "known_all_physical_unsafe_selection_count"
            ],
            "validation_chosen_test_abstention": chosen_test[
                "abstention_count"
            ],
        },
        "new_simulator_rollout_count": 0,
        "retraining_performed": False,
        "test_used_for_margin_or_arm_choice": False,
        "test_is_opened_posthoc_diagnostic": True,
        "prospective_paper_test_authorized": False,
        "correction_safety_authorized": False,
        "QP_authorized": False,
        "denoising_authorized": False,
        "CBF_claim_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "margins": result["frozen_validation_parameters"],
        "validation_choice": result["validation_only_arm_choice"],
        "comparison": result["diagnostic_comparison"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
