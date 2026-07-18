#!/usr/bin/env python3
"""Validate action invariance and opt-in AEGIS failure diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
sys.path.insert(0, str(MAIN))
try:
    import aegis_failure_diagnostics as diagnostics
finally:
    sys.path.remove(str(MAIN))


class DiagnosticValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticValidationError(
            f"cannot read JSON {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise DiagnosticValidationError(f"{path}: root must be an object")
    return value


def recompute_action_ledger(result: Mapping[str, Any]) -> dict[str, Any]:
    actions = result.get("actions")
    queries = result.get("policy_queries")
    if not isinstance(actions, list) or not isinstance(queries, list):
        raise DiagnosticValidationError(
            "result lacks action/policy-query ledgers"
        )
    return diagnostics.action_invariance_ledger(
        actions=actions,
        policy_queries=queries,
    )


def validate_embedded_action_ledger(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    embedded = result.get("action_invariance_ledger")
    if not isinstance(embedded, Mapping):
        raise DiagnosticValidationError(
            "result lacks embedded action-invariance ledger"
        )
    recomputed = recompute_action_ledger(result)
    differing = diagnostics.compare_action_ledgers(embedded, recomputed)
    if differing:
        raise DiagnosticValidationError(
            f"embedded action ledger is stale/tampered: {differing}"
        )
    return recomputed


def validate_canary_reference(
    result: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    if reference.get("schema_version") != diagnostics.REFERENCE_SCHEMA:
        raise DiagnosticValidationError("unexpected reference schema")
    if result.get("case_id") != reference.get("case_id"):
        raise DiagnosticValidationError("reference case ID mismatch")
    arm = str(result.get("arm", ""))
    expected = reference.get("arms", {}).get(arm)
    if not isinstance(expected, Mapping):
        raise DiagnosticValidationError(f"reference has no arm {arm!r}")
    ledger = recompute_action_ledger(result)
    checked = [
        key
        for key in expected
        if key.endswith("_sha256")
        and key != "source_result_file_sha256"
    ] + ["action_count", "policy_query_count"]
    mismatches = [
        key for key in checked if ledger.get(key) != expected.get(key)
    ]
    if mismatches:
        raise DiagnosticValidationError(
            f"canary action reference mismatch: {sorted(mismatches)}"
        )
    return ledger


def _pair_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    pairing = result.get("pairing")
    if not isinstance(pairing, Mapping):
        raise DiagnosticValidationError("result lacks pairing record")
    keys = (
        "manifest_row_sha256",
        "initial_state_sha256",
        "initial_observation_sha256",
        "settled_simulator_state_sha256",
        "settled_active_obstacle_position_sha256",
        "policy_noise_schedule_id",
        "policy_noise_schedule_sha256",
        "initial_policy_action_chunk_sha256",
        "semantic_label_record_sha256",
        "semantic_label_settled_agentview_sha256",
        "semantic_obstacle_label",
        "max_steps",
        "model_action_horizon",
        "replan_steps",
    )
    return {key: pairing.get(key) for key in keys}


def validate_action_invariance_pair(
    diagnostics_off: Mapping[str, Any],
    diagnostics_on: Mapping[str, Any],
) -> dict[str, Any]:
    for key in ("case_id", "arm", "mode", "protocol_id"):
        if diagnostics_off.get(key) != diagnostics_on.get(key):
            raise DiagnosticValidationError(
                f"diagnostics on/off {key} mismatch"
            )
    if _pair_binding(diagnostics_off) != _pair_binding(diagnostics_on):
        raise DiagnosticValidationError(
            "diagnostics on/off pair binding changed"
        )
    off_ledger = recompute_action_ledger(diagnostics_off)
    on_ledger = validate_embedded_action_ledger(diagnostics_on)
    differing = diagnostics.compare_action_ledgers(
        off_ledger,
        on_ledger,
        include_diagnostic_only_fields=False,
    )
    if differing:
        raise DiagnosticValidationError(
            f"diagnostics changed action bytes/schedule: {differing}"
        )
    off_flag = diagnostics_off.get("failure_diagnostics", {}).get(
        "enabled", False
    )
    on_flag = diagnostics_on.get("failure_diagnostics", {}).get("enabled")
    if off_flag is not False or on_flag is not True:
        raise DiagnosticValidationError(
            "diagnostics on/off enable flags are invalid"
        )
    return {
        "case_id": diagnostics_on["case_id"],
        "arm": diagnostics_on["arm"],
        "action_ledger": on_ledger,
        "status": "action_invariant",
    }


def _validate_npz_artifact(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    import numpy as np

    try:
        diagnostics.validate_artifact_descriptor(
            descriptor, output_root=output_root
        )
    except ValueError as error:
        raise DiagnosticValidationError(str(error)) from error
    path = output_root / str(descriptor["path"])
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = descriptor.get("keys")
        if sorted(archive.files) != expected_keys:
            raise DiagnosticValidationError(
                "geometry NPZ key inventory changed"
            )
        metadata = descriptor.get("arrays")
        if not isinstance(metadata, Mapping):
            raise DiagnosticValidationError(
                "geometry array metadata is missing"
            )
        for key in archive.files:
            if diagnostics.array_descriptor(archive[key]) != metadata.get(key):
                raise DiagnosticValidationError(
                    f"geometry NPZ array metadata mismatch: {key}"
                )


def _validate_geometry(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
    require_ready: bool,
) -> None:
    if descriptor.get("schema_version") != diagnostics.GEOMETRY_SCHEMA:
        raise DiagnosticValidationError("unexpected geometry schema")
    _validate_npz_artifact(descriptor, output_root=output_root)
    record = descriptor.get("record")
    if not isinstance(record, Mapping):
        raise DiagnosticValidationError("geometry record is missing")
    status = record.get("status")
    if status not in {
        "complete",
        "failure",
        "method_failure_passthrough",
    }:
        raise DiagnosticValidationError(
            f"geometry diagnostics are incomplete: {status}"
        )
    views = record.get("views")
    if not isinstance(views, Mapping) or set(views) != {
        "agentview",
        "backview",
    }:
        raise DiagnosticValidationError(
            "geometry record lacks both released camera views"
        )
    for view_name, view in views.items():
        if not isinstance(view, Mapping):
            raise DiagnosticValidationError(f"{view_name}: invalid view")
        detections = view.get("detections")
        request = view.get("request")
        if not isinstance(detections, Mapping) or not isinstance(
            request, Mapping
        ):
            raise DiagnosticValidationError(
                f"{view_name}: DINO request/detections missing"
            )
        count = detections.get("count")
        selected = detections.get("selected_index")
        if not isinstance(count, int) or selected != (
            0 if count > 0 else None
        ):
            raise DiagnosticValidationError(
                f"{view_name}: released first-box selection changed"
            )
        phrases = detections.get("returned_order_phrases")
        boxes = detections.get("returned_order_boxes_cxcywh")
        logits = detections.get("returned_order_logits")
        if (
            not isinstance(phrases, list)
            or not isinstance(boxes, list)
            or not isinstance(logits, list)
            or len(phrases) != count
            or len(boxes) != count
            or len(logits) != count
        ):
            raise DiagnosticValidationError(
                f"{view_name}: returned-order DINO ledger is inconsistent"
            )
        if count == 0 and view.get("no_detection", {}).get(
            "explicit"
        ) is not True:
            raise DiagnosticValidationError(
                f"{view_name}: no-detection state is implicit"
            )
        crop = detections.get("selected_crop")
        if count > 0 and (
            not isinstance(crop, Mapping)
            or not isinstance(crop.get("rgb_array"), Mapping)
            or not isinstance(crop.get("metric_depth_array"), Mapping)
            or not isinstance(crop.get("released_rgb_depth_crop_xyxy"), list)
        ):
            raise DiagnosticValidationError(
                f"{view_name}: selected crop/image/depth binding is missing"
            )
    if status != "complete":
        failure = record.get("failure")
        if (
            not isinstance(failure, Mapping)
            or not isinstance(failure.get("component"), str)
            or not isinstance(failure.get("type"), str)
        ):
            raise DiagnosticValidationError(
                "geometry method failure lacks typed context"
            )
        if require_ready:
            raise DiagnosticValidationError(
                f"canary geometry did not reach ready: {status}"
            )
        return
    filtering = record.get("filtering")
    if (
        not isinstance(filtering, Mapping)
        or filtering.get("status") != "captured"
        or filtering.get("shadow_matches_released") is not True
        or filtering.get("centroid_trim", {}).get("fraction_kept") != 0.8
        or filtering.get("dbscan", {}).get("eps") != 0.0001
        or filtering.get("dbscan", {}).get("min_points") != 50
    ):
        raise DiagnosticValidationError(
            "released point-filter stages were not exactly reconstructed"
        )
    mvee = record.get("mvee")
    if (
        not isinstance(mvee, Mapping)
        or mvee.get("status") != "captured"
        or not isinstance(mvee.get("solver_calls"), list)
        or not mvee["solver_calls"]
        or "matrix_A" not in mvee
        or "eigenvalues" not in mvee
        or "semiaxes" not in mvee
    ):
        raise DiagnosticValidationError(
            "released ConvexHull/MVEE diagnostics are incomplete"
        )


def _validate_contacts(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
    action_count: int,
) -> None:
    if descriptor.get("schema_version") != diagnostics.CONTACT_SCHEMA:
        raise DiagnosticValidationError("unexpected contact schema")
    try:
        diagnostics.validate_artifact_descriptor(
            descriptor, output_root=output_root
        )
    except ValueError as error:
        raise DiagnosticValidationError(str(error)) from error
    path = output_root / str(descriptor["path"])
    try:
        raw = gzip.decompress(path.read_bytes())
        payload = json.loads(raw)
    except Exception as error:
        raise DiagnosticValidationError(
            f"invalid contact artifact: {error}"
        ) from error
    if (
        diagnostics.sha256_bytes(raw)
        != descriptor.get("uncompressed_payload_sha256")
        or payload.get("schema_version") != diagnostics.CONTACT_SCHEMA
    ):
        raise DiagnosticValidationError("contact payload binding changed")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != action_count + 1:
        raise DiagnosticValidationError(
            "contact artifact must cover settle step -1 and every action"
        )
    if [snapshot.get("step") for snapshot in snapshots] != list(
        range(-1, action_count)
    ):
        raise DiagnosticValidationError("contact snapshot steps changed")
    all_events = []
    robot_events = []
    nonrobot_events = []
    for snapshot in snapshots:
        if snapshot.get("status") != "available":
            raise DiagnosticValidationError(
                f"contact snapshot unavailable at step {snapshot.get('step')}"
            )
        for event in snapshot.get("events", []):
            all_events.append(event)
            obstacle = event.get("obstacle")
            other = event.get("other")
            if (
                not isinstance(obstacle, Mapping)
                or not isinstance(other, Mapping)
                or other.get("classification")
                not in {"robot", "nonrobot"}
                or not isinstance(event.get("distance"), (int, float))
                or not isinstance(event.get("position"), list)
                or len(event["position"]) != 3
                or not isinstance(
                    event.get("normal_obstacle_to_other"), list
                )
                or len(event["normal_obstacle_to_other"]) != 3
            ):
                raise DiagnosticValidationError(
                    "contact event lacks canonical physical context"
                )
            expected_hash = diagnostics.sha256_bytes(
                diagnostics.canonical_json_bytes(
                    {
                        key: value
                        for key, value in event.items()
                        if key != "event_sha256"
                    }
                )
            )
            if event.get("event_sha256") != expected_hash:
                raise DiagnosticValidationError(
                    "contact event hash mismatch"
                )
            if other["classification"] == "robot":
                robot_events.append(event)
            else:
                nonrobot_events.append(event)
    expected_summary = {
        "snapshot_count": len(snapshots),
        "event_count": len(all_events),
        "robot_event_count": len(robot_events),
        "nonrobot_event_count": len(nonrobot_events),
        "steps_with_any_contact": sorted(
            {
                int(snapshot["step"])
                for snapshot in snapshots
                if snapshot.get("events")
            }
        ),
        "steps_with_robot_contact": sorted(
            {int(event["step"]) for event in robot_events}
        ),
        "steps_with_nonrobot_contact": sorted(
            {int(event["step"]) for event in nonrobot_events}
        ),
        "first_robot_contact_step": (
            None
            if not robot_events
            else min(int(event["step"]) for event in robot_events)
        ),
        "first_nonrobot_contact_step": (
            None
            if not nonrobot_events
            else min(int(event["step"]) for event in nonrobot_events)
        ),
    }
    for key, expected in expected_summary.items():
        if descriptor.get(key) != expected:
            raise DiagnosticValidationError(
                f"contact artifact summary mismatch: {key}"
            )


def _validate_qp_contexts(result: Mapping[str, Any]) -> None:
    if result.get("mode") != "aegis":
        return
    for action in result.get("actions", []):
        if action.get("control_path") != "aegis_qp":
            continue
        qp = action.get("qp")
        context = qp.get("context") if isinstance(qp, Mapping) else None
        required = {
            "p1",
            "R1",
            "q1_diag",
            "p2",
            "R2",
            "Q2_diag",
            "z_before",
            "reference",
            "weights_diagonal",
            "cbf",
            "solver_stats",
            "u_solution",
            "solution_lhs",
            "solution_slack",
            "solution_violation",
            "constraint_dual",
            "objective",
            "z_after",
            "executed_action",
            "executed_action_array_sha256",
            "executed_action_canonical_sha256",
        }
        if not isinstance(context, Mapping) or not required.issubset(context):
            raise DiagnosticValidationError(
                f"step {action.get('step')}: incomplete QP context"
            )
        if context.get("status") != "solved":
            raise DiagnosticValidationError(
                f"step {action.get('step')}: QP observer failed"
            )
        if context.get("executed_action") != action.get("executed"):
            raise DiagnosticValidationError(
                f"step {action.get('step')}: executed action binding changed"
            )
        cbf = context.get("cbf")
        if not isinstance(cbf, Mapping) or not {
            "a_v",
            "a_omega",
            "a_u_v",
            "a_u_z",
            "mu_row",
            "h",
            "alpha_gain",
            "constant",
            "reference_lhs",
            "reference_slack",
            "reference_violation",
        }.issubset(cbf):
            raise DiagnosticValidationError(
                f"step {action.get('step')}: CBF reconstruction is incomplete"
            )
        try:
            import numpy as np

            executed_array_hash = diagnostics.array_sha256(
                np.asarray(action["executed"], dtype=float)
            )
        except Exception as error:
            raise DiagnosticValidationError(
                f"step {action.get('step')}: cannot hash executed action"
            ) from error
        executed_canonical_hash = diagnostics.sha256_bytes(
            diagnostics.canonical_json_bytes(action["executed"])
        )
        if (
            context.get("executed_action_array_sha256")
            != executed_array_hash
            or context.get("executed_action_canonical_sha256")
            != executed_canonical_hash
            or context.get("z_after") != qp.get("z_after")
            or context.get("solution_lhs") != qp.get("constraint_lhs")
        ):
            raise DiagnosticValidationError(
                f"step {action.get('step')}: QP/action hash binding changed"
            )
    method_failure = result.get("method_failure")
    if (
        isinstance(method_failure, Mapping)
        and method_failure.get("component") == "aegis_qp"
    ):
        context = method_failure.get("diagnostics")
        if (
            not isinstance(context, Mapping)
            or context.get("status") != "failure"
            or context.get("failure_type")
            not in {
                "cbf_coefficient_exception",
                "qp_construction_exception",
                "solver_exception",
                "no_solution",
                "invalid_solution",
                "degenerate_virtual_direction",
                "nonfinite_diagnostics",
            }
        ):
            raise DiagnosticValidationError(
                "terminal QP failure lacks typed reconstruction context"
            )


def validate_diagnostic_result(
    result: Mapping[str, Any],
    *,
    output_root: Path,
    require_ready_geometry: bool = False,
) -> dict[str, Any]:
    payload_hash = result.get("result_payload_sha256")
    if (
        not isinstance(payload_hash, str)
        or payload_hash
        != diagnostics.sha256_bytes(
            diagnostics.canonical_json_bytes(
                {
                    key: value
                    for key, value in result.items()
                    if key != "result_payload_sha256"
                }
            )
        )
    ):
        raise DiagnosticValidationError("result payload hash mismatch")
    ledger = validate_embedded_action_ledger(result)
    record = result.get("failure_diagnostics")
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != diagnostics.DIAGNOSTICS_SCHEMA
        or record.get("enabled") is not True
        or record.get("status") != "published"
        or record.get("control_effect") != "read_only_observation"
    ):
        raise DiagnosticValidationError(
            "result has no complete opt-in diagnostics"
        )
    geometry = record.get("geometry")
    if result.get("mode") == "aegis":
        if not isinstance(geometry, Mapping):
            raise DiagnosticValidationError("AEGIS geometry is missing")
        _validate_geometry(
            geometry,
            output_root=output_root,
            require_ready=require_ready_geometry,
        )
    contacts = record.get("contacts")
    if not isinstance(contacts, Mapping):
        raise DiagnosticValidationError("contact artifact is missing")
    _validate_contacts(
        contacts,
        output_root=output_root,
        action_count=int(ledger["action_count"]),
    )
    _validate_qp_contexts(result)
    return {
        "case_id": result.get("case_id"),
        "arm": result.get("arm"),
        "status": "valid",
        "action_invariance_ledger": ledger,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate AEGIS failure diagnostics and action invariance"
    )
    parser.add_argument("--diagnostics-off-result", type=Path)
    parser.add_argument("--diagnostics-on-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--require-ready-geometry",
        action="store_true",
        help="paired-canary gate: require both views, filtering, and MVEE",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    on_result = _load_json(args.diagnostics_on_result.resolve())
    receipt: dict[str, Any] = {
        "schema_version": "vlsa_table1_diagnostics_validation.v1",
        "diagnostic_result": validate_diagnostic_result(
            on_result,
            output_root=args.output_root.resolve(),
            require_ready_geometry=args.require_ready_geometry,
        ),
    }
    if args.diagnostics_off_result is not None:
        off_result = _load_json(args.diagnostics_off_result.resolve())
        receipt["action_invariance"] = validate_action_invariance_pair(
            off_result,
            on_result,
        )
    if args.reference is not None:
        reference = _load_json(args.reference.resolve())
        receipt["canary_reference"] = validate_canary_reference(
            on_result, reference
        )
    receipt["receipt_payload_sha256"] = diagnostics.sha256_bytes(
        diagnostics.canonical_json_bytes(receipt)
    )
    payload = json.dumps(
        receipt,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    diagnostics._atomic_write_bytes(
        args.receipt, payload.encode("utf-8")
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
