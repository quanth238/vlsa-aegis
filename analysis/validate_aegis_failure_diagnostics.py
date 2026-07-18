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
        view_status = view.get("status")
        if view_status in {"not_attempted", "observer_or_runtime_failure"}:
            if status == "complete":
                raise DiagnosticValidationError(
                    f"{view_name}: complete geometry contains an unobserved view"
                )
            if (
                view.get("view") != view_name
                or (
                    view_status == "not_attempted"
                    and not isinstance(view.get("reason"), str)
                )
                or (
                    view_status == "observer_or_runtime_failure"
                    and not isinstance(view.get("failure"), Mapping)
                )
            ):
                raise DiagnosticValidationError(
                    f"{view_name}: explicit failed-view context is incomplete"
                )
            continue
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
    initial_direction = record.get("initial_direction")
    if (
        not isinstance(initial_direction, Mapping)
        or initial_direction.get("formula")
        != "normalize(mvee_center - stale_pre_settle_eef_proxy_center)"
        or not isinstance(initial_direction.get("z_initial"), list)
        or len(initial_direction["z_initial"]) != 3
        or not isinstance(initial_direction.get("arrays"), Mapping)
        or set(initial_direction["arrays"])
        != {"stale_proxy_center", "z_initial"}
    ):
        raise DiagnosticValidationError(
            "complete geometry lacks the exact initial virtual direction"
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
        events = snapshot.get("events")
        if not isinstance(events, list):
            raise DiagnosticValidationError(
                f"contact events are invalid at step {snapshot.get('step')}"
            )
        for event in events:
            if (
                not isinstance(event, Mapping)
                or event.get("step") != snapshot.get("step")
            ):
                raise DiagnosticValidationError(
                    "contact event step is not bound to its snapshot"
                )
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


def _vector_hat(value: Any, np: Any) -> Any:
    x, y, z = value
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _recompute_qp_context(context: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    p1 = np.asarray(context["p1"], dtype=float)
    R1 = np.asarray(context["R1"], dtype=float)
    q1 = np.asarray(context["q1_diag"], dtype=float)
    p2 = np.asarray(context["p2"], dtype=float)
    R2 = np.asarray(context["R2"], dtype=float)
    q2 = np.asarray(context["Q2_diag"], dtype=float)
    z_input = np.asarray(context["z_before"], dtype=float)
    nominal = np.asarray(context["nominal_translational"], dtype=float)
    if (
        p1.shape != (3,)
        or R1.shape != (3, 3)
        or q1.shape != (3,)
        or p2.shape != (3,)
        or R2.shape != (3, 3)
        or q2.shape != (3,)
        or z_input.shape != (3,)
        or nominal.shape != (7,)
    ):
        raise DiagnosticValidationError("QP context array shape changed")
    values = (p1, R1, q1, p2, R2, q2, z_input, nominal)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise DiagnosticValidationError("QP context contains non-finite input")

    eps = 1e-10
    Q1 = np.diag(q1)
    Q2 = np.diag(q2)
    Qbar1 = R1 @ Q1 @ R1.T
    Qbar2 = R2 @ Q2 @ R2.T
    Qbar1_inv = np.linalg.inv(Qbar1)
    Qbar1_inv2 = Qbar1_inv @ Qbar1_inv
    Qbar2_sq = Qbar2 @ Qbar2
    z = z_input / (np.linalg.norm(z_input) + eps)
    a_vec = Qbar1_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = Qbar2 @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = 1.0 - (p2 - p1).T @ a_vec + term1
    eta_row = -(1.0 / denom) * (z.T @ Qbar1_inv)
    mu_row = (
        (rho / (denom**3 + eps)) * (z.T @ Qbar1_inv2)
        + (1.0 / denom) * ((p2 - p1).T @ Qbar1_inv)
        - (1.0 / sigma)
        * (z.T @ Qbar1_inv @ Qbar2_sq @ Qbar1_inv)
    )
    tmp1 = z.T @ Qbar1_inv2 @ _vector_hat(z, np)
    left_vec = z.T @ Qbar1_inv @ Qbar2_sq
    tmp2 = left_vec @ (
        _vector_hat(a_vec, np)
        - Qbar1_inv @ _vector_hat(z, np)
    )
    tmp3 = (
        (p2 - p1).T @ Qbar1_inv @ _vector_hat(z, np)
        + z.T @ Qbar1_inv @ _vector_hat(p2 - p1, np)
    )
    zeta = (
        rho * (1.0 / (denom**3 + eps)) * tmp1
        + (1.0 / sigma) * tmp2
        + (1.0 / denom) * tmp3
    )
    a_v = np.asarray(eta_row @ R1).reshape(-1)
    a_omega = np.asarray(zeta @ R1).reshape(-1)
    projected_z = z / (np.linalg.norm(z) + 1e-12)
    cbf_projection = np.eye(3) - np.outer(projected_z, projected_z)
    a_u_z = np.asarray(mu_row @ cbf_projection).reshape(-1)

    h_z = z / np.linalg.norm(z)
    h_a = Qbar1_inv @ h_z
    h = (
        -np.linalg.norm(Qbar2 @ h_a)
        + (p2 - p1).T @ h_a
        - 1.0
    ) / np.linalg.norm(h_a)
    a_u_v = 0.2 * a_v
    v_ref = R1.T @ nominal[:3]
    u_v_ref = 5.0 * v_ref
    u_z_ref = 10.0 * mu_row
    reference = np.hstack([u_v_ref, u_z_ref])
    weights = np.asarray([1.0 / 25.0] * 3 + [1.0] * 3)
    reference_lhs = float(
        a_u_v @ u_v_ref + a_u_z @ u_z_ref + 10.0 * h
    )
    solution = np.asarray(context["u_solution"], dtype=float)
    if solution.shape != (6,) or not np.all(np.isfinite(solution)):
        raise DiagnosticValidationError("QP solution is invalid")
    solution_lhs = float(
        a_u_v @ solution[:3]
        + a_u_z @ solution[3:]
        + 10.0 * h
    )
    objective = float(np.sum(weights * (solution - reference) ** 2))
    # The released controller updates the virtual direction with the original
    # stored z, not compute_h_coeffs_3d's epsilon-normalized local copy.
    flow_projection = np.eye(3) - np.outer(z_input, z_input)
    next_z = z_input + flow_projection @ solution[3:] * 0.05
    next_z_norm = np.linalg.norm(next_z)
    if not np.isfinite(next_z_norm) or next_z_norm <= 1e-12:
        raise DiagnosticValidationError(
            "QP reconstruction produced a degenerate virtual direction"
        )
    next_z = next_z / next_z_norm
    executed = np.zeros(7, dtype=float)
    executed[:3] = 0.2 * R1 @ solution[:3]
    executed[6] = nominal[6]
    return {
        "a_v": a_v,
        "a_omega": a_omega,
        "a_u_v": a_u_v,
        "a_u_z": a_u_z,
        "mu_row": np.asarray(mu_row).reshape(-1),
        "h": float(h),
        "constant": float(10.0 * h),
        "v_ref": v_ref,
        "u_v_reference": u_v_ref,
        "u_z_reference": u_z_ref,
        "reference": reference,
        "weights_diagonal": weights,
        "reference_lhs": reference_lhs,
        "solution_lhs": solution_lhs,
        "objective": objective,
        "z_after": next_z,
        "executed": executed,
    }


def _assert_numeric_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    import numpy as np

    try:
        actual_array = np.asarray(actual, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
    except Exception as error:
        raise DiagnosticValidationError(
            f"QP reconstruction value is invalid: {label}"
        ) from error
    if (
        actual_array.shape != expected_array.shape
        or not np.all(np.isfinite(actual_array))
        or not np.all(np.isfinite(expected_array))
    ):
        raise DiagnosticValidationError(
            f"QP reconstruction shape/finiteness mismatch: {label}"
        )
    if not np.allclose(
        actual_array,
        expected_array,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise DiagnosticValidationError(f"QP reconstruction mismatch: {label}")


def _validate_qp_contexts(result: Mapping[str, Any]) -> None:
    if result.get("mode") != "aegis":
        return
    previous_z_after = None
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
            "nominal_translational",
            "v_ref",
            "u_v_reference",
            "u_z_reference",
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
        if (
            qp.get("solver") != "OSQP"
            or context.get("solver") != "OSQP"
            or qp.get("solver_status") != context.get("solver_status")
            or not isinstance(context.get("solver_stats"), Mapping)
        ):
            raise DiagnosticValidationError(
                f"step {action.get('step')}: QP solver context changed"
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
            reconstructed = _recompute_qp_context(context)
        except DiagnosticValidationError:
            raise
        except Exception as error:
            raise DiagnosticValidationError(
                f"step {action.get('step')}: QP reconstruction failed"
            ) from error
        for key in (
            "a_v",
            "a_omega",
            "a_u_v",
            "a_u_z",
            "mu_row",
            "h",
            "constant",
        ):
            _assert_numeric_close(
                cbf[key],
                reconstructed[key],
                label=f"cbf.{key}",
            )
        for key in (
            "v_ref",
            "u_v_reference",
            "u_z_reference",
            "reference",
            "weights_diagonal",
        ):
            _assert_numeric_close(
                context[key],
                reconstructed[key],
                label=key,
            )
        for prefix, expected_lhs in (
            ("reference", reconstructed["reference_lhs"]),
            ("solution", reconstructed["solution_lhs"]),
        ):
            _assert_numeric_close(
                (
                    cbf[f"{prefix}_lhs"]
                    if prefix == "reference"
                    else context[f"{prefix}_lhs"]
                ),
                expected_lhs,
                label=f"{prefix}_lhs",
            )
            stored_slack = (
                cbf[f"{prefix}_slack"]
                if prefix == "reference"
                else context[f"{prefix}_slack"]
            )
            stored_violation = (
                cbf[f"{prefix}_violation"]
                if prefix == "reference"
                else context[f"{prefix}_violation"]
            )
            _assert_numeric_close(
                stored_slack,
                expected_lhs,
                label=f"{prefix}_slack",
            )
            _assert_numeric_close(
                stored_violation,
                max(0.0, -expected_lhs),
                label=f"{prefix}_violation",
            )
        _assert_numeric_close(
            context["objective"],
            reconstructed["objective"],
            label="objective",
        )
        _assert_numeric_close(
            context["z_after"],
            reconstructed["z_after"],
            label="z_after",
        )
        _assert_numeric_close(
            context["executed_action"],
            reconstructed["executed"],
            label="executed_action",
        )
        _assert_numeric_close(
            qp["barrier_h"],
            reconstructed["h"],
            label="top_level.barrier_h",
        )
        _assert_numeric_close(
            qp["constraint_lhs"],
            reconstructed["solution_lhs"],
            label="top_level.constraint_lhs",
        )
        _assert_numeric_close(
            qp["u_solution"],
            context["u_solution"],
            label="top_level.u_solution",
        )
        _assert_numeric_close(
            qp["objective"],
            context["objective"],
            label="top_level.objective",
        )
        _assert_numeric_close(
            qp["z_before"],
            context["z_before"],
            label="top_level.z_before",
        )
        _assert_numeric_close(
            qp["z_after"],
            context["z_after"],
            label="top_level.z_after",
        )
        if previous_z_after is not None:
            _assert_numeric_close(
                context["z_before"],
                previous_z_after,
                label="cross_step.z_before",
            )
        previous_z_after = context["z_after"]
        if cbf.get("alpha_gain") != 10.0:
            raise DiagnosticValidationError(
                f"step {action.get('step')}: CBF alpha gain changed"
            )
        try:
            import numpy as np

            dual = np.asarray(
                context["constraint_dual"], dtype=float
            ).reshape(-1)
        except Exception as error:
            raise DiagnosticValidationError(
                f"step {action.get('step')}: invalid QP dual"
            ) from error
        if (
            dual.size != 1
            or not np.all(np.isfinite(dual))
            or float(dual[0]) < -1e-8
            or reconstructed["solution_lhs"] < -1e-6
        ):
            raise DiagnosticValidationError(
                f"step {action.get('step')}: invalid QP feasibility evidence"
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
