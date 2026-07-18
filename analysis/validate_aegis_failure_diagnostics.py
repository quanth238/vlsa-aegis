#!/usr/bin/env python3
"""Validate action invariance and opt-in AEGIS failure diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
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


_MANIFEST_CASES: dict[str, dict[str, Any]] | None = None


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


def _trusted_manifest_case(
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    global _MANIFEST_CASES

    if _MANIFEST_CASES is None:
        path = ROOT / "manifests" / "vlsa_table1_population.jsonl"
        cases: dict[str, dict[str, Any]] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = (
                    row.get("case_id")
                    if isinstance(row, dict)
                    else None
                )
                if (
                    not isinstance(row, dict)
                    or not isinstance(case_id, str)
                    or not case_id
                    or case_id in cases
                ):
                    raise ValueError("invalid or duplicate manifest row")
                cases[case_id] = row
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise DiagnosticValidationError(
                f"trusted Table-1 manifest cannot be loaded: {error}"
            ) from error
        if len(cases) != 1600:
            raise DiagnosticValidationError(
                f"trusted Table-1 manifest has {len(cases)} cases, expected 1600"
            )
        _MANIFEST_CASES = cases
    case_id = result.get("case_id")
    case = result.get("case")
    expected = _MANIFEST_CASES.get(str(case_id))
    if (
        expected is None
        or not isinstance(case, Mapping)
        or diagnostics.canonical_json_bytes(case)
        != diagnostics.canonical_json_bytes(expected)
    ):
        raise DiagnosticValidationError(
            "result case does not match the immutable Table-1 manifest"
        )
    pairing = result.get("pairing")
    if (
        not isinstance(pairing, Mapping)
        or pairing.get("manifest_row_sha256")
        != diagnostics.sha256_bytes(
            diagnostics.canonical_json_bytes(expected)
        )
    ):
        raise DiagnosticValidationError(
            "manifest-row hash is not bound to the immutable Table-1 case"
        )
    return expected


def _parse_sexpression(
    tokens: Sequence[str],
    start: int,
) -> tuple[Any, int]:
    if start >= len(tokens):
        raise DiagnosticValidationError("truncated BDDL expression")
    if tokens[start] != "(":
        return tokens[start], start + 1
    values: list[Any] = []
    cursor = start + 1
    while cursor < len(tokens) and tokens[cursor] != ")":
        value, cursor = _parse_sexpression(tokens, cursor)
        values.append(value)
    if cursor >= len(tokens):
        raise DiagnosticValidationError("unterminated BDDL expression")
    return values, cursor + 1


def _trusted_native_goal_atoms(
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    case = _trusted_manifest_case(result)
    relative = Path(str(case.get("bddl_path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise DiagnosticValidationError("immutable BDDL path is unsafe")
    path = ROOT / relative
    if (
        not path.is_file()
        or case.get("bddl_sha256") != diagnostics.sha256_path(path)
    ):
        raise DiagnosticValidationError(
            "immutable BDDL file/hash binding changed"
        )
    text = re.sub(
        r";;[^\n]*",
        "",
        path.read_text(encoding="utf-8"),
    )
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    goal_start = next(
        (
            index
            for index in range(len(tokens) - 1)
            if tokens[index] == "("
            and tokens[index + 1].lower() == ":goal"
        ),
        None,
    )
    if goal_start is None:
        raise DiagnosticValidationError("immutable BDDL goal is missing")
    goal, cursor = _parse_sexpression(tokens, goal_start)
    if (
        cursor <= goal_start
        or not isinstance(goal, list)
        or len(goal) != 2
        or str(goal[0]).lower() != ":goal"
        or not isinstance(goal[1], list)
        or not goal[1]
        or str(goal[1][0]).lower() != "and"
    ):
        raise DiagnosticValidationError(
            "immutable BDDL goal structure is unsupported"
        )
    atoms: list[dict[str, Any]] = []
    for index, atom in enumerate(goal[1][1:]):
        if (
            not isinstance(atom, list)
            or len(atom) != 3
            or str(atom[0]).lower() not in {"in", "on"}
            or not all(
                isinstance(argument, str) and argument
                for argument in atom[1:]
            )
        ):
            raise DiagnosticValidationError(
                "immutable BDDL goal atom is unsupported"
            )
        atoms.append(
            {
                "index": index,
                "predicate": str(atom[0]).lower(),
                "arguments": [str(atom[1]), str(atom[2])],
            }
        )
    if not atoms:
        raise DiagnosticValidationError("immutable BDDL goal has no atoms")
    return atoms


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


def _validated_settled_contract(
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    pairing = result.get("pairing")
    if not isinstance(pairing, Mapping):
        raise DiagnosticValidationError("result lacks pairing record")
    contract = pairing.get("initial_observation_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema_version")
        != "vlsa_table1_settled_input.v1"
        or pairing.get("initial_observation_sha256")
        != diagnostics.sha256_bytes(
            diagnostics.canonical_json_bytes(contract)
        )
    ):
        raise DiagnosticValidationError(
            "settled-input contract hash/schema changed"
        )
    return contract


def _pair_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    pairing = result.get("pairing")
    _validated_settled_contract(result)
    assert isinstance(pairing, Mapping)
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


def _terminal_attempt_binding(
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    method_failure = result.get("method_failure")
    if (
        not isinstance(method_failure, Mapping)
        or method_failure.get("component") != "aegis_qp"
    ):
        return None
    binding = {
        "phase": method_failure.get("phase"),
        "step": method_failure.get("step"),
        "nominal_raw": method_failure.get("nominal_raw"),
        "nominal_translational": method_failure.get(
            "nominal_translational"
        ),
    }
    if (
        binding["phase"] != "control"
        or not isinstance(binding["step"], int)
        or not isinstance(binding["nominal_raw"], list)
        or not isinstance(binding["nominal_translational"], list)
    ):
        raise DiagnosticValidationError(
            "terminal AEGIS attempt binding is incomplete"
        )
    return binding


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
    if _terminal_attempt_binding(
        diagnostics_off
    ) != _terminal_attempt_binding(diagnostics_on):
        raise DiagnosticValidationError(
            "diagnostics changed the terminal unexecuted AEGIS attempt"
        )
    off_flag = "failure_diagnostics" in diagnostics_off
    on_flag = diagnostics_on.get("failure_diagnostics", {}).get("enabled")
    if off_flag or on_flag is not True:
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
) -> dict[str, Any]:
    import numpy as np

    try:
        diagnostics.validate_artifact_descriptor(
            descriptor, output_root=output_root
        )
    except ValueError as error:
        raise DiagnosticValidationError(str(error)) from error
    path = output_root / str(descriptor["path"])
    retained: dict[str, Any] = {}
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
            retained[key] = np.asarray(archive[key]).copy()
    return retained


def _validate_array_descriptor_values(
    values: Any,
    descriptor: Any,
    *,
    label: str,
) -> None:
    import numpy as np

    if not isinstance(descriptor, Mapping):
        raise DiagnosticValidationError(
            f"{label}: array descriptor is missing"
        )
    try:
        value = np.asarray(
            values, dtype=np.dtype(descriptor["dtype"])
        ).reshape(tuple(descriptor["shape"]))
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: descriptor values cannot be reconstructed"
        ) from error
    if diagnostics.array_descriptor(value) != dict(descriptor):
        raise DiagnosticValidationError(
            f"{label}: descriptor/value hash binding changed"
        )


def _validate_geometry(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
    require_ready: bool,
    expected_case_id: str | None = None,
    expected_suite: str | None = None,
    expected_label: str | None = None,
    settled_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if descriptor.get("schema_version") != diagnostics.GEOMETRY_SCHEMA:
        raise DiagnosticValidationError("unexpected geometry schema")
    arrays = _validate_npz_artifact(
        descriptor, output_root=output_root
    )
    record = descriptor.get("record")
    if not isinstance(record, Mapping):
        raise DiagnosticValidationError("geometry record is missing")
    for key, expected in (
        ("case_id", expected_case_id),
        ("suite", expected_suite),
        ("obstacle_label", expected_label),
    ):
        if expected is not None and record.get(key) != expected:
            raise DiagnosticValidationError(
                f"geometry identity mismatch: {key}"
            )
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
        if expected_label is not None and (
            request.get("caption") != expected_label
            or request.get("box_threshold") != 0.35
            or request.get("text_threshold") != 0.25
            or not isinstance(request.get("device"), str)
            or not request.get("device")
        ):
            raise DiagnosticValidationError(
                f"{view_name}: released DINO request changed"
            )
        if settled_contract is not None:
            image_key = (
                "agentview_array_sha256"
                if view_name == "agentview"
                else "backview_array_sha256"
            )
            depth_key = (
                "agentview_depth_array_sha256"
                if view_name == "agentview"
                else "backview_depth_array_sha256"
            )
            input_image = view.get("input_image")
            input_depth = view.get("input_depth")
            if (
                not isinstance(input_image, Mapping)
                or not isinstance(input_depth, Mapping)
                or input_image.get("array_sha256")
                != settled_contract.get(image_key)
                or input_depth.get("array_sha256")
                != settled_contract.get(depth_key)
            ):
                raise DiagnosticValidationError(
                    f"{view_name}: DINO input image/depth binding changed"
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
        _validate_array_descriptor_values(
            boxes,
            detections.get("returned_order_boxes_cxcywh_array"),
            label=f"{view_name}.boxes_cxcywh",
        )
        boxes_xyxy = detections.get("returned_order_boxes_xyxy")
        if not isinstance(boxes_xyxy, list) or len(boxes_xyxy) != count:
            raise DiagnosticValidationError(
                f"{view_name}: returned xyxy boxes are inconsistent"
            )
        _validate_array_descriptor_values(
            boxes_xyxy,
            detections.get("returned_order_boxes_xyxy_array"),
            label=f"{view_name}.boxes_xyxy",
        )
        _validate_array_descriptor_values(
            logits,
            detections.get("returned_order_logits_array"),
            label=f"{view_name}.logits",
        )
        raw_key = f"{view_name}_raw_points"
        raw_descriptor = view.get("raw_points_array")
        if (
            raw_key not in arrays
            or not isinstance(raw_descriptor, Mapping)
            or diagnostics.array_descriptor(arrays[raw_key])
            != dict(raw_descriptor)
            or view.get("returned_point_cloud") != raw_descriptor
        ):
            raise DiagnosticValidationError(
                f"{view_name}: raw point-cloud/NPZ binding changed"
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
        return None
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
    filtering_arrays = filtering.get("arrays")
    required_filtering_arrays = {
        "fused_points",
        "range_mask",
        "range_filtered_points",
        "centroid_distances",
        "centroid_sorted_indices",
        "nearest80_points",
        "dbscan_labels",
        "shadow_filtered_points",
        "released_filtered_points",
    }
    if (
        not isinstance(filtering_arrays, Mapping)
        or not required_filtering_arrays.issubset(filtering_arrays)
    ):
        raise DiagnosticValidationError(
            "released filtering arrays are missing"
        )
    for key, metadata in filtering_arrays.items():
        if (
            key not in arrays
            or diagnostics.array_descriptor(arrays[key]) != metadata
        ):
            raise DiagnosticValidationError(
                f"released filtering array binding changed: {key}"
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
    mvee_arrays = mvee.get("arrays")
    required_mvee_arrays = {
        "hull_vertices",
        "hull_simplices",
        "hull_equations",
        "mvee_input",
        "center",
        "matrix_A",
        "released_center",
        "released_rotation",
        "released_semiaxes",
        "matrix_A_eigenvalues",
        "matrix_A_eigenvectors",
        "hull_quadratic_values",
    }
    if (
        not isinstance(mvee_arrays, Mapping)
        or not required_mvee_arrays.issubset(mvee_arrays)
    ):
        raise DiagnosticValidationError("MVEE array ledger is missing")
    for key, metadata in mvee_arrays.items():
        if (
            key not in arrays
            or diagnostics.array_descriptor(arrays[key]) != metadata
        ):
            raise DiagnosticValidationError(
                f"MVEE array binding changed: {key}"
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
        != {
            "stale_proxy_center",
            "stale_proxy_rotation",
            "z_initial",
        }
    ):
        raise DiagnosticValidationError(
            "complete geometry lacks the exact initial virtual direction"
        )
    bindings = {
        "p1": initial_direction.get("stale_proxy_center"),
        "R1": initial_direction.get("stale_proxy_rotation"),
        "p2": mvee.get("center"),
        "R2": mvee.get("rotation"),
        "Q2_diag": mvee.get("semiaxes"),
        "z_initial": initial_direction.get("z_initial"),
    }
    array_bindings = {
        "p1": "stale_proxy_center",
        "R1": "stale_proxy_rotation",
        "p2": "released_center",
        "R2": "released_rotation",
        "Q2_diag": "released_semiaxes",
        "z_initial": "z_initial",
    }
    for key, array_key in array_bindings.items():
        if array_key not in arrays:
            raise DiagnosticValidationError(
                f"geometry binding array is missing: {array_key}"
            )
        _assert_numeric_close(
            bindings[key],
            arrays[array_key],
            label=f"geometry.{key}",
        )
    for record_key, array_key in (
        ("center", "center"),
        ("matrix_A", "matrix_A"),
        ("eigenvalues", "matrix_A_eigenvalues"),
        ("semiaxes", "released_semiaxes"),
        ("rotation", "released_rotation"),
    ):
        _assert_numeric_close(
            mvee.get(record_key),
            arrays[array_key],
            label=f"geometry.mvee.{record_key}",
        )
    _assert_numeric_close(
        initial_direction.get("obstacle_center"),
        bindings["p2"],
        label="geometry.initial_obstacle_center",
    )
    try:
        import numpy as np

        delta = np.asarray(bindings["p2"], dtype=float) - np.asarray(
            bindings["p1"], dtype=float
        )
        delta_norm = float(np.linalg.norm(delta))
    except Exception as error:
        raise DiagnosticValidationError(
            "initial virtual-direction formula inputs are invalid"
        ) from error
    if not np.isfinite(delta_norm) or delta_norm <= 1e-12:
        raise DiagnosticValidationError(
            "initial virtual-direction formula is degenerate"
        )
    _assert_numeric_close(
        bindings["z_initial"],
        delta / delta_norm,
        label="geometry.initial_direction_formula",
    )
    return bindings


def _validate_contacts(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
    action_count: int,
    expected_case_id: str | None = None,
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
        or (
            expected_case_id is not None
            and payload.get("case_id") != expected_case_id
        )
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_goal_snapshot(
    snapshot: Any,
    *,
    atoms: Sequence[Mapping[str, Any]],
    expected_step: int,
    previous_values: Sequence[bool] | None,
) -> list[bool]:
    import numpy as np

    if not isinstance(snapshot, Mapping):
        raise DiagnosticValidationError(
            f"goal snapshot {expected_step} is missing"
        )
    values = snapshot.get("values")
    if (
        type(snapshot.get("step")) is not int
        or snapshot.get("step") != expected_step
        or not isinstance(values, list)
        or len(values) != len(atoms)
        or not all(type(value) is bool for value in values)
    ):
        raise DiagnosticValidationError(
            f"goal snapshot {expected_step} vector changed"
        )
    satisfied = sum(values)
    fraction = snapshot.get("fraction")
    if type(fraction) is not float:
        raise DiagnosticValidationError(
            f"goal snapshot {expected_step} fraction changed"
        )
    if (
        type(snapshot.get("satisfied_count")) is not int
        or snapshot.get("satisfied_count") != satisfied
        or snapshot.get("all_satisfied") is not all(values)
        or not np.isclose(
            fraction,
            satisfied / len(values),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise DiagnosticValidationError(
            f"goal snapshot {expected_step} aggregate changed"
        )
    prior = (
        [False] * len(values)
        if previous_values is None
        else list(previous_values)
    )
    expected_new = [
        index
        for index, (old, new) in enumerate(zip(prior, values))
        if not old and new
    ]
    expected_regressed = [
        index
        for index, (old, new) in enumerate(zip(prior, values))
        if old and not new
    ]
    try:
        transition_indices_match = (
            diagnostics.canonical_json_bytes(
                snapshot.get("newly_satisfied_indices")
            )
            == diagnostics.canonical_json_bytes(expected_new)
            and diagnostics.canonical_json_bytes(
                snapshot.get("regressed_indices")
            )
            == diagnostics.canonical_json_bytes(expected_regressed)
        )
    except (TypeError, ValueError):
        transition_indices_match = False
    if (
        not transition_indices_match
        or snapshot.get("inert") is not True
        or not _is_sha256(
            snapshot.get("simulator_state_sha256_before")
        )
        or snapshot.get("simulator_state_sha256_before")
        != snapshot.get("simulator_state_sha256_after")
    ):
        raise DiagnosticValidationError(
            f"goal snapshot {expected_step} transition/inertness changed"
        )
    pose_groups = snapshot.get("argument_poses")
    if not isinstance(pose_groups, list) or len(pose_groups) != len(atoms):
        raise DiagnosticValidationError(
            f"goal snapshot {expected_step} pose ledger changed"
        )
    for atom_index, (atom, pose_group) in enumerate(
        zip(atoms, pose_groups)
    ):
        arguments = atom.get("arguments")
        poses = (
            pose_group.get("arguments")
            if isinstance(pose_group, Mapping)
            else None
        )
        if (
            not isinstance(pose_group, Mapping)
            or type(pose_group.get("atom_index")) is not int
            or pose_group.get("atom_index") != atom_index
            or not isinstance(arguments, list)
            or not isinstance(poses, list)
            or len(arguments) != 2
            or len(poses) != 2
        ):
            raise DiagnosticValidationError(
                f"goal snapshot {expected_step} atom-pose binding changed"
            )
        for expected_name, pose in zip(arguments, poses):
            if (
                not isinstance(pose, Mapping)
                or pose.get("name") != expected_name
                or pose.get("object_state_type")
                not in {"object", "site"}
            ):
                raise DiagnosticValidationError(
                    f"goal snapshot {expected_step} argument identity changed"
                )
            try:
                position = np.asarray(pose.get("position"), dtype=float)
                quaternion = np.asarray(
                    pose.get("quaternion"), dtype=float
                )
            except Exception as error:
                raise DiagnosticValidationError(
                    f"goal snapshot {expected_step} argument pose is invalid"
                ) from error
            if (
                position.shape != (3,)
                or quaternion.shape != (4,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(quaternion))
            ):
                raise DiagnosticValidationError(
                    f"goal snapshot {expected_step} argument pose changed"
                )
    return list(values)


def _validate_goal_progress(
    result: Mapping[str, Any],
    *,
    action_count: int,
) -> str:
    goal = result.get("goal_progress")
    if not isinstance(goal, Mapping):
        raise DiagnosticValidationError("native goal progress is missing")
    atoms = goal.get("goal_atoms")
    if (
        goal.get("schema_version") != "safelibero_goal_progress.v1"
        or goal.get("source") != "native_bddl_goal_predicates"
        or goal.get("logic") != "conjunction"
        or not isinstance(atoms, list)
        or not atoms
    ):
        raise DiagnosticValidationError(
            "native goal definition changed"
        )
    trusted_atoms = _trusted_native_goal_atoms(result)
    if diagnostics.canonical_json_bytes(atoms) != (
        diagnostics.canonical_json_bytes(trusted_atoms)
    ):
        raise DiagnosticValidationError(
            "native goal atoms do not match the immutable BDDL task"
        )
    for index, atom in enumerate(atoms):
        if (
            not isinstance(atom, Mapping)
            or atom.get("index") != index
            or atom.get("predicate") not in {"in", "on"}
            or not isinstance(atom.get("arguments"), list)
            or len(atom["arguments"]) != 2
            or not all(
                isinstance(value, str) and value
                for value in atom["arguments"]
            )
        ):
            raise DiagnosticValidationError(
                "native goal atom ledger changed"
            )
    definition = {
        "schema_version": goal["schema_version"],
        "source": goal["source"],
        "logic": goal["logic"],
        "goal_atoms": atoms,
    }
    if goal.get("goal_definition_sha256") != diagnostics.sha256_bytes(
        diagnostics.canonical_json_bytes(definition)
    ):
        raise DiagnosticValidationError(
            "native goal definition hash changed"
        )
    pairing = result.get("pairing")
    initial_state_hash = (
        pairing.get("settled_simulator_state_sha256")
        if isinstance(pairing, Mapping)
        else None
    )
    previous = _validate_goal_snapshot(
        goal.get("initial"),
        atoms=atoms,
        expected_step=-1,
        previous_values=None,
    )
    if goal["initial"].get(
        "simulator_state_sha256_before"
    ) != initial_state_hash:
        raise DiagnosticValidationError(
            "initial goal snapshot is not bound to the settled state"
        )
    snapshots = [goal["initial"]]
    actions = result.get("actions")
    if not isinstance(actions, list) or len(actions) != action_count:
        raise DiagnosticValidationError("action inventory changed")
    for expected_step, action in enumerate(actions):
        if (
            not isinstance(action, Mapping)
            or type(action.get("step")) is not int
            or action.get("step") != expected_step
            or type(action.get("done")) is not bool
        ):
            raise DiagnosticValidationError(
                "executed action step sequence changed"
            )
        snapshot = action.get("goal_progress")
        previous = _validate_goal_snapshot(
            snapshot,
            atoms=atoms,
            expected_step=expected_step,
            previous_values=previous,
        )
        if snapshot["all_satisfied"] is not action["done"]:
            raise DiagnosticValidationError(
                "native goal snapshot disagrees with env.step done"
            )
        snapshots.append(snapshot)
    expected_final = snapshots[-1]
    try:
        final_matches = diagnostics.canonical_json_bytes(
            goal.get("final")
        ) == diagnostics.canonical_json_bytes(expected_final)
    except (TypeError, ValueError):
        final_matches = False
    if not final_matches:
        raise DiagnosticValidationError(
            "final goal snapshot is not the final simulator action state"
        )
    values = [list(snapshot["values"]) for snapshot in snapshots]
    summary = {
        "initial_values": list(snapshots[0]["values"]),
        "final_values": list(expected_final["values"]),
        "initial_satisfied_count": int(
            snapshots[0]["satisfied_count"]
        ),
        "final_satisfied_count": int(
            expected_final["satisfied_count"]
        ),
        "maximum_satisfied_count": max(
            int(snapshot["satisfied_count"]) for snapshot in snapshots
        ),
        "initial_fraction": float(snapshots[0]["fraction"]),
        "final_fraction": float(expected_final["fraction"]),
        "maximum_fraction": max(
            float(snapshot["fraction"]) for snapshot in snapshots
        ),
        "ever_satisfied": [
            any(vector[index] for vector in values)
            for index in range(len(atoms))
        ],
        "first_satisfied_step": [
            next(
                (
                    int(snapshot["step"])
                    for snapshot in snapshots
                    if snapshot["values"][atom_index]
                ),
                None,
            )
            for atom_index in range(len(atoms))
        ],
        "first_all_satisfied_step": next(
            (
                int(snapshot["step"])
                for snapshot in snapshots
                if snapshot["all_satisfied"]
            ),
            None,
        ),
        "regression_count": sum(
            len(snapshot["regressed_indices"])
            for snapshot in snapshots[1:]
        ),
    }
    try:
        summary_matches = diagnostics.canonical_json_bytes(
            goal.get("summary")
        ) == diagnostics.canonical_json_bytes(summary)
    except (TypeError, ValueError):
        summary_matches = False
    if not summary_matches:
        raise DiagnosticValidationError(
            "native goal summary is stale/tampered"
        )
    if (
        type(result.get("task_success")) is not bool
        or expected_final["all_satisfied"] is not result["task_success"]
    ):
        raise DiagnosticValidationError(
            "final native goal satisfaction disagrees with task success"
        )
    return str(expected_final["simulator_state_sha256_after"])


def _validate_terminal_frame_artifact(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    if (
        descriptor.get("schema_version")
        != diagnostics.TERMINAL_FRAME_SCHEMA
        or descriptor.get("format") != "numpy_npy"
        or not isinstance(descriptor.get("array"), Mapping)
    ):
        raise DiagnosticValidationError(
            "terminal lossless frame descriptor changed"
        )
    try:
        diagnostics.validate_artifact_descriptor(
            descriptor,
            output_root=output_root,
        )
    except ValueError as error:
        raise DiagnosticValidationError(str(error)) from error
    path = output_root / str(descriptor["path"])
    try:
        frame = np.load(path, allow_pickle=False)
    except Exception as error:
        raise DiagnosticValidationError(
            "terminal lossless frame cannot be loaded"
        ) from error
    actual = diagnostics.array_descriptor(frame)
    if actual != descriptor["array"]:
        raise DiagnosticValidationError(
            "terminal lossless frame array binding changed"
        )
    if (
        actual.get("dtype") != "|u1"
        or actual.get("shape") != [1024, 1024, 3]
        or actual.get("finite") is not True
    ):
        raise DiagnosticValidationError(
            "terminal lossless frame is not the 1024x1024 uint8 RGB image"
        )
    return actual, frame


def _probe_episode_video(
    path: Path,
    *,
    expected_frame_count: int,
    terminal_source: Any,
) -> dict[str, Any]:
    import math
    import numpy as np

    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise DiagnosticValidationError(
            "imageio is required to decode the episode video"
        ) from error
    try:
        reader = imageio.get_reader(str(path))
        try:
            metadata = reader.get_meta_data()
            fps = metadata.get("fps")
            if (
                isinstance(fps, bool)
                or not isinstance(fps, (int, float))
                or not math.isfinite(float(fps))
                or not math.isclose(
                    float(fps),
                    30.0,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise DiagnosticValidationError(
                    "decoded episode video fps is not 30"
                )
            decoded_count = 0
            last_frame = None
            for raw_frame in reader:
                if decoded_count >= expected_frame_count:
                    raise DiagnosticValidationError(
                        "decoded episode video has extra frames"
                    )
                frame = np.asarray(raw_frame)
                if (
                    frame.dtype != np.uint8
                    or frame.shape != (1024, 1024, 3)
                ):
                    raise DiagnosticValidationError(
                        "decoded episode frame is not 1024x1024 uint8 RGB"
                    )
                last_frame = np.ascontiguousarray(frame).copy()
                decoded_count += 1
        finally:
            reader.close()
    except DiagnosticValidationError:
        raise
    except Exception as error:
        raise DiagnosticValidationError(
            f"episode video cannot be fully decoded: {error}"
        ) from error
    if decoded_count != expected_frame_count or last_frame is None:
        raise DiagnosticValidationError(
            "decoded episode video frame count changed"
        )
    source = np.asarray(terminal_source)
    difference = (
        last_frame.astype(np.float64) - source.astype(np.float64)
    )
    mean_absolute_error = float(np.mean(np.abs(difference)))
    mean_squared_error = float(np.mean(difference * difference))
    peak_signal_to_noise_ratio_db = (
        float("inf")
        if mean_squared_error == 0.0
        else float(
            20.0
            * math.log10(255.0 / math.sqrt(mean_squared_error))
        )
    )
    if (
        not math.isfinite(mean_absolute_error)
        or mean_absolute_error > 12.0
        or (
            math.isfinite(peak_signal_to_noise_ratio_db)
            and peak_signal_to_noise_ratio_db < 20.0
        )
    ):
        raise DiagnosticValidationError(
            "decoded terminal video frame does not match the lossless source"
        )
    return {
        "decoder": "imageio.v2",
        "decoded_frame_count": decoded_count,
        "decoded_fps": float(fps),
        "decoded_resolution": [1024, 1024],
        "decoded_channels": 3,
        "decoded_dtype": "|u1",
        "terminal_mean_absolute_error": mean_absolute_error,
        "terminal_peak_signal_to_noise_ratio_db": (
            None
            if not math.isfinite(peak_signal_to_noise_ratio_db)
            else peak_signal_to_noise_ratio_db
        ),
        "status": "fully_decoded",
    }


def _validate_terminal_and_video(
    result: Mapping[str, Any],
    *,
    output_root: Path,
    terminal_frame: Mapping[str, Any],
    terminal_frame_array: Any,
    action_count: int,
    final_goal_state_sha256: str,
) -> dict[str, Any]:
    mode = result.get("mode")
    case_id = result.get("case_id")
    if (
        mode not in {"pi05", "aegis"}
        or not isinstance(case_id, str)
        or not case_id
    ):
        raise DiagnosticValidationError(
            "terminal result mode/case identity changed"
        )
    if (
        result.get("scientific_result") is not True
        or result.get("status")
        not in {"complete", "method_failure", "method_failure_passthrough"}
    ):
        raise DiagnosticValidationError(
            "partial/apparatus result cannot publish diagnostics"
        )
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DiagnosticValidationError("terminal metrics are missing")
    reason = result.get("terminal_reason")
    if (
        type(metrics.get("executed_action_count")) is not int
        or metrics.get("executed_action_count") != action_count
        or metrics.get("termination_reason") != reason
        or type(metrics.get("task_success")) is not bool
        or metrics.get("task_success") is not (reason == "task_success")
        or type(result.get("task_success")) is not bool
        or result.get("task_success") is not metrics.get("task_success")
        or reason not in {"task_success", "time_limit", "method_failure"}
    ):
        raise DiagnosticValidationError(
            "terminal metric/action binding changed"
        )
    if (
        (result.get("status") == "method_failure")
        is not (reason == "method_failure")
    ):
        raise DiagnosticValidationError(
            "terminal status/reason binding changed"
        )
    terminal = result.get("terminal_observation")
    if (
        not isinstance(terminal, Mapping)
        or type(terminal.get("frame_index")) is not int
        or terminal.get("frame_index") != action_count
        or type(terminal.get("after_executed_action_count")) is not int
        or terminal.get("after_executed_action_count") != action_count
        or terminal.get("agentview_array_sha256")
        != terminal_frame.get("array_sha256")
        or terminal.get("simulator_state_sha256")
        != final_goal_state_sha256
        or not _is_sha256(terminal.get("simulator_state_sha256"))
    ):
        raise DiagnosticValidationError(
            "terminal observation/frame/state binding changed"
        )
    video = result.get("video")
    if not isinstance(video, Mapping):
        raise DiagnosticValidationError("episode video receipt is missing")
    try:
        relative = Path(str(video["path"]))
    except Exception as error:
        raise DiagnosticValidationError("episode video path is invalid") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise DiagnosticValidationError("episode video path is unsafe")
    path = output_root / relative
    if (
        relative != Path(mode) / case_id / "episode.mp4"
        or not path.is_file()
        or video.get("sha256") != diagnostics.sha256_path(path)
        or video.get("complete_episode") is not True
        or type(video.get("frames")) is not int
        or video.get("frames") != action_count + 1
        or type(video.get("fps")) is not int
        or video.get("fps") != 30
        or video.get("terminal_source_array_sha256")
        != terminal_frame.get("array_sha256")
    ):
        raise DiagnosticValidationError(
            "episode video/terminal-frame binding changed"
        )
    return _probe_episode_video(
        path,
        expected_frame_count=action_count + 1,
        terminal_source=terminal_frame_array,
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


def _assert_numeric_equivalent(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    """Compare finite values tightly and non-finite values by exact class."""

    import numpy as np

    try:
        actual_array = np.asarray(actual, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
    except Exception as error:
        raise DiagnosticValidationError(
            f"QP numeric evidence is invalid: {label}"
        ) from error
    if actual_array.shape != expected_array.shape:
        raise DiagnosticValidationError(
            f"QP numeric evidence shape mismatch: {label}"
        )
    if (
        not np.array_equal(np.isnan(actual_array), np.isnan(expected_array))
        or not np.array_equal(
            np.isposinf(actual_array), np.isposinf(expected_array)
        )
        or not np.array_equal(
            np.isneginf(actual_array), np.isneginf(expected_array)
        )
    ):
        raise DiagnosticValidationError(
            f"QP non-finite evidence mismatch: {label}"
        )
    finite = np.isfinite(expected_array)
    if not np.allclose(
        actual_array[finite],
        expected_array[finite],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise DiagnosticValidationError(
            f"QP finite evidence mismatch: {label}"
        )


def _validate_qp_core_inputs(
    context: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    import numpy as np

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
    }
    if not required.issubset(context):
        raise DiagnosticValidationError(
            f"{label}: incomplete QP source inputs"
        )
    try:
        arrays = {
            "p1": np.asarray(context["p1"], dtype=float),
            "R1": np.asarray(context["R1"], dtype=float),
            "q1_diag": np.asarray(context["q1_diag"], dtype=float),
            "p2": np.asarray(context["p2"], dtype=float),
            "R2": np.asarray(context["R2"], dtype=float),
            "Q2_diag": np.asarray(context["Q2_diag"], dtype=float),
            "z_before": np.asarray(context["z_before"], dtype=float),
            "nominal_translational": np.asarray(
                context["nominal_translational"], dtype=float
            ),
        }
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: invalid QP source inputs"
        ) from error
    expected_shapes = {
        "p1": (3,),
        "R1": (3, 3),
        "q1_diag": (3,),
        "p2": (3,),
        "R2": (3, 3),
        "Q2_diag": (3,),
        "z_before": (3,),
        "nominal_translational": (7,),
    }
    for key, value in arrays.items():
        if (
            value.shape != expected_shapes[key]
            or not np.all(np.isfinite(value))
        ):
            raise DiagnosticValidationError(
                f"{label}: invalid QP source input {key}"
            )
    if float(np.linalg.norm(arrays["z_before"])) <= 1e-12:
        raise DiagnosticValidationError(
            f"{label}: degenerate QP source direction"
        )
    v_ref = arrays["R1"].T @ arrays["nominal_translational"][:3]
    u_v_reference = 5.0 * v_ref
    _assert_numeric_close(
        context["v_ref"],
        v_ref,
        label=f"{label}.v_ref",
    )
    _assert_numeric_close(
        context["u_v_reference"],
        u_v_reference,
        label=f"{label}.u_v_reference",
    )
    return arrays


def _validate_prepared_qp_context(
    context: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    required = {
        "solver",
        "u_z_reference",
        "reference",
        "weights_diagonal",
        "cbf",
    }
    if not required.issubset(context) or context.get("solver") != "OSQP":
        raise DiagnosticValidationError(
            f"{label}: incomplete prepared QP context"
        )
    _validate_qp_core_inputs(context, label=label)
    cbf = context.get("cbf")
    cbf_required = {
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
    }
    if not isinstance(cbf, Mapping) or not cbf_required.issubset(cbf):
        raise DiagnosticValidationError(
            f"{label}: incomplete prepared CBF context"
        )
    reconstruction_input = dict(context)
    reconstruction_input["u_solution"] = [0.0] * 6
    try:
        reconstructed = _recompute_qp_context(reconstruction_input)
    except DiagnosticValidationError:
        raise
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: prepared QP reconstruction failed"
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
            label=f"{label}.cbf.{key}",
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
            label=f"{label}.{key}",
        )
    _assert_numeric_close(
        cbf["reference_lhs"],
        reconstructed["reference_lhs"],
        label=f"{label}.reference_lhs",
    )
    _assert_numeric_close(
        cbf["reference_slack"],
        reconstructed["reference_lhs"],
        label=f"{label}.reference_slack",
    )
    _assert_numeric_close(
        cbf["reference_violation"],
        max(0.0, -reconstructed["reference_lhs"]),
        label=f"{label}.reference_violation",
    )
    if cbf.get("alpha_gain") != 10.0:
        raise DiagnosticValidationError(
            f"{label}: CBF alpha gain changed"
        )
    return reconstructed


def _require_typed_failure(
    context: Mapping[str, Any],
    *,
    label: str,
) -> None:
    failure = context.get("failure")
    if (
        not isinstance(failure, Mapping)
        or not isinstance(failure.get("type"), str)
        or not failure.get("type")
        or not isinstance(failure.get("message"), str)
    ):
        raise DiagnosticValidationError(
            f"{label}: missing typed exception evidence"
        )


def _require_solver_evidence(
    context: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if (
        not isinstance(context.get("solver_status"), str)
        or not context.get("solver_status")
        or not isinstance(context.get("solver_stats"), Mapping)
        or not isinstance(
            context.get("solver_stats", {}).get("available"), bool
        )
    ):
        raise DiagnosticValidationError(
            f"{label}: missing solver termination evidence"
        )


def _validate_invalid_solution_evidence(
    context: Mapping[str, Any],
    *,
    label: str,
) -> None:
    import numpy as np

    descriptor = context.get("solution_descriptor")
    raw_hex = context.get("solution_raw_bytes_hex")
    if (
        not isinstance(descriptor, Mapping)
        or not isinstance(raw_hex, str)
        or descriptor.get("shape") != context.get("solution_shape")
    ):
        raise DiagnosticValidationError(
            f"{label}: incomplete invalid-solution evidence"
        )
    try:
        dtype = np.dtype(descriptor["dtype"])
        raw = bytes.fromhex(raw_hex)
        solution = np.frombuffer(raw, dtype=dtype).copy()
        solution = solution.reshape(tuple(descriptor["shape"]))
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: invalid solution bytes cannot be reconstructed"
        ) from error
    if diagnostics.array_descriptor(solution) != dict(descriptor):
        raise DiagnosticValidationError(
            f"{label}: invalid solution descriptor/hash mismatch"
        )
    if diagnostics._json_safe(solution.tolist()) != context.get(
        "solution_values"
    ):
        raise DiagnosticValidationError(
            f"{label}: invalid solution values mismatch"
        )
    if solution.shape == (6,) and bool(np.all(np.isfinite(solution))):
        raise DiagnosticValidationError(
            f"{label}: recorded solution is not invalid"
        )


def _validate_qp_controller_binding(
    context: Mapping[str, Any],
    controller_binding: Mapping[str, Any],
    *,
    label: str,
    bind_initial_direction: bool,
) -> None:
    for key in ("p1", "R1", "q1_diag", "p2", "R2", "Q2_diag"):
        if key not in controller_binding:
            raise DiagnosticValidationError(
                f"{label}: controller binding lacks {key}"
            )
        _assert_numeric_close(
            context.get(key),
            controller_binding[key],
            label=f"{label}.controller_binding.{key}",
        )
    if bind_initial_direction:
        _assert_numeric_close(
            context.get("z_before"),
            controller_binding.get("z_initial"),
            label=f"{label}.controller_binding.z_initial",
        )


def _validate_post_step_controller_proxy(
    action: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    import numpy as np

    record = action.get("post_step_controller_proxy")
    if not isinstance(record, Mapping):
        raise DiagnosticValidationError(
            f"{label}: post-step controller proxy is missing"
        )
    try:
        position = np.asarray(record["eef_position"], dtype=float)
        quaternion = np.asarray(
            record["eef_quaternion_xyzw"], dtype=float
        )
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: post-step controller observation is invalid"
        ) from error
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
    ):
        raise DiagnosticValidationError(
            f"{label}: post-step controller observation shape changed"
        )
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-12:
        raise DiagnosticValidationError(
            f"{label}: post-step quaternion is degenerate"
        )
    x, y, z, w = quaternion / quaternion_norm
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )
    center = position + rotation @ np.asarray([0.0, 0.0, -0.08])
    _assert_numeric_close(
        record.get("R1"),
        rotation,
        label=f"{label}.post_step.R1",
    )
    _assert_numeric_close(
        record.get("p1"),
        center,
        label=f"{label}.post_step.p1",
    )
    return {"p1": center, "R1": rotation}


def _validate_nonfinite_qp_context(
    context: Mapping[str, Any],
    *,
    label: str,
) -> None:
    import numpy as np

    _require_solver_evidence(context, label=label)
    required = {
        "solver",
        "u_z_reference",
        "reference",
        "weights_diagonal",
        "cbf",
        "u_solution",
        "solution_lhs",
        "z_after",
        "executed_action",
        "nonfinite_values",
    }
    if not required.issubset(context) or context.get("solver") != "OSQP":
        raise DiagnosticValidationError(
            f"{label}: incomplete non-finite QP context"
        )
    cbf = context.get("cbf")
    if not isinstance(cbf, Mapping):
        raise DiagnosticValidationError(
            f"{label}: non-finite CBF context is absent"
        )
    solution = np.asarray(context.get("u_solution"), dtype=float)
    if solution.shape != (6,) or not np.all(np.isfinite(solution)):
        raise DiagnosticValidationError(
            f"{label}: non-finite failure has no finite QP solution"
        )
    try:
        reconstructed = _recompute_qp_context(context)
    except DiagnosticValidationError:
        raise
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: non-finite QP reconstruction failed"
        ) from error
    evidence = context.get("nonfinite_values")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "barrier_h",
        "solution_lhs",
    }:
        raise DiagnosticValidationError(
            f"{label}: non-finite fields are not exact"
        )
    _assert_numeric_equivalent(
        evidence["barrier_h"],
        reconstructed["h"],
        label=f"{label}.nonfinite.barrier_h",
    )
    _assert_numeric_equivalent(
        evidence["solution_lhs"],
        reconstructed["solution_lhs"],
        label=f"{label}.nonfinite.solution_lhs",
    )
    if np.all(
        np.isfinite(
            [
                float(evidence["barrier_h"]),
                float(evidence["solution_lhs"]),
            ]
        )
    ):
        raise DiagnosticValidationError(
            f"{label}: non-finite failure predicate is false"
        )
    for key in (
        "a_v",
        "a_omega",
        "a_u_v",
        "a_u_z",
        "mu_row",
        "h",
        "constant",
    ):
        _assert_numeric_equivalent(
            cbf.get(key),
            reconstructed[key],
            label=f"{label}.cbf.{key}",
        )
    for key in (
        "v_ref",
        "u_v_reference",
        "u_z_reference",
        "reference",
        "weights_diagonal",
        "z_after",
        "executed_action",
    ):
        expected = (
            reconstructed["executed"]
            if key == "executed_action"
            else reconstructed[key]
        )
        _assert_numeric_equivalent(
            context.get(key),
            expected,
            label=f"{label}.{key}",
        )
    _assert_numeric_equivalent(
        cbf.get("reference_lhs"),
        reconstructed["reference_lhs"],
        label=f"{label}.reference_lhs",
    )
    _assert_numeric_equivalent(
        context.get("solution_lhs"),
        reconstructed["solution_lhs"],
        label=f"{label}.solution_lhs",
    )
    if cbf.get("alpha_gain") != 10.0:
        raise DiagnosticValidationError(
            f"{label}: CBF alpha gain changed"
        )


def _validate_terminal_qp_failure(
    result: Mapping[str, Any],
    method_failure: Mapping[str, Any],
    *,
    previous_z_after: Any,
    controller_binding: Mapping[str, Any] | None,
    require_controller_binding: bool,
) -> None:
    import math
    import numpy as np

    action_count = len(result.get("actions", []))
    label = f"terminal QP failure at step {method_failure.get('step')}"
    if (
        method_failure.get("status") != "method_failure"
        or method_failure.get("phase") != "control"
        or method_failure.get("step") != action_count
        or method_failure.get("safety_by_no_execution")
        != (action_count == 0)
        or not isinstance(method_failure.get("type"), str)
        or not method_failure.get("type")
        or not isinstance(method_failure.get("message"), str)
        or not method_failure.get("message")
    ):
        raise DiagnosticValidationError(
            f"{label}: terminal/action-step binding changed"
        )
    try:
        nominal_raw = np.asarray(
            method_failure["nominal_raw"], dtype=float
        )
        nominal = np.asarray(
            method_failure["nominal_translational"], dtype=float
        )
    except Exception as error:
        raise DiagnosticValidationError(
            f"{label}: terminal nominal action is invalid"
        ) from error
    if (
        nominal_raw.shape != (7,)
        or nominal.shape != (7,)
        or not np.all(np.isfinite(nominal_raw))
        or not np.all(np.isfinite(nominal))
    ):
        raise DiagnosticValidationError(
            f"{label}: terminal nominal action shape/finiteness changed"
        )
    expected_nominal = np.zeros(7, dtype=float)
    expected_nominal[:3] = nominal_raw[:3]
    expected_nominal[6] = nominal_raw[6]
    _assert_numeric_close(
        nominal,
        expected_nominal,
        label=f"{label}.nominal_translational",
    )

    context = method_failure.get("diagnostics")
    if not isinstance(context, Mapping):
        raise DiagnosticValidationError(
            f"{label}: diagnostics payload is absent"
        )
    expected_payload_hash = diagnostics.sha256_bytes(
        diagnostics.canonical_json_bytes(context)
    )
    if method_failure.get("diagnostics_payload_sha256") != (
        expected_payload_hash
    ):
        raise DiagnosticValidationError(
            f"{label}: diagnostics payload hash mismatch"
        )
    failure_type = context.get("failure_type")
    allowed = {
        "cbf_coefficient_exception",
        "qp_construction_exception",
        "solver_exception",
        "no_solution",
        "invalid_solution",
        "degenerate_virtual_direction",
        "nonfinite_diagnostics",
    }
    if context.get("status") != "failure" or failure_type not in allowed:
        raise DiagnosticValidationError(
            f"{label}: terminal QP failure is not typed"
        )
    core = _validate_qp_core_inputs(context, label=label)
    if require_controller_binding and controller_binding is None:
        raise DiagnosticValidationError(
            f"{label}: immutable controller/geometry binding is missing"
        )
    if controller_binding is not None:
        _validate_qp_controller_binding(
            context,
            controller_binding,
            label=label,
            bind_initial_direction=previous_z_after is None,
        )
    _assert_numeric_close(
        context["nominal_translational"],
        nominal,
        label=f"{label}.nominal_binding",
    )
    if previous_z_after is not None:
        _assert_numeric_close(
            context["z_before"],
            previous_z_after,
            label=f"{label}.cross_step.z_before",
        )

    if failure_type == "cbf_coefficient_exception":
        _require_typed_failure(context, label=label)
        return
    if failure_type == "nonfinite_diagnostics":
        _validate_nonfinite_qp_context(context, label=label)
        return

    reconstructed = _validate_prepared_qp_context(
        context,
        label=label,
    )
    if failure_type == "qp_construction_exception":
        _require_typed_failure(context, label=label)
        return
    _require_solver_evidence(context, label=label)
    if failure_type == "solver_exception":
        _require_typed_failure(context, label=label)
        cause = context.get("failure", {}).get("cause")
        if (
            not isinstance(cause, Mapping)
            or not isinstance(cause.get("type"), str)
            or not cause.get("type")
            or not isinstance(cause.get("message"), str)
        ):
            raise DiagnosticValidationError(
                f"{label}: solver exception cause is missing"
            )
        return
    if failure_type == "no_solution":
        observation = context.get("solution_observation")
        if (
            not isinstance(observation, Mapping)
            or observation.get("variable_value_is_none") is not True
            or observation.get("variable_shape") != [6]
            or "problem_value" not in observation
            or context.get("solver_status", "").lower()
            in {"optimal", "optimal_inaccurate"}
        ):
            raise DiagnosticValidationError(
                f"{label}: no-solution predicate was not observed"
            )
        return
    if failure_type == "invalid_solution":
        _validate_invalid_solution_evidence(context, label=label)
        return

    solution = np.asarray(context.get("u_solution"), dtype=float)
    if solution.shape != (6,) or not np.all(np.isfinite(solution)):
        raise DiagnosticValidationError(
            f"{label}: invalid terminal QP solution"
        )
    z_before = core["z_before"]
    next_z = (
        z_before
        + (np.eye(3) - np.outer(z_before, z_before))
        @ solution[3:]
        * 0.05
    )
    next_z_norm = float(np.linalg.norm(next_z))
    if failure_type == "degenerate_virtual_direction":
        if math.isfinite(next_z_norm) and next_z_norm > 1e-12:
            raise DiagnosticValidationError(
                f"{label}: degenerate-direction predicate is false"
            )
        return
    raise DiagnosticValidationError(
        f"{label}: unsupported terminal failure semantics"
    )


def _validate_qp_contexts(
    result: Mapping[str, Any],
    *,
    controller_binding: Mapping[str, Any] | None = None,
    require_controller_binding: bool = False,
) -> None:
    if result.get("mode") != "aegis":
        return
    active_controller_binding = (
        None
        if controller_binding is None
        else dict(controller_binding)
    )
    previous_z_after = None
    for action in result.get("actions", []):
        if action.get("control_path") != "aegis_qp":
            if active_controller_binding is not None:
                raise DiagnosticValidationError(
                    f"step {action.get('step')}: complete AEGIS geometry "
                    "did not execute the released QP path"
                )
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
        if require_controller_binding and active_controller_binding is None:
            raise DiagnosticValidationError(
                f"step {action.get('step')}: immutable controller/geometry "
                "binding is missing"
            )
        if active_controller_binding is not None:
            _validate_qp_controller_binding(
                context,
                active_controller_binding,
                label=f"step {action.get('step')}",
                bind_initial_direction=previous_z_after is None,
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
        if active_controller_binding is not None:
            post_step_proxy = _validate_post_step_controller_proxy(
                action,
                label=f"step {action.get('step')}",
            )
            active_controller_binding.update(post_step_proxy)
    method_failure = result.get("method_failure")
    if result.get("status") == "method_failure_passthrough":
        if (
            not isinstance(method_failure, Mapping)
            or method_failure.get("component") != "aegis_perception"
            or method_failure.get("status")
            != "method_failure_passthrough"
            or method_failure.get("corrected_execution")
            != "corrected_translational_nominal"
            or method_failure.get("phase") is not None
        ):
            raise DiagnosticValidationError(
                "perception fail-open method record changed"
            )
        return
    terminal_method_failure = (
        result.get("status") == "method_failure"
        or result.get("terminal_reason") == "method_failure"
        or isinstance(method_failure, Mapping)
    )
    if terminal_method_failure and not isinstance(method_failure, Mapping):
        raise DiagnosticValidationError(
            "terminal method failure record is missing"
        )
    if terminal_method_failure and isinstance(method_failure, Mapping):
        if (
            result.get("status") != "method_failure"
            or result.get("terminal_reason") != "method_failure"
        ):
            raise DiagnosticValidationError(
                "terminal method-failure status/reason changed"
            )
        terminal_tuple = (
            method_failure.get("component"),
            method_failure.get("phase"),
        )
        if terminal_tuple == ("aegis_geometry", "precontrol"):
            if (
                method_failure.get("step") != 0
                or method_failure.get("safety_by_no_execution") is not True
                or result.get("actions")
                or controller_binding is not None
            ):
                raise DiagnosticValidationError(
                    "precontrol geometry failure binding changed"
                )
            return
        if terminal_tuple != ("aegis_qp", "control"):
            raise DiagnosticValidationError(
                "terminal AEGIS method-failure component/phase is invalid"
            )
    if isinstance(method_failure, Mapping) and (
        method_failure.get("component"),
        method_failure.get("phase"),
    ) == ("aegis_qp", "control"):
        _validate_terminal_qp_failure(
            result,
            method_failure,
            previous_z_after=previous_z_after,
            controller_binding=active_controller_binding,
            require_controller_binding=require_controller_binding,
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
    settled_contract = _validated_settled_contract(result)
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
    action_count = int(ledger["action_count"])
    terminal_frame_descriptor = record.get("terminal_frame")
    if not isinstance(terminal_frame_descriptor, Mapping):
        raise DiagnosticValidationError(
            "terminal lossless frame artifact is missing"
        )
    expected_terminal_path = (
        Path(str(result.get("mode")))
        / str(result.get("case_id"))
        / "terminal_agentview.npy"
    )
    if Path(str(terminal_frame_descriptor.get("path"))) != (
        expected_terminal_path
    ):
        raise DiagnosticValidationError(
            "terminal lossless frame path/case binding changed"
        )
    terminal_frame, terminal_frame_array = (
        _validate_terminal_frame_artifact(
            terminal_frame_descriptor,
            output_root=output_root,
        )
    )
    final_goal_state_sha256 = _validate_goal_progress(
        result,
        action_count=action_count,
    )
    video_decode = _validate_terminal_and_video(
        result,
        output_root=output_root,
        terminal_frame=terminal_frame,
        terminal_frame_array=terminal_frame_array,
        action_count=action_count,
        final_goal_state_sha256=final_goal_state_sha256,
    )
    geometry = record.get("geometry")
    controller_binding: dict[str, Any] | None = None
    if result.get("mode") == "aegis":
        if not isinstance(geometry, Mapping):
            raise DiagnosticValidationError("AEGIS geometry is missing")
        geometry_binding = _validate_geometry(
            geometry,
            output_root=output_root,
            require_ready=require_ready_geometry,
            expected_case_id=str(result.get("case_id")),
            expected_suite=str(result.get("suite")),
            expected_label=str(
                result.get("pairing", {}).get(
                    "semantic_obstacle_label"
                )
            ),
            settled_contract=settled_contract,
        )
        if geometry_binding is not None:
            prompt = settled_contract.get("prompt")
            if not isinstance(prompt, str):
                raise DiagnosticValidationError(
                    "task prompt is missing from the settled-input contract"
                )
            controller_binding = dict(geometry_binding)
            controller_binding["q1_diag"] = (
                [0.06, 0.12, 0.2]
                if any(
                    token in prompt
                    for token in (
                        "orange juice",
                        "milk",
                        "alphabet soup",
                    )
                )
                else [0.06, 0.12, 0.11]
            )
    contacts = record.get("contacts")
    if not isinstance(contacts, Mapping):
        raise DiagnosticValidationError("contact artifact is missing")
    _validate_contacts(
        contacts,
        output_root=output_root,
        action_count=action_count,
        expected_case_id=str(result.get("case_id")),
    )
    _validate_qp_contexts(
        result,
        controller_binding=controller_binding,
        require_controller_binding=result.get("mode") == "aegis",
    )
    return {
        "case_id": result.get("case_id"),
        "arm": result.get("arm"),
        "status": "valid",
        "action_invariance_ledger": ledger,
        "video_decode": video_decode,
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
