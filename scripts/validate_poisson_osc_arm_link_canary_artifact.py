#!/usr/bin/env python3
"""Independent complete-only consumer for the post-OSC arm-link canary."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts import run_poisson_fast_feasibility as fast


class OscCanaryValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OscCanaryValidationError(message)


def _canonical_sha256(value: Any) -> str:
    return fast._sha256(fast._canonical(value))


def _float64_sha256(value: Any, np: Any) -> str:
    array = np.asarray(value, dtype=np.float64)
    _require(np.all(np.isfinite(array)), "array hash input is non-finite")
    contiguous = np.ascontiguousarray(array)
    header = fast._canonical(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _float64_vector7_bytes_and_sha256(value: Any) -> Tuple[bytes, str]:
    """Validate and hash one 7D vector without an allocation-only NumPy import."""

    _require(
        isinstance(value, (list, tuple)) and len(value) == 7,
        "7D float64 vector is malformed",
    )
    numbers = []
    for item in value:
        _require(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item)),
            "7D float64 vector is non-finite",
        )
        numbers.append(float(item))
    endian = "<" if sys.byteorder == "little" else ">"
    payload = struct.pack(endian + "7d", *numbers)
    header = fast._canonical({"dtype": endian + "f8", "shape": [7]})
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(payload)
    return payload, digest.hexdigest()


def _validated_sample_rows(
    evidence: Mapping[str, Any],
    *,
    expected_geom_ids: Sequence[int],
    expected_body_ids: Sequence[int],
    expected_hash: str,
    label: str,
    require_declared_count: bool = True,
) -> Tuple[Dict[int, int], int]:
    """Reconstruct a serialized local-frame sample ledger and its identity."""

    rows = evidence.get("samples")
    _require(isinstance(rows, list) and rows, "%s raw samples are absent" % label)
    _require(_canonical_sha256(rows) == expected_hash, "%s sample hash differs" % label)
    geom_ids = {int(value) for value in expected_geom_ids}
    body_ids = {int(value) for value in expected_body_ids}
    counts: Dict[int, int] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), "%s sample row is invalid" % label)
        point = row.get("point_body_local_m")
        _require(
            row.get("sample_id") == index
            and int(row.get("geom_id", -1)) in geom_ids
            and int(row.get("body_id", -1)) in body_ids
            and isinstance(row.get("geom_name"), str)
            and bool(row.get("geom_name"))
            and isinstance(row.get("body_name"), str)
            and bool(row.get("body_name"))
            and row.get("source") == "collision_geom_surface"
            and isinstance(point, list)
            and len(point) == 3
            and all(math.isfinite(float(value)) for value in point),
            "%s sample provenance differs at row %d" % (label, index),
        )
        geom_id = int(row["geom_id"])
        counts[geom_id] = counts.get(geom_id, 0) + 1
    if require_declared_count:
        _require(
            int(evidence.get("sample_count", -1)) == len(rows),
            "%s sample count differs" % label,
        )
    return counts, len(rows)


def _strict_integer_list(value: Any, label: str, *, nonempty: bool = True) -> List[int]:
    """Parse an ordered JSON integer ledger without accepting bools or coercions."""

    _require(isinstance(value, list), "%s must be a JSON list" % label)
    output: List[int] = []
    for index, item in enumerate(value):
        _require(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0,
            "%s row %d is not a nonnegative integer" % (label, index),
        )
        output.append(int(item))
    _require(not nonempty or bool(output), "%s is empty" % label)
    return output


def _validate_compact_array_identity(
    evidence: Any,
    *,
    expected_shape: Sequence[int],
    label: str,
    np: Any,
    raw_array: Any = None,
) -> None:
    """Validate one compact float64 record, optionally against raw values."""

    _require(isinstance(evidence, Mapping), "%s record is absent" % label)
    _require(
        set(evidence) == {"dtype", "shape", "sha256", "minimum"}
        and evidence.get("dtype") == np.dtype(np.float64).str
        and evidence.get("shape") == [int(value) for value in expected_shape]
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))))
        and not isinstance(evidence.get("minimum"), bool)
        and isinstance(evidence.get("minimum"), (int, float))
        and math.isfinite(float(evidence.get("minimum"))),
        "%s record differs" % label,
    )
    if raw_array is not None:
        array = np.asarray(raw_array, dtype=np.float64)
        _require(
            array.shape == tuple(int(value) for value in expected_shape)
            and np.all(np.isfinite(array))
            and evidence.get("sha256") == _float64_sha256(array, np),
            "%s raw-array identity differs" % label,
        )
        _exact_float(evidence.get("minimum"), float(np.min(array)), label + " minimum")


def _compact_zero_float64_array_record(shape: Sequence[int]) -> Dict[str, Any]:
    """Construct the canonical compact identity of an all-zero float64 array."""

    dimensions = [int(value) for value in shape]
    _require(
        dimensions and all(value > 0 for value in dimensions),
        "zero-array shape is invalid",
    )
    dtype = "<f8" if sys.byteorder == "little" else ">f8"
    header = fast._canonical({"dtype": dtype, "shape": dimensions})
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(b"\0" * (8 * math.prod(dimensions)))
    return {
        "dtype": dtype,
        "shape": dimensions,
        "sha256": digest.hexdigest(),
        "minimum": 0.0,
    }


def _validate_compact_array_header(
    evidence: Any, *, expected_shape: Sequence[int], label: str
) -> None:
    """Validate compact array metadata when raw producer values are unavailable."""

    expected_dtype = "<f8" if sys.byteorder == "little" else ">f8"
    _require(
        isinstance(evidence, Mapping)
        and set(evidence) == {"dtype", "shape", "sha256", "minimum"}
        and evidence.get("dtype") == expected_dtype
        and evidence.get("shape") == [int(value) for value in expected_shape]
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))))
        and not isinstance(evidence.get("minimum"), bool)
        and isinstance(evidence.get("minimum"), (int, float))
        and math.isfinite(float(evidence.get("minimum"))),
        "%s record differs" % label,
    )


def _validate_structural_robot_geom_partition(
    *,
    apparatus: Mapping[str, Any],
    resolved_geometry: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reconstruct the v3 movable/fixed robot-geometry partition.

    This consumer deliberately does not trust the producer's classification
    booleans.  It reconstructs each geom's influencing-qvel set from the
    serialized qvel joint-body ownership and complete body-ancestry ledgers.
    The clean producer commit remains authority for the compiled MuJoCo model,
    while this check prevents an internally inconsistent, name-selected, or
    contact-unmonitored partition from passing the independent consumer.

    Expected v3 apparatus records are intentionally explicit:

    * ``robot_geom_influence_partition`` is the exact output schema of
      ``partition_robot_geoms_by_qvel_influence``;
    * ``movable_manipulator_sampling`` and ``fixed_infrastructure_sampling``
      are the stable, reindexed subsequences of ``all_robot_sampling.samples``
      selected by the two reconstructed geom-ID sets; and
    * ``fixed_infrastructure_settled_certificate`` binds the complementary
      full-ledger sample IDs, an exact-zero settled Jacobian certificate, the
      complete contact-monitor scope identity, and an empty settled contact
      ledger.
    """

    partition = apparatus.get("robot_geom_influence_partition")
    movable = apparatus.get("movable_manipulator_sampling")
    fixed_sampling = apparatus.get("fixed_infrastructure_sampling")
    fixed = apparatus.get("fixed_infrastructure_settled_certificate")
    full = apparatus.get("all_robot_sampling")
    binding = apparatus.get("shield_sampling_binding")
    scope = apparatus.get("registered_contact_scope")
    _require(isinstance(partition, Mapping), "robot geom influence partition is absent")
    _require(isinstance(movable, Mapping), "movable manipulator sampling is absent")
    _require(isinstance(fixed_sampling, Mapping), "fixed infrastructure sampling is absent")
    _require(isinstance(fixed, Mapping), "fixed infrastructure certificate is absent")
    _require(isinstance(full, Mapping), "all-robot sampling evidence is absent")
    _require(isinstance(binding, Mapping), "shield sampling binding is absent")
    _require(isinstance(scope, Mapping), "registered contact scope is absent")
    partition_sha256 = _canonical_sha256(partition)
    _require(
        apparatus.get("robot_geom_influence_partition_sha256")
        == partition_sha256,
        "robot geom influence partition hash differs",
    )
    _require(
        apparatus.get("movable_manipulator_sampling_sha256")
        == _canonical_sha256(movable),
        "movable manipulator sampling hash differs",
    )
    _require(
        apparatus.get("fixed_infrastructure_sampling_sha256")
        == _canonical_sha256(fixed_sampling),
        "fixed infrastructure sampling hash differs",
    )
    _require(
        apparatus.get("all_robot_sampling_sha256") == _canonical_sha256(full),
        "all-robot sampling hash differs",
    )
    _require(
        apparatus.get("fixed_infrastructure_settled_certificate_sha256")
        == _canonical_sha256(fixed),
        "fixed infrastructure certificate hash differs",
    )

    resolved_geom_ids = _strict_integer_list(
        resolved_geometry.get("robot_geom_ids"), "resolved robot geom IDs"
    )
    resolved_geom_names = resolved_geometry.get("robot_geom_names")
    resolved_body_ids = _strict_integer_list(
        resolved_geometry.get("robot_body_ids"), "resolved robot body IDs"
    )
    resolved_root_body_ids = _strict_integer_list(
        resolved_geometry.get("robot_root_body_ids"),
        "resolved robot root body IDs",
    )
    resolved_body_names = resolved_geometry.get("robot_body_names")
    link56_geom_ids = _strict_integer_list(
        resolved_geometry.get("link56_geom_ids"), "resolved link56 geom IDs"
    )
    _require(
        isinstance(resolved_geom_names, list)
        and len(resolved_geom_names) == len(resolved_geom_ids)
        and all(isinstance(value, str) and value for value in resolved_geom_names)
        and isinstance(resolved_body_names, list)
        and len(resolved_body_names) == len(resolved_body_ids)
        and all(isinstance(value, str) and value for value in resolved_body_names),
        "resolved robot ID/name ledgers differ",
    )
    _require(
        resolved_root_body_ids == sorted(set(resolved_root_body_ids))
        and set(resolved_root_body_ids).issubset(resolved_body_ids),
        "resolved robot roots are outside the robot body ledger",
    )
    geom_name_by_id = dict(zip(resolved_geom_ids, resolved_geom_names))
    body_name_by_id = dict(zip(resolved_body_ids, resolved_body_names))

    expected_partition_fields = {
        "schema_version",
        "selection_rule",
        "contact_monitor_robot_geom_ids",
        "shield_manipulator_geom_ids",
        "fixed_robot_infrastructure_geom_ids",
        "robot_qvel_indices",
        "robot_qvel_joint_body_ids",
        "robot_body_parent_records",
        "geom_records",
        "all_contact_geoms_partitioned",
        "all_fixed_geoms_have_zero_structural_qvel_influence",
    }
    _require(
        set(partition) == expected_partition_fields
        and partition.get("schema_version")
        == "vlsa_poisson_robot_geom_influence_partition.v1"
        and partition.get("selection_rule")
        == (
            "geom_body_self_or_ancestor_owns_at_least_one_authoritative_"
            "robot_tree_qvel"
        ),
        "robot geom influence partition schema or selection rule differs",
    )
    monitored_geom_ids = _strict_integer_list(
        partition.get("contact_monitor_robot_geom_ids"),
        "partition contact-monitor robot geom IDs",
    )
    movable_geom_ids = _strict_integer_list(
        partition.get("shield_manipulator_geom_ids"),
        "partition movable manipulator geom IDs",
    )
    fixed_geom_ids = _strict_integer_list(
        partition.get("fixed_robot_infrastructure_geom_ids"),
        "partition fixed infrastructure geom IDs",
    )
    qvel_indices = _strict_integer_list(
        partition.get("robot_qvel_indices"), "partition robot qvel indices"
    )
    qvel_joint_body_ids = _strict_integer_list(
        partition.get("robot_qvel_joint_body_ids"),
        "partition qvel joint-body IDs",
    )
    _require(
        monitored_geom_ids == resolved_geom_ids
        and movable_geom_ids == sorted(movable_geom_ids)
        and fixed_geom_ids == sorted(fixed_geom_ids)
        and not set(movable_geom_ids) & set(fixed_geom_ids)
        and set(movable_geom_ids) | set(fixed_geom_ids) == set(resolved_geom_ids)
        and set(link56_geom_ids).issubset(movable_geom_ids)
        and qvel_indices
        == _strict_integer_list(binding.get("robot_qvel_indices"), "bound robot qvel indices")
        and len(qvel_joint_body_ids) == len(qvel_indices),
        "robot geom influence partition is incomplete or link56 is unshielded",
    )
    qvel_records = binding.get("robot_qvel_records")
    _require(
        isinstance(qvel_records, list)
        and len(qvel_records) == len(qvel_indices)
        and [record.get("qvel_index") for record in qvel_records] == qvel_indices
        and [record.get("joint_body_id") for record in qvel_records]
        == qvel_joint_body_ids
        and all(body_id in body_name_by_id for body_id in qvel_joint_body_ids),
        "partition qvel joint-body ownership differs from shield authority",
    )

    parent_records = partition.get("robot_body_parent_records")
    _require(
        isinstance(parent_records, list)
        and len(parent_records) == len(resolved_body_ids),
        "robot body parent ledger differs",
    )
    ancestry_by_body: Dict[int, List[int]] = {}
    parent_by_body: Dict[int, int] = {}
    expected_parent_fields = {
        "body_id",
        "body_name",
        "parent_body_id",
        "parent_body_name",
        "body_ancestry_ids",
    }
    for expected_body_id, record in zip(resolved_body_ids, parent_records):
        _require(isinstance(record, Mapping), "robot body parent row is invalid")
        ancestry = _strict_integer_list(
            record.get("body_ancestry_ids"), "robot body ancestry"
        )
        parent_id = record.get("parent_body_id")
        _require(
            set(record) == expected_parent_fields
            and record.get("body_id") == expected_body_id
            and record.get("body_name") == body_name_by_id[expected_body_id]
            and isinstance(parent_id, int)
            and not isinstance(parent_id, bool)
            and parent_id >= 0
            and isinstance(record.get("parent_body_name"), str)
            and bool(record.get("parent_body_name"))
            and ancestry[0] == expected_body_id
            and len(ancestry) == len(set(ancestry))
            and parent_id == (ancestry[1] if len(ancestry) > 1 else 0)
            and (parent_id != 0 or record.get("parent_body_name") == "world"),
            "robot body ancestry row differs for body %d" % expected_body_id,
        )
        ancestry_by_body[expected_body_id] = ancestry
        parent_by_body[expected_body_id] = int(parent_id)
    for body_id, ancestry in ancestry_by_body.items():
        for position, ancestor in enumerate(ancestry[:-1]):
            if ancestor in parent_by_body:
                _require(
                    parent_by_body[ancestor] == ancestry[position + 1],
                    "robot body ancestry chains disagree",
                )
    _require(
        all(
            parent_by_body[root_body_id] == 0
            and ancestry_by_body[root_body_id] == [root_body_id]
            for root_body_id in resolved_root_body_ids
        ),
        "resolved robot roots are not world-mounted",
    )

    geom_records = partition.get("geom_records")
    _require(
        isinstance(geom_records, list)
        and len(geom_records) == len(resolved_geom_ids),
        "robot geom influence record ledger differs",
    )
    expected_geom_record_fields = {
        "geom_id",
        "geom_name",
        "body_id",
        "body_name",
        "body_ancestry_ids",
        "influencing_robot_qvel_indices",
        "classification",
    }
    reconstructed_movable: List[int] = []
    reconstructed_fixed: List[int] = []
    sample_body_by_geom: Dict[int, int] = {}
    full_samples = full.get("samples")
    _require(isinstance(full_samples, list) and full_samples, "full-robot samples are absent")
    for sample in full_samples:
        _require(isinstance(sample, Mapping), "full-robot sample row is invalid")
        geom_id = sample.get("geom_id")
        body_id = sample.get("body_id")
        _require(
            isinstance(geom_id, int)
            and not isinstance(geom_id, bool)
            and geom_id in geom_name_by_id
            and isinstance(body_id, int)
            and not isinstance(body_id, bool)
            and body_id in body_name_by_id,
            "full-robot sample identity is outside resolved geometry",
        )
        if geom_id in sample_body_by_geom:
            _require(
                sample_body_by_geom[geom_id] == body_id,
                "one robot geom appears under multiple bodies",
            )
        sample_body_by_geom[int(geom_id)] = int(body_id)
    _require(
        set(sample_body_by_geom) == set(resolved_geom_ids),
        "full-robot samples omit an authoritative geom",
    )
    for expected_geom_id, record in zip(resolved_geom_ids, geom_records):
        _require(isinstance(record, Mapping), "robot geom influence row is invalid")
        body_id = record.get("body_id")
        influence = _strict_integer_list(
            record.get("influencing_robot_qvel_indices"),
            "geom influencing robot qvel indices",
            nonempty=False,
        )
        _require(
            set(record) == expected_geom_record_fields
            and record.get("geom_id") == expected_geom_id
            and record.get("geom_name") == geom_name_by_id[expected_geom_id]
            and isinstance(body_id, int)
            and not isinstance(body_id, bool)
            and body_id == sample_body_by_geom[expected_geom_id]
            and record.get("body_name") == body_name_by_id.get(body_id)
            and record.get("body_ancestry_ids") == ancestry_by_body.get(body_id),
            "robot geom influence identity differs for geom %d" % expected_geom_id,
        )
        expected_influence = [
            qvel_index
            for qvel_index, joint_body_id in zip(qvel_indices, qvel_joint_body_ids)
            if joint_body_id in ancestry_by_body[int(body_id)]
        ]
        expected_classification = (
            "kinematically_movable_manipulator_surface"
            if expected_influence
            else "kinematically_fixed_robot_infrastructure"
        )
        _require(
            influence == expected_influence
            and record.get("classification") == expected_classification,
            "robot geom structural influence was not reconstructed for geom %d"
            % expected_geom_id,
        )
        (reconstructed_movable if expected_influence else reconstructed_fixed).append(
            expected_geom_id
        )
    _require(
        reconstructed_movable == movable_geom_ids
        and reconstructed_fixed == fixed_geom_ids
        and partition.get("all_contact_geoms_partitioned") is True
        and partition.get("all_fixed_geoms_have_zero_structural_qvel_influence")
        is True,
        "producer robot geom partition flags differ from reconstructed partition",
    )

    surface_fields = {
        "sample_count",
        "sample_ledger_sha256",
        "samples",
        "geom_records",
        "epsilon_m",
        "maximum_surface_cover_radius_m",
        "coverage_semantics",
        "roundtrip",
    }

    def validate_surface_subset(
        evidence: Mapping[str, Any], geom_ids: Sequence[int], label: str
    ) -> List[Mapping[str, Any]]:
        expected_samples = []
        geom_id_set = set(geom_ids)
        for sample in full_samples:
            if int(sample["geom_id"]) in geom_id_set:
                row = dict(sample)
                row["sample_id"] = len(expected_samples)
                expected_samples.append(row)
        samples = evidence.get("samples")
        records = evidence.get("geom_records")
        _require(
            set(evidence) == surface_fields
            and isinstance(samples, list)
            and fast._canonical(samples) == fast._canonical(expected_samples)
            and evidence.get("sample_count") == len(expected_samples)
            and evidence.get("sample_ledger_sha256")
            == _canonical_sha256(expected_samples)
            and isinstance(records, list)
            and [record.get("geom_id") for record in records] == list(geom_ids),
            "%s samples are not the exact reindexed all-robot subsequence" % label,
        )
        return samples

    _require(
        full.get("sample_count") == len(full_samples)
        and full.get("sample_ledger_sha256") == _canonical_sha256(full_samples),
        "all-robot sample count or ledger hash differs",
    )
    expected_movable_samples = validate_surface_subset(
        movable, movable_geom_ids, "movable manipulator"
    )
    expected_fixed_samples = validate_surface_subset(
        fixed_sampling, fixed_geom_ids, "fixed infrastructure"
    )

    expected_fixed_fields = {
        "schema_version",
        "fixed_geom_ids",
        "fixed_geom_ids_sha256",
        "fixed_sample_ids",
        "fixed_sample_ids_sha256",
        "fixed_sample_count",
        "fixed_sample_ledger_sha256",
        "fixed_world_points_array_record",
        "fixed_point_jacobian_array_record",
        "structural_partition_sha256",
        "all_fixed_geoms_zero_structural_qvel_influence",
        "all_fixed_sample_jacobians_exactly_zero",
        "maximum_abs_fixed_sample_jacobian",
        "settled_registered_contact_count",
        "settled_contact_records",
        "settled_contact_free",
        "all_fixed_geoms_contact_monitored",
        "contact_monitor_scope_identity_sha256",
    }
    fixed_sample_ids = _strict_integer_list(
        fixed.get("fixed_sample_ids"), "fixed infrastructure sample IDs"
    )
    expected_fixed_sample_ids = list(range(len(expected_fixed_samples)))
    fixed_contacts = fixed.get("settled_contact_records")
    scope_without_hash = dict(scope)
    scope_identity_sha256 = scope_without_hash.pop("identity_sha256", None)
    _require(
        set(fixed) == expected_fixed_fields
        and fixed.get("schema_version")
        == "vlsa_poisson_fixed_infrastructure_settled_certificate.v1"
        and fixed.get("structural_partition_sha256") == partition_sha256
        and fixed.get("fixed_geom_ids") == fixed_geom_ids
        and fixed.get("fixed_geom_ids_sha256") == _canonical_sha256(fixed_geom_ids)
        and fixed_sample_ids == expected_fixed_sample_ids
        and fixed.get("fixed_sample_ids_sha256")
        == _canonical_sha256(fixed_sample_ids)
        and fixed.get("fixed_sample_count") == len(fixed_sample_ids)
        and fixed.get("fixed_sample_ledger_sha256")
        == fixed_sampling.get("sample_ledger_sha256")
        and fixed.get("all_fixed_geoms_zero_structural_qvel_influence") is True
        and fixed.get("all_fixed_sample_jacobians_exactly_zero") is True
        and not isinstance(fixed.get("maximum_abs_fixed_sample_jacobian"), bool)
        and fixed.get("maximum_abs_fixed_sample_jacobian") == 0.0
        and fixed_contacts == []
        and fixed.get("settled_registered_contact_count") == 0
        and fixed.get("settled_contact_free") is True
        and fixed.get("all_fixed_geoms_contact_monitored") is True
        and scope_identity_sha256 == _canonical_sha256(scope_without_hash)
        and fixed.get("contact_monitor_scope_identity_sha256")
        == scope_identity_sha256,
        "fixed infrastructure zero-influence or settled contact-free certificate differs",
    )
    _validate_compact_array_header(
        fixed.get("fixed_world_points_array_record"),
        expected_shape=(len(expected_fixed_samples), 3),
        label="fixed infrastructure world points",
    )
    expected_zero_jacobian_record = _compact_zero_float64_array_record(
        (len(expected_fixed_samples), 3, len(qvel_indices))
    )
    _require(
        fast._canonical(fixed.get("fixed_point_jacobian_array_record"))
        == fast._canonical(expected_zero_jacobian_record),
        "fixed infrastructure point Jacobian array is not exactly zero",
    )
    scope_robot_geom_ids = _strict_integer_list(
        scope.get("robot_geom_ids"), "registered contact robot geom IDs"
    )
    _require(
        scope_robot_geom_ids == resolved_geom_ids
        and set(fixed_geom_ids).issubset(scope_robot_geom_ids)
        and set(movable_geom_ids).issubset(scope_robot_geom_ids),
        "registered contact scope omits authoritative robot geometry",
    )
    return {
        "all_structurally_movable_manipulator_collision_surfaces_shielded": True,
        "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free": True,
        "all_authoritative_robot_collision_surfaces_contact_monitored": True,
        "movable_geom_ids": movable_geom_ids,
        "fixed_geom_ids": fixed_geom_ids,
        "movable_sample_count": len(expected_movable_samples),
        "fixed_sample_count": len(expected_fixed_samples),
        "partition_sha256": partition_sha256,
    }


def _validate_shield_sampling_binding(
    *,
    apparatus: Mapping[str, Any],
    resolved_geometry: Mapping[str, Any],
    controller: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    np: Any,
) -> Dict[str, Any]:
    """Bind v3 shield constraints to movable samples and all robot qvels."""

    structural = _validate_structural_robot_geom_partition(
        apparatus=apparatus,
        resolved_geometry=resolved_geometry,
    )
    full = apparatus.get("movable_manipulator_sampling")
    binding = apparatus.get("shield_sampling_binding")
    _require(isinstance(full, Mapping), "movable manipulator sampling evidence is absent")
    _require(isinstance(binding, Mapping), "movable shield sampling binding is absent")
    _require(
        apparatus.get("shield_sampling_binding_sha256")
        == _canonical_sha256(binding),
        "movable-manipulator shield sampling binding hash differs",
    )
    expected_fields = {
        "schema_version",
        "scope",
        "sample_source",
        "sample_count",
        "sample_ledger_sha256",
        "resolved_robot_geom_ids",
        "resolved_robot_geom_ids_sha256",
        "shield_manipulator_geom_ids",
        "shield_manipulator_geom_ids_sha256",
        "fixed_robot_infrastructure_geom_ids",
        "fixed_robot_infrastructure_geom_ids_sha256",
        "robot_geom_influence_partition_sha256",
        "all_robot_contact_monitor_scope_identity_sha256",
        "robot_qvel_selection_rule",
        "robot_qvel_indices",
        "robot_qvel_indices_sha256",
        "robot_qvel_records",
        "robot_qvel_records_sha256",
        "velocity_dimension",
        "arm_qvel_indices",
        "arm_qvel_indices_included",
        "decision_arm_actuator_ids",
        "decision_dimension",
        "nonarm_robot_qvel_included",
        "nonarm_ctrl_policy",
        "model_actuator_count",
        "nonarm_ctrl_selection_rule",
        "nonarm_ctrl_indices",
        "nonarm_ctrl_indices_sha256",
        "nonarm_ctrl_count",
        "live_nonarm_ctrl_array_hash_format",
        "field_seed_sample_scope",
        "settled_field_query_certificate",
        "settled_zero_jacobian_sample_count",
        "settled_nonzero_jacobian_sample_count",
    }
    _require(
        set(binding) == expected_fields
        and binding.get("schema_version")
        == "vlsa_poisson_movable_manipulator_shield_sampling_binding.v1"
        and binding.get("scope")
        == (
            "all_structurally_movable_manipulator_collision_surfaces_"
            "vs_selected_obstacle"
        )
        and binding.get("sample_source")
        == "apparatus.movable_manipulator_sampling.samples"
        and binding.get("robot_qvel_selection_rule")
        == "ascending_dof_index_whose_dof_joint_body_is_in_resolved_robot_body_ids"
        and binding.get("nonarm_ctrl_policy")
        == "unchanged_byte_exact_in_all_cloned_and_live_transitions"
        and binding.get("nonarm_ctrl_selection_rule")
        == "ascending_actuator_index_excluding_decision_arm_actuator_ids"
        and binding.get("live_nonarm_ctrl_array_hash_format")
        == "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
        and binding.get("field_seed_sample_scope")
        == "link56_only_not_shield_scope",
        "movable shield sampling binding schema or semantics differ",
    )

    samples = full.get("samples")
    _require(isinstance(samples, list) and samples, "movable shield samples are absent")
    sample_count = len(samples)
    sample_hash = _canonical_sha256(samples)
    _require(
        binding.get("sample_count") == sample_count
        and full.get("sample_count") == sample_count
        and binding.get("sample_ledger_sha256") == sample_hash
        and full.get("sample_ledger_sha256") == sample_hash,
        "movable shield sample count or ledger hash differs",
    )

    resolved_geom_ids = _strict_integer_list(
        resolved_geometry.get("robot_geom_ids"), "resolved robot geom IDs"
    )
    bound_geom_ids = _strict_integer_list(
        binding.get("resolved_robot_geom_ids"), "bound robot geom IDs"
    )
    _require(
        bound_geom_ids == resolved_geom_ids
        and len(set(bound_geom_ids)) == len(bound_geom_ids)
        and binding.get("resolved_robot_geom_ids_sha256")
        == _canonical_sha256(bound_geom_ids),
        "resolved robot geom ordering or hash differs",
    )
    shield_geom_ids = _strict_integer_list(
        binding.get("shield_manipulator_geom_ids"),
        "bound movable manipulator geom IDs",
    )
    fixed_geom_ids = _strict_integer_list(
        binding.get("fixed_robot_infrastructure_geom_ids"),
        "bound fixed infrastructure geom IDs",
    )
    contact_scope = apparatus.get("registered_contact_scope")
    _require(
        shield_geom_ids == structural["movable_geom_ids"]
        and fixed_geom_ids == structural["fixed_geom_ids"]
        and binding.get("shield_manipulator_geom_ids_sha256")
        == _canonical_sha256(shield_geom_ids)
        and binding.get("fixed_robot_infrastructure_geom_ids_sha256")
        == _canonical_sha256(fixed_geom_ids)
        and binding.get("robot_geom_influence_partition_sha256")
        == structural["partition_sha256"]
        and isinstance(contact_scope, Mapping)
        and binding.get("all_robot_contact_monitor_scope_identity_sha256")
        == contact_scope.get("identity_sha256"),
        "shield/fixed structural partition binding differs",
    )
    bound_geom_id_set = set(shield_geom_ids)
    sampled_geom_ids = []
    sampled_body_ids = []
    expected_sample_fields = {
        "sample_id",
        "body_id",
        "body_name",
        "geom_id",
        "geom_name",
        "point_body_local_m",
        "source",
    }
    for index, sample in enumerate(samples):
        _require(isinstance(sample, Mapping), "movable sample row is invalid")
        sample_id = sample.get("sample_id")
        geom_id = sample.get("geom_id")
        body_id = sample.get("body_id")
        point = sample.get("point_body_local_m")
        _require(
            set(sample) == expected_sample_fields
            and isinstance(sample_id, int)
            and not isinstance(sample_id, bool)
            and sample_id == index
            and isinstance(geom_id, int)
            and not isinstance(geom_id, bool)
            and geom_id in bound_geom_id_set
            and isinstance(sample.get("geom_name"), str)
            and bool(sample.get("geom_name"))
            and isinstance(sample.get("body_name"), str)
            and bool(sample.get("body_name"))
            and sample.get("source") == "collision_geom_surface"
            and isinstance(point, list)
            and len(point) == 3
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in point
            ),
            "movable sample geom differs at row %d" % index,
        )
        sampled_geom_ids.append(int(geom_id))
        _require(
            isinstance(body_id, int) and not isinstance(body_id, bool),
            "movable sample body differs at row %d" % index,
        )
        sampled_body_ids.append(int(body_id))
    geom_order = {geom_id: index for index, geom_id in enumerate(shield_geom_ids)}
    _require(
        set(sampled_geom_ids) == bound_geom_id_set,
        "movable shield samples omit a structurally movable robot geom",
    )
    _require(
        [geom_order[geom_id] for geom_id in sampled_geom_ids]
        == sorted(geom_order[geom_id] for geom_id in sampled_geom_ids),
        "movable shield sample rows differ from structural geom order",
    )

    robot_body_ids = _strict_integer_list(
        resolved_geometry.get("robot_body_ids"), "resolved robot body IDs"
    )
    robot_body_names = resolved_geometry.get("robot_body_names")
    _require(
        isinstance(robot_body_names, list)
        and len(robot_body_names) == len(robot_body_ids)
        and all(isinstance(value, str) and value for value in robot_body_names),
        "resolved robot body names differ",
    )
    body_name_by_id = dict(zip(robot_body_ids, robot_body_names))
    for index, (body_id, sample) in enumerate(zip(sampled_body_ids, samples)):
        _require(
            body_id in body_name_by_id
            and sample.get("body_name") == body_name_by_id[body_id],
            "movable sample body identity differs at row %d" % index,
        )

    qvel_indices = _strict_integer_list(
        binding.get("robot_qvel_indices"), "robot-tree qvel indices"
    )
    _require(
        qvel_indices == sorted(qvel_indices)
        and len(set(qvel_indices)) == len(qvel_indices)
        and binding.get("robot_qvel_indices_sha256")
        == _canonical_sha256(qvel_indices),
        "robot-tree qvel ordering or hash differs",
    )
    qvel_records = binding.get("robot_qvel_records")
    _require(
        isinstance(qvel_records, list)
        and len(qvel_records) == len(qvel_indices)
        and binding.get("robot_qvel_records_sha256")
        == _canonical_sha256(qvel_records),
        "robot-tree qvel record count or hash differs",
    )
    expected_record_fields = {
        "qvel_index",
        "joint_id",
        "joint_name",
        "joint_body_id",
        "joint_body_name",
    }
    for index, (qvel_index, record) in enumerate(zip(qvel_indices, qvel_records)):
        _require(
            isinstance(record, Mapping)
            and set(record) == expected_record_fields
            and record.get("qvel_index") == qvel_index
            and isinstance(record.get("joint_id"), int)
            and not isinstance(record.get("joint_id"), bool)
            and int(record["joint_id"]) >= 0
            and isinstance(record.get("joint_name"), str)
            and bool(record.get("joint_name"))
            and isinstance(record.get("joint_body_id"), int)
            and not isinstance(record.get("joint_body_id"), bool)
            and record.get("joint_body_id") in body_name_by_id
            and record.get("joint_body_name")
            == body_name_by_id.get(record.get("joint_body_id")),
            "robot-tree qvel record differs at row %d" % index,
        )

    velocity_dimension = len(qvel_indices)
    arm_qvel_indices = _strict_integer_list(
        binding.get("arm_qvel_indices"), "bound arm qvel indices"
    )
    controller_arm_qvel = _strict_integer_list(
        controller.get("arm_qvel_indexes"), "controller arm qvel indices"
    )
    decision_actuators = _strict_integer_list(
        binding.get("decision_arm_actuator_ids"), "decision arm actuator IDs"
    )
    controller_actuators = _strict_integer_list(
        controller.get("arm_actuator_indexes"), "controller arm actuator IDs"
    )
    model_actuator_count = binding.get("model_actuator_count")
    _require(
        isinstance(model_actuator_count, int)
        and not isinstance(model_actuator_count, bool)
        and model_actuator_count > 7,
        "producer model actuator-count authority differs",
    )
    nonarm_ctrl_indices = _strict_integer_list(
        binding.get("nonarm_ctrl_indices"), "non-arm control indices"
    )
    expected_nonarm_ctrl_indices = [
        index
        for index in range(int(model_actuator_count))
        if index not in set(decision_actuators)
    ]
    _require(
        binding.get("velocity_dimension") == velocity_dimension
        and velocity_dimension > 7
        and arm_qvel_indices == controller_arm_qvel
        and len(arm_qvel_indices) == 7
        and len(set(arm_qvel_indices)) == 7
        and set(arm_qvel_indices).issubset(qvel_indices)
        and binding.get("arm_qvel_indices_included") is True
        and decision_actuators == controller_actuators
        and len(decision_actuators) == 7
        and len(set(decision_actuators)) == 7
        and all(
            0 <= actuator_id < int(model_actuator_count)
            for actuator_id in decision_actuators
        )
        and binding.get("decision_dimension") == 7
        and binding.get("nonarm_robot_qvel_included") is True
        and bool(set(qvel_indices) - set(arm_qvel_indices)),
        "full-robot velocity or seven-arm decision binding differs",
    )
    _require(
        nonarm_ctrl_indices == expected_nonarm_ctrl_indices
        and binding.get("nonarm_ctrl_count") == len(nonarm_ctrl_indices)
        and binding.get("nonarm_ctrl_indices_sha256")
        == _canonical_sha256(nonarm_ctrl_indices),
        "producer non-arm control complement authority differs",
    )

    settled = binding.get("settled_field_query_certificate")
    _require(
        isinstance(settled, Mapping)
        and set(settled)
        == {
            "schema_version",
            "sample_count",
            "sample_ledger_sha256",
            "all_queries_valid",
            "all_h_strictly_positive",
            "minimum_h_m2",
            "h_m2",
            "h_array_record",
            "world_points_m",
            "world_points_array_record",
            "outer_boundary_clearance_m",
            "outer_boundary_clearance_array_record",
            "required_outer_boundary_clearance_m",
            "minimum_outer_boundary_clearance_m",
            "outer_boundary_clearance_comparison_tolerance_m",
            "all_outer_boundary_clearances_pass",
        }
        and settled.get("schema_version")
        == "vlsa_poisson_movable_manipulator_settled_field_query.v1"
        and settled.get("sample_count") == sample_count
        and settled.get("sample_ledger_sha256") == sample_hash
        and settled.get("all_queries_valid") is True
        and settled.get("all_h_strictly_positive") is True,
        "settled movable-manipulator field-query certificate differs",
    )
    h_values = settled.get("h_m2")
    _require(
        isinstance(h_values, list)
        and len(h_values) == sample_count
        and all(
            isinstance(value, float) and math.isfinite(value) and value > 0.0
            for value in h_values
        ),
        "settled movable-manipulator h ledger is invalid",
    )
    h_array = np.asarray(h_values, dtype=np.float64)
    _exact_float(
        settled.get("minimum_h_m2"),
        float(np.min(h_array)),
        "settled movable-manipulator minimum h",
    )
    _validate_compact_array_identity(
        settled.get("h_array_record"),
        expected_shape=(sample_count,),
        label="settled movable-manipulator h",
        np=np,
        raw_array=h_array,
    )
    _validate_compact_array_identity(
        settled.get("world_points_array_record"),
        expected_shape=(sample_count, 3),
        label="settled movable-manipulator world points",
        np=np,
        raw_array=settled.get("world_points_m"),
    )
    world_points = np.asarray(settled.get("world_points_m"), dtype=np.float64)
    outer_clearance_values = settled.get("outer_boundary_clearance_m")
    _require(
        isinstance(outer_clearance_values, list)
        and len(outer_clearance_values) == sample_count
        and all(
            isinstance(value, float) and math.isfinite(value)
            for value in outer_clearance_values
        ),
        "settled movable-manipulator outer-clearance ledger is invalid",
    )
    outer_clearance = np.asarray(outer_clearance_values, dtype=np.float64)
    _validate_compact_array_identity(
        settled.get("outer_boundary_clearance_array_record"),
        expected_shape=(sample_count,),
        label="settled movable-manipulator outer clearance",
        np=np,
        raw_array=outer_clearance,
    )
    workspace = runtime_protocol.get("workspace")
    occupancy = runtime_protocol.get("occupancy")
    _require(
        isinstance(workspace, Mapping) and isinstance(occupancy, Mapping),
        "runtime workspace/occupancy authority is absent",
    )
    workspace_lower = np.asarray(workspace.get("minimum_m"), dtype=np.float64)
    workspace_upper = np.asarray(workspace.get("maximum_m"), dtype=np.float64)
    _require(
        workspace_lower.shape == workspace_upper.shape == (3,)
        and np.all(np.isfinite(workspace_lower))
        and np.all(np.isfinite(workspace_upper))
        and np.all(workspace_upper > workspace_lower),
        "runtime workspace authority differs",
    )
    direct_outer_clearance = np.min(
        np.concatenate(
            (
                world_points - workspace_lower[None, :],
                workspace_upper[None, :] - world_points,
            ),
            axis=1,
        ),
        axis=1,
    )
    required_outer_clearance = float(
        occupancy.get("outer_boundary_clearance_m")
    )
    workspace_scale = max(
        1.0,
        float(np.max(np.abs(workspace_lower))),
        float(np.max(np.abs(workspace_upper))),
    )
    comparison_tolerance = (
        64.0 * float(np.finfo(np.float64).eps) * workspace_scale
    )
    _exact_float(
        settled.get("required_outer_boundary_clearance_m"),
        required_outer_clearance,
        "settled required outer clearance",
    )
    _exact_float(
        settled.get("outer_boundary_clearance_comparison_tolerance_m"),
        comparison_tolerance,
        "settled outer-clearance comparison tolerance",
    )
    _exact_float(
        settled.get("minimum_outer_boundary_clearance_m"),
        float(np.min(outer_clearance)),
        "settled minimum outer clearance",
    )
    _require(
        np.allclose(
            outer_clearance,
            direct_outer_clearance,
            rtol=0.0,
            atol=1.0e-12,
        )
        and settled.get("all_outer_boundary_clearances_pass")
        is bool(
            np.all(
                outer_clearance + comparison_tolerance
                >= required_outer_clearance
            )
        )
        and settled.get("all_outer_boundary_clearances_pass") is True,
        "settled movable-manipulator samples violate runtime outer-boundary clearance",
    )
    zero_count = binding.get("settled_zero_jacobian_sample_count")
    nonzero_count = binding.get("settled_nonzero_jacobian_sample_count")
    _require(
        isinstance(zero_count, int)
        and not isinstance(zero_count, bool)
        and zero_count >= 0
        and isinstance(nonzero_count, int)
        and not isinstance(nonzero_count, bool)
        and nonzero_count > 0
        and zero_count + nonzero_count == sample_count,
        "settled movable-manipulator Jacobian row counts differ",
    )
    return {
        "samples": samples,
        "sample_count": sample_count,
        "sample_ledger_sha256": sample_hash,
        "robot_qvel_indices": qvel_indices,
        "velocity_dimension": velocity_dimension,
        "arm_qvel_indices": arm_qvel_indices,
        "decision_arm_actuator_ids": decision_actuators,
        "nonarm_ctrl_indices": nonarm_ctrl_indices,
        "nonarm_ctrl_indices_sha256": _canonical_sha256(nonarm_ctrl_indices),
        "structural_partition": structural,
    }


def _validate_protected_link_shield_sampling_binding(
    *,
    apparatus: Mapping[str, Any],
    resolved_geometry: Mapping[str, Any],
    target_resolved_geometry: Mapping[str, Any],
    protected_resolved_geometry: Mapping[str, Any],
    expected_target_link_body_names: Sequence[str],
    expected_protected_link_body_names: Sequence[str],
    shield_contract: Mapping[str, Any],
    controller: Mapping[str, Any],
    np: Any,
) -> Dict[str, Any]:
    """Bind every v4 QP row to the shared configured protected-link ledger."""

    binding = apparatus.get("shield_sampling_binding")
    protected = apparatus.get("protected_sampling")
    scope = apparatus.get("registered_contact_scope")
    _require(
        isinstance(binding, Mapping), "protected-link shield binding is absent"
    )
    _require(isinstance(protected, Mapping), "protected field sampling is absent")
    _require(isinstance(scope, Mapping), "registered contact scope is absent")
    _require(
        apparatus.get("shield_sampling_binding_sha256")
        == _canonical_sha256(binding),
        "protected-link shield binding hash differs",
    )
    samples = binding.get("samples")
    if samples is None:
        samples = protected.get("samples")
    protected_samples = protected.get("samples")
    _require(
        binding.get("schema_version")
        == "vlsa_poisson_protected_link_shield_sampling_binding.v1"
        and binding.get("scope") == shield_contract.get("binding_scope")
        and binding.get("sample_source") == "apparatus.protected_sampling.samples"
        and binding.get("protected_samples_contract")
        == shield_contract.get("protected_samples")
        and binding.get("field_bundle_samples_contract")
        == shield_contract.get("field_bundle_samples")
        and binding.get("shield_body_authority")
        == shield_contract.get("shield_body_authority")
        and binding.get("shield_geometry_selection")
        == shield_contract.get("shield_geometry_selection")
        and isinstance(samples, list)
        and samples
        and isinstance(protected_samples, list)
        and fast._canonical(samples) == fast._canonical(protected_samples),
        "protected-link shield source or ordered sample payload differs",
    )
    sample_count = len(samples)
    sample_hash = _canonical_sha256(samples)
    _require(
        binding.get("sample_count") == sample_count
        and binding.get("sample_ledger_sha256") == sample_hash
        and binding.get("field_bundle_protected_sample_count") == sample_count
        and binding.get("field_bundle_protected_samples_sha256")
        == _canonical_sha256(protected)
        and binding.get("exact_ordered_field_bundle_protected_sample_ledger")
        is True,
        "protected-link shield count or ledger hash differs",
    )
    _require(
        binding.get("field_seed_sample_scope")
        == shield_contract.get("field_bundle_samples"),
        "protected-link field seed scope differs from the protocol",
    )

    broad_link56_geom_ids = _strict_integer_list(
        resolved_geometry.get("link56_geom_ids"), "resolved link5/link6 geom IDs"
    )
    target_body_ids = _strict_integer_list(
        target_resolved_geometry.get("target_link_body_ids"),
        "resolved target-link body IDs",
    )
    target_body_names = target_resolved_geometry.get("target_link_body_names")
    target_geom_ids = _strict_integer_list(
        target_resolved_geometry.get("target_link_geom_ids"),
        "resolved target-link geom IDs",
    )
    target_geom_names = target_resolved_geometry.get("target_link_geom_names")
    protected_body_ids = _strict_integer_list(
        protected_resolved_geometry.get("protected_link_body_ids"),
        "resolved protected-link body IDs",
    )
    protected_body_names = protected_resolved_geometry.get(
        "protected_link_body_names"
    )
    protected_geom_ids = _strict_integer_list(
        protected_resolved_geometry.get("protected_link_geom_ids"),
        "resolved protected-link geom IDs",
    )
    protected_geom_names = protected_resolved_geometry.get(
        "protected_link_geom_names"
    )
    broad_robot_body_ids = _strict_integer_list(
        resolved_geometry.get("robot_body_ids"), "broad resolved robot body IDs"
    )
    broad_robot_body_names = resolved_geometry.get("robot_body_names")
    broad_robot_geom_ids = _strict_integer_list(
        resolved_geometry.get("robot_geom_ids"), "broad resolved robot geom IDs"
    )
    broad_robot_geom_names = resolved_geometry.get("robot_geom_names")
    _require(
        isinstance(broad_robot_body_names, list)
        and len(broad_robot_body_names) == len(broad_robot_body_ids)
        and isinstance(broad_robot_geom_names, list)
        and len(broad_robot_geom_names) == len(broad_robot_geom_ids),
        "broad resolved robot name ledgers differ",
    )
    broad_body_name_by_id = dict(
        zip(broad_robot_body_ids, broad_robot_body_names)
    )
    broad_geom_name_by_id = dict(
        zip(broad_robot_geom_ids, broad_robot_geom_names)
    )
    _require(
        target_resolved_geometry.get("schema_version")
        == "vlsa_poisson_target_link_resolved_geometry.v1"
        and apparatus.get("target_link_resolved_geometry_sha256")
        == _canonical_sha256(target_resolved_geometry)
        and target_body_names == [str(value) for value in expected_target_link_body_names]
        and isinstance(target_geom_names, list)
        and len(target_geom_names) == len(target_geom_ids)
        and len(target_body_ids) == len(target_body_names)
        and len(target_body_ids) == len(set(target_body_ids))
        and len(target_body_names) == len(set(target_body_names))
        and bool(target_body_ids)
        and bool(target_geom_ids)
        and set(target_geom_ids).issubset(broad_link56_geom_ids)
        and [broad_body_name_by_id.get(value) for value in target_body_ids]
        == target_body_names
        and [broad_geom_name_by_id.get(value) for value in target_geom_ids]
        == target_geom_names
        and target_resolved_geometry.get("geometry_selection")
        == (
            "collision_enabled_geoms_directly_attached_to_exact_configured_"
            "target_link_bodies_no_descendants"
        )
        and target_resolved_geometry.get(
            "broad_robot_obstacle_authority_unchanged"
        )
        is True
        and target_resolved_geometry.get(
            "target_link_geoms_subset_of_broad_literal_link56_geoms"
        )
        is True,
        "configured target-link resolved geometry differs",
    )
    _require(
        protected_resolved_geometry.get("schema_version")
        == "vlsa_poisson_protected_link_resolved_geometry.v1"
        and apparatus.get("protected_link_resolved_geometry_sha256")
        == _canonical_sha256(protected_resolved_geometry)
        and protected_body_names
        == [str(value) for value in expected_protected_link_body_names]
        and isinstance(protected_geom_names, list)
        and len(protected_geom_names) == len(protected_geom_ids)
        and len(protected_body_ids) == len(protected_body_names) == 2
        and len(set(protected_body_ids)) == len(protected_body_ids)
        and len(set(protected_body_names)) == len(protected_body_names)
        and bool(protected_geom_ids)
        and protected_geom_ids == broad_link56_geom_ids
        and set(target_body_ids).issubset(protected_body_ids)
        and set(target_geom_ids).issubset(protected_geom_ids)
        and [broad_body_name_by_id.get(value) for value in protected_body_ids]
        == protected_body_names
        and [broad_geom_name_by_id.get(value) for value in protected_geom_ids]
        == protected_geom_names
        and protected_resolved_geometry.get("geometry_selection")
        == (
            "collision_enabled_geoms_directly_attached_to_exact_configured_"
            "protected_link_bodies_no_descendants"
        )
        and protected_resolved_geometry.get(
            "broad_robot_obstacle_authority_unchanged"
        )
        is True
        and protected_resolved_geometry.get(
            "target_link_geoms_subset_of_protected_link_geoms"
        )
        is True
        and protected_resolved_geometry.get(
            "protected_link_geoms_subset_of_broad_literal_link56_geoms"
        )
        is True,
        "shared protected-link resolved geometry differs",
    )
    shield_geom_ids = _strict_integer_list(
        binding.get("shield_manipulator_geom_ids"),
        "bound protected-link shield geom IDs",
    )
    _require(
        shield_geom_ids == protected_geom_ids
        and _strict_integer_list(
            binding.get("shield_protected_link_geom_ids"),
            "bound protected-link geom IDs",
        )
        == protected_geom_ids
        and binding.get("protected_link_body_ids") == protected_body_ids
        and binding.get("protected_link_body_names") == protected_body_names
        and binding.get("field_bundle_protected_body_ids")
        == protected_body_ids
        and binding.get("field_bundle_protected_body_names")
        == protected_body_names
        and binding.get("evaluation_target_link_body_names")
        == target_body_names
        and binding.get("evaluation_target_link_geom_ids") == target_geom_ids
        and _strict_integer_list(
            binding.get("resolved_link56_geom_ids"),
            "bound broad resolved link5/link6 geom IDs",
        )
        == broad_link56_geom_ids
        and binding.get("resolved_link56_geom_ids_sha256")
        == _canonical_sha256(broad_link56_geom_ids)
        and binding.get("shield_protected_link_geom_ids_sha256")
        == _canonical_sha256(protected_geom_ids)
        and binding.get("shield_manipulator_geom_ids_sha256")
        == _canonical_sha256(shield_geom_ids),
        "shield geom ledger is not exactly the shared protected-link ledger",
    )
    sampled_geom_ids = [int(row.get("geom_id", -1)) for row in samples]
    geom_order = {
        geom_id: index for index, geom_id in enumerate(protected_geom_ids)
    }
    _require(
        set(sampled_geom_ids) == set(protected_geom_ids)
        and all(geom_id in geom_order for geom_id in sampled_geom_ids)
        and [geom_order[geom_id] for geom_id in sampled_geom_ids]
        == sorted(geom_order[geom_id] for geom_id in sampled_geom_ids),
        "protected field samples do not cover exactly the ordered shared-link geoms",
    )
    robot_geom_ids = _strict_integer_list(
        resolved_geometry.get("robot_geom_ids"), "resolved robot geom IDs"
    )
    scope_robot_geom_ids = _strict_integer_list(
        scope.get("robot_geom_ids"), "registered whole-robot contact geom IDs"
    )
    _require(
        scope_robot_geom_ids == robot_geom_ids
        and binding.get("all_robot_contact_monitor_scope_identity_sha256")
        == scope.get("identity_sha256"),
        "protected-link shield is not paired with the broad contact monitor",
    )

    qvel_authority = binding.get("protected_link_qvel_authority")
    _require(
        isinstance(qvel_authority, Mapping)
        and qvel_authority.get("schema_version")
        == "vlsa_poisson_protected_link_arm_qvel_binding.v1"
        and fast._canonical(apparatus.get("active_shield_qvel_authority"))
        == fast._canonical(qvel_authority),
        "protected-link active qvel authority differs",
    )
    qvel_indices = _strict_integer_list(
        qvel_authority.get("robot_qvel_indices"),
        "protected-link shield qvel indices",
    )
    qvel_records = qvel_authority.get("robot_qvel_records")
    full_qvel_authority = apparatus.get("full_robot_qvel_authority")
    _require(
        isinstance(full_qvel_authority, Mapping),
        "full robot-tree qvel authority is absent",
    )
    full_qvel_indices = _strict_integer_list(
        full_qvel_authority.get("robot_qvel_indices"),
        "full robot-tree qvel indices",
    )
    full_qvel_records = full_qvel_authority.get("robot_qvel_records")
    _require(
        qvel_indices == sorted(qvel_indices)
        and len(qvel_indices) == len(set(qvel_indices))
        and qvel_authority.get("robot_qvel_indices_sha256")
        == _canonical_sha256(qvel_indices)
        and isinstance(qvel_records, list)
        and len(qvel_records) == len(qvel_indices)
        and qvel_authority.get("robot_qvel_records_sha256")
        == _canonical_sha256(qvel_records)
        and [row.get("qvel_index") for row in qvel_records] == qvel_indices,
        "protected-link shield velocity authority differs",
    )
    _require(
        full_qvel_indices == sorted(full_qvel_indices)
        and len(full_qvel_indices) == len(set(full_qvel_indices))
        and set(qvel_indices).issubset(full_qvel_indices)
        and full_qvel_authority.get("robot_qvel_indices_sha256")
        == _canonical_sha256(full_qvel_indices)
        and isinstance(full_qvel_records, list)
        and len(full_qvel_records) == len(full_qvel_indices)
        and full_qvel_authority.get("robot_qvel_records_sha256")
        == _canonical_sha256(full_qvel_records)
        and [row.get("qvel_index") for row in full_qvel_records]
        == full_qvel_indices
        and [
            row
            for row in full_qvel_records
            if int(row.get("qvel_index", -1)) in set(qvel_indices)
        ]
        == qvel_records,
        "protected-link qvel rows differ from the full robot-tree ledger",
    )
    arm_qvel_indices = _strict_integer_list(
        qvel_authority.get("arm_qvel_indices"),
        "protected-link shield arm qvel indices",
    )
    controller_arm_qvel = _strict_integer_list(
        controller.get("arm_qvel_indexes"), "controller arm qvel indices"
    )
    decision_actuators = _strict_integer_list(
        qvel_authority.get("decision_arm_actuator_ids"),
        "decision arm actuator IDs",
    )
    controller_actuators = _strict_integer_list(
        controller.get("arm_actuator_indexes"), "controller arm actuator IDs"
    )
    model_actuator_count = binding.get("model_actuator_count")
    nonarm_ctrl_indices = _strict_integer_list(
        binding.get("nonarm_ctrl_indices"), "non-arm control indices"
    )
    expected_nonarm_ctrl_indices = [
        index
        for index in range(int(model_actuator_count))
        if index not in set(decision_actuators)
    ]
    _require(
        isinstance(model_actuator_count, int)
        and not isinstance(model_actuator_count, bool)
        and int(model_actuator_count) > 7
        and qvel_authority.get("velocity_dimension") == len(qvel_indices)
        and arm_qvel_indices == controller_arm_qvel
        and len(arm_qvel_indices) == len(set(arm_qvel_indices)) == 7
        and set(arm_qvel_indices).issubset(qvel_indices)
        and decision_actuators == controller_actuators
        and len(decision_actuators) == len(set(decision_actuators)) == 7
        and qvel_authority.get("decision_dimension") == 7
        and nonarm_ctrl_indices == expected_nonarm_ctrl_indices
        and binding.get("nonarm_ctrl_indices_sha256")
        == _canonical_sha256(nonarm_ctrl_indices)
        and binding.get("nonarm_ctrl_count") == len(nonarm_ctrl_indices)
        and binding.get("nonarm_ctrl_policy")
        == "unchanged_byte_exact_in_all_cloned_and_live_transitions",
        "protected-link shield decision/control authority differs",
    )
    influencing_qvel_indices = _strict_integer_list(
        qvel_authority.get(
            "structurally_influencing_protected_link_qvel_indices"
        ),
        "structurally influencing protected-link qvel indices",
    )
    omitted_qvel_indices = _strict_integer_list(
        qvel_authority.get(
            "omitted_robot_tree_qvels_proven_noninfluential_for_protected_link"
        ),
        "omitted noninfluential robot-tree qvel indices",
    )
    expected_omitted_qvel_indices = sorted(
        set(full_qvel_indices) - set(qvel_indices)
    )
    _require(
        binding.get("robot_qvel_selection_rule")
        == shield_contract.get("point_velocity_scope")
        and qvel_authority.get("resolved_protected_link_geom_ids")
        == protected_geom_ids
        and influencing_qvel_indices
        and set(influencing_qvel_indices).issubset(arm_qvel_indices)
        and qvel_authority.get(
            "structurally_influencing_protected_link_qvel_indices_sha256"
        )
        == _canonical_sha256(influencing_qvel_indices)
        and qvel_authority.get(
            "all_structural_protected_link_qvel_influences_included"
        )
        is True
        and qvel_authority.get("nonarm_robot_qvel_included") is False,
        "protected-link structural qvel authority differs",
    )
    _require(
        full_qvel_authority.get("arm_qvel_indices") == arm_qvel_indices
        and full_qvel_authority.get("decision_arm_actuator_ids")
        == decision_actuators
        and full_qvel_authority.get("nonarm_robot_qvel_included") is True
        and omitted_qvel_indices == expected_omitted_qvel_indices
        and set(omitted_qvel_indices).isdisjoint(influencing_qvel_indices),
        "omitted protected-link qvel provenance differs",
    )

    settled = binding.get("settled_field_query_certificate")
    _require(
        isinstance(settled, Mapping),
        "settled protected-link field query is absent",
    )
    h_values = settled.get("h_m2")
    _require(
        settled.get("sample_count") == sample_count
        and settled.get("sample_ledger_sha256") == sample_hash
        and settled.get("all_queries_valid") is True
        and settled.get("all_h_strictly_positive") is True
        and isinstance(h_values, list)
        and len(h_values) == sample_count
        and all(
            isinstance(value, float) and math.isfinite(value) and value > 0.0
            for value in h_values
        ),
        "settled protected-link field-query certificate differs",
    )
    _validate_compact_array_identity(
        settled.get("h_array_record"),
        expected_shape=(sample_count,),
        label="settled protected-link h",
        np=np,
        raw_array=h_values,
    )
    _validate_compact_array_identity(
        settled.get("world_points_array_record"),
        expected_shape=(sample_count, 3),
        label="settled protected-link world points",
        np=np,
        raw_array=settled.get("world_points_m"),
    )
    outer_clearance = settled.get("outer_boundary_clearance_m")
    _require(
        isinstance(outer_clearance, list)
        and len(outer_clearance) == sample_count
        and all(math.isfinite(float(value)) for value in outer_clearance)
        and settled.get("all_outer_boundary_clearances_pass") is True,
        "settled protected-link field boundary certificate differs",
    )
    _validate_compact_array_identity(
        settled.get("outer_boundary_clearance_array_record"),
        expected_shape=(sample_count,),
        label="settled protected-link outer clearance",
        np=np,
        raw_array=outer_clearance,
    )
    return {
        "samples": samples,
        "sample_count": sample_count,
        "sample_ledger_sha256": sample_hash,
        "robot_qvel_indices": qvel_indices,
        "velocity_dimension": len(qvel_indices),
        "arm_qvel_indices": arm_qvel_indices,
        "decision_arm_actuator_ids": decision_actuators,
        "nonarm_ctrl_indices": nonarm_ctrl_indices,
        "nonarm_ctrl_indices_sha256": _canonical_sha256(nonarm_ctrl_indices),
        "shield_geom_ids": shield_geom_ids,
        "target_body_ids": target_body_ids,
        "target_body_names": target_body_names,
        "protected_body_ids": protected_body_ids,
        "protected_body_names": protected_body_names,
        "exact_protected_field_ledger": True,
    }


def _validate_field_sample_static_evidence(
    *,
    apparatus: Mapping[str, Any],
    treatment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    runtime_protocol_sha256: str,
    runtime_parameter_block_sha256: str,
    target_link_v4: bool,
    expected_target_link_body_names: Sequence[str],
    expected_protected_link_body_names: Sequence[str],
) -> Dict[str, Any]:
    """Reconstruct field identities and admissibility from serialized rows."""

    resolved = apparatus.get("resolved_geometry")
    _require(isinstance(resolved, Mapping), "resolved geometry is absent")
    robot_geom_ids = [int(value) for value in resolved.get("robot_geom_ids", ())]
    robot_body_ids = [int(value) for value in resolved.get("robot_body_ids", ())]
    broad_link_geom_ids = [int(value) for value in resolved.get("link56_geom_ids", ())]
    broad_link_body_ids = [int(value) for value in resolved.get("link56_body_ids", ())]
    obstacle_geom_ids = [int(value) for value in resolved.get("obstacle_geom_ids", ())]
    _require(
        robot_geom_ids and robot_body_ids and broad_link_geom_ids and broad_link_body_ids
        and obstacle_geom_ids,
        "resolved field/sample identities are empty",
    )
    if target_link_v4:
        target_resolved = apparatus.get("target_link_resolved_geometry")
        protected_resolved = apparatus.get("protected_link_resolved_geometry")
        _require(
            isinstance(target_resolved, Mapping)
            and isinstance(protected_resolved, Mapping),
            "target/protected resolved geometry is absent",
        )
        target_geom_ids = _strict_integer_list(
            target_resolved.get("target_link_geom_ids"),
            "field evaluation target-link geom IDs",
        )
        target_body_ids = _strict_integer_list(
            target_resolved.get("target_link_body_ids"),
            "field evaluation target-link body IDs",
        )
        link_geom_ids = _strict_integer_list(
            protected_resolved.get("protected_link_geom_ids"),
            "field protected-link geom IDs",
        )
        link_body_ids = _strict_integer_list(
            protected_resolved.get("protected_link_body_ids"),
            "field protected-link body IDs",
        )
        _require(
            target_resolved.get("schema_version")
            == "vlsa_poisson_target_link_resolved_geometry.v1"
            and apparatus.get("target_link_resolved_geometry_sha256")
            == _canonical_sha256(target_resolved)
            and target_resolved.get("target_link_body_names")
            == [str(value) for value in expected_target_link_body_names]
            and protected_resolved.get("schema_version")
            == "vlsa_poisson_protected_link_resolved_geometry.v1"
            and apparatus.get("protected_link_resolved_geometry_sha256")
            == _canonical_sha256(protected_resolved)
            and protected_resolved.get("protected_link_body_names")
            == [str(value) for value in expected_protected_link_body_names]
            and link_geom_ids == broad_link_geom_ids
            and link_body_ids == broad_link_body_ids
            and set(target_geom_ids).issubset(link_geom_ids)
            and set(target_body_ids).issubset(link_body_ids),
            "field shield geometry is not the shared protected-link set",
        )
    else:
        link_geom_ids = broad_link_geom_ids
        link_body_ids = broad_link_body_ids

    full = apparatus.get("all_robot_sampling")
    _require(isinstance(full, Mapping), "all-robot sampling evidence is absent")
    _require(
        apparatus.get("all_robot_sampling_sha256") == _canonical_sha256(full),
        "all-robot sampling block hash differs",
    )
    full_counts, full_count = _validated_sample_rows(
        full,
        expected_geom_ids=robot_geom_ids,
        expected_body_ids=robot_body_ids,
        expected_hash=str(full.get("sample_ledger_sha256", "")),
        label="all-robot",
    )
    from main.poisson_fullbody.surface_sampling import validate_robot_sample_evidence

    validate_robot_sample_evidence(
        full,
        resolved_geom_ids=robot_geom_ids,
        resolved_geom_names=[str(value) for value in resolved.get("robot_geom_names", ())],
        resolved_body_ids=robot_body_ids,
        roundtrip_field="roundtrip",
    )
    records = full.get("geom_records")
    _require(isinstance(records, list), "all-robot component ledger is absent")
    _require(
        {int(row.get("geom_id", -1)): int(row.get("sample_count", -1)) for row in records}
        == full_counts,
        "all-robot component counts differ from raw samples",
    )

    protected = apparatus.get("protected_sampling")
    hashes = apparatus.get("field_bundle_hashes")
    diagnostics = apparatus.get("field_diagnostics")
    boxes = apparatus.get("field_obstacle_boxes")
    _require(isinstance(protected, Mapping), "protected sampling evidence is absent")
    _require(isinstance(hashes, Mapping), "field bundle hashes are absent")
    _require(isinstance(diagnostics, Mapping), "field diagnostics are absent")
    _require(isinstance(boxes, list) and boxes, "field obstacle boxes are absent")
    for name, value in hashes.items():
        _require(
            isinstance(name, str)
            and isinstance(value, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
            "field component hash is malformed",
        )
    _require(
        hashes.get("protocol_sha256") == runtime_protocol_sha256
        and hashes.get("parameter_block_sha256") == runtime_parameter_block_sha256,
        "field bundle/runtime identity differs",
    )
    _require(
        _canonical_sha256(boxes) == hashes.get("obstacle_geometry_sha256"),
        "field obstacle geometry hash differs",
    )
    protected_counts, protected_count = _validated_sample_rows(
        protected,
        expected_geom_ids=link_geom_ids,
        expected_body_ids=link_body_ids,
        expected_hash=_canonical_sha256(protected.get("samples")),
        label="protected",
        require_declared_count=False,
    )
    components = protected.get("components")
    _require(isinstance(components, list), "protected component ledger is absent")
    _require(
        {int(row.get("geom_id", -1)): int(row.get("sample_count", -1)) for row in components}
        == protected_counts,
        "protected component counts differ from raw samples",
    )
    epsilon = float(protected.get("epsilon_m"))
    maximum_cover = float(protected.get("maximum_surface_cover_radius_m"))
    _require(
        _canonical_sha256(protected) == hashes.get("protected_samples_sha256"),
        "protected sample payload hash differs",
    )
    _require(
        math.isfinite(epsilon)
        and epsilon > 0.0
        and math.isfinite(maximum_cover)
        and 0.0 <= maximum_cover < epsilon
        and all(
            0.0 <= float(row.get("certified_surface_cover_radius_m")) < epsilon
            for row in components
        ),
        "protected surface cover certificate differs",
    )

    expected_bundle_identity = {
        "protocol_sha256": hashes["protocol_sha256"],
        "parameter_block_sha256": hashes["parameter_block_sha256"],
        "obstacle_geometry_sha256": hashes["obstacle_geometry_sha256"],
        "occupancy_sha256": hashes["occupancy_sha256"],
        "domain_sha256": hashes["domain_sha256"],
        "system_sha256": hashes["system_sha256"],
        "field_sha256": hashes["field_sha256"],
        "protected_samples_sha256": hashes["protected_samples_sha256"],
        "diagnostics": diagnostics,
    }
    _require(
        _canonical_sha256(expected_bundle_identity) == hashes.get("bundle_sha256"),
        "field bundle aggregate hash differs",
    )
    field_certificate = {
        "field_bundle_hashes": dict(hashes),
        "field_diagnostics": dict(diagnostics),
        "field_obstacle_boxes": list(boxes),
        "protected_sampling": dict(protected),
    }
    _require(
        apparatus.get("serialized_field_certificate_sha256")
        == _canonical_sha256(field_certificate),
        "serialized field certificate hash differs",
    )

    poisson = diagnostics.get("poisson")
    _require(isinstance(poisson, Mapping), "Poisson diagnostics are absent")
    tolerance = float(runtime_protocol["poisson"]["normalized_backward_error_tolerance"])
    _require(
        diagnostics.get("obstacle_geom_count") == len(boxes) == len(obstacle_geom_ids)
        and diagnostics.get("protected_surface_component_count") == len(components)
        and diagnostics.get("protected_sample_count") == protected_count
        and int(diagnostics.get("active_vertex_count", -1))
        == int(diagnostics.get("boundary_vertex_count", -2))
        + int(diagnostics.get("interior_vertex_count", -2))
        and int(poisson.get("unknown_count", -1))
        == int(diagnostics.get("interior_vertex_count", -2))
        and diagnostics.get("poisson_method") == "red_black_sor"
        and 0 <= int(diagnostics.get("poisson_iterations", -1))
        <= int(runtime_protocol["poisson"]["max_iterations"])
        and poisson.get("finite") is True
        and poisson.get("passed") is True
        and poisson.get("expected_sign") == "nonnegative"
        and int(poisson.get("sign_violation_count", -1)) == 0
        and float(poisson.get("backward_error_linf")) <= tolerance
        and float(poisson.get("boundary_linf")) <= 1e-12
        and float(poisson.get("interior_min")) > 0.0
        and float(diagnostics.get("minimum_initial_h_m2")) > 0.0
        and math.isclose(
            float(diagnostics.get("required_outer_boundary_clearance_m")),
            float(runtime_protocol["occupancy"]["outer_boundary_clearance_m"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and float(diagnostics.get("minimum_outer_boundary_clearance_m"))
        >= float(diagnostics.get("required_outer_boundary_clearance_m")),
        "field diagnostics fail independent consistency checks",
    )

    static_rows = treatment.get("static_obstacle_trace")
    physics_rows = treatment.get("physics_trace")
    _require(isinstance(static_rows, list) and static_rows, "static trace is absent")
    _require(isinstance(physics_rows, list), "physics trace is absent")
    _require(
        treatment.get("static_obstacle_trace_sha256") == _canonical_sha256(static_rows),
        "static trace hash differs",
    )
    static_fields = (
        "translation_drift_m",
        "rotation_drift_rad",
        "surface_drift_m",
        "maximum_body_linear_speed_m_s",
        "maximum_body_angular_speed_rad_s",
    )
    for index, row in enumerate(static_rows):
        _require(
            isinstance(row, Mapping)
            and set(row) == set(static_fields)
            and all(math.isfinite(float(row[field])) and float(row[field]) >= 0.0 for field in static_fields),
            "static trace row %d is invalid" % index,
        )
    _require(
        len(static_rows) == len(physics_rows) + 1
        and fast._canonical(static_rows[0]) == fast._canonical(apparatus.get("settled_obstacle"))
        and all(
            fast._canonical(static_rows[index + 1])
            == fast._canonical(row.get("selected_obstacle"))
            for index, row in enumerate(physics_rows)
        ),
        "static trace is not bound to every physics row",
    )
    limits = runtime_protocol["admissibility"]
    admissible = bool(
        max(float(row["translation_drift_m"]) for row in static_rows)
        <= float(limits["max_selected_geom_translation_drift_m"])
        and max(float(row["rotation_drift_rad"]) for row in static_rows)
        <= float(limits["max_selected_geom_rotation_drift_rad"])
        and max(float(row["surface_drift_m"]) for row in static_rows)
        <= float(limits["max_selected_geom_surface_drift_m"])
        and max(float(row["maximum_body_linear_speed_m_s"]) for row in static_rows)
        <= float(limits["max_selected_body_linear_speed_m_s"])
        and max(float(row["maximum_body_angular_speed_rad_s"]) for row in static_rows)
        <= float(limits["max_selected_body_angular_speed_rad_s"])
    )
    _require(
        metrics.get("static_field_admissible") is admissible,
        "static admissibility flag differs from raw trace",
    )
    return {
        "all_robot_sample_count": full_count,
        "protected_sample_count": protected_count,
        "static_row_count": len(static_rows),
        "static_admissible": admissible,
        "bundle_sha256": hashes["bundle_sha256"],
    }


def _reconstruct_finite_difference_resolution(
    *,
    base: float,
    lower: float,
    upper: float,
    requested: float,
    scale: float,
) -> Dict[str, Any]:
    """Reconstruct the producer's exact no-clipping binary64 stencil."""

    _require(
        all(math.isfinite(value) for value in (base, lower, upper, requested, scale))
        and lower <= base <= upper
        and requested > 0.0
        and 0.0 < scale <= 1.0,
        "finite-difference reconstruction inputs are invalid",
    )
    negative_room = float(base - lower)
    positive_room = float(upper - base)
    if requested <= negative_room and requested <= positive_room:
        stencil = "centered"
        signed_full = (-requested, requested)
    elif min(requested, positive_room) >= min(requested, negative_room):
        stencil = "forward"
        signed_full = (0.0, min(requested, positive_room))
    else:
        stencil = "backward"
        signed_full = (-min(requested, negative_room), 0.0)

    requested_deltas = tuple(float(value * scale) for value in signed_full)
    candidates = tuple(float(base + value) for value in requested_deltas)
    exact_deltas = []
    for requested_delta, candidate in zip(requested_deltas, candidates):
        _require(
            math.isfinite(candidate) and lower <= candidate <= upper,
            "reconstructed stencil candidate violates actuator bounds",
        )
        if requested_delta == 0.0:
            exact_deltas.append(0.0)
            continue
        actual_delta = float(candidate - base)
        _require(
            math.isfinite(actual_delta)
            and actual_delta != 0.0
            and math.copysign(1.0, actual_delta)
            == math.copysign(1.0, requested_delta),
            "reconstructed stencil has no signed binary64 resolution",
        )
        exact_deltas.append(actual_delta)
    denominator = float(exact_deltas[1] - exact_deltas[0])
    _require(
        math.isfinite(denominator) and denominator > 0.0,
        "reconstructed stencil denominator is nonpositive",
    )
    return {
        "stencil": stencil,
        "requested_deltas_nm": requested_deltas,
        "candidate_torques_nm": candidates,
        "sample_deltas_nm": tuple(exact_deltas),
        "denominator_nm": denominator,
        "bound_adapted": bool(
            stencil != "centered"
            or max(abs(value) for value in requested_deltas) < requested
        ),
    }


def _validate_shield_diagnostic_input_binding(
    diagnostics: Mapping[str, Any],
    *,
    velocity_dimension: int,
    protocol: Mapping[str, Any],
    sensitivity_max_absolute_error: float,
) -> None:
    """Bind producer diagnostics, including its omitted 7D defaults."""

    _require(
        diagnostics.get("schema")
        == "vlsa_poisson_post_osc_torque_shield.v1",
        "shield diagnostic schema differs",
    )
    _require(
        diagnostics.get("constraint_equation")
        == "a@(v_nom_next+S@delta_tau)+alpha*h>=margin",
        "shield diagnostic constraint_equation differs",
    )
    _require(
        diagnostics.get("sensitivity_already_includes_dt_and_contact_effects")
        is True,
        "shield diagnostic sensitivity_already_includes_dt_and_contact_effects differs",
    )

    for field, expected in (
        ("dt_seconds", 0.002),
        ("alpha", float(protocol["shield"]["alpha_gain_per_s"])),
        ("margin", float(protocol["shield"]["margin_m2_per_s"])),
    ):
        _exact_float(
            diagnostics.get(field),
            expected,
            "shield diagnostic %s" % field,
        )

    observed_sensitivity_error = diagnostics.get(
        "sensitivity_max_absolute_error"
    )
    _require(
        not isinstance(observed_sensitivity_error, bool)
        and isinstance(observed_sensitivity_error, (int, float))
        and math.isfinite(float(observed_sensitivity_error))
        and math.isclose(
            float(observed_sensitivity_error),
            float(sensitivity_max_absolute_error),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "shield diagnostic sensitivity_max_absolute_error differs",
    )

    _require(
        isinstance(velocity_dimension, int)
        and not isinstance(velocity_dimension, bool)
        and velocity_dimension > 0,
        "registered shield velocity dimension is invalid",
    )
    if velocity_dimension == 7:
        _require(
            "velocity_dimension" not in diagnostics,
            "shield diagnostic velocity_dimension must be omitted for the 7D default",
        )
        _require(
            "torque_dimension" not in diagnostics,
            "shield diagnostic torque_dimension must be omitted for the 7D default",
        )
        observed_velocity_dimension = 7
        observed_torque_dimension = 7
    else:
        _require(
            "velocity_dimension" in diagnostics,
            "shield diagnostic velocity_dimension is absent",
        )
        _require(
            "torque_dimension" in diagnostics,
            "shield diagnostic torque_dimension is absent",
        )
        observed_velocity_dimension = diagnostics.get("velocity_dimension")
        observed_torque_dimension = diagnostics.get("torque_dimension")
    _require(
        isinstance(observed_velocity_dimension, int)
        and not isinstance(observed_velocity_dimension, bool)
        and observed_velocity_dimension == velocity_dimension,
        "shield diagnostic velocity_dimension differs",
    )
    _require(
        isinstance(observed_torque_dimension, int)
        and not isinstance(observed_torque_dimension, bool)
        and observed_torque_dimension == 7,
        "shield diagnostic torque_dimension differs",
    )


def _validate_solved_qp_certificate(
    row: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    controller: Mapping[str, Any],
    torque_actuators: Sequence[Mapping[str, Any]],
    robot_qvel_indices: Sequence[int],
    velocity_dimension: int,
    np: Any,
) -> Dict[str, float]:
    """Independently reconstruct the solved torque QP and its KKT optimum."""

    try:
        from scipy.optimize import nnls
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise OscCanaryValidationError(
            "SciPy is required for independent QP optimality validation"
        ) from error

    sensitivity = row.get("sensitivity")
    diagnostics = row.get("shield_diagnostics")
    _require(isinstance(sensitivity, Mapping), "solved row sensitivity is absent")
    _require(isinstance(diagnostics, Mapping), "solved row diagnostics are absent")
    full = np.asarray(sensitivity.get("full_epsilon_sensitivity"), dtype=np.float64)
    half = np.asarray(sensitivity.get("half_epsilon_sensitivity"), dtype=np.float64)
    selected = np.asarray(
        sensitivity.get("torque_to_next_output_qvel_sensitivity"), dtype=np.float64
    )
    epsilon = np.asarray(sensitivity.get("torque_epsilon_nm"), dtype=np.float64)
    _require(
        full.shape == half.shape == selected.shape == (int(velocity_dimension), 7)
        and epsilon.shape == (7,)
        and np.all(np.isfinite(full))
        and np.all(np.isfinite(half))
        and np.all(np.isfinite(selected))
        and np.all(np.isfinite(epsilon))
        and np.array_equal(selected, half)
        and np.all(epsilon == float(protocol["shield"]["torque_epsilon_nm"])),
        "two-resolution sensitivity shape or selected estimate differs",
    )
    atol = float(sensitivity.get("agreement_atol"))
    rtol = float(sensitivity.get("agreement_rtol"))
    _require(
        atol == float(protocol["shield"]["sensitivity_agreement_atol"])
        and rtol == float(protocol["shield"]["sensitivity_agreement_rtol"]),
        "sensitivity agreement tolerances differ",
    )
    difference = np.abs(full - half)
    agreement_scale = atol + rtol * np.maximum(np.abs(full), np.abs(half))
    max_absolute = float(np.max(difference))
    max_scaled = float(np.max(difference / agreement_scale))
    _require(
        np.all(difference <= agreement_scale)
        and math.isclose(
            float(sensitivity.get("maximum_epsilon_agreement_absolute_error")),
            max_absolute,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(sensitivity.get("maximum_epsilon_agreement_scaled_error")),
            max_scaled,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "two-resolution sensitivity agreement differs",
    )

    snapshot = sensitivity.get("snapshot")
    _require(isinstance(snapshot, Mapping), "sensitivity snapshot is absent")
    lower_torque = np.asarray(snapshot.get("arm_torque_lower_nm"), dtype=np.float64)
    upper_torque = np.asarray(snapshot.get("arm_torque_upper_nm"), dtype=np.float64)
    nominal_torque = np.asarray(row.get("nominal_torque_nm"), dtype=np.float64)
    command_torque = np.asarray(row.get("command_torque_nm"), dtype=np.float64)
    delta = command_torque - nominal_torque
    nominal_next = np.asarray(
        row.get("nominal_predicted_next_shield_qvel_rad_s"), dtype=np.float64
    )
    range_by_actuator = {
        int(actuator["actuator_id"]): actuator["control_range"]
        for actuator in torque_actuators
    }
    expected_control_ranges = np.asarray(
        [
            range_by_actuator[int(actuator_id)]
            for actuator_id in controller.get("arm_actuator_indexes", ())
        ],
        dtype=np.float64,
    )
    _require(
        lower_torque.shape == upper_torque.shape == nominal_torque.shape
        == command_torque.shape == (7,)
        and nominal_next.shape == (int(velocity_dimension),)
        and np.all(np.isfinite(lower_torque))
        and np.all(np.isfinite(upper_torque))
        and expected_control_ranges.shape == (7, 2)
        and np.array_equal(lower_torque, expected_control_ranges[:, 0])
        and np.array_equal(upper_torque, expected_control_ranges[:, 1])
        and np.array_equal(
            np.asarray(snapshot.get("nominal_arm_torque_nm"), dtype=np.float64),
            nominal_torque,
        )
        and np.array_equal(
            np.asarray(
                sensitivity.get("nominal_next_output_qvel_rad_s"),
                dtype=np.float64,
            ),
            nominal_next,
        )
        and tuple(sensitivity.get("output_qvel_indices", ()))
        == tuple(robot_qvel_indices)
        and tuple(snapshot.get("arm_actuator_ids", ()))
        == tuple(controller.get("arm_actuator_indexes", ()))
        and tuple(snapshot.get("arm_qpos_indices", ()))
        == tuple(controller.get("arm_qpos_indexes", ()))
        and tuple(snapshot.get("arm_qvel_indices", ()))
        == tuple(controller.get("arm_qvel_indexes", ()))
        and tuple(snapshot.get("output_qvel_indices", ()))
        == tuple(robot_qvel_indices)
        and float(snapshot.get("timestep_seconds")) == 0.002
        and sensitivity.get("non_arm_ctrl_preserved") is True
        and math.isfinite(float(sensitivity.get("nominal_next_time_seconds")))
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("integration_state_sha256", ""))))
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("all_ctrl_sha256", ""))))
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(sensitivity.get("nominal_next_integration_state_sha256", "")),
            )
        ),
        "sensitivity snapshot, perturbation bounds, or controller indexes differ",
    )
    stencil_rows = sensitivity.get("finite_difference_column_stencils")
    _require(
        isinstance(stencil_rows, list) and len(stencil_rows) == 7,
        "finite-difference stencil ledger is absent",
    )
    for column, plan in enumerate(stencil_rows):
        _require(isinstance(plan, Mapping), "finite-difference plan is invalid")
        base = float(nominal_torque[column])
        low = float(lower_torque[column])
        high = float(upper_torque[column])
        requested = float(epsilon[column])
        negative_room = base - low
        positive_room = high - base
        expected_full = _reconstruct_finite_difference_resolution(
            base=base,
            lower=low,
            upper=high,
            requested=requested,
            scale=1.0,
        )
        expected_stencil = str(expected_full["stencil"])
        _require(
            int(plan.get("column_index", -1)) == column
            and float(plan.get("nominal_torque_nm")) == base
            and float(plan.get("lower_bound_nm")) == low
            and float(plan.get("upper_bound_nm")) == high
            and float(plan.get("requested_full_epsilon_nm")) == requested
            and math.isclose(float(plan.get("available_negative_delta_nm")), negative_room, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(float(plan.get("available_positive_delta_nm")), positive_room, rel_tol=0.0, abs_tol=1e-15)
            and plan.get("difference_formula")
            == "(v_next(delta_1)-v_next(delta_0))/(delta_1-delta_0)",
            "finite-difference plan authority differs at column %d" % column,
        )
        for resolution_name, scale in (("full_resolution", 1.0), ("half_resolution", 0.5)):
            resolution = plan.get(resolution_name)
            _require(isinstance(resolution, Mapping), "stencil resolution is absent")
            deltas = resolution.get("sample_deltas_nm")
            _require(
                resolution.get("stencil") == expected_stencil
                and isinstance(deltas, (list, tuple))
                and len(deltas) == 2,
                "stencil type or samples differ at column %d" % column,
            )
            left, right = (float(deltas[0]), float(deltas[1]))
            denominator = right - left
            expected_resolution = _reconstruct_finite_difference_resolution(
                base=base,
                lower=low,
                upper=high,
                requested=requested,
                scale=scale,
            )
            expected_exact_deltas = tuple(
                float(value)
                for value in expected_resolution["sample_deltas_nm"]
            )
            _require(
                math.isfinite(left)
                and math.isfinite(right)
                and denominator > 0.0
                and math.isclose(
                    float(resolution.get("denominator_nm")),
                    denominator,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and all(
                    struct.pack("<d", observed)
                    == struct.pack("<d", expected)
                    for observed, expected in zip(
                        (left, right), expected_exact_deltas
                    )
                ),
                "stencil perturbation violates bounds at column %d" % column,
            )
            if expected_stencil == "centered":
                _require(left < 0.0 < right, "centered stencil is not two-sided")
            elif expected_stencil == "forward":
                _require(left == 0.0 < right, "forward stencil differs")
            else:
                _require(left < 0.0 == right, "backward stencil differs")
        expected_bound_adapted = bool(expected_full["bound_adapted"])
        _require(
            plan.get("bound_adapted") is expected_bound_adapted,
            "bound-adapted stencil flag differs at column %d" % column,
        )
    for field, observed in (
        ("nominal_torque_nm", nominal_torque),
        ("torque_lower_nm", lower_torque),
        ("torque_upper_nm", upper_torque),
        ("nominal_next_qvel_rad_s", nominal_next),
    ):
        _require(
            np.array_equal(np.asarray(diagnostics.get(field), dtype=np.float64), observed),
            "shield diagnostic %s differs" % field,
        )
    _validate_shield_diagnostic_input_binding(
        diagnostics,
        velocity_dimension=int(velocity_dimension),
        protocol=protocol,
        sensitivity_max_absolute_error=max_absolute,
    )

    h = np.asarray(row.get("poisson_h_m2"), dtype=np.float64)
    gradient_rows = np.asarray(
        row.get("joint_gradient_rows_m2_per_rad"), dtype=np.float64
    )
    _require(
        h.ndim == 1
        and h.size > 0
        and gradient_rows.shape == (h.size, int(velocity_dimension)),
        "solved CBF rows are invalid",
    )
    gain_rows = gradient_rows @ selected
    required_gain = (
        float(protocol["shield"]["margin_m2_per_s"])
        - float(protocol["shield"]["alpha_gain_per_s"]) * h
        - gradient_rows @ nominal_next
    )
    row_scales = np.max(np.abs(gain_rows), axis=1)
    uncertainty = np.sum(np.abs(gradient_rows), axis=1) * max_absolute
    controllable = row_scales - uncertainty > 0.0
    _require(
        not np.any((~controllable) & (required_gain > 0.0)),
        "serialized solved QP contains an uncontrollable violated row",
    )
    normalized_rows = gain_rows[controllable] / row_scales[controllable, None]
    normalized_lower = required_gain[controllable] / row_scales[controllable]
    delta_lower = lower_torque - nominal_torque
    delta_upper = upper_torque - nominal_torque
    feasibility_tolerance = max(
        5e-7,
        20.0 * float(protocol["shield"]["solver_eps_abs"]),
        20.0 * float(protocol["shield"]["solver_eps_rel"]),
    )
    normalized_slack = normalized_rows @ delta - normalized_lower
    _require(
        np.all(normalized_slack >= -feasibility_tolerance)
        and np.all(delta >= delta_lower - feasibility_tolerance)
        and np.all(delta <= delta_upper + feasibility_tolerance),
        "independently reconstructed solved QP is infeasible",
    )

    # For the strictly convex minimum-norm QP, primal feasibility plus a
    # nonnegative KKT multiplier certificate is sufficient and necessary for
    # the unique global optimum.  Reconstruct the certificate without OSQP.
    active_tolerance = max(2e-6, 50.0 * feasibility_tolerance)
    active_normals: List[Any] = [
        normalized_rows[index]
        for index, slack in enumerate(normalized_slack)
        if float(slack) <= active_tolerance
    ]
    identity = np.eye(7, dtype=np.float64)
    active_normals.extend(
        identity[index]
        for index in range(7)
        if float(delta[index] - delta_lower[index]) <= active_tolerance
    )
    active_normals.extend(
        -identity[index]
        for index in range(7)
        if float(delta_upper[index] - delta[index]) <= active_tolerance
    )
    _require(active_normals, "nonzero solved correction has no active QP constraint")
    active = np.asarray(active_normals, dtype=np.float64)
    multipliers, kkt_residual = nnls(active.T, delta)
    kkt_tolerance = max(2e-5, 200.0 * feasibility_tolerance) * max(
        1.0, float(np.linalg.norm(delta))
    )
    _require(
        np.all(np.isfinite(multipliers))
        and math.isfinite(float(kkt_residual))
        and float(kkt_residual) <= kkt_tolerance,
        "candidate is feasible but lacks an independent minimum-norm KKT certificate",
    )
    return {
        "maximum_sensitivity_absolute_error": max_absolute,
        "maximum_sensitivity_scaled_error": max_scaled,
        "minimum_normalized_qp_slack": (
            float(np.min(normalized_slack)) if normalized_slack.size else math.inf
        ),
        "kkt_stationarity_residual": float(kkt_residual),
        "correction_l2_nm": float(np.linalg.norm(delta)),
    }


def _manifest_case(path: Path, case_id: str) -> tuple[Dict[str, Any], str]:
    matches = []
    with path.open("rb") as stream:
        for raw in stream:
            if raw.strip():
                value = json.loads(raw)
                if value.get("case_id") == case_id:
                    matches.append(
                        (value, hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest())
                    )
    _require(len(matches) == 1, "contact manifest case is absent or duplicated")
    return matches[0]


def _bound_file(root: Path, relative: str, expected_sha256: str) -> Path:
    _require(
        isinstance(relative, str)
        and bool(relative)
        and not Path(relative).is_absolute()
        and all(part not in ("", ".", "..") for part in Path(relative).parts),
        "bound relative path is invalid",
    )
    current = root
    _require(current.is_dir() and not current.is_symlink(), "bound root is invalid")
    for part in Path(relative).parts:
        current = current / part
        _require(not current.is_symlink(), "bound path traverses a symlink")
    _require(current.is_file(), "bound file is absent")
    _require(fast._file_sha256(current) == expected_sha256, "bound file hash differs")
    return current


def _validate_historical_target_contact_gzip(
    path: Path,
    *,
    expected_payload_sha256: str,
    expected_case_id: str,
    expected_obstacle_name: str,
    expected_obstacle_root_body_name: str,
    expected_contact_action: int,
    expected_target_body_names: Sequence[str],
    literal_link56_body_names: Sequence[str],
) -> Dict[str, Any]:
    """Prove the frozen AEGIS contact directly from its detailed gzip.

    The manifest and compact historical result are useful indexes, but neither
    independently proves which robot body participated at one action.  This
    consumer therefore authenticates and reconstructs the complete detailed
    contact payload, then finds the first nonpositive-distance robot contact
    for every configured evaluation target link.
    """

    compressed = path.read_bytes()
    try:
        raw = gzip.decompress(compressed)
    except Exception as error:
        raise OscCanaryValidationError(
            "historical detailed contact gzip is invalid: %s" % error
        ) from error
    _require(
        hashlib.sha256(raw).hexdigest() == expected_payload_sha256,
        "historical detailed contact payload hash differs",
    )
    try:
        payload = json.loads(raw)
    except Exception as error:
        raise OscCanaryValidationError(
            "historical detailed contact payload is invalid JSON: %s" % error
        ) from error
    _require(
        isinstance(payload, Mapping)
        and fast._canonical(payload) == raw
        and payload.get("schema_version")
        == "vlsa_table1_active_obstacle_contacts.v3"
        and payload.get("case_id") == expected_case_id
        and payload.get("active_obstacle_name") == expected_obstacle_name,
        "historical detailed contact payload authority differs",
    )
    snapshots = payload.get("snapshots")
    _require(isinstance(snapshots, list) and snapshots, "contact snapshots are absent")
    target_body_names = [str(value) for value in expected_target_body_names]
    target_body_set = set(target_body_names)
    target_events_by_body: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in target_body_names
    }
    configured_action_link56_events: List[Dict[str, Any]] = []
    configured_action_robot_pair_link56_bodies: List[str] = []
    literal_link56 = {str(value) for value in literal_link56_body_names}
    _require(
        len(literal_link56) == 2
        and all(bool(value) for value in literal_link56)
        and target_body_names
        and len(target_body_names) == len(target_body_set)
        and target_body_set.issubset(literal_link56),
        "protocol-declared link5/link6 body authority differs",
    )
    action_snapshot_count = 0
    for snapshot_index, snapshot in enumerate(snapshots):
        _require(isinstance(snapshot, Mapping), "contact snapshot is malformed")
        step = snapshot.get("step")
        _require(
            isinstance(step, int)
            and not isinstance(step, bool)
            and snapshot.get("status") == "available"
            and snapshot.get("active_obstacle_name") == expected_obstacle_name
            and snapshot.get("role_authority", {}).get("status") == "complete",
            "contact snapshot authority differs at row %d" % snapshot_index,
        )
        if int(step) == int(expected_contact_action):
            action_snapshot_count += 1
        events = snapshot.get("events")
        robot_pairs = snapshot.get("robot_pairs")
        _require(
            isinstance(events, list) and isinstance(robot_pairs, list),
            "contact event or robot-pair ledger is absent",
        )
        if int(step) == int(expected_contact_action):
            for pair_index, pair in enumerate(robot_pairs):
                _require(
                    isinstance(pair, Mapping),
                    "configured contact robot-pair row is malformed",
                )
                lineage1 = pair.get("body_lineage1")
                lineage2 = pair.get("body_lineage2")
                _require(
                    isinstance(lineage1, list)
                    and lineage1
                    and isinstance(lineage2, list)
                    and lineage2,
                    "configured contact robot-pair lineage is absent at row %d"
                    % pair_index,
                )
                root1 = str(lineage1[0])
                root2 = str(lineage2[0])
                if root1 == expected_obstacle_root_body_name and root2 in literal_link56:
                    configured_action_robot_pair_link56_bodies.append(root2)
                elif root2 == expected_obstacle_root_body_name and root1 in literal_link56:
                    configured_action_robot_pair_link56_bodies.append(root1)
        for event_index, event in enumerate(events):
            _require(isinstance(event, Mapping), "contact event is malformed")
            unhashed = dict(event)
            declared_event_hash = unhashed.pop("event_sha256", None)
            _require(
                declared_event_hash == _canonical_sha256(unhashed)
                and event.get("step") == step,
                "contact event hash or step differs at snapshot %d event %d"
                % (snapshot_index, event_index),
            )
            obstacle = event.get("obstacle")
            other = event.get("other")
            if not isinstance(obstacle, Mapping) or not isinstance(other, Mapping):
                continue
            distance = event.get("distance")
            literal_robot_contact = bool(
                other.get("classification") == "robot"
                and isinstance(distance, (int, float))
                and not isinstance(distance, bool)
                and math.isfinite(float(distance))
                and float(distance) <= 0.0
            )
            if literal_robot_contact and other.get("body_name") in literal_link56:
                _require(
                    obstacle.get("body_name") == expected_obstacle_root_body_name,
                    "literal link5/link6 event obstacle root differs",
                )
                contact_row = {
                    "step": int(step),
                    "snapshot_index": snapshot_index,
                    "event_index": event_index,
                    "distance_m": float(distance),
                    "robot_body_name": str(other["body_name"]),
                    "robot_geom_name": str(other.get("geom_name")),
                    "obstacle_body_name": str(obstacle.get("body_name")),
                    "obstacle_geom_name": str(obstacle.get("geom_name")),
                    "event_sha256": str(declared_event_hash),
                }
                if int(step) == int(expected_contact_action):
                    configured_action_link56_events.append(contact_row)
                if other.get("body_name") in target_body_set:
                    target_events_by_body[str(other["body_name"])].append(
                        contact_row
                    )
    _require(
        action_snapshot_count == 1,
        "configured historical contact action is absent or duplicated",
    )
    _require(
        all(target_events_by_body.values()),
        "a configured target link has no literal historical contact",
    )
    first_target_action_by_body = {
        name: min(int(row["step"]) for row in target_events_by_body[name])
        for name in target_body_names
    }
    action_target_events = [
        row
        for row in configured_action_link56_events
        if str(row["robot_body_name"]) in target_body_set
    ]
    _require(
        all(
            value == int(expected_contact_action)
            for value in first_target_action_by_body.values()
        )
        and {str(row["robot_body_name"]) for row in action_target_events}
        == target_body_set,
        "configured action is not the first literal contact for every target link",
    )
    configured_action_link56_bodies = sorted(
        {str(row["robot_body_name"]) for row in configured_action_link56_events}
    )
    configured_action_robot_pair_link56_bodies = sorted(
        set(configured_action_robot_pair_link56_bodies)
    )
    _require(
        target_body_set.issubset(configured_action_link56_bodies)
        and target_body_set.issubset(
            configured_action_robot_pair_link56_bodies
        ),
        "configured action omits a target from the literal link5/link6 contacts",
    )
    return {
        "schema_version": "vlsa_poisson_historical_target_contact_audit.v1",
        "detailed_contact_file_sha256": fast._file_sha256(path),
        "detailed_contact_payload_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_count": len(snapshots),
        "target_link_body_names": target_body_names,
        "first_target_link_contact_source_action_by_body": (
            first_target_action_by_body
        ),
        "configured_action_literal_target_contact_count": len(action_target_events),
        "configured_action_target_nonpositive_distances_m": [
            float(row["distance_m"]) for row in action_target_events
        ],
        "configured_action_literal_link56_body_names": (
            configured_action_link56_bodies
        ),
        "configured_action_robot_pair_link56_body_names": (
            configured_action_robot_pair_link56_bodies
        ),
        "configured_action_target_geom_pairs": [
            {
                "target_link_body_name": row["robot_body_name"],
                "target_link_geom_name": row["robot_geom_name"],
                "obstacle_root_body_name": row["obstacle_body_name"],
                "obstacle_geom_name": row["obstacle_geom_name"],
            }
            for row in action_target_events
        ],
        "configured_action_event_sha256": _canonical_sha256(action_target_events),
    }


def _motion(rows: Sequence[Mapping[str, Any]], first: Any) -> Dict[str, float]:
    import numpy as np

    if first is None:
        return {"joint": 0.0, "eef": 0.0, "zero": 1.0}
    selected = [row for row in rows if int(row["physical_boundary"]) >= int(first)]
    if not selected:
        return {"joint": 0.0, "eef": 0.0, "zero": 1.0}
    joint = sum(
        float(row["measured_arm_qvel_l2_rad_s"]) * 0.002 for row in selected
    )
    positions = [np.asarray(row["eef_position_world_m"], dtype=np.float64) for row in selected]
    eef = sum(float(np.linalg.norm(right - left)) for left, right in zip(positions, positions[1:]))
    zero = sum(float(row["torque_correction_l2_nm"]) < 1e-12 for row in selected) / len(selected)
    return {"joint": float(joint), "eef": float(eef), "zero": float(zero)}


def _close(left: Any, right: Any, label: str, tolerance: float = 1e-12) -> None:
    _require(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance),
        "%s differs" % label,
    )


def _exact_float(left: Any, right: Any, label: str) -> None:
    _require(
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and not isinstance(right, bool)
        and isinstance(right, (int, float))
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and struct.pack(">d", float(left)) == struct.pack(">d", float(right)),
        "%s differs" % label,
    )


def _strict_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "%s must be an integer" % label,
    )
    return int(value)


def _strict_serialized_float64_vector3(value: Any, label: str, np: Any) -> Any:
    """Parse one producer JSON float64 vector without coercing other types."""

    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, float) and math.isfinite(item) for item in value),
        "%s must be a three-float JSON array" % label,
    )
    array = np.asarray(value, dtype=np.float64)
    _require(
        array.shape == (3,)
        and array.dtype == np.dtype(np.float64)
        and np.all(np.isfinite(array)),
        "%s float64 reconstruction differs" % label,
    )
    return array


def _validate_paper_car_cadence_and_historical_binding(
    car_ledger: Sequence[Mapping[str, Any]],
    *,
    action_count: int,
    historical_settled_position_sha256: str,
) -> None:
    """Require exact settled/20 Hz CAR rows and immutable settled pairing."""

    _require(
        isinstance(action_count, int)
        and not isinstance(action_count, bool)
        and action_count >= 0
        and isinstance(car_ledger, Sequence)
        and not isinstance(car_ledger, (str, bytes))
        and len(car_ledger) == action_count + 1,
        "paper CAR endpoint count differs",
    )
    _require(bool(car_ledger), "paper CAR settled row is absent")
    settled = car_ledger[0]
    _require(isinstance(settled, Mapping), "paper CAR settled row is invalid")
    settled_index = settled.get("source_action_index")
    _require(
        isinstance(settled_index, int)
        and not isinstance(settled_index, bool)
        and settled_index == -1
        and settled.get("snapshot_kind") == "settled_pre_action"
        and settled.get("active_obstacle_position_observation_array_sha256")
        == historical_settled_position_sha256,
        "paper CAR settled row or historical binding differs",
    )
    _exact_float(
        settled.get("l1_displacement_from_settled_m"),
        0.0,
        "paper CAR settled displacement",
    )
    for expected_index, row in enumerate(car_ledger[1:]):
        _require(isinstance(row, Mapping), "paper CAR endpoint row is invalid")
        observed_index = row.get("source_action_index")
        _require(
            isinstance(observed_index, int)
            and not isinstance(observed_index, bool)
            and observed_index == expected_index
            and row.get("snapshot_kind") == "completed_high_level_endpoint",
            "paper CAR endpoint cadence differs at row %d" % (expected_index + 1),
        )


def _require_resolved_body_name(
    resolved: Mapping[str, Any],
    *,
    ids_field: str,
    names_field: str,
    expected_body_id: int,
    expected_body_name: str,
    label: str,
) -> Dict[int, str]:
    """Validate one parallel resolved ID/name ledger and its selected row."""

    ids = resolved.get(ids_field)
    names = resolved.get(names_field)
    _require(
        isinstance(ids, Sequence)
        and not isinstance(ids, (str, bytes))
        and isinstance(names, Sequence)
        and not isinstance(names, (str, bytes))
        and len(ids) > 0
        and len(ids) == len(names),
        "%s resolved body ID/name ledger differs" % label,
    )
    parsed_ids = []
    parsed_names = []
    for body_id, body_name in zip(ids, names):
        _require(
            not isinstance(body_id, bool)
            and isinstance(body_id, int)
            and isinstance(body_name, str)
            and bool(body_name),
            "%s resolved body ID/name row is malformed" % label,
        )
        parsed_ids.append(int(body_id))
        parsed_names.append(str(body_name))
    _require(
        len(set(parsed_ids)) == len(parsed_ids),
        "%s resolved body IDs are duplicated" % label,
    )
    name_by_id = dict(zip(parsed_ids, parsed_names))
    _require(
        name_by_id.get(int(expected_body_id)) == str(expected_body_name),
        "%s selected root body name differs" % label,
    )
    return name_by_id


def _validate_paper_car_endpoint_row(
    row: Mapping[str, Any],
    *,
    index: int,
    car_key: str,
    obstacle_root_body_id: int,
    settled_car_position: Any,
    np: Any,
) -> float:
    """Independently reconstruct one exact paper-CAR observation endpoint."""

    observed_position = _strict_serialized_float64_vector3(
        row.get("active_obstacle_position_observation_world_m"),
        "paper CAR returned observation row %d" % index,
        np,
    )
    observable_value = _strict_serialized_float64_vector3(
        row.get("native_observable_value_world_m"),
        "paper CAR Observable.obs row %d" % index,
        np,
    )
    observation_cache_value = _strict_serialized_float64_vector3(
        row.get("native_observation_cache_value_world_m"),
        "paper CAR observation cache row %d" % index,
        np,
    )
    live_body_position = _strict_serialized_float64_vector3(
        row.get("active_obstacle_root_position_world_m"),
        "paper CAR live-root diagnostic row %d" % index,
        np,
    )
    forwarded_body_position = _strict_serialized_float64_vector3(
        row.get(
            "active_obstacle_root_position_post_integration_forwarded_world_m"
        ),
        "paper CAR forwarded diagnostic row %d" % index,
        np,
    )
    forwarded_component_delta = _strict_serialized_float64_vector3(
        row.get("observation_post_integration_forwarded_component_delta_m"),
        "paper CAR forwarded delta row %d" % index,
        np,
    )
    settled = _strict_serialized_float64_vector3(
        settled_car_position,
        "paper CAR settled observation",
        np,
    )
    observed_hash = _float64_sha256(observed_position, np)
    observable_hash = _float64_sha256(observable_value, np)
    cache_hash = _float64_sha256(observation_cache_value, np)
    live_hash = _float64_sha256(live_body_position, np)
    forwarded_hash = _float64_sha256(forwarded_body_position, np)
    reconstructed_observable_equal = bool(
        np.array_equal(observed_position, observable_value)
        and observed_hash == observable_hash
    )
    reconstructed_cache_equal = bool(
        np.array_equal(observed_position, observation_cache_value)
        and observed_hash == cache_hash
    )
    reconstructed_live_delta = live_body_position - observed_position
    reconstructed_live_l1_delta = float(
        np.sum(np.abs(reconstructed_live_delta))
    )
    reconstructed_live_linf_delta = float(
        np.max(np.abs(reconstructed_live_delta))
    )
    reconstructed_forwarded_delta = forwarded_body_position - observed_position
    forwarded_delta_hash = _float64_sha256(forwarded_component_delta, np)
    reconstructed_forwarded_delta_hash = _float64_sha256(
        reconstructed_forwarded_delta, np
    )
    _require(
        row.get("active_obstacle_position_observation_dtype")
        == np.dtype(np.float64).str
        and row.get("native_observable_value_dtype") == np.dtype(np.float64).str
        and row.get("native_observation_cache_value_dtype")
        == np.dtype(np.float64).str
        and row.get("active_obstacle_position_observation_shape") == [3]
        and row.get("native_observable_value_shape") == [3]
        and row.get("native_observation_cache_value_shape") == [3]
        and observed_position.shape == (3,)
        and observable_value.shape == (3,)
        and observation_cache_value.shape == (3,)
        and live_body_position.shape == (3,)
        and forwarded_body_position.shape == (3,)
        and forwarded_component_delta.shape == (3,)
        and settled.shape == (3,)
        and np.all(np.isfinite(observed_position))
        and np.all(np.isfinite(observable_value))
        and np.all(np.isfinite(observation_cache_value))
        and np.all(np.isfinite(live_body_position))
        and np.all(np.isfinite(forwarded_body_position))
        and np.all(np.isfinite(forwarded_component_delta))
        and np.all(np.isfinite(settled))
        and np.array_equal(forwarded_component_delta, reconstructed_forwarded_delta)
        and forwarded_delta_hash == reconstructed_forwarded_delta_hash
        and row.get("observation_observable_value_bitwise_equal")
        is reconstructed_observable_equal
        and row.get("observation_cache_value_bitwise_equal")
        is reconstructed_cache_equal
        and row.get("native_observable_cache_binding_exact") is True
        and reconstructed_observable_equal
        and reconstructed_cache_equal
        and row.get("paper_car_observation_key") == car_key,
        "paper CAR native observable/cache binding differs at row %d" % index,
    )
    _require(
        row.get("paper_car_position_source")
        == "selected_obstacle_pos_native_observation"
        and _strict_integer(
            row.get("active_obstacle_root_body_id"),
            "paper CAR row root body ID",
        )
        == int(obstacle_root_body_id)
        and row.get("active_obstacle_root_position_phase")
        == "live_body_xpos_after_cached_observation_return"
        and row.get("live_root_position_role")
        == "phase_diagnostic_only_not_authority_or_paper_car_metric"
        and row.get("post_integration_forwarded_pose_role")
        == "phase_diagnostic_only_not_paper_car_metric"
        and row.get("active_obstacle_position_observation_array_sha256")
        == observed_hash
        and row.get("native_observable_value_array_sha256") == observable_hash
        and row.get("native_observation_cache_value_array_sha256") == cache_hash
        and row.get("active_obstacle_root_position_array_sha256") == live_hash
        and row.get(
            "active_obstacle_root_position_post_integration_array_sha256"
        )
        == forwarded_hash,
        "paper CAR phase or array identity differs at row %d" % index,
    )
    _require(
        row.get(
            "observation_post_integration_forwarded_component_delta_array_sha256"
        )
        == forwarded_delta_hash,
        "paper CAR forwarded component-delta identity differs at row %d" % index,
    )
    _exact_float(
        row.get("observation_live_root_l1_delta_m"),
        reconstructed_live_l1_delta,
        "paper CAR live-root L1 delta row %d" % index,
    )
    _exact_float(
        row.get("observation_live_root_linf_delta_m"),
        reconstructed_live_linf_delta,
        "paper CAR live-root Linf delta row %d" % index,
    )
    _exact_float(
        row.get("observation_post_integration_forwarded_l1_delta_m"),
        float(np.sum(np.abs(reconstructed_forwarded_delta))),
        "paper CAR forwarded L1 diagnostic row %d" % index,
    )
    _exact_float(
        row.get("observation_post_integration_forwarded_linf_delta_m"),
        float(np.max(np.abs(reconstructed_forwarded_delta))),
        "paper CAR forwarded Linf diagnostic row %d" % index,
    )
    displacement = float(np.sum(np.abs(observed_position - settled)))
    _exact_float(
        row.get("l1_displacement_from_settled_m"),
        displacement,
        "paper CAR row %d" % index,
    )
    return displacement


def _validate_compact_constraint_trace(
    row: Mapping[str, Any],
    *,
    expected_sample_count: int,
    expected_sample_ledger_sha256: str,
    expected_robot_qvel_indices: Sequence[int],
    expected_robot_qvel_indices_sha256: str,
    expected_velocity_dimension: int,
    np: Any,
) -> Mapping[str, Mapping[str, Any]]:
    """Validate the bounded per-substep ledger without inventing raw arrays."""

    trace = row.get("constraint_trace")
    _require(isinstance(trace, Mapping), "compact constraint trace is absent")
    _require(
        set(trace)
        == {
            "schema_version",
            "array_hash_format",
            "sample_count",
            "sample_ledger_sha256",
            "robot_qvel_indices",
            "robot_qvel_indices_sha256",
            "velocity_dimension",
            "arrays",
        }
        and
        trace.get("schema_version")
        == "vlsa_poisson_compact_constraint_trace.v2"
        and trace.get("array_hash_format")
        == "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
        and int(trace.get("sample_count", -1)) == int(expected_sample_count),
        "compact constraint trace identity differs",
    )
    trace_qvel_indices = _strict_integer_list(
        trace.get("robot_qvel_indices"), "compact robot qvel indices"
    )
    _require(
        trace.get("sample_ledger_sha256") == expected_sample_ledger_sha256
        and trace_qvel_indices == [int(value) for value in expected_robot_qvel_indices]
        and trace.get("robot_qvel_indices_sha256")
        == expected_robot_qvel_indices_sha256
        == _canonical_sha256(trace_qvel_indices)
        and trace.get("velocity_dimension") == int(expected_velocity_dimension)
        == len(trace_qvel_indices),
        "compact constraint sample/qvel binding differs",
    )
    arrays = trace.get("arrays")
    expected_shapes = {
        "poisson_h_m2": [int(expected_sample_count)],
        "joint_gradient_rows_m2_per_rad": [
            int(expected_sample_count),
            int(expected_velocity_dimension),
        ],
        "actual_hdot_m2_per_s": [int(expected_sample_count)],
        "candidate_exact_clone_hdot_m2_per_s": [int(expected_sample_count)],
        "nominal_exact_clone_hdot_m2_per_s": [int(expected_sample_count)],
    }
    _require(
        isinstance(arrays, Mapping) and set(arrays) == set(expected_shapes),
        "compact constraint array ledger differs",
    )
    expected_dtype = np.dtype(np.float64).str
    for name, shape in expected_shapes.items():
        evidence = arrays.get(name)
        _require(
            isinstance(evidence, Mapping)
            and set(evidence) == {"dtype", "shape", "sha256", "minimum"}
            and evidence.get("dtype") == expected_dtype
            and evidence.get("shape") == shape
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))))
            and not isinstance(evidence.get("minimum"), bool)
            and math.isfinite(float(evidence.get("minimum"))),
            "compact %s evidence is malformed" % name,
        )
    _require(
        float(arrays["poisson_h_m2"]["minimum"]) > 0.0,
        "compact Poisson minimum is nonpositive",
    )
    return arrays


def _validate_live_nonarm_ctrl_evidence(
    row: Mapping[str, Any],
    *,
    expected_nonarm_ctrl_indices: Sequence[int],
    expected_nonarm_ctrl_indices_sha256: str,
    np: Any,
) -> None:
    """Validate live, not cloned, non-arm controls around one MuJoCo step."""

    count = len(expected_nonarm_ctrl_indices)
    _require(count > 0, "non-arm live-control authority is empty")
    before = row.get("live_nonarm_ctrl_before_array_record")
    after = row.get("live_nonarm_ctrl_after_array_record")
    _validate_compact_array_identity(
        before,
        expected_shape=(count,),
        label="live non-arm ctrl before",
        np=np,
    )
    _validate_compact_array_identity(
        after,
        expected_shape=(count,),
        label="live non-arm ctrl after",
        np=np,
    )
    _require(
        row.get("live_nonarm_ctrl_count") == count
        and row.get("live_nonarm_ctrl_indices_sha256")
        == expected_nonarm_ctrl_indices_sha256
        == _canonical_sha256(
            [int(value) for value in expected_nonarm_ctrl_indices]
        )
        and before == after
        and row.get("live_nonarm_ctrl_byte_identical") is True,
        "live non-arm controls changed across the executed transition",
    )


def _validate_compact_shield_diagnostics(
    row: Mapping[str, Any], *, expected_sample_count: int
) -> Mapping[str, Any]:
    evidence = row.get("compact_shield_diagnostics")
    status = row.get("shield_status")
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("schema_version")
        == "vlsa_poisson_compact_shield_diagnostics.v1"
        and evidence.get("shield_status") == status
        and int(evidence.get("sample_count", -1)) == int(expected_sample_count)
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(evidence.get("full_diagnostics_sha256", "")),
            )
        ),
        "compact shield diagnostics identity differs",
    )
    scalars = evidence.get("scalars")
    _require(isinstance(scalars, Mapping), "compact shield scalar ledger is absent")
    for name, value in scalars.items():
        _require(
            isinstance(name, str)
            and (
                value is None
                or (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                )
            ),
            "compact shield scalar %s is invalid" % name,
        )
    if status == "nominal_safe_exact_clone":
        _require(
            evidence.get("solver_attempted") is False
            and evidence.get("solver") is None
            and evidence.get("solver_status") is None
            and evidence.get("sensitivity_certificate_sha256") is None,
            "compact nominal-pass diagnostics differ",
        )
    else:
        _require(
            status == "solved"
            and evidence.get("solver_attempted") is True
            and evidence.get("solver") == "osqp"
            and str(evidence.get("solver_status", "")).lower()
            in ("solved", "solved inaccurate")
            and int(scalars.get("input_constraint_count", -1))
            == int(expected_sample_count)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(evidence.get("sensitivity_certificate_sha256", "")),
                )
            ),
            "compact solved-QP diagnostics differ",
        )
    return evidence


def _independent_constraint_attribution(
    nominal_residuals: Any,
    shield_samples: Sequence[Mapping[str, Any]],
    *,
    np: Any,
    sample_scope: str = (
        "all_structurally_movable_manipulator_collision_surfaces"
    ),
) -> Dict[str, Any]:
    """Reconstruct which movable-manipulator samples made the nominal step unsafe."""

    residuals = np.asarray(nominal_residuals, dtype=np.float64)
    _require(
        residuals.ndim == 1
        and residuals.size > 0
        and residuals.size == len(shield_samples)
        and np.all(np.isfinite(residuals)),
        "constraint attribution inputs differ",
    )
    minimum = float(np.min(residuals))
    minimum_index = int(np.argmin(residuals))
    tolerance = 1e-12
    near_indices = np.flatnonzero(np.abs(residuals - minimum) <= tolerance)
    negative_indices = np.flatnonzero(residuals < 0.0)
    return {
        "schema_version": "vlsa_poisson_constraint_attribution.v2",
        "sample_scope": str(sample_scope),
        "selection_rule": "first_np_argmin_of_nominal_exact_clone_cbf_residual",
        "minimum_nominal_residual_m2_per_s": minimum,
        "minimum_sample_index": minimum_index,
        "minimum_sample": dict(shield_samples[minimum_index]),
        "near_minimum_absolute_tolerance_m2_per_s": tolerance,
        "near_minimum_sample_indices": [int(value) for value in near_indices],
        "near_minimum_body_names": sorted(
            {
                str(shield_samples[int(value)]["body_name"])
                for value in near_indices
            }
        ),
        "negative_nominal_residual_sample_count": int(negative_indices.size),
        "negative_nominal_residual_body_names": sorted(
            {
                str(shield_samples[int(value)]["body_name"])
                for value in negative_indices
            }
        ),
    }


def _validate_first_divergence_full_certificate(
    row: Mapping[str, Any],
    *,
    compact_arrays: Mapping[str, Mapping[str, Any]],
    compact_diagnostics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    controller: Mapping[str, Any],
    torque_actuators: Sequence[Mapping[str, Any]],
    shield_samples: Sequence[Mapping[str, Any]],
    sample_ledger_sha256: str,
    robot_qvel_indices: Sequence[int],
    robot_qvel_indices_sha256: str,
    velocity_dimension: int,
    expected_certificate_roles: Sequence[str],
    np: Any,
) -> Dict[str, Any]:
    """Reconstruct raw residuals and the QP at a registered causal row."""

    certificate = row.get("full_constraint_qp_certificate")
    _require(isinstance(certificate, Mapping), "first divergence certificate is absent")
    target_link_v4 = bool(
        protocol.get("schema_version")
        == "vlsa_poisson_osc_target_link_canary_protocol.v4"
    )
    expected_schema = (
        "vlsa_poisson_divergence_or_material_full_qp_certificate.v4"
        if target_link_v4
        else "vlsa_poisson_first_divergence_full_qp_certificate.v2"
    )
    expected_roles = [str(value) for value in expected_certificate_roles]
    _require(
        certificate.get("schema_version") == expected_schema
        and (
            certificate.get("certificate_roles") == expected_roles
            if target_link_v4
            else "certificate_roles" not in certificate
        )
        and int(certificate.get("physical_boundary", -1))
        == int(row.get("physical_boundary", -2))
        and int(certificate.get("source_action_index", -1))
        == int(row.get("source_action_index", -2))
        and certificate.get("callback_endpoint_action_inner_substep")
        == row.get("callback_endpoint_action_inner_substep")
        and int(certificate.get("sample_count", -1))
        == int(row["constraint_trace"]["sample_count"])
        and certificate.get("sample_ledger_sha256") == sample_ledger_sha256
        and certificate.get("robot_qvel_indices")
        == [int(value) for value in robot_qvel_indices]
        and certificate.get("robot_qvel_indices_sha256")
        == robot_qvel_indices_sha256
        and certificate.get("velocity_dimension") == int(velocity_dimension)
        and row.get("full_constraint_qp_certificate_sha256")
        == _canonical_sha256(certificate),
        "first divergence certificate identity differs",
    )
    sample_count = int(certificate["sample_count"])
    arrays = {
        name: np.asarray(certificate.get(name), dtype=np.float64)
        for name in (
            "poisson_h_m2",
            "joint_gradient_rows_m2_per_rad",
            "actual_hdot_m2_per_s",
            "candidate_exact_clone_hdot_m2_per_s",
            "nominal_exact_clone_hdot_m2_per_s",
        )
    }
    expected_shapes = {
        "poisson_h_m2": (sample_count,),
        "joint_gradient_rows_m2_per_rad": (
            sample_count,
            int(velocity_dimension),
        ),
        "actual_hdot_m2_per_s": (sample_count,),
        "candidate_exact_clone_hdot_m2_per_s": (sample_count,),
        "nominal_exact_clone_hdot_m2_per_s": (sample_count,),
    }
    for name, array in arrays.items():
        _require(
            array.shape == expected_shapes[name]
            and np.all(np.isfinite(array))
            and compact_arrays[name].get("sha256") == _float64_sha256(array, np),
            "full certificate %s differs from its compact identity" % name,
        )
        _close(
            compact_arrays[name].get("minimum"),
            float(np.min(array)),
            "full certificate %s minimum" % name,
        )
    sensitivity = certificate.get("sensitivity")
    diagnostics = certificate.get("shield_diagnostics")
    _require(
        isinstance(sensitivity, Mapping)
        and isinstance(diagnostics, Mapping)
        and compact_diagnostics.get("sensitivity_certificate_sha256")
        == _canonical_sha256(sensitivity)
        and compact_diagnostics.get("full_diagnostics_sha256")
        == _canonical_sha256(diagnostics),
        "full sensitivity/diagnostic certificate hashes differ",
    )
    h = arrays["poisson_h_m2"]
    gradients = arrays["joint_gradient_rows_m2_per_rad"]
    measured = np.asarray(
        row.get("measured_shield_qvel_rad_s"), dtype=np.float64
    )
    candidate_qvel = np.asarray(
        row.get("predicted_next_shield_qvel_rad_s"), dtype=np.float64
    )
    nominal_qvel = np.asarray(
        row.get("nominal_predicted_next_shield_qvel_rad_s"), dtype=np.float64
    )
    _require(
        measured.shape == candidate_qvel.shape == nominal_qvel.shape
        == (int(velocity_dimension),)
        and np.all(np.isfinite(measured))
        and np.all(np.isfinite(candidate_qvel))
        and np.all(np.isfinite(nominal_qvel))
        and np.allclose(
            arrays["actual_hdot_m2_per_s"],
            gradients @ measured,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            arrays["candidate_exact_clone_hdot_m2_per_s"],
            gradients @ candidate_qvel,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            arrays["nominal_exact_clone_hdot_m2_per_s"],
            gradients @ nominal_qvel,
            rtol=0.0,
            atol=1e-12,
        ),
        "full divergence hdot reconstruction differs",
    )
    alpha_h_minus_margin = (
        float(protocol["shield"]["alpha_gain_per_s"]) * h
        - float(protocol["shield"]["margin_m2_per_s"])
    )
    residuals = {
        "minimum_actual_cbf_residual_m2_per_s": (
            arrays["actual_hdot_m2_per_s"] + alpha_h_minus_margin
        ),
        "candidate_exact_clone_minimum_cbf_residual_m2_per_s": (
            arrays["candidate_exact_clone_hdot_m2_per_s"]
            + alpha_h_minus_margin
        ),
        "nominal_exact_clone_minimum_cbf_residual_m2_per_s": (
            arrays["nominal_exact_clone_hdot_m2_per_s"]
            + alpha_h_minus_margin
        ),
    }
    for field, values in residuals.items():
        _close(row.get(field), float(np.min(values)), "full divergence %s" % field)
    independent_attribution = _independent_constraint_attribution(
        residuals["nominal_exact_clone_minimum_cbf_residual_m2_per_s"],
        shield_samples,
        np=np,
        sample_scope=(
            str(protocol["shield"]["binding_scope"])
            if target_link_v4
            else "all_structurally_movable_manipulator_collision_surfaces"
        ),
    )
    _require(
        fast._canonical(certificate.get("constraint_attribution"))
        == fast._canonical(independent_attribution),
        "first divergence full-robot constraint attribution differs",
    )
    tolerance = float(
        protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"]
    )
    _require(
        float(np.min(residuals["minimum_actual_cbf_residual_m2_per_s"]))
        >= -tolerance
        and float(
            np.min(
                residuals[
                    "candidate_exact_clone_minimum_cbf_residual_m2_per_s"
                ]
            )
        )
        >= -tolerance,
        "full divergence residual reconstruction failed",
    )
    certificate_row = dict(row)
    certificate_row.update(certificate)
    qp_audit = _validate_solved_qp_certificate(
        certificate_row,
        protocol=protocol,
        controller=controller,
        torque_actuators=torque_actuators,
        robot_qvel_indices=robot_qvel_indices,
        velocity_dimension=velocity_dimension,
        np=np,
    )
    return {
        **qp_audit,
        "constraint_attribution": independent_attribution,
        "nominal_exact_cbf_residuals_m2_per_s": residuals[
            "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
        ].tolist(),
    }


def _validate_registered_contact_evidence(
    *,
    scope: Any,
    measurement: Any,
    physics: Sequence[Mapping[str, Any]],
    resolved_geometry: Mapping[str, Any],
    physics_substeps_per_action: int = 25,
    physics_substeps_per_controller_update: int = 5,
    target_link_v4: bool = False,
) -> Dict[str, Any]:
    """Reconstruct every forbidden category from serialized authoritative IDs."""

    _require(isinstance(scope, Mapping), "registered contact scope is absent")
    scope_without_hash = dict(scope)
    identity_sha256 = scope_without_hash.pop("identity_sha256", None)
    _require(
        scope.get("schema_version") == "vlsa_poisson_registered_contact_scope.v1"
        and identity_sha256 == _canonical_sha256(scope_without_hash),
        "registered contact scope identity differs",
    )
    geom_count = int(scope.get("model_geom_count", -1))

    def id_set(field: str) -> set:
        values = scope.get(field)
        _require(
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and list(values) == sorted({int(value) for value in values}),
            "registered contact %s is malformed" % field,
        )
        return {int(value) for value in values}

    robot = id_set("robot_geom_ids")
    robot_owned = id_set("robot_owned_geom_ids")
    selected = id_set("selected_obstacle_geom_ids")
    link56 = id_set("link56_geom_ids")
    external = id_set("external_nonrobot_geom_ids")
    _require(
        geom_count > 0
        and robot
        and selected
        and link56
        and robot_owned | external == set(range(geom_count))
        and not robot_owned & external
        and robot <= robot_owned
        and selected <= external
        and link56 <= robot
        and robot
        == {int(value) for value in resolved_geometry.get("robot_geom_ids", ())}
        and selected
        == {int(value) for value in resolved_geometry.get("obstacle_geom_ids", ())}
        and link56
        == {int(value) for value in resolved_geometry.get("link56_geom_ids", ())},
        "registered contact ID partition differs",
    )
    identities = scope.get("geom_identities")
    _require(
        isinstance(identities, Sequence)
        and len(identities) == geom_count
        and [int(row.get("geom_id", -1)) for row in identities]
        == list(range(geom_count)),
        "registered geom identity ledger differs",
    )
    identity = {int(row["geom_id"]): row for row in identities}
    robot_body_ids = {
        int(value) for value in resolved_geometry.get("robot_body_ids", ())
    }
    _require(
        robot_body_ids
        and robot_owned
        == {
            geom_id
            for geom_id, row in identity.items()
            if int(row.get("body_id", -1)) in robot_body_ids
        },
        "registered robot-owned geom set differs from the robot body tree",
    )
    for geom_id, row in identity.items():
        _require(
            isinstance(row, Mapping)
            and isinstance(row.get("geom_name"), str)
            and isinstance(row.get("body_name"), str)
            and isinstance(row.get("body_id"), int)
            and row.get("is_robot_geom") is (geom_id in robot)
            and row.get("is_robot_owned_geom") is (geom_id in robot_owned)
            and row.get("is_selected_obstacle_geom") is (geom_id in selected)
            and row.get("is_link56_geom") is (geom_id in link56)
            and row.get("is_external_nonrobot_geom") is (geom_id in external),
            "registered geom identity role differs",
        )

    def validate_record(record: Any, *, settled: bool) -> Dict[str, Any]:
        _require(isinstance(record, Mapping), "registered contact row is invalid")
        unhashed = dict(record)
        record_sha256 = unhashed.pop("record_sha256", None)
        geom1 = int(record.get("geom1_id", -1))
        geom2 = int(record.get("geom2_id", -1))
        selected_contact = bool(
            (geom1 in robot and geom2 in selected)
            or (geom2 in robot and geom1 in selected)
        )
        link_external_contact = bool(
            (geom1 in link56 and geom2 in external)
            or (geom2 in link56 and geom1 in external)
        )
        _require(
            record.get("schema_version")
            == (
                "vlsa_poisson_registered_forbidden_contact.v1"
                if settled or not target_link_v4
                else "vlsa_poisson_registered_forbidden_contact.v2"
            )
            and geom1 in identity
            and geom2 in identity
            and geom1 != geom2
            and math.isfinite(float(record.get("contact_distance_m")))
            and float(record.get("contact_distance_m")) <= 0.0
            and (selected_contact or link_external_contact)
            and record_sha256 == _canonical_sha256(unhashed),
            "registered forbidden contact row identity differs",
        )
        robot_geom_id = geom1 if geom1 in robot else geom2
        external_geom_id = geom2 if robot_geom_id == geom1 else geom1
        expected_categories = []
        if selected_contact:
            expected_categories.append("any_robot_vs_selected_obstacle")
        if link_external_contact:
            expected_categories.append("link56_vs_external_nonrobot")
        _require(
            record.get("robot_geom_id") == robot_geom_id
            and record.get("external_geom_id") == external_geom_id
            and record.get("robot_geom_name")
            == identity[robot_geom_id]["geom_name"]
            and record.get("robot_body_id")
            == identity[robot_geom_id]["body_id"]
            and record.get("robot_body_name")
            == identity[robot_geom_id]["body_name"]
            and record.get("external_geom_name")
            == identity[external_geom_id]["geom_name"]
            and record.get("external_body_id")
            == identity[external_geom_id]["body_id"]
            and record.get("external_body_name")
            == identity[external_geom_id]["body_name"]
            and record.get("contact_categories") == expected_categories
            and record.get("any_robot_selected_obstacle_contact")
            is selected_contact
            and record.get("link56_external_nonrobot_contact")
            is link_external_contact
            and record.get("link56_nonselected_external_contact")
            is bool(link_external_contact and external_geom_id not in selected),
            "registered forbidden contact categories differ",
        )
        if settled:
            _require(
                record.get("source_phase") == "settled_forwarded"
                and record.get("physical_boundary") is None
                and record.get("executed_transition_start_boundary") is None
                and record.get("observed_state_boundary") == 0
                and record.get("source_action_index") is None
                and record.get("physics_substep_index") is None,
                "settled registered contact cadence differs",
            )
        else:
            boundary = int(record.get("physical_boundary", -1))
            flat_substep = boundary % int(physics_substeps_per_action)
            expected_controller_update = (
                flat_substep // int(physics_substeps_per_controller_update)
            )
            expected_local_substep = (
                flat_substep % int(physics_substeps_per_controller_update)
            )
            _require(
                record.get("source_phase")
                in (
                    "live_solver_phase_preintegration_geometry",
                    "post_integration_recomputed",
                )
                and 0 <= boundary < len(physics)
                and record.get("executed_transition_start_boundary") == boundary
                and record.get("observed_state_boundary") == boundary + 1
                and int(record.get("source_action_index", -1))
                == boundary // int(physics_substeps_per_action)
                and int(record.get("physics_substep_index", -1))
                == flat_substep
                and (
                    not target_link_v4
                    or (
                        int(record.get("controller_update_index", -1))
                        == expected_controller_update
                        and int(
                            record.get(
                                "physics_substep_within_controller_update",
                                -1,
                            )
                        )
                        == expected_local_substep
                        and record.get(
                            "callback_endpoint_action_inner_substep"
                        )
                        == [
                            boundary // int(physics_substeps_per_action),
                            expected_controller_update,
                            expected_local_substep,
                        ]
                    )
                ),
                "rollout registered contact cadence differs",
            )
        return dict(record)

    _require(
        isinstance(measurement, Mapping)
        and measurement.get("schema_version")
        == (
            "vlsa_poisson_registered_contact_measurement.v2"
            if target_link_v4
            else "vlsa_poisson_registered_contact_measurement.v1"
        )
        and measurement.get("scope_identity_sha256") == identity_sha256
        and int(measurement.get("observed_physics_substeps", -1)) == len(physics),
        "registered contact measurement header differs",
    )
    if target_link_v4:
        _require(
            measurement.get("callback_cadence")
            == {
                "controller_updates_per_action": (
                    int(physics_substeps_per_action)
                    // int(physics_substeps_per_controller_update)
                ),
                "physics_substeps_per_controller_update": int(
                    physics_substeps_per_controller_update
                ),
                "physics_substeps_per_action": int(
                    physics_substeps_per_action
                ),
            },
            "registered contact measurement callback cadence differs",
        )
    settled_records = measurement.get("settled_contact_records")
    rollout_records = measurement.get("rollout_contact_records")
    _require(
        isinstance(settled_records, Sequence)
        and isinstance(rollout_records, Sequence),
        "registered contact record ledgers are absent",
    )
    settled_validated = [
        validate_record(record, settled=True) for record in settled_records
    ]
    rollout_validated = [
        validate_record(record, settled=False) for record in rollout_records
    ]
    flattened_trace = []
    for boundary, row in enumerate(physics):
        records = row.get("registered_forbidden_contacts")
        _require(
            row.get("registered_contact_scope_checked") is True
            and isinstance(records, Sequence)
            and int(row.get("physical_boundary", -1)) == boundary,
            "physics row lacks registered contact monitoring",
        )
        validated = [validate_record(record, settled=False) for record in records]
        if target_link_v4:
            _require(
                all(
                    int(record.get("physical_boundary", -1)) == boundary
                    and record.get("source_action_index")
                    == row.get("source_action_index")
                    and record.get("physics_substep_index")
                    == row.get("physics_substep_index")
                    and record.get("controller_update_index")
                    == row.get("controller_update_index")
                    and record.get(
                        "physics_substep_within_controller_update"
                    )
                    == row.get("physics_substep_within_controller_update")
                    and record.get(
                        "callback_endpoint_action_inner_substep"
                    )
                    == row.get("callback_endpoint_action_inner_substep")
                    for record in validated
                ),
                "embedded registered contact phase differs from its physics row",
            )
        flattened_trace.extend(validated)
        categories = sorted(
            {
                category
                for record in validated
                for category in record["contact_categories"]
            }
        )
        _require(
            row.get("registered_forbidden_contact_seen") is bool(validated)
            and row.get("registered_contact_categories") == categories,
            "physics registered contact flags differ",
        )
        candidate = row.get("candidate_exact_clone_contact")
        _require(
            isinstance(candidate, Mapping)
            and candidate.get("literal_contact") is False
            and candidate.get("literal_contact_count") == 0
            and candidate.get("contacts") == []
            and candidate.get("contact_categories") == [],
            "executed candidate clone was not forbidden-contact free",
        )
    _require(
        _canonical_sha256(flattened_trace) == _canonical_sha256(rollout_validated),
        "registered contact trace and aggregate ledger differ",
    )
    all_records = settled_validated + rollout_validated
    categories = sorted(
        {
            category
            for record in rollout_validated
            for category in record["contact_categories"]
        }
    )
    selected_any = any(
        record["any_robot_selected_obstacle_contact"] for record in all_records
    )
    link_external_any = any(
        record["link56_external_nonrobot_contact"] for record in all_records
    )
    shifted_any = any(
        record["link56_nonselected_external_contact"] for record in all_records
    )
    _require(
        measurement.get("settled_forbidden_contact") is bool(settled_validated)
        and measurement.get("rollout_forbidden_contact")
        is bool(rollout_validated)
        and measurement.get("rollout_contact_categories") == categories
        and measurement.get("any_registered_forbidden_contact")
        is bool(all_records)
        and measurement.get("any_robot_selected_obstacle_contact")
        is selected_any
        and measurement.get("any_link56_external_nonrobot_contact")
        is link_external_any
        and measurement.get("any_link56_nonselected_external_contact")
        is shifted_any,
        "registered contact aggregate flags differ",
    )
    return {
        "settled_contact": bool(settled_validated),
        "rollout_contact": bool(rollout_validated),
        "rollout_contact_categories": categories,
        "rollout_contact_records_sha256": _canonical_sha256(rollout_validated),
        "any_robot_selected_obstacle_contact": selected_any,
        "any_link56_external_nonrobot_contact": link_external_any,
        "any_link56_nonselected_external_contact": shifted_any,
    }


def _validate_partial_action_ledger(
    *,
    terminal_kind: str,
    actions: Sequence[Mapping[str, Any]],
    planner_actions: Sequence[Mapping[str, Any]],
    partial_action_record: Any,
    physics_substep_count: int,
    physics_substeps_per_action: int = 25,
) -> Dict[str, Any]:
    """Validate the sole planner row allowed beyond completed actions."""

    partial_terminal = terminal_kind in (
        "literal_registered_forbidden_contact",
        "safety_method_stop_before_physics",
    )
    if not partial_terminal:
        _require(
            partial_action_record is None and len(planner_actions) == len(actions),
            "complete terminal contains a partial action",
        )
        return {"present": False, "completed_physics_substeps": 0}
    _require(
        isinstance(partial_action_record, Mapping)
        and len(planner_actions) == len(actions) + 1,
        "partial terminal does not contain exactly one extra planner action",
    )
    required_fields = {
        "schema_version",
        "terminal_kind",
        "source_action_index",
        "planner_action_trace_index",
        "executed_high_level_action",
        "executed_high_level_action_sha256",
        "planner_executed_action_sha256",
        "planner_native_observation_sha256",
        "pre_action_observation_sha256",
        "completed_physics_substeps",
        "physics_boundary_start",
        "physics_boundary_end_exclusive",
        "terminal_observation_kind",
        "terminal_observation_sha256",
        "terminal_forbidden_contact_categories",
        "terminal_forbidden_contact_records_sha256",
    }
    _require(
        set(partial_action_record) == required_fields
        and partial_action_record.get("schema_version")
        == "vlsa_poisson_partial_action_record.v1"
        and partial_action_record.get("terminal_kind") == terminal_kind,
        "partial action schema or terminal differs",
    )
    source_index = len(actions)
    planner_row = planner_actions[-1]
    completed = int(partial_action_record.get("completed_physics_substeps", -1))
    expected_completed = (
        int(physics_substep_count)
        - source_index * int(physics_substeps_per_action)
    )
    valid_count = (
        1 <= completed <= int(physics_substeps_per_action)
        if terminal_kind == "literal_registered_forbidden_contact"
        else 0 <= completed < int(physics_substeps_per_action)
    )
    action_bytes, action_hash = _float64_vector7_bytes_and_sha256(
        partial_action_record.get("executed_high_level_action")
    )
    planner_action_bytes, planner_action_hash = _float64_vector7_bytes_and_sha256(
        planner_row.get("executed")
    )
    for field in (
        "executed_high_level_action_sha256",
        "planner_executed_action_sha256",
        "planner_native_observation_sha256",
        "pre_action_observation_sha256",
        "terminal_observation_sha256",
        "terminal_forbidden_contact_records_sha256",
    ):
        _require(
            bool(
                re.fullmatch(
                    r"[0-9a-f]{64}", str(partial_action_record.get(field, ""))
                )
            ),
            "partial action %s is malformed" % field,
        )
    _require(
        int(partial_action_record.get("source_action_index", -1)) == source_index
        and int(partial_action_record.get("planner_action_trace_index", -1))
        == source_index
        and int(planner_row.get("source_action_index", -1)) == source_index
        and partial_action_record.get("executed_high_level_action_sha256")
        == action_hash
        and partial_action_record.get("planner_executed_action_sha256")
        == action_hash
        and planner_row.get("executed_action_sha256") == planner_action_hash
        and planner_action_hash == action_hash
        and planner_action_bytes == action_bytes
        and partial_action_record.get("planner_native_observation_sha256")
        == planner_row.get("native_observation_sha256")
        == partial_action_record.get("pre_action_observation_sha256")
        and completed == expected_completed
        and valid_count
        and int(partial_action_record.get("physics_boundary_start", -1))
        == source_index * int(physics_substeps_per_action)
        and int(partial_action_record.get("physics_boundary_end_exclusive", -1))
        == int(physics_substep_count),
        "partial action planner/physics binding differs",
    )
    if actions:
        _require(
            partial_action_record.get("pre_action_observation_sha256")
            == actions[-1].get("observation_sha256"),
            "partial action does not consume the last completed observation",
        )
    expected_kind = (
        "post_contact_observable_refresh"
        if terminal_kind == "literal_registered_forbidden_contact"
        else "post_method_stop_observable_refresh"
    )
    _require(
        partial_action_record.get("terminal_observation_kind") == expected_kind,
        "partial terminal observation kind differs",
    )
    categories = partial_action_record.get(
        "terminal_forbidden_contact_categories"
    )
    allowed_categories = {
        "any_robot_vs_selected_obstacle",
        "link56_vs_external_nonrobot",
    }
    _require(
        isinstance(categories, Sequence)
        and not isinstance(categories, (str, bytes))
        and list(categories) == sorted(set(categories))
        and set(categories) <= allowed_categories
        and (
            bool(categories)
            if terminal_kind == "literal_registered_forbidden_contact"
            else not categories
        ),
        "partial terminal forbidden-contact categories differ",
    )
    return {
        "present": True,
        "completed_physics_substeps": completed,
        "terminal_forbidden_contact_categories": list(categories),
        "terminal_forbidden_contact_records_sha256": partial_action_record[
            "terminal_forbidden_contact_records_sha256"
        ],
    }


def _validate_safety_method_stop(
    record: Any,
    *,
    terminal_kind: str,
    action_count: int,
    physics_substep_count: int,
    sample_count: int,
    sample_ledger_sha256: str,
    robot_qvel_indices: Sequence[int],
    robot_qvel_indices_sha256: str,
    velocity_dimension: int,
    physics_substeps_per_action: int = 25,
    physics_substeps_per_controller_update: int = 5,
    phase_correct_callback_required: bool = False,
) -> bool:
    """Bind a pre-physics stop to its unexecuted full-robot candidate boundary."""

    if terminal_kind != "safety_method_stop_before_physics":
        _require(record is None, "non-method terminal contains a method-stop record")
        return False
    _require(isinstance(record, Mapping), "method-stop record is absent")
    required_fields = {
        "schema_version",
        "source_action_index",
        "physics_substep_index",
        "physics_boundary_before_unexecuted_step",
        "unexecuted_post_integration_boundary",
        "reason",
        "sample_count",
        "sample_ledger_sha256",
        "robot_qvel_indices",
        "robot_qvel_indices_sha256",
        "velocity_dimension",
        "record_payload_sha256",
    }
    if phase_correct_callback_required:
        required_fields.add("callback_endpoint_action_inner_substep")
    _require(
        required_fields.issubset(record)
        and record.get("schema_version") == "vlsa_poisson_safety_method_stop.v2",
        "method-stop v2 schema or full-robot binding is absent",
    )
    source_action = record.get("source_action_index")
    substep = record.get("physics_substep_index")
    boundary = record.get("physics_boundary_before_unexecuted_step")
    post_boundary = record.get("unexecuted_post_integration_boundary")
    for value, label in (
        (source_action, "source action"),
        (substep, "physics substep"),
        (boundary, "candidate boundary"),
        (post_boundary, "post-integration boundary"),
    ):
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            "method-stop %s is invalid" % label,
        )
    expected_substep = (
        int(physics_substep_count)
        - int(action_count) * int(physics_substeps_per_action)
    )
    expected_callback_endpoint = [
        int(source_action),
        int(substep) // int(physics_substeps_per_controller_update),
        int(substep) % int(physics_substeps_per_controller_update),
    ]
    _require(
        int(source_action) == int(action_count)
        and 0 <= expected_substep < int(physics_substeps_per_action)
        and int(substep) == expected_substep
        and int(boundary) == int(physics_substep_count)
        == int(source_action) * int(physics_substeps_per_action) + int(substep)
        and int(post_boundary) == int(boundary) + 1,
        "method-stop candidate boundary arithmetic differs",
    )
    if (
        phase_correct_callback_required
        or "callback_endpoint_action_inner_substep" in record
    ):
        _require(
            record.get("callback_endpoint_action_inner_substep")
            == expected_callback_endpoint,
            "method-stop phase-correct callback endpoint differs",
        )
    bound_qvel_indices = _strict_integer_list(
        record.get("robot_qvel_indices"), "method-stop robot qvel indices"
    )
    _require(
        record.get("sample_count") == int(sample_count)
        and record.get("sample_ledger_sha256") == sample_ledger_sha256
        and bound_qvel_indices == [int(value) for value in robot_qvel_indices]
        and record.get("robot_qvel_indices_sha256")
        == robot_qvel_indices_sha256
        == _canonical_sha256(bound_qvel_indices)
        and record.get("velocity_dimension") == int(velocity_dimension)
        == len(bound_qvel_indices),
        "method-stop full-robot sample/qvel binding differs",
    )
    _require(
        isinstance(record.get("reason"), str)
        and record["reason"].endswith("stop_before_physics"),
        "method-stop reason differs",
    )
    unhashed = dict(record)
    payload_hash = unhashed.pop("record_payload_sha256", None)
    _require(
        isinstance(payload_hash, str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", payload_hash))
        and payload_hash == _canonical_sha256(unhashed),
        "method-stop payload hash differs",
    )
    return True


def validate(
    result_path: Path,
    protocol_path: Path,
    historical_result_root: Path,
    numeric_validation_result: Path,
    *,
    expected_case_id: str,
    expected_protocol_relative_path: str,
    expected_numeric_job_id: str,
    expected_producer_job_id: str,
    expected_producer_commit: str,
    expected_consumer_job_id: str,
    expected_consumer_commit: str,
) -> Dict[str, Any]:
    from main.poisson_fullbody.contracts import load_hashed_json
    from main.poisson_fullbody import osc_arm_link_canary as canary_contract
    from main.poisson_fullbody.osc_numeric_prerequisite import (
        validate_numeric_prerequisite_artifact,
    )

    protocol = fast._json(protocol_path, "post-OSC canary protocol")
    target_link_v4 = bool(
        protocol.get("schema_version")
        == "vlsa_poisson_osc_target_link_canary_protocol.v4"
    )
    if target_link_v4:
        result_schema = canary_contract.TARGET_LINK_RESULT_SCHEMA
        validate_protocol = canary_contract.validate_osc_target_link_canary_protocol
        classify_canary = canary_contract.classify_osc_target_link_canary
    else:
        result_schema = canary_contract.RESULT_SCHEMA
        validate_protocol = canary_contract.validate_osc_arm_link_canary_protocol
        classify_canary = canary_contract.classify_osc_arm_link_canary
    derived = validate_protocol(protocol)
    _require(
        bool(re.fullmatch(r"vlsa-t1-[A-Za-z0-9][A-Za-z0-9._-]{0,127}", expected_case_id)),
        "expected case ID is malformed",
    )
    _require(
        bool(
            re.fullmatch(
                r"configs/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json",
                expected_protocol_relative_path,
            )
        ),
        "expected protocol relative path is malformed",
    )
    for value, label in (
        (expected_numeric_job_id, "numeric job ID"),
        (expected_producer_job_id, "producer job ID"),
        (expected_consumer_job_id, "consumer job ID"),
    ):
        _require(bool(re.fullmatch(r"[0-9]+", value)), "%s is malformed" % label)
    for value, label in (
        (expected_producer_commit, "producer commit"),
        (expected_consumer_commit, "consumer commit"),
    ):
        _require(
            bool(re.fullmatch(r"[0-9a-f]{40}", value)),
            "%s is malformed" % label,
        )
    _require(
        len(
            {
                expected_numeric_job_id,
                expected_producer_job_id,
                expected_consumer_job_id,
            }
        )
        == 3,
        "numeric, producer, and consumer must use distinct Slurm allocations",
    )
    consumer_slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    _require(
        consumer_slurm_job_id == expected_consumer_job_id,
        "consumer Slurm job ID differs",
    )
    repo_root = protocol_path.resolve().parents[1]
    _require(
        protocol_path.resolve()
        == (repo_root / expected_protocol_relative_path).resolve()
        and protocol_path.resolve().relative_to(repo_root).as_posix()
        == expected_protocol_relative_path
        and protocol.get("case", {}).get("case_id") == expected_case_id,
        "case or protocol path differs from the explicit Slurm registration",
    )
    consumer_source = fast._git(repo_root)
    _require(
        consumer_source.get("commit") == expected_consumer_commit,
        "consumer source commit differs",
    )
    registered_result_root = result_path.parent.parent
    _require(
        numeric_validation_result.is_absolute()
        and numeric_validation_result.name == "result.json"
        and numeric_validation_result.parent.parent == registered_result_root
        and numeric_validation_result.parent.name.startswith(
            "vlsa-poisson-post-osc-numeric-"
        ),
        "numeric prerequisite is outside the registered result namespace",
    )
    numeric_prerequisite = validate_numeric_prerequisite_artifact(
        numeric_validation_result,
        contract=derived["numeric_prerequisite"],
        expected_job_id=expected_numeric_job_id,
        expected_commit=expected_producer_commit,
        expected_python_executable=(
            "/mnt/data/quanth/venvs/safety_vla/main/bin/python"
        ),
        expected_parent=numeric_validation_result.parent,
        forbidden_job_id=expected_consumer_job_id,
    )
    selection = protocol["selection"]
    from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol

    runtime_path = repo_root / protocol["runtime"]["relative_path"]
    _require(
        fast._file_sha256(runtime_path) == protocol["runtime"]["raw_file_sha256"],
        "runtime protocol raw hash differs",
    )
    runtime_protocol, runtime_hashes = load_feasibility_protocol(
        runtime_path,
        expected_protocol_sha256=protocol["runtime"]["semantic_sha256"],
    )
    _require(
        runtime_hashes.parameter_block_sha256
        == protocol["runtime"]["parameter_block_sha256"],
        "runtime protocol parameter block differs",
    )
    study_protocol_path = repo_root / selection["study_protocol_relative_path"]
    manifest_path = repo_root / selection["manifest_relative_path"]
    manifest_receipt_path = repo_root / selection["manifest_receipt_relative_path"]
    _require(
        fast._file_sha256(study_protocol_path)
        == selection["study_protocol_file_sha256"],
        "study protocol hash differs",
    )
    _require(
        fast._file_sha256(manifest_path) == selection["manifest_file_sha256"],
        "contact manifest hash differs",
    )
    _require(
        fast._file_sha256(manifest_receipt_path)
        == selection["manifest_receipt_file_sha256"],
        "contact manifest receipt hash differs",
    )
    remote_receipt_path = repo_root / selection["remote_artifact_receipt_relative_path"]
    _require(
        fast._file_sha256(remote_receipt_path)
        == selection["remote_artifact_receipt_file_sha256"],
        "remote artifact receipt hash differs",
    )
    case, case_row_sha256 = _manifest_case(manifest_path, protocol["case"]["case_id"])
    _require(case_row_sha256 == selection["case_row_sha256"], "case row hash differs")
    historical_binding = case.get("historical_result")
    _require(isinstance(historical_binding, Mapping), "historical binding is absent")
    historical_path = _bound_file(
        historical_result_root,
        str(historical_binding.get("relative_path")),
        str(historical_binding.get("file_sha256")),
    )
    historical_value = fast._json(historical_path, "historical AEGIS result")
    from main.poisson_fullbody.shadow_replay import load_historical_action_replay

    replay = load_historical_action_replay(
        historical_path,
        expected_case_id=protocol["case"]["case_id"],
        expected_arm=protocol["historical_control"]["source_arm"],
    )
    _require(
        replay.result_file_sha256
        == protocol["historical_control"]["result_file_sha256"]
        and replay.result_payload_sha256
        == protocol["historical_control"]["result_payload_sha256"]
        and replay.executed_sequence_sha256
        == protocol["historical_control"]["executed_action_sequence_sha256"],
        "historical replay binding differs",
    )
    remote_receipt = fast._json(remote_receipt_path, "remote artifact receipt")
    _require(
        remote_receipt.get("case_id") == case["case_id"]
        and Path(str(remote_receipt.get("historical_root", ""))).resolve()
        == historical_result_root.resolve(),
        "remote receipt authority differs",
    )
    remote_rows = remote_receipt.get("artifacts")
    _require(isinstance(remote_rows, Sequence) and len(remote_rows) == 3, "remote receipt rows differ")
    auxiliary = case["artifact_bindings"]
    expected_remote = {
        "result.json": historical_binding["file_sha256"],
        "episode.mp4": auxiliary["video_sha256"],
        "active_obstacle_contacts.json.gz": auxiliary[
            "detailed_contact_gzip_sha256"
        ],
    }
    historical_relative = Path(str(historical_binding["relative_path"]))
    expected_remote_relative = {
        "result.json": historical_relative.as_posix(),
        "episode.mp4": (historical_relative.parent / "episode.mp4").as_posix(),
        "active_obstacle_contacts.json.gz": (
            historical_relative.parent / "active_obstacle_contacts.json.gz"
        ).as_posix(),
    }
    remote_targets: Dict[str, Path] = {}
    for remote_row in remote_rows:
        _require(isinstance(remote_row, Mapping), "remote receipt row is invalid")
        basename = Path(str(remote_row.get("relative_path", ""))).name
        _require(basename in expected_remote, "remote receipt filename differs")
        _require(
            str(remote_row.get("relative_path")) == expected_remote_relative[basename],
            "remote receipt path differs from the bound historical case directory",
        )
        target = _bound_file(
            historical_result_root,
            str(remote_row["relative_path"]),
            str(expected_remote[basename]),
        )
        _require(
            remote_row.get("regular_file") is True
            and remote_row.get("symlink") is False
            and int(remote_row.get("byte_count", -1)) == int(target.stat().st_size),
            "remote artifact regular-file receipt differs",
        )
        _require(basename not in remote_targets, "remote artifact filename is duplicated")
        remote_targets[basename] = target
    _require(
        set(remote_targets) == set(expected_remote),
        "remote source-artifact receipt is incomplete",
    )
    target_link_body_names = protocol["case"].get("target_link_body_names")
    if target_link_body_names is None and not target_link_v4:
        target_link_body_names = [
            protocol["case"].get("historical_direct_contact_body")
        ]
    _require(
        isinstance(target_link_body_names, list)
        and target_link_body_names
        and len(target_link_body_names) == len(set(target_link_body_names))
        and all(
            isinstance(value, str) and bool(value)
            for value in target_link_body_names
        )
        and (
            not target_link_v4
            or target_link_body_names
            == list(derived["target_link_body_names"])
        ),
        "configured target-link authority differs",
    )
    target_link_body_names = [str(value) for value in target_link_body_names]
    target_link_body_name_set = set(target_link_body_names)
    _require(
        target_link_body_name_set.issubset(
            set(case.get("literal_link_contact_bodies", ()))
        ),
        "a configured target link is absent from the frozen contact manifest",
    )
    protected_link_body_names = (
        list(derived.get("protected_link_body_names", ()))
        if target_link_v4
        else list(protocol["case"]["literal_link56_body_names"])
    )
    _require(
        len(protected_link_body_names) == 2
        and len(set(protected_link_body_names)) == 2
        and all(
            isinstance(value, str) and bool(value)
            for value in protected_link_body_names
        )
        and target_link_body_name_set.issubset(protected_link_body_names)
        and set(protected_link_body_names)
        == set(protocol["case"]["literal_link56_body_names"])
        and (
            not target_link_v4
            or protocol["shield"].get("protected_link_body_names")
            == protected_link_body_names
        ),
        "shared protected-link body authority differs",
    )
    historical_target_contact_audit = _validate_historical_target_contact_gzip(
        remote_targets["active_obstacle_contacts.json.gz"],
        expected_payload_sha256=str(auxiliary["detailed_contact_payload_sha256"]),
        expected_case_id=expected_case_id,
        expected_obstacle_name=str(protocol["case"]["selected_obstacle_name"]),
        expected_obstacle_root_body_name=str(
            protocol["case"]["selected_obstacle_root_body_name"]
        ),
        expected_contact_action=int(derived["historical_contact_action"]),
        expected_target_body_names=target_link_body_names,
        literal_link56_body_names=protocol["case"][
            "literal_link56_body_names"
        ],
    )
    manifest_direct_pairs = case.get("direct_link_active_obstacle_pairs")
    detailed_geom_pairs = {
        (
            str(row["target_link_body_name"]),
            str(row["target_link_geom_name"]),
            str(row["obstacle_root_body_name"]),
            str(row["obstacle_geom_name"]),
        )
        for row in historical_target_contact_audit[
            "configured_action_target_geom_pairs"
        ]
    }
    manifest_target_pairs = {
        (
            str(row.get("actual_link_body_name")),
            str(row.get("link_geom_name")),
            str(row.get("active_obstacle_actual_body_name")),
            str(row.get("active_obstacle_geom_name")),
        )
        for row in manifest_direct_pairs or ()
        if isinstance(row, Mapping)
        and row.get("actual_link_body_name") in target_link_body_name_set
    }
    _require(
        int(case.get("first_sampled_link_contact_control_step", -1))
        == int(derived["historical_contact_action"])
        and case.get("first_link_step_is_timing_resolved") is True
        and case.get("only_link56_selected_obstacle_robot_contact") is True
        and case.get("direct_link_active_obstacle_pairs_sha256")
        == _canonical_sha256(manifest_direct_pairs)
        and detailed_geom_pairs
        and detailed_geom_pairs.issubset(manifest_target_pairs),
        "detailed target contact does not match the frozen manifest action/pair",
    )
    result = load_hashed_json(result_path)
    _require(result.get("schema_version") == result_schema, "result schema differs")
    _require(result.get("status") == "complete", "partial or failed result rejected")
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result.get("partial_output_interpreted") is False, "partial-output flag differs")
    _require(result.get("protocol_id") == protocol["protocol_id"], "protocol ID differs")
    _require(
        result.get("case_id") == protocol["case"]["case_id"] == expected_case_id,
        "case differs",
    )
    _require(result.get("claim_scope") == protocol["claim_scope"], "claim scope differs")
    provenance = result.get("provenance")
    _require(isinstance(provenance, Mapping), "provenance is absent")
    _require(
        fast._canonical(provenance.get("numeric_prerequisite"))
        == fast._canonical(numeric_prerequisite),
        "producer numeric prerequisite binding differs",
    )
    source = provenance.get("source")
    _require(isinstance(source, Mapping) and not source.get("status_short"), "source is dirty")
    _require(
        source.get("commit") == expected_producer_commit,
        "producer source commit differs",
    )
    allocation = provenance.get("allocation")
    _require(isinstance(allocation, Mapping), "Slurm allocation is absent")
    _require(
        str(allocation.get("slurm_job_id")) == expected_producer_job_id,
        "producer Slurm job ID differs",
    )
    _require(
        isinstance(allocation.get("gpu_name"), str)
        and "H100" in allocation["gpu_name"],
        "producer H100 allocation is not verified",
    )
    expected_evaluation_python = Path(
        "/mnt/data/quanth/venvs/safety_vla/main/bin/python"
    ).resolve()
    _require(
        Path(str(provenance.get("python_executable", ""))).resolve()
        == expected_evaluation_python
        and Path(sys.executable).resolve() == expected_evaluation_python,
        "producer or consumer evaluation Python differs",
    )
    producer_versions = provenance.get("evaluation_package_versions")
    _require(isinstance(producer_versions, Mapping), "producer package versions are absent")
    consumer_versions = {
        package: importlib.metadata.version(package)
        for package in ("mujoco", "numpy", "scipy", "osqp", "robosuite")
    }
    _require(
        dict(producer_versions) == consumer_versions,
        "producer and consumer evaluation package versions differ",
    )
    _require(provenance.get("historical_control_rerun") is False, "baseline was rerun")
    _require(
        provenance.get("protocol_file_sha256") == fast._file_sha256(protocol_path),
        "protocol raw hash differs",
    )
    for field, expected in (
        ("study_protocol_file_sha256", selection["study_protocol_file_sha256"]),
        ("manifest_file_sha256", selection["manifest_file_sha256"]),
        ("manifest_receipt_file_sha256", selection["manifest_receipt_file_sha256"]),
        (
            "remote_artifact_receipt_file_sha256",
            selection["remote_artifact_receipt_file_sha256"],
        ),
        ("manifest_case_row_sha256", selection["case_row_sha256"]),
    ):
        _require(provenance.get(field) == expected, "%s differs" % field)
    checkpoint_identity = provenance.get("checkpoint_identity")
    _require(isinstance(checkpoint_identity, Mapping), "checkpoint identity is absent")
    from scripts.run_poisson_closed_loop_canary import _checkpoint_identity

    independently_observed_checkpoint = _checkpoint_identity(protocol["online_policy"])
    _require(
        fast._canonical(checkpoint_identity)
        == fast._canonical(independently_observed_checkpoint)
        and checkpoint_identity.get("full_content_tree_sha256")
        == protocol["online_policy"]["checkpoint_tree_sha256"]
        and checkpoint_identity.get("receipt_file_sha256")
        == protocol["online_policy"]["checkpoint_receipt_sha256"]
        and checkpoint_identity.get("current_filesystem_identity_matches_receipt")
        is True,
        "producer checkpoint identity differs from independent filesystem receipt",
    )

    treatment = result.get("treatment")
    metrics = result.get("metrics")
    planner = result.get("planner")
    video = result.get("video")
    historical = result.get("historical_control")
    apparatus = result.get("apparatus")
    for value, label in (
        (treatment, "treatment"),
        (metrics, "metrics"),
        (planner, "planner"),
        (video, "video"),
        (historical, "historical_control"),
        (apparatus, "apparatus"),
    ):
        _require(isinstance(value, Mapping), "%s is absent" % label)
    active_cadence = runtime_protocol.get("cadence", {}).get("active", {})
    _require(
        isinstance(active_cadence, Mapping),
        "active callback cadence is absent from the frozen runtime",
    )
    controller_updates_per_action = _strict_integer(
        active_cadence.get("filter_updates_per_high_level_action"),
        "controller updates per high-level action",
    )
    physics_substeps_per_controller_update = _strict_integer(
        active_cadence.get("physics_substeps_per_filter_update"),
        "physics substeps per controller update",
    )
    physics_substeps_per_action = (
        controller_updates_per_action
        * physics_substeps_per_controller_update
    )
    callback_cadence = apparatus.get("callback_cadence")
    _require(
        controller_updates_per_action == 5
        and physics_substeps_per_controller_update == 5
        and physics_substeps_per_action == 25
        and isinstance(callback_cadence, Mapping)
        and callback_cadence.get("schema_version")
        == "vlsa_poisson_callback_cadence.v1"
        and callback_cadence.get("high_level_actions_hz")
        == int(protocol["execution"]["high_level_frequency_hz"])
        == int(runtime_protocol["cadence"]["high_level_frequency_hz"])
        and callback_cadence.get("controller_updates_per_high_level_action")
        == controller_updates_per_action
        and callback_cadence.get("physics_substeps_per_controller_update")
        == physics_substeps_per_controller_update
        and callback_cadence.get("physics_substeps_per_high_level_action")
        == physics_substeps_per_action
        and callback_cadence.get("typed_endpoint_order")
        == [
            "source_action_index",
            "controller_update_index",
            "physics_substep_within_controller_update",
        ],
        "serialized callback cadence differs from the frozen 5x5 runtime",
    )
    field_audit = _validate_field_sample_static_evidence(
        apparatus=apparatus,
        treatment=treatment,
        metrics=metrics,
        runtime_protocol=runtime_protocol,
        runtime_protocol_sha256=runtime_hashes.protocol_sha256,
        runtime_parameter_block_sha256=runtime_hashes.parameter_block_sha256,
        target_link_v4=target_link_v4,
        expected_target_link_body_names=target_link_body_names,
        expected_protected_link_body_names=protected_link_body_names,
    )
    _require(
        metrics.get("contact_scope") == protocol["execution"]["contact_scope"],
        "registered union contact scope differs",
    )
    _require(historical.get("rerun") is False, "historical baseline rerun differs")
    direct_pairs = case.get("direct_link_active_obstacle_pairs")
    historical_telemetry = historical_value.get("contact_telemetry")
    historical_unique_pairs = (
        historical_telemetry.get("unique_contact_pairs")
        if isinstance(historical_telemetry, Mapping)
        else None
    )
    historical_geom_pairs = {
        (str(row.get("geom1")), str(row.get("geom2")))
        for row in historical_unique_pairs or ()
        if isinstance(row, Mapping)
    }
    direct_contact_bodies = {
        str(row.get("actual_link_body_name"))
        for row in direct_pairs or ()
        if isinstance(row, Mapping)
        and row.get("actual_link_body_name") in target_link_body_name_set
        and (
            (
                str(row.get("active_obstacle_geom_name")),
                str(row.get("link_geom_name")),
            )
            in historical_geom_pairs
            or (
                str(row.get("link_geom_name")),
                str(row.get("active_obstacle_geom_name")),
            )
            in historical_geom_pairs
        )
    }
    direct_historical_contact = bool(
        isinstance(direct_pairs, Sequence)
        and direct_pairs
        and direct_contact_bodies == target_link_body_name_set
        and historical_telemetry.get("first_contact_step")
        == int(derived["historical_contact_action"])
        and all(
            value == int(derived["historical_contact_action"])
            for value in historical_target_contact_audit[
                "first_target_link_contact_source_action_by_body"
            ].values()
        )
    )
    if target_link_v4:
        producer_target_evidence = apparatus.get(
            "historical_target_link_contact_evidence"
        )
        _require(
            direct_historical_contact
            and historical.get("archived_direct_target_link_contact") is True
            and isinstance(producer_target_evidence, Mapping)
            and producer_target_evidence.get("schema_version")
            == "vlsa_poisson_historical_target_link_contact_evidence.v1"
            and producer_target_evidence.get("case_id") == expected_case_id
            and producer_target_evidence.get("selected_obstacle_name")
            == protocol["case"]["selected_obstacle_name"]
            and producer_target_evidence.get("selected_obstacle_root_body_name")
            == protocol["case"]["selected_obstacle_root_body_name"]
            and producer_target_evidence.get("target_link_body_names")
            == target_link_body_names
            and producer_target_evidence.get(
                "first_target_link_contact_source_action_by_body"
            )
            == historical_target_contact_audit[
                "first_target_link_contact_source_action_by_body"
            ]
            and producer_target_evidence.get(
                "configured_historical_contact_source_action"
            )
            == int(derived["historical_contact_action"])
            and producer_target_evidence.get(
                "exact_action_link56_contact_body_names"
            )
            == historical_target_contact_audit[
                "configured_action_literal_link56_body_names"
            ]
            and producer_target_evidence.get(
                "exact_action_robot_pair_link56_body_names"
            )
            == historical_target_contact_audit[
                "configured_action_robot_pair_link56_body_names"
            ]
            and producer_target_evidence.get(
                "exact_action_target_link_nonpositive_contact_distances_m"
            )
            == historical_target_contact_audit[
                "configured_action_target_nonpositive_distances_m"
            ]
            and producer_target_evidence.get("detailed_contact_file_sha256")
            == historical_target_contact_audit["detailed_contact_file_sha256"]
            and producer_target_evidence.get("compact_result_binding_verified")
            is True
            and producer_target_evidence.get(
                "detailed_timing_and_target_attribution_verified"
            )
            is True
            and producer_target_evidence.get("verified") is True
            and metrics.get("historical_direct_target_link_contact_verified")
            is True,
            "historical direct target-link contact is absent",
        )
    else:
        _require(
            direct_historical_contact
            and historical.get("direct_link56_selected_obstacle_contact_verified")
            is True
            and metrics.get("historical_direct_link56_contact_verified") is True,
            "historical direct contact is absent",
        )
    _require(
        replay.historical_task_success is True
        and historical.get("historical_task_success") is True
        and metrics.get("historical_control_task_success") is True,
        "historical task failed",
    )
    controller = apparatus.get("controller")
    _require(
        isinstance(controller, Mapping)
        and controller.get("original_osc_verified") is True
        and controller.get("controller_name") == "OSC_POSE"
        and controller.get("controller_class_qualname")
        == "OperationalSpaceController"
        and str(controller.get("controller_class_module", "")).endswith(".osc")
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(controller.get("controller_implementation_file_sha256", "")),
            )
        )
        and controller.get("controller_control_dim") == 6
        and controller.get("environment_action_dim") == 7
        and controller.get("control_frequency_hz") == 20
        and controller.get("physics_frequency_hz") == 500,
        "original OSC cadence is not verified",
    )
    controller_module = importlib.import_module(
        str(controller["controller_class_module"])
    )
    controller_class = getattr(
        controller_module, str(controller["controller_class_qualname"]), None
    )
    controller_source_path = inspect.getsourcefile(controller_class)
    _require(
        controller_source_path is not None
        and fast._file_sha256(Path(controller_source_path))
        == controller["controller_implementation_file_sha256"],
        "consumer-installed OSC implementation differs from producer",
    )
    import numpy as np

    if target_link_v4:
        shield_binding = _validate_protected_link_shield_sampling_binding(
            apparatus=apparatus,
            resolved_geometry=apparatus["resolved_geometry"],
            target_resolved_geometry=apparatus["target_link_resolved_geometry"],
            protected_resolved_geometry=apparatus[
                "protected_link_resolved_geometry"
            ],
            expected_target_link_body_names=target_link_body_names,
            expected_protected_link_body_names=protected_link_body_names,
            shield_contract=protocol["shield"],
            controller=controller,
            np=np,
        )
    else:
        shield_binding = _validate_shield_sampling_binding(
            apparatus=apparatus,
            resolved_geometry=apparatus["resolved_geometry"],
            controller=controller,
            runtime_protocol=runtime_protocol,
            np=np,
        )
    torque_units = apparatus.get("arm_actuator_torque_units")
    _require(
        apparatus.get("field_frame")
        == "world_frame_frozen_at_post_settling_selected_obstacle_pose",
        "Poisson obstacle-field frame differs",
    )
    _require(
        isinstance(torque_units, Mapping)
        and torque_units.get("schema_version")
        == "vlsa_poisson_direct_hinge_torque_units.v1"
        and torque_units.get("control_unit") == "newton_metre",
        "arm actuator torque-unit contract is absent",
    )
    torque_actuators = torque_units.get("actuators")
    _require(
        isinstance(torque_actuators, Sequence) and len(torque_actuators) == 7,
        "torque-unit actuator ledger differs",
    )
    actuator_ids = set()
    joint_ids = set()
    torque_units_verified = True
    required_torque_checks = {
        "joint_transmission",
        "transmission_joint_matches",
        "hinge_joint",
        "fixed_gain",
        "unit_gain",
        "no_bias",
        "no_activation_dynamics",
        "control_limit_enabled",
        "unit_joint_gear",
        "force_limit_cannot_clip_control_range",
    }
    for actuator in torque_actuators:
        _require(isinstance(actuator, Mapping), "torque actuator row is invalid")
        actuator_ids.add(int(actuator.get("actuator_id", -1)))
        joint_ids.add(int(actuator.get("joint_id", -1)))
        checks = actuator.get("checks")
        control_range = actuator.get("control_range")
        force_range = actuator.get("force_range")
        gain_parameters = actuator.get("gain_parameters")
        bias_parameters = actuator.get("bias_parameters")
        gear = actuator.get("gear")
        raw_limit_valid = bool(
            actuator.get("control_limited") is True
            and isinstance(control_range, list)
            and len(control_range) == 2
            and all(math.isfinite(float(value)) for value in control_range)
            and float(control_range[0]) < float(control_range[1])
            and isinstance(force_range, list)
            and len(force_range) == 2
            and all(math.isfinite(float(value)) for value in force_range)
            and (
                actuator.get("force_limited") is False
                or (
                    actuator.get("force_limited") is True
                    and float(force_range[0]) <= float(control_range[0])
                    and float(force_range[1]) >= float(control_range[1])
                )
            )
            and isinstance(gain_parameters, list)
            and len(gain_parameters) == 10
            and float(gain_parameters[0]) == 1.0
            and all(float(value) == 0.0 for value in gain_parameters[1:])
            and isinstance(bias_parameters, list)
            and len(bias_parameters) == 10
            and all(float(value) == 0.0 for value in bias_parameters)
            and isinstance(gear, list)
            and len(gear) == 6
            and float(gear[0]) == 1.0
            and all(float(value) == 0.0 for value in gear[1:])
        )
        row_verified = bool(
            isinstance(checks, Mapping)
            and set(checks) == required_torque_checks
            and all(checks.get(field) is True for field in required_torque_checks)
            and actuator.get("verified") is True
            and raw_limit_valid
        )
        torque_units_verified = torque_units_verified and row_verified
    torque_units_verified = bool(
        torque_units_verified
        and len(actuator_ids) == 7
        and len(joint_ids) == 7
        and torque_units.get("verified") is True
    )
    _require(
        torque_units_verified
        and metrics.get("direct_unit_gain_hinge_torque_actuators_verified") is True,
        "direct unit-gain hinge torque authority differs",
    )

    physics = treatment.get("physics_trace")
    actions = treatment.get("action_trace")
    parity = treatment.get("parity_trace")
    goals = treatment.get("goal_progress_ledger")
    car_ledger = treatment.get("paper_car_endpoint_ledger")
    _require(isinstance(physics, Sequence), "physics trace is absent")
    _require(isinstance(actions, Sequence), "action trace is absent")
    _require(isinstance(parity, Sequence), "parity trace is absent")
    _require(isinstance(goals, Sequence), "goal trace is absent")
    _require(isinstance(car_ledger, Sequence) and car_ledger, "paper CAR ledger is absent")
    _require(
        treatment.get("paper_car_endpoint_ledger_schema_version")
        == "vlsa_poisson_paper_car_endpoint_ledger.v3",
        "paper CAR ledger schema differs",
    )
    car_authority = apparatus.get("paper_car_authority")
    resolved_car_geometry = apparatus.get("resolved_geometry")
    resolved_obstacle_roots = (
        list(resolved_car_geometry.get("obstacle_root_body_ids", ()))
        if isinstance(resolved_car_geometry, Mapping)
        else []
    )
    _require(isinstance(car_authority, Mapping), "paper CAR authority is absent")
    observable_car_root_id = _strict_integer(
        car_authority.get("observable_root_body_id"),
        "paper CAR observable root body ID",
    )
    contact_car_root_id = _strict_integer(
        car_authority.get("contact_authority_root_body_id"),
        "paper CAR contact root body ID",
    )
    _require(
        len(resolved_obstacle_roots) == 1,
        "paper CAR resolved obstacle root count differs",
    )
    resolved_car_root_id = _strict_integer(
        resolved_obstacle_roots[0], "paper CAR resolved obstacle root body ID"
    )
    _require(
        car_authority.get("schema_version")
        == "vlsa_poisson_paper_car_authority.v2"
        and car_authority.get("metric")
        == protocol["paper_car_measurement"]["metric"]
        and car_authority.get("paper_car_observation_key")
        == "%s_pos" % protocol["case"]["selected_obstacle_name"]
        and car_authority.get("selected_obstacle_name")
        == protocol["case"]["selected_obstacle_name"]
        and observable_car_root_id == contact_car_root_id == resolved_car_root_id
        and car_authority.get("observable_root_body_name")
        == protocol["case"]["selected_obstacle_root_body_name"]
        and car_authority.get("contact_authority_root_body_name")
        == protocol["case"]["selected_obstacle_root_body_name"]
        and car_authority.get("observable_and_contact_root_body_ids_equal") is True
        and car_authority.get("position_source")
        == protocol["paper_car_measurement"]["position_source"]
        and car_authority.get("root_body_binding")
        == protocol["paper_car_measurement"]["root_body_binding"]
        and car_authority.get("native_observable_binding")
        == protocol["paper_car_measurement"]["native_observable_binding"]
        and car_authority.get("live_root_position_role")
        == protocol["paper_car_measurement"]["live_root_position_role"]
        and car_authority.get("post_integration_forwarded_pose_role")
        == protocol["paper_car_measurement"][
            "post_integration_forwarded_pose_role"
        ]
        and car_authority.get(
            "historical_settled_active_obstacle_position_sha256"
        )
        == replay.settled_active_obstacle_position_sha256,
        "paper CAR observation/root-body authority differs",
    )
    native_observable = car_authority.get("native_observable")
    expected_observable_checks = {
        "observable_name_matches_key": True,
        "observable_enabled": True,
        "observable_active": True,
        "observable_modality_is_object": True,
        "observable_sampling_timestep_is_20hz": True,
        "sensor_function_is_obj_pos": True,
        "sensor_nonlocal_object_name_matches": True,
        "sensor_nonlocal_environment_matches": True,
    }
    _require(
        isinstance(native_observable, Mapping)
        and native_observable.get("schema_version")
        == "vlsa_poisson_native_object_observable.v1"
        and native_observable.get("observation_key")
        == "%s_pos" % protocol["case"]["selected_obstacle_name"]
        and native_observable.get("observable_name")
        == native_observable.get("observation_key")
        and native_observable.get("observable_modality") == "object"
        and native_observable.get("sensor_function_name") == "obj_pos"
        and native_observable.get("sensor_nonlocal_object_name")
        == protocol["case"]["selected_obstacle_name"]
        and native_observable.get("checks") == expected_observable_checks
        and native_observable.get("all_checks_passed") is True,
        "paper CAR native observable identity differs",
    )
    _exact_float(
        native_observable.get("sampling_timestep_s"),
        0.05,
        "paper CAR native observable sampling timestep",
    )
    _require_resolved_body_name(
        resolved_car_geometry,
        ids_field="obstacle_body_ids",
        names_field="obstacle_body_names",
        expected_body_id=observable_car_root_id,
        expected_body_name=protocol["case"]["selected_obstacle_root_body_name"],
        label="paper CAR obstacle",
    )
    shield_samples = shield_binding["samples"]
    if target_link_v4:
        structural_partition = None
        _require(
            isinstance(shield_samples, Sequence)
            and len(shield_samples) == int(shield_binding["sample_count"])
            == int(field_audit["protected_sample_count"])
            and shield_binding["exact_protected_field_ledger"] is True
            and metrics.get("protected_link_shield_sampling_exact") is True,
            "configured protected-link shield attribution ledger differs",
        )
        _require(
            treatment.get("physics_trace_schema_version")
            == "vlsa_poisson_osc_movable_manipulator_compact_physics_trace.v3"
            and treatment.get("full_qp_certificate_scope")
            == (
                "first_byte_divergence_and_first_material_correction_rows_"
                "deduplicated"
            ),
            "target-link compact physics trace contract differs",
        )
    else:
        structural_partition = shield_binding["structural_partition"]
        _require(
            isinstance(shield_samples, Sequence)
            and len(shield_samples) == int(shield_binding["sample_count"])
            == int(structural_partition["movable_sample_count"])
            and int(field_audit["all_robot_sample_count"])
            == int(structural_partition["movable_sample_count"])
            + int(structural_partition["fixed_sample_count"]),
            "movable-manipulator shield attribution ledger differs",
        )
        _require(
            treatment.get("physics_trace_schema_version")
            == "vlsa_poisson_osc_movable_manipulator_compact_physics_trace.v3"
            and treatment.get("full_qp_certificate_scope")
            == "first_byte_different_torque_row_only",
            "compact physics trace contract differs",
        )
    _require(
        [int(row["physical_boundary"]) for row in physics] == list(range(len(physics))),
        "physics boundaries are incomplete or duplicated",
    )
    _require(
        all(
            int(row["source_action_index"])
            == int(row["physical_boundary"]) // physics_substeps_per_action
            and int(row["physics_substep_index"])
            == int(row["physical_boundary"]) % physics_substeps_per_action
            and int(row["controller_update_index"])
            == int(row["physics_substep_index"])
            // physics_substeps_per_controller_update
            and int(row["physics_substep_within_controller_update"])
            == int(row["physics_substep_index"])
            % physics_substeps_per_controller_update
            and row.get("callback_endpoint_action_inner_substep")
            == [
                int(row["source_action_index"]),
                int(row["controller_update_index"]),
                int(row["physics_substep_within_controller_update"]),
            ]
            for row in physics
        ),
        "physics/action cadence differs",
    )
    _require(
        all(
            row.get("shield_status")
            in ("nominal_safe_exact_clone", "solved")
            for row in physics
        ),
        "invalid shield decision is present",
    )
    _require(
        all(
            isinstance(row.get("candidate_exact_clone_contact"), Mapping)
            and row["candidate_exact_clone_contact"].get("literal_contact") is False
            and float(row["candidate_exact_clone_minimum_cbf_residual_m2_per_s"])
            >= -float(protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"])
            for row in physics
        ),
        "exact nonlinear candidate pre-physics postcheck failed",
    )
    solved_qp_count = 0
    independently_validated_qp_audits: List[Dict[str, Any]] = []
    independently_validated_qp_audits_by_boundary: Dict[int, Dict[str, Any]] = {}
    first_byte_divergence_seen = False
    first_material_certificate_seen = False
    shield_qvel_indices = list(shield_binding["robot_qvel_indices"])
    shield_velocity_dimension = int(shield_binding["velocity_dimension"])
    nonarm_ctrl_indices = list(shield_binding["nonarm_ctrl_indices"])
    nonarm_ctrl_indices_sha256 = str(
        shield_binding["nonarm_ctrl_indices_sha256"]
    )
    shield_qvel_index_to_position = {
        qvel_index: position
        for position, qvel_index in enumerate(shield_qvel_indices)
    }
    arm_positions = [
        shield_qvel_index_to_position[qvel_index]
        for qvel_index in shield_binding["arm_qvel_indices"]
    ]
    for index, row in enumerate(physics):
        _require(
            not {
                "measured_qvel_rad_s",
                "measured_qvel_l2_rad_s",
                "predicted_next_qvel_rad_s",
                "nominal_predicted_next_qvel_rad_s",
                "prediction_error_l2_rad_s",
                "candidate_exact_clone_matches_live_qvel",
            }.intersection(row),
            "stale ambiguous v1 qvel fields appear at physics row %d" % index,
        )
        compact_arrays = _validate_compact_constraint_trace(
            row,
            expected_sample_count=int(shield_binding["sample_count"]),
            expected_sample_ledger_sha256=str(
                shield_binding["sample_ledger_sha256"]
            ),
            expected_robot_qvel_indices=shield_qvel_indices,
            expected_robot_qvel_indices_sha256=_canonical_sha256(
                shield_qvel_indices
            ),
            expected_velocity_dimension=shield_velocity_dimension,
            np=np,
        )
        compact_diagnostics = _validate_compact_shield_diagnostics(
            row,
            expected_sample_count=int(shield_binding["sample_count"]),
        )
        _validate_live_nonarm_ctrl_evidence(
            row,
            expected_nonarm_ctrl_indices=nonarm_ctrl_indices,
            expected_nonarm_ctrl_indices_sha256=nonarm_ctrl_indices_sha256,
            np=np,
        )
        nominal_torque = np.asarray(row.get("nominal_torque_nm"), dtype=np.float64)
        command_torque = np.asarray(row.get("command_torque_nm"), dtype=np.float64)
        torque_delta = np.asarray(row.get("torque_delta_nm"), dtype=np.float64)
        measured_arm_qvel = np.asarray(
            row.get("measured_arm_qvel_rad_s"), dtype=np.float64
        )
        measured_shield_qvel = np.asarray(
            row.get("measured_shield_qvel_rad_s"), dtype=np.float64
        )
        candidate_arm_qvel = np.asarray(
            row.get("predicted_next_arm_qvel_rad_s"), dtype=np.float64
        )
        nominal_arm_qvel = np.asarray(
            row.get("nominal_predicted_next_arm_qvel_rad_s"), dtype=np.float64
        )
        candidate_shield_qvel = np.asarray(
            row.get("predicted_next_shield_qvel_rad_s"), dtype=np.float64
        )
        nominal_shield_qvel = np.asarray(
            row.get("nominal_predicted_next_shield_qvel_rad_s"), dtype=np.float64
        )
        _require(
            nominal_torque.shape == (7,)
            and command_torque.shape == (7,)
            and torque_delta.shape == (7,)
            and measured_arm_qvel.shape == (7,)
            and candidate_arm_qvel.shape == (7,)
            and nominal_arm_qvel.shape == (7,)
            and measured_shield_qvel.shape == (shield_velocity_dimension,)
            and candidate_shield_qvel.shape == (shield_velocity_dimension,)
            and nominal_shield_qvel.shape == (shield_velocity_dimension,),
            "compact physics vectors are invalid at physics row %d" % index,
        )
        for array in (
            nominal_torque,
            command_torque,
            torque_delta,
            measured_arm_qvel,
            candidate_arm_qvel,
            nominal_arm_qvel,
            measured_shield_qvel,
            candidate_shield_qvel,
            nominal_shield_qvel,
        ):
            _require(np.all(np.isfinite(array)), "non-finite physics array at row %d" % index)
        _require(
            np.array_equal(measured_arm_qvel, measured_shield_qvel[arm_positions])
            and np.array_equal(
                candidate_arm_qvel, candidate_shield_qvel[arm_positions]
            )
            and np.array_equal(
                nominal_arm_qvel, nominal_shield_qvel[arm_positions]
            ),
            "arm and full-robot qvel projections differ at row %d" % index,
        )
        _require(
            row.get("nominal_non_arm_ctrl_preserved") is True
            and row.get("candidate_non_arm_ctrl_preserved") is True,
            "non-arm controls changed at physics row %d" % index,
        )
        reconstructed_delta = command_torque - nominal_torque
        _require(
            np.allclose(reconstructed_delta, torque_delta, rtol=0.0, atol=1e-12),
            "torque delta differs at row %d" % index,
        )
        _close(
            row.get("torque_correction_l2_nm"),
            float(np.linalg.norm(torque_delta)),
            "torque correction row %d" % index,
        )
        _close(
            row.get("measured_arm_qvel_l2_rad_s"),
            float(np.linalg.norm(measured_arm_qvel)),
            "arm qvel norm row %d" % index,
        )
        _close(
            row.get("shield_prediction_error_l2_rad_s"),
            float(np.linalg.norm(measured_shield_qvel - candidate_shield_qvel)),
            "shield prediction error row %d" % index,
        )
        byte_equal = bool(
            command_torque.tobytes(order="C")
            == nominal_torque.tobytes(order="C")
        )
        _require(
            row.get("command_byte_identical_to_nominal") is byte_equal
            and row.get("nominal_torque_sha256")
            == hashlib.sha256(nominal_torque.tobytes(order="C")).hexdigest()
            and row.get("command_torque_sha256")
            == hashlib.sha256(command_torque.tobytes(order="C")).hexdigest(),
            "torque byte authority differs at row %d" % index,
        )
        _require(
            bool(
                np.allclose(
                    measured_shield_qvel,
                    candidate_shield_qvel,
                    rtol=0.0,
                    atol=1e-10,
                )
            )
            is bool(
                row.get("candidate_exact_clone_matches_live_shield_qvel")
            ),
            "candidate/live equality flag differs at row %d" % index,
        )
        candidate_contact = row.get("candidate_exact_clone_contact")
        _require(
            isinstance(candidate_contact, Mapping)
            and candidate_contact.get("literal_contact") is False
            and int(candidate_contact.get("literal_contact_count", -1)) == 0
            and candidate_contact.get("contacts") == []
            and candidate_contact.get("phases_checked")
            == [
                "live_solver_phase_preintegration_geometry",
                "post_integration_recomputed",
            ],
            "candidate clone contact union differs at row %d" % index,
        )
        if row.get("shield_status") == "nominal_safe_exact_clone":
            _require(
                byte_equal
                and np.array_equal(torque_delta, np.zeros(7))
                and compact_diagnostics.get("solver_attempted") is False,
                "nominal exact pass-through evidence differs",
            )
        else:
            _require(
                row.get("shield_status") == "solved"
                and compact_diagnostics.get("solver_attempted") is True,
                "solved QP evidence differs",
            )
            solved_qp_count += 1
        is_first_byte_divergence = bool(
            not byte_equal and not first_byte_divergence_seen
        )
        is_material_correction = bool(
            float(row["torque_correction_l2_nm"])
            >= float(derived["acceptance"]["material_torque_correction_l2_nm"])
        )
        is_first_material_correction = bool(
            is_material_correction and not first_material_certificate_seen
        )
        certificate_roles = []
        if is_first_byte_divergence:
            certificate_roles.append("first_byte_divergence")
        if target_link_v4 and is_first_material_correction:
            certificate_roles.append("first_material_correction")
        if certificate_roles:
            _require(
                row.get("shield_status") == "solved",
                "registered certificate row is not a solved QP",
            )
            audit = _validate_first_divergence_full_certificate(
                row,
                compact_arrays=compact_arrays,
                compact_diagnostics=compact_diagnostics,
                protocol=protocol,
                controller=controller,
                torque_actuators=torque_actuators,
                shield_samples=shield_samples,
                sample_ledger_sha256=str(
                    shield_binding["sample_ledger_sha256"]
                ),
                robot_qvel_indices=shield_qvel_indices,
                robot_qvel_indices_sha256=_canonical_sha256(
                    shield_qvel_indices
                ),
                velocity_dimension=shield_velocity_dimension,
                expected_certificate_roles=certificate_roles,
                np=np,
            )
            audit = {
                **audit,
                "physical_boundary": int(row["physical_boundary"]),
                "certificate_roles": certificate_roles,
            }
            independently_validated_qp_audits.append(audit)
            independently_validated_qp_audits_by_boundary[
                int(row["physical_boundary"])
            ] = audit
            first_byte_divergence_seen = True
        else:
            _require(
                row.get("full_constraint_qp_certificate") is None
                and row.get("full_constraint_qp_certificate_sha256") is None,
                "full QP certificate appears outside a registered audit row",
            )
        if is_first_material_correction:
            first_material_certificate_seen = True
        residual_scalars = (
            "minimum_actual_cbf_residual_m2_per_s",
            "candidate_exact_clone_minimum_cbf_residual_m2_per_s",
            "nominal_exact_clone_minimum_cbf_residual_m2_per_s",
        )
        _require(
            all(
                not isinstance(row.get(field), bool)
                and math.isfinite(float(row.get(field)))
                for field in residual_scalars
            ),
            "compact residual minima are invalid at row %d" % index,
        )
        _require(
            float(row["candidate_exact_clone_minimum_cbf_residual_m2_per_s"])
            >= -float(protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"])
            and float(row["minimum_actual_cbf_residual_m2_per_s"])
            >= -float(protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"]),
            "compact CBF residual minimum failed at physics row %d" % index,
        )
    nonarm_controls_unchanged = all(
        row.get("nominal_non_arm_ctrl_preserved") is True
        and row.get("candidate_non_arm_ctrl_preserved") is True
        and row.get("live_nonarm_ctrl_byte_identical") is True
        and row.get("live_nonarm_ctrl_before_array_record")
        == row.get("live_nonarm_ctrl_after_array_record")
        for row in physics
    )
    if target_link_v4:
        _require(
            metrics.get("protected_link_shield_sampling_exact") is True
            and shield_binding["exact_protected_field_ledger"] is True
            and metrics.get(
                "all_authoritative_robot_collision_surfaces_contact_monitored"
            )
            is True
            and metrics.get("seven_arm_torque_decision_verified") is True
            and metrics.get("nonarm_controls_unchanged")
            is nonarm_controls_unchanged
            and nonarm_controls_unchanged,
            "protected-link shield/contact apparatus metrics differ from raw evidence",
        )
    else:
        link56_seed_only = bool(
            int(field_audit["protected_sample_count"])
            < int(shield_binding["sample_count"])
        )
        _require(
            metrics.get(
                "all_structurally_movable_manipulator_collision_surfaces_shielded"
            )
            is structural_partition[
                "all_structurally_movable_manipulator_collision_surfaces_shielded"
            ]
            and metrics.get(
                "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free"
            )
            is structural_partition[
                "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free"
            ]
            and metrics.get(
                "all_authoritative_robot_collision_surfaces_contact_monitored"
            )
            is structural_partition[
                "all_authoritative_robot_collision_surfaces_contact_monitored"
            ]
            and metrics.get("robot_tree_qvel_scope_verified") is True
            and metrics.get("seven_arm_torque_decision_verified") is True
            and metrics.get("nonarm_controls_unchanged")
            is nonarm_controls_unchanged
            and metrics.get("link56_bundle_samples_field_seed_only")
            is link56_seed_only
            and link56_seed_only,
            "structural shield/contact apparatus metrics differ from raw evidence",
        )
    _require(
        int(metrics.get("solved_qp_count", -1)) == solved_qp_count
        and int(metrics.get("first_divergence_full_qp_certificate_count", -1))
        == len(independently_validated_qp_audits),
        "producer QP/certificate counts differ",
    )
    _require(
        all(
            row.get("actual_cbf_residual_postcheck_pass") is True
            and row.get("candidate_exact_clone_matches_live_shield_qvel") is True
            for row in physics
        ),
        "actual CBF residual postcheck failed",
    )
    first_material_rows = [
        row
        for row in physics
        if float(row["torque_correction_l2_nm"])
        >= derived["acceptance"]["material_torque_correction_l2_nm"]
    ]
    first_material = (
        int(first_material_rows[0]["physical_boundary"])
        if first_material_rows
        else None
    )
    first_action = (
        None
        if first_material is None
        else first_material // physics_substeps_per_action
    )
    divergence_rows = [
        row
        for row in physics
        if row.get("command_byte_identical_to_nominal") is False
    ]
    first_divergence = (
        int(divergence_rows[0]["physical_boundary"])
        if divergence_rows
        else None
    )
    first_divergence_action = (
        None
        if first_divergence is None
        else first_divergence // physics_substeps_per_action
    )
    first_divergence_correction = (
        None
        if not divergence_rows
        else float(divergence_rows[0]["torque_correction_l2_nm"])
    )
    first_divergence_is_material = bool(
        first_divergence is not None
        and first_divergence == first_material
        and first_divergence_correction
        >= derived["acceptance"]["material_torque_correction_l2_nm"]
    )
    _require(
        metrics.get("first_torque_divergence_physical_boundary")
        == first_divergence
        and metrics.get("first_torque_divergence_source_action_index")
        == first_divergence_action
        and metrics.get("first_torque_divergence_is_material")
        is first_divergence_is_material,
        "first byte-different torque divergence differs",
    )
    divergence_audit = (
        None
        if first_divergence is None
        else independently_validated_qp_audits_by_boundary.get(first_divergence)
    )
    first_material_audit = (
        None
        if first_material is None
        else independently_validated_qp_audits_by_boundary.get(first_material)
    )
    _require(
        (first_divergence is None or divergence_audit is not None)
        and (
            not target_link_v4
            or first_material is None
            or first_material_audit is not None
        ),
        "registered divergence/material full QP certificate is absent",
    )
    divergence_attribution = (
        None
        if divergence_audit is None
        else divergence_audit["constraint_attribution"]
    )
    material_attribution = (
        divergence_attribution
        if not target_link_v4
        else (
            None
            if first_material_audit is None
            else first_material_audit["constraint_attribution"]
        )
    )
    if target_link_v4:
        _require(
            fast._canonical(
                treatment.get("first_material_constraint_attribution")
            )
            == fast._canonical(material_attribution),
            "serialized first-material constraint attribution differs",
        )
        protected_row_diagnostics = treatment.get(
            "first_material_protected_row_diagnostics"
        )
        if first_material_audit is None:
            _require(
                protected_row_diagnostics is None,
                "protected-row diagnostics appear without a material correction",
            )
        else:
            protected_residuals = np.asarray(
                first_material_audit[
                    "nominal_exact_cbf_residuals_m2_per_s"
                ],
                dtype=np.float64,
            )
            _require(
                protected_residuals.shape == (len(shield_samples),)
                and np.all(np.isfinite(protected_residuals)),
                "first-material protected residual vector differs",
            )
            minimum_index = int(np.argmin(protected_residuals))
            negative_indices = np.flatnonzero(protected_residuals < 0.0)
            expected_protected_row_diagnostics = {
                "schema_version": (
                    "vlsa_poisson_first_material_protected_rows.v1"
                ),
                "sample_scope": protocol["shield"]["binding_scope"],
                "minimum_row": {
                    "sample_index": minimum_index,
                    "sample": dict(shield_samples[minimum_index]),
                    "nominal_exact_cbf_residual_m2_per_s": float(
                        protected_residuals[minimum_index]
                    ),
                },
                "negative_nominal_exact_cbf_rows": [
                    {
                        "sample_index": int(index),
                        "sample_id": int(
                            shield_samples[int(index)]["sample_id"]
                        ),
                        "body_name": str(
                            shield_samples[int(index)]["body_name"]
                        ),
                        "geom_name": str(
                            shield_samples[int(index)]["geom_name"]
                        ),
                        "nominal_exact_cbf_residual_m2_per_s": float(
                            protected_residuals[int(index)]
                        ),
                    }
                    for index in negative_indices
                ],
                "negative_nominal_exact_cbf_row_count": int(
                    negative_indices.size
                ),
            }
            _require(
                fast._canonical(protected_row_diagnostics)
                == fast._canonical(expected_protected_row_diagnostics),
                "first-material protected-row diagnostics differ",
            )
    resolved_body_ids = _strict_integer_list(
        apparatus["resolved_geometry"].get("robot_body_ids"),
        "resolved robot body IDs",
    )
    resolved_body_names = apparatus["resolved_geometry"].get("robot_body_names")
    link56_body_ids = _strict_integer_list(
        apparatus["resolved_geometry"].get("link56_body_ids"),
        "resolved literal link56 body IDs",
    )
    _require(
        isinstance(resolved_body_names, list)
        and len(resolved_body_names) == len(resolved_body_ids)
        and set(link56_body_ids).issubset(resolved_body_ids),
        "resolved robot body names differ",
    )
    body_name_by_id = dict(zip(resolved_body_ids, resolved_body_names))
    literal_link56_body_names = {
        str(body_name_by_id[body_id]) for body_id in link56_body_ids
    }
    _require(
        literal_link56_body_names
        == {str(value) for value in protocol["case"]["literal_link56_body_names"]},
        "resolved literal link56 body-name authority differs",
    )
    first_material_minimum_is_link56 = bool(
        material_attribution is not None
        and material_attribution["minimum_sample"]["body_name"]
        in literal_link56_body_names
    )
    first_material_minimum_is_target_link = bool(
        material_attribution is not None
        and material_attribution["minimum_sample"]["body_name"]
        in target_link_body_name_set
    )
    protected_link_body_name_set = set(protected_link_body_names)
    first_material_minimum_is_protected_link = bool(
        material_attribution is not None
        and material_attribution["minimum_sample"]["body_name"]
        in protected_link_body_name_set
        and set(material_attribution["near_minimum_body_names"]).issubset(
            protected_link_body_name_set
        )
        and set(
            material_attribution["negative_nominal_residual_body_names"]
        ).issubset(protected_link_body_name_set)
    )
    _require(
        metrics.get("first_divergence_minimum_constraint_body_name")
        == (
            None
            if divergence_attribution is None
            else divergence_attribution["minimum_sample"]["body_name"]
        )
        and metrics.get("first_divergence_near_minimum_constraint_body_names")
        == (
            []
            if divergence_attribution is None
            else divergence_attribution["near_minimum_body_names"]
        )
        and metrics.get("first_divergence_negative_constraint_body_names")
        == (
            []
            if divergence_attribution is None
            else divergence_attribution["negative_nominal_residual_body_names"]
        )
        and (
            metrics.get(
                "first_material_correction_minimum_constraint_is_target_link"
            )
            is first_material_minimum_is_target_link
            and metrics.get(
                "first_material_correction_minimum_constraint_is_protected_link"
            )
            is first_material_minimum_is_protected_link
            and metrics.get(
                "first_material_correction_target_link_attributed"
            )
            is first_material_minimum_is_target_link
            and metrics.get(
                "first_material_correction_protected_link_attributed"
            )
            is first_material_minimum_is_protected_link
            and metrics.get(
                "first_material_correction_minimum_constraint_is_literal_link56"
            )
            is first_material_minimum_is_link56
            and metrics.get(
                "first_material_correction_minimum_constraint_body_name"
            )
            == (
                None
                if material_attribution is None
                else material_attribution["minimum_sample"]["body_name"]
            )
            and metrics.get(
                "first_material_correction_near_minimum_constraint_body_names"
            )
            == (
                []
                if material_attribution is None
                else material_attribution["near_minimum_body_names"]
            )
            and metrics.get(
                "first_material_correction_negative_nominal_constraint_body_names"
            )
            == (
                []
                if material_attribution is None
                else material_attribution[
                    "negative_nominal_residual_body_names"
                ]
            )
            if target_link_v4
            else metrics.get(
                "first_material_correction_minimum_constraint_is_literal_link56"
            )
            is first_material_minimum_is_link56
        ),
        "first divergence protected-link body attribution differs",
    )
    if first_divergence_correction is None:
        _require(
            metrics.get("first_torque_divergence_correction_l2_nm") is None,
            "absent first divergence correction differs",
        )
    else:
        _close(
            metrics.get("first_torque_divergence_correction_l2_nm"),
            first_divergence_correction,
            "first divergence correction",
        )
    first_material_residual_improvement = (
        None
        if not first_material_rows
        else float(
            first_material_rows[0][
                "candidate_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
            - first_material_rows[0][
                "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
        )
    )
    first_material_next_qvel_change = (
        None
        if not first_material_rows
        else float(
            np.linalg.norm(
                np.asarray(
                    first_material_rows[0]["predicted_next_arm_qvel_rad_s"],
                    dtype=np.float64,
                )
                - np.asarray(
                    first_material_rows[0][
                        "nominal_predicted_next_arm_qvel_rad_s"
                    ],
                    dtype=np.float64,
                )
            )
        )
    )
    _require(
        metrics.get("first_material_correction_physical_boundary") == first_material,
        "first material correction boundary differs",
    )
    _require(
        metrics.get("first_material_correction_source_action_index") == first_action,
        "first material correction action differs",
    )
    _require(
        metrics.get("material_correction_present") is (first_material is not None),
        "material-correction flag differs",
    )
    if first_material_residual_improvement is None:
        if target_link_v4:
            _close(
                metrics.get(
                    "first_material_exact_cbf_residual_improvement_m2_per_s"
                ),
                0.0,
                "absent first material exact CBF residual improvement",
            )
            _close(
                metrics.get("first_material_exact_next_qvel_change_l2_rad_s"),
                0.0,
                "absent first material exact next-qvel change",
            )
        else:
            _require(
                metrics.get(
                    "first_material_exact_cbf_residual_improvement_m2_per_s"
                )
                is None
                and metrics.get(
                    "first_material_exact_next_qvel_change_l2_rad_s"
                )
                is None,
                "absent directed safety improvement differs",
            )
    else:
        _close(
            metrics.get(
                "first_material_exact_cbf_residual_improvement_m2_per_s"
            ),
            first_material_residual_improvement,
            "first material exact CBF residual improvement",
        )
        _close(
            metrics.get("first_material_exact_next_qvel_change_l2_rad_s"),
            first_material_next_qvel_change,
            "first material exact next-qvel change",
        )
    nominal_negative = bool(
        divergence_rows
        and float(
            divergence_rows[0][
                "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
        )
        < 0.0
    )
    first_material_nominal_negative = bool(
        first_material_rows
        and float(
            first_material_rows[0][
                "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
        )
        < 0.0
    )
    _require(
        metrics.get("first_divergence_nominal_exact_cbf_residual_negative")
        is nominal_negative,
        "first divergence nominal residual sign differs",
    )
    if target_link_v4:
        _require(
            metrics.get(
                "first_material_nominal_exact_protected_link_cbf_residual_negative"
            )
            is bool(
                first_material_nominal_negative
                and first_material_minimum_is_protected_link
            )
            and metrics.get(
                "first_material_nominal_exact_target_link_cbf_residual_negative"
            )
            is bool(
                first_material_nominal_negative
                and first_material_minimum_is_target_link
            ),
            "first-material protected/target nominal residual signs differ",
        )
    correction_goal = treatment.get("first_material_correction_goal_snapshot")
    expected_task_incomplete = False if target_link_v4 else None
    if first_material is not None:
        _require(isinstance(correction_goal, Mapping), "correction-time goal snapshot is absent")
        _require(
            int(correction_goal.get("step", -1)) == first_action
            and correction_goal.get("inert") is True
            and correction_goal.get("simulator_state_sha256_before")
            == correction_goal.get("simulator_state_sha256_after"),
            "correction-time goal snapshot is not state-inert",
        )
        expected_task_incomplete = not bool(correction_goal.get("all_satisfied"))
    _require(
        metrics.get("task_incomplete_at_first_material_correction")
        is expected_task_incomplete,
        "task state at first correction differs",
    )
    prior_live_contact_free = bool(
        first_material is not None
        and all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if int(row["physical_boundary"]) < int(first_material)
        )
    )
    predivergence_exact = all(
        row.get("command_byte_identical_to_nominal") is True
        for row in physics
        if first_divergence is None
        or int(row["physical_boundary"]) < first_divergence
    )
    prefix_contact_free = bool(
        first_divergence is not None
        and all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if int(row["physical_boundary"]) < first_divergence
        )
    )
    v4_predivergence_contact_free = bool(
        all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if first_divergence is None
            or int(row["physical_boundary"]) < int(first_divergence)
        )
    )
    expected_parity_count = (
        len(actions) if first_divergence_action is None else first_divergence_action
    )
    _require(len(parity) == expected_parity_count, "pre-correction parity count differs")
    _require(
        [int(row.get("source_action_index", -1)) for row in parity]
        == list(range(expected_parity_count)),
        "historical parity indexes differ",
    )
    _require(
        [int(row.get("source_action_index", -1)) for row in actions]
        == list(range(len(actions))),
        "completed action indexes differ",
    )
    goal_by_index = {
        int(row.get("source_action_index", -1)): row for row in goals
    }
    _require(len(goal_by_index) == len(goals), "goal indexes are duplicated")
    for index in range(expected_parity_count):
        parity_row = parity[index]
        action_row = actions[index]
        goal_row = goal_by_index.get(index)
        historical_step = replay.steps[index]
        _require(isinstance(goal_row, Mapping), "historical goal row is absent")
        _require(
            parity_row.get("historical_match") is True
            and parity_row.get("simulator_state_sha256")
            == historical_step.simulator_state_sha256
            and action_row.get("state_sha256")
            == historical_step.simulator_state_sha256
            and parity_row.get("observation_sha256")
            == action_row.get("observation_sha256")
            == goal_row.get("observation_sha256")
            and np.array_equal(
                np.asarray(action_row.get("executed_high_level_action"), dtype=np.float64),
                np.asarray(historical_step.action, dtype=np.float64),
            )
            and float(goal_row.get("reward")) == historical_step.reward
            and bool(goal_row.get("done")) is historical_step.done
            and tuple(goal_row.get("values", ())) == historical_step.goal_values,
            "historical action/state/reward/goal parity differs at action %d" % index,
        )

    historical_contact_action = int(derived["historical_contact_action"])
    historical_same_action_exception = bool(
        first_divergence_action is not None
        and (
            first_divergence_action < historical_contact_action
            or (
                first_divergence_action == historical_contact_action
                and len(parity) == historical_contact_action
            )
        )
    )
    prior_endpoint_parity_exact = False
    if target_link_v4:
        historical_prior_endpoint_action = int(
            derived["historical_prior_endpoint_action"]
        )
        _require(
            historical_prior_endpoint_action == historical_contact_action - 1,
            "configured historical prior endpoint is not immediately before contact",
        )
        prior_rows = [
            row
            for row in parity
            if int(row.get("source_action_index", -1))
            == historical_prior_endpoint_action
        ]
        prior_endpoint_parity_exact = bool(
            len(prior_rows) == 1
            and prior_rows[0].get("historical_match") is True
        )
    prior_endpoint_contact_free = bool(
        target_link_v4
        and any(
            int(row.get("source_action_index", -1))
            == int(derived["historical_prior_endpoint_action"])
            for row in actions
        )
        and all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if int(row["source_action_index"])
            <= int(derived["historical_prior_endpoint_action"])
        )
    )
    same_action_live_substeps_before_material_contact_free = bool(
        first_material is not None
        and first_action == historical_contact_action
        and all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if int(row["source_action_index"]) == historical_contact_action
            and int(row["physical_boundary"]) < int(first_material)
        )
    )
    first_material_is_same_action_initial_callback = bool(
        first_material_rows
        and first_action == historical_contact_action
        and int(first_material_rows[0]["physical_boundary"])
        == historical_contact_action * physics_substeps_per_action
        and int(first_material_rows[0]["physics_substep_index"]) == 0
    )
    if target_link_v4:
        expected_before = bool(
            first_action is not None
            and (
                first_action < historical_contact_action
                or (
                    first_action == historical_contact_action
                    and first_material_is_same_action_initial_callback
                    and prior_endpoint_parity_exact
                    and prior_endpoint_contact_free
                    and prior_live_contact_free
                    and same_action_live_substeps_before_material_contact_free
                )
            )
        )
    else:
        expected_before = bool(
            first_action is not None
            and first_divergence_is_material
            and (
                first_action < historical_contact_action
                or (
                    first_action == historical_contact_action
                    and prior_live_contact_free
                    and prefix_contact_free
                    and predivergence_exact
                    and historical_same_action_exception
                )
            )
        )
    if target_link_v4:
        first_material_callback_endpoint = (
            None
            if not first_material_rows
            else [
                int(first_material_rows[0]["source_action_index"]),
                int(first_material_rows[0]["physics_substep_index"])
                // physics_substeps_per_controller_update,
                int(first_material_rows[0]["physics_substep_index"])
                % physics_substeps_per_controller_update,
            ]
        )
        timing_certificate = treatment.get("first_material_timing_certificate")
        _require(
            isinstance(timing_certificate, Mapping)
            and timing_certificate.get("schema_version")
            == "vlsa_poisson_first_material_historical_contact_timing.v1"
            and timing_certificate.get("historical_target_contact_source_action")
            == historical_contact_action
            and timing_certificate.get(
                "historical_within_action_contact_substep_known"
            )
            is False
            and timing_certificate.get(
                "historical_prior_completed_action_endpoint"
            )
            == int(derived["historical_prior_endpoint_action"])
            and timing_certificate.get(
                "historical_prior_completed_endpoint_parity_exact"
            )
            is prior_endpoint_parity_exact
            and timing_certificate.get(
                "historical_prior_completed_endpoint_contact_free"
            )
            is prior_endpoint_contact_free
            and timing_certificate.get(
                "first_material_callback_endpoint_action_inner_substep"
            )
            == first_material_callback_endpoint
            and timing_certificate.get(
                "same_source_action_correction_is_prephysics_callback_zero"
            )
            is first_material_is_same_action_initial_callback
            and timing_certificate.get(
                "all_earlier_live_substeps_registered_contact_free"
            )
            is prior_live_contact_free
            and timing_certificate.get(
                "correction_strictly_before_historical_target_contact"
            )
            is expected_before
            and timing_certificate.get(
                "independently_reconstructible_strict_before_result"
            )
            is expected_before
            and timing_certificate.get(
                "strict_same_action_timing_rule_enforced"
            )
            is True
            and timing_certificate.get("same_source_action_positive_rule")
            == (
                "only_callback_endpoint_action_0_0_with_exact_contact_free_"
                "prior_completed_endpoint"
            )
            and timing_certificate.get(
                "registered_same_source_action_callback_suffix"
            )
            == list(
                derived[
                    "same_source_action_contact_exception_callback_suffix"
                ]
            )
            and metrics.get(
                "first_material_correction_callback_endpoint_action_inner_substep"
            )
            == first_material_callback_endpoint
            and metrics.get(
                "same_source_action_correction_is_prephysics_callback_zero"
            )
            is first_material_is_same_action_initial_callback,
            "first-material historical-contact timing certificate differs",
        )
    _require(
        metrics.get("material_correction_before_historical_contact")
        is expected_before,
        "correction/contact ordering differs",
    )
    _require(
        metrics.get("live_substeps_before_material_correction_contact_free")
        is prior_live_contact_free,
        "live pre-correction contact-free flag differs",
    )
    # This apparatus flag certifies that the exact rule and its timing inputs
    # were serialized and audited; it is not the scientific before-contact
    # outcome.  A coherent late/no-material/stop record therefore remains an
    # interpretable negative rather than an apparatus rejection.
    v4_same_source_action_exception_valid = True
    _require(
        metrics.get("all_predivergence_torque_commands_byte_identical_nominal")
        is predivergence_exact
        and metrics.get("all_live_substeps_before_first_divergence_contact_free")
        is (
            v4_predivergence_contact_free if target_link_v4 else prefix_contact_free
        )
        and (
            metrics.get(
                "historical_contact_same_source_action_exception_requires_"
                "exact_callback_action_0_0_and_exact_contact_free_prior_"
                "completed_endpoint"
            )
            is v4_same_source_action_exception_valid
            if target_link_v4
            else metrics.get(
                "action62_same_action_exception_has_action61_endpoint_parity_exact"
            )
            is historical_same_action_exception
        ),
        "predivergence authority differs",
    )

    trace_contact = any(
        row.get("literal_robot_selected_obstacle_contact_seen") is True
        for row in physics
    )
    measurement = treatment.get("measurement")
    _require(isinstance(measurement, Mapping), "authoritative measurement is absent")
    _require(
        int(measurement.get("observed_physics_substeps")) == len(physics),
        "measurement contact/cadence differs from physics trace",
    )
    physical_count = int(
        measurement.get("total_physical_contact_point_record_count")
    )
    rollout_count = int(
        measurement.get("rollout_phase_physical_contact_point_record_count")
    )
    post_records = measurement.get("post_state_physical_contact_point_records")
    live_records = measurement.get("live_solver_phase_contact_point_records")
    _require(
        isinstance(post_records, Sequence) and isinstance(live_records, Sequence),
        "measurement physical-contact record ledgers are absent",
    )
    settled_measurement = measurement.get("settled_state")
    _require(isinstance(settled_measurement, Mapping), "settled contact evidence is absent")
    settled_records = settled_measurement.get("physical_contact_point_records")
    _require(isinstance(settled_records, Sequence), "settled physical records are absent")
    for record in live_records:
        _require(isinstance(record, Mapping), "live contact row is invalid")
        distance = float(record.get("contact_distance_m"))
        _require(
            math.isfinite(distance)
            and record.get("is_physical_nonpositive_distance_contact")
            is (distance <= 0.0),
            "live contact distance/flag differs",
        )
    live_physical = [
        row for row in live_records if float(row["contact_distance_m"]) <= 0.0
    ]
    _require(
        all(
            isinstance(row, Mapping)
            and math.isfinite(float(row.get("contact_distance_m")))
            and float(row.get("contact_distance_m")) <= 0.0
            and row.get("is_physical_nonpositive_distance_contact") is True
            for row in list(settled_records) + list(post_records)
        ),
        "physical record ledger contains a nonphysical row",
    )
    rollout_union = live_physical + list(post_records)
    all_physical_records = list(settled_records) + rollout_union
    measured_contact = bool(all_physical_records)
    _require(
        trace_contact is bool(rollout_union)
        and metrics.get("any_robot_selected_obstacle_contact") is measured_contact
        and measurement.get("any_robot_obstacle_contact") is measured_contact
        and measurement.get("rollout_any_robot_obstacle_contact")
        is bool(rollout_union),
        "contact union differs from trace or metrics",
    )
    resolved_geometry = apparatus.get("resolved_geometry")
    _require(isinstance(resolved_geometry, Mapping), "resolved geometry is absent")
    robot_geom_ids = {int(value) for value in resolved_geometry.get("robot_geom_ids", ())}
    obstacle_geom_ids = {int(value) for value in resolved_geometry.get("obstacle_geom_ids", ())}
    link_geom_ids = {int(value) for value in resolved_geometry.get("link56_geom_ids", ())}
    _require(robot_geom_ids and obstacle_geom_ids and link_geom_ids, "resolved contact IDs are absent")
    for record in all_physical_records:
        _require(
            int(record.get("robot_geom_id", -1)) in robot_geom_ids
            and int(record.get("obstacle_geom_id", -1)) in obstacle_geom_ids,
            "physical contact record lies outside registered scope",
        )
    reconstructed_link_contact = any(
        int(record.get("robot_geom_id", -1)) in link_geom_ids
        for record in all_physical_records
    )
    _require(
        measurement.get("link56_obstacle_contact") is reconstructed_link_contact
        and metrics.get("link56_selected_obstacle_contact")
        is reconstructed_link_contact,
        "link5/6 contact union differs",
    )
    _require(
        physical_count == len(all_physical_records)
        and rollout_count == len(rollout_union)
        and int(measurement.get("post_state_physical_contact_point_record_count"))
        == len(post_records)
        and int(measurement.get("live_solver_nonpositive_contact_point_record_count"))
        == len(live_physical),
        "physical contact counts differ from record union",
    )
    if measured_contact:
        _require(
            physical_count > 0
            and rollout_count > 0
            and measurement.get("first_physical_contact_point_record") is not None,
            "contact flag lacks authoritative physical records",
        )
    else:
        _require(
            physical_count == 0
            and rollout_count == 0
            and measurement.get("first_physical_contact_point_record") is None
            and not post_records
            and not any(
                row.get("is_physical_nonpositive_distance_contact") is True
                for row in live_records
                if isinstance(row, Mapping)
            ),
            "contact-free flag conflicts with physical-contact records",
        )
    registered_contact_audit = _validate_registered_contact_evidence(
        scope=apparatus.get("registered_contact_scope"),
        measurement=treatment.get("registered_contact_measurement"),
        physics=physics,
        resolved_geometry=resolved_geometry,
        physics_substeps_per_action=physics_substeps_per_action,
        physics_substeps_per_controller_update=(
            physics_substeps_per_controller_update
        ),
        phase_correct_callback_required=target_link_v4,
        target_link_v4=target_link_v4,
    )
    registered_contact = bool(registered_contact_audit["rollout_contact"])
    _require(
        registered_contact_audit["settled_contact"] is False
        and metrics.get("any_registered_forbidden_contact")
        is registered_contact
        and metrics.get("any_robot_selected_obstacle_contact")
        is registered_contact_audit["any_robot_selected_obstacle_contact"]
        and metrics.get("any_link56_external_nonrobot_contact")
        is registered_contact_audit["any_link56_external_nonrobot_contact"]
        and metrics.get("any_link56_nonselected_external_contact")
        is registered_contact_audit["any_link56_nonselected_external_contact"]
        and measured_contact
        is registered_contact_audit["any_robot_selected_obstacle_contact"]
        and metrics.get("every_executed_substep_registered_contact_scope_monitored")
        is True
        and metrics.get("every_executed_substep_contact_monitored") is True,
        "registered union contact metrics differ",
    )
    _require(
        int(metrics.get("executed_physics_substep_count")) == len(physics)
        and int(metrics.get("shield_decision_count")) == len(physics),
        "shield exposure count differs",
    )
    _require(
        metrics.get("no_policy_query_before_divergence") is True
        and planner.get("no_policy_query_before_divergence") is True,
        "policy was queried before divergence",
    )
    divergence = planner.get("divergence_source_action_index")
    _require(
        divergence == first_divergence_action
        and planner.get("divergence_physical_boundary") == first_divergence,
        "planner divergence differs from first byte-different torque",
    )
    planner_actions = planner.get("action_trace")
    policy_queries = planner.get("policy_queries")
    _require(
        isinstance(planner_actions, Sequence)
        and isinstance(policy_queries, Sequence)
        and [int(row.get("source_action_index", -1)) for row in planner_actions]
        == list(range(len(planner_actions))),
        "planner action cadence differs",
    )
    _require(
        planner.get("action_trace_sha256") == _canonical_sha256(planner_actions)
        and planner.get("policy_query_trace_sha256")
        == _canonical_sha256(policy_queries),
        "planner trace aggregate hash differs",
    )
    terminal_kind = treatment.get("terminal_kind")
    partial_action_record = treatment.get("partial_action_record")
    partial_action_audit = _validate_partial_action_ledger(
        terminal_kind=str(terminal_kind),
        actions=actions,
        planner_actions=planner_actions,
        partial_action_record=partial_action_record,
        physics_substep_count=len(physics),
        physics_substeps_per_action=physics_substeps_per_action,
    )
    if terminal_kind == "literal_registered_forbidden_contact":
        _require(
            partial_action_audit.get("terminal_forbidden_contact_categories")
            == registered_contact_audit["rollout_contact_categories"]
            and partial_action_audit.get(
                "terminal_forbidden_contact_records_sha256"
            )
            == registered_contact_audit["rollout_contact_records_sha256"],
            "partial contact terminal is not bound to registered contact evidence",
        )
    policy_server = planner.get("policy_server")
    endpoint = planner.get("policy_server_endpoint")
    if policy_queries:
        _require(
            isinstance(policy_server, Mapping)
            and policy_server.get("status") == "available"
            and isinstance(endpoint, Mapping)
            and endpoint.get("host") == "127.0.0.1"
            and isinstance(endpoint.get("port"), int)
            and 20000 <= int(endpoint["port"]) < 50000
            and planner.get("policy_server_identity_sha256")
            == _canonical_sha256(policy_server),
            "fresh policy server identity is absent or unbound",
        )
    else:
        _require(
            policy_server is None
            and planner.get("policy_server_identity_sha256") is None,
            "unused policy server identity differs",
        )
    query_by_index: Dict[int, Mapping[str, Any]] = {}
    for query in policy_queries:
        query_index = int(query.get("query_index", -1))
        _require(query_index not in query_by_index, "policy query index is duplicated")
        chunk = np.asarray(query.get("returned_actions"), dtype=np.float64)
        _require(
            chunk.shape == (int(protocol["execution"]["model_action_horizon"]), 7)
            and np.all(np.isfinite(chunk))
            and query.get("returned_actions_sha256") == _float64_sha256(chunk, np)
            and query.get("returned_action_shape") == list(chunk.shape)
            and query.get("source") == "fresh_pi05_from_treatment_observation",
            "fresh policy response bytes or shape differ",
        )
        query_by_index[query_index] = query
    cached_end = planner.get("cached_current_chunk_end_source_action_index")
    expected_cached = 0
    expected_fresh = 0
    expected_aegis_z = (
        None
        if divergence is None
        else np.asarray(
            historical_value["actions"][int(divergence)]["qp"]["z_after"],
            dtype=np.float64,
        )
    )
    for planner_row in planner_actions:
        source_index = int(planner_row["source_action_index"])
        source_label = planner_row.get("source")
        executed = np.asarray(planner_row.get("executed"), dtype=np.float64)
        action_binding = (
            actions[source_index]
            if source_index < len(actions)
            else partial_action_record
        )
        _require(
            isinstance(action_binding, Mapping)
            and executed.shape == (7,)
            and np.all(np.isfinite(executed))
            and planner_row.get("executed_action_sha256")
            == _float64_sha256(executed, np)
            and np.array_equal(
                executed,
                np.asarray(
                    action_binding.get("executed_high_level_action"),
                    dtype=np.float64,
                ),
            )
            and action_binding.get("executed_high_level_action_sha256")
            == _float64_sha256(executed, np),
            "planner/treatment executed action chain differs",
        )
        expected_observation = (
            None
            if source_index == 0
            else actions[source_index - 1].get("observation_sha256")
        )
        if expected_observation is not None:
            _require(
                planner_row.get("native_observation_sha256") == expected_observation,
                "planner action does not consume the preceding treatment observation",
            )
        if divergence is None or source_index <= int(divergence):
            _require(
                source_label == "archived_aegis_executed_pre_divergence"
                and source_index < len(replay.actions)
                and np.array_equal(
                    np.asarray(planner_row.get("executed"), dtype=np.float64),
                    np.asarray(replay.actions[source_index], dtype=np.float64),
                ),
                "archived planner prefix differs",
            )
        elif source_index <= int(cached_end):
            expected_cached += 1
            archived_raw = np.asarray(
                historical_value["actions"][source_index].get("nominal_raw"),
                dtype=np.float64,
            )
            planner_raw = np.asarray(planner_row.get("nominal_raw"), dtype=np.float64)
            _require(
                source_label
                == "archived_current_pi05_chunk_reprocessed_live_aegis",
                "cached treatment chunk source differs",
            )
            _require(
                archived_raw.shape == planner_raw.shape == (7,)
                and np.array_equal(planner_raw, archived_raw)
                and planner_row.get("nominal_raw_sha256")
                == _float64_sha256(planner_raw, np),
                "cached historical raw chunk differs from the bound result",
            )
        else:
            expected_fresh += 1
            query_index = source_index // 5
            query = query_by_index.get(query_index)
            planner_raw = np.asarray(planner_row.get("nominal_raw"), dtype=np.float64)
            returned_chunk = (
                np.asarray(query.get("returned_actions"), dtype=np.float64)
                if isinstance(query, Mapping)
                else np.asarray([])
            )
            _require(
                source_label == "fresh_pi05_reprocessed_live_aegis"
                and int(planner_row.get("query_index", -1)) == query_index
                and int(planner_row.get("query_chunk_offset", -1))
                == source_index % 5,
                "fresh treatment action provenance differs",
            )
            _require(
                isinstance(query, Mapping)
                and returned_chunk.shape[0] > source_index % 5
                and planner_raw.shape == (7,)
                and np.array_equal(
                    planner_raw, returned_chunk[source_index % 5]
                )
                and planner_row.get("nominal_raw_sha256")
                == _float64_sha256(planner_raw, np),
                "fresh policy response is not bound to the executed action chain",
            )
        if divergence is not None and source_index > int(divergence):
            raw = np.asarray(planner_row.get("nominal_raw"), dtype=np.float64)
            nominal_translational = np.asarray(
                planner_row.get("nominal_translational"), dtype=np.float64
            )
            expected_nominal = np.zeros(7, dtype=np.float64)
            expected_nominal[:3] = raw[:3]
            expected_nominal[6] = raw[6]
            z_before = np.asarray(planner_row.get("aegis_z_before"), dtype=np.float64)
            z_after = np.asarray(planner_row.get("aegis_z_after"), dtype=np.float64)
            aegis_qp = planner_row.get("aegis_qp")
            context = aegis_qp.get("context") if isinstance(aegis_qp, Mapping) else None
            _require(
                nominal_translational.shape == (7,)
                and np.array_equal(nominal_translational, expected_nominal)
                and executed[3] == executed[4] == executed[5] == 0.0
                and executed[6] == raw[6]
                and expected_aegis_z is not None
                and np.array_equal(z_before, expected_aegis_z)
                and z_after.shape == (3,)
                and np.all(np.isfinite(z_after))
                and math.isclose(float(np.linalg.norm(z_after)), 1.0, rel_tol=0.0, abs_tol=1e-10)
                and isinstance(aegis_qp, Mapping)
                and aegis_qp.get("status") == "solved"
                and str(aegis_qp.get("solver_status")) in ("optimal", "optimal_inaccurate")
                and np.array_equal(
                    np.asarray(aegis_qp.get("z_before"), dtype=np.float64), z_before
                )
                and np.array_equal(
                    np.asarray(aegis_qp.get("z_after"), dtype=np.float64), z_after
                )
                and isinstance(context, Mapping)
                and np.array_equal(
                    np.asarray(context.get("executed_action"), dtype=np.float64),
                    executed,
                )
                and context.get("executed_action_array_sha256")
                == _float64_sha256(executed, np),
                "post-divergence AEGIS action or virtual-state chain differs",
            )
            expected_aegis_z = z_after
    _require(
        int(planner.get("cached_current_chunk_action_count", -1)) == expected_cached
        and int(planner.get("fresh_own_observation_action_count", -1))
        == expected_fresh,
        "planner cached/fresh action counts differ",
    )
    expected_query_sources = sorted(
        {
            int(row["source_action_index"])
            for row in planner_actions
            if row.get("source") == "fresh_pi05_reprocessed_live_aegis"
            and int(row["source_action_index"]) % 5 == 0
        }
    )
    _require(
        [int(row.get("source_action_index", -1)) for row in policy_queries]
        == expected_query_sources,
        "fresh policy query schedule differs",
    )
    base_noise_seed = int(case["source_case"]["policy_noise_seed"])
    for query in policy_queries:
        _require(
            divergence is not None
            and int(query["source_action_index"]) > int(divergence),
            "policy query does not use a post-divergence observation",
        )
        _require(
            int(query["source_action_index"]) % 5 == 0
            and int(query["query_index"])
            == int(query["source_action_index"]) // 5
            and int(query["rng_seed"])
            == base_noise_seed + int(query["query_index"]),
            "policy query seed or cadence differs",
        )
        _require(
            query.get("native_observation_sha256")
            == planner_actions[int(query["source_action_index"])].get(
                "native_observation_sha256"
            ),
            "policy query does not bind the planner observation",
        )
    motion = _motion(physics, first_material)
    _close(metrics.get("post_correction_joint_motion_integral_rad"), motion["joint"], "joint motion")
    _close(metrics.get("post_correction_eef_path_length_m"), motion["eef"], "EEF path")
    _close(metrics.get("post_correction_zero_torque_delta_fraction"), motion["zero"], "zero fraction")
    _require(
        len(goals) == len(actions)
        and [int(row.get("source_action_index", -1)) for row in goals]
        == list(range(len(goals))),
        "goal/action terminal cadence differs",
    )
    method_stop = treatment.get("safety_method_stop")
    method_stopped = _validate_safety_method_stop(
        method_stop,
        terminal_kind=terminal_kind,
        action_count=len(actions),
        physics_substep_count=len(physics),
        sample_count=int(shield_binding["sample_count"]),
        sample_ledger_sha256=str(shield_binding["sample_ledger_sha256"]),
        robot_qvel_indices=shield_qvel_indices,
        robot_qvel_indices_sha256=_canonical_sha256(shield_qvel_indices),
        velocity_dimension=shield_velocity_dimension,
        physics_substeps_per_action=physics_substeps_per_action,
        physics_substeps_per_controller_update=(
            physics_substeps_per_controller_update
        ),
    )
    _require(
        partial_action_audit.get("present")
        is bool(
            terminal_kind == "literal_registered_forbidden_contact"
            or method_stopped
        ),
        "partial action record does not match the terminal condition",
    )
    _require(
        metrics.get("safety_method_stop_before_physics") is method_stopped
        and metrics.get("all_shield_decisions_valid")
        is (True if target_link_v4 else not method_stopped),
        "safety method-stop evidence differs",
    )
    task_success = bool(
        goals
        and goals[-1].get("all_satisfied") is True
        and goals[-1].get("done") is True
        and actions[-1].get("task_success") is True
        and terminal_kind == "native_task_success"
    )
    complete_terminal = bool(
        (
            terminal_kind == "native_task_success"
            and task_success
            and not registered_contact
            and len(physics)
            == len(actions) * physics_substeps_per_action
        )
        or (
            terminal_kind == "literal_registered_forbidden_contact"
            and registered_contact
            and 1
            <= len(physics) - len(actions) * physics_substeps_per_action
            <= physics_substeps_per_action
        )
        or (
            terminal_kind == "maximum_action_count"
            and not registered_contact
            and not task_success
            and len(actions) == int(derived["maximum_action_count"])
            and len(physics)
            == len(actions) * physics_substeps_per_action
        )
        or (
            method_stopped
            and not registered_contact
            and len(actions) * physics_substeps_per_action
            <= len(physics)
            < (len(actions) + 1) * physics_substeps_per_action
        )
    )
    _require(
        metrics.get("complete_terminal_condition_reached") is complete_terminal,
        "terminal condition differs",
    )
    _require(metrics.get("native_task_success") is task_success, "native task flag differs")
    _validate_paper_car_cadence_and_historical_binding(
        car_ledger,
        action_count=len(actions),
        historical_settled_position_sha256=(
            replay.settled_active_obstacle_position_sha256
        ),
    )
    car_key = "%s_pos" % protocol["case"]["selected_obstacle_name"]
    settled_car_position = car_ledger[0].get(
        "active_obstacle_position_observation_world_m"
    )
    reconstructed_car = [
        _validate_paper_car_endpoint_row(
            row,
            index=index,
            car_key=car_key,
            obstacle_root_body_id=observable_car_root_id,
            settled_car_position=settled_car_position,
            np=np,
        )
        for index, row in enumerate(car_ledger)
    ]
    maximum_car = max(reconstructed_car)
    _exact_float(
        metrics.get(
            "maximum_active_obstacle_l1_displacement_at_completed_action_endpoints_m"
        ),
        maximum_car,
        "maximum paper CAR displacement",
    )
    expected_car_safe = bool(
        maximum_car
        <= float(protocol["acceptance"]["paper_car_l1_displacement_threshold_m"])
    )
    _require(
        metrics.get("treatment_paper_car_avoided") is expected_car_safe,
        "paper CAR outcome differs",
    )

    video_path = result_path.parent / str(video.get("path"))
    _require(
        not video_path.is_symlink()
        and video_path.is_file()
        and fast._file_sha256(video_path) == video.get("sha256"),
        "video file or hash differs",
    )
    _require(
        video.get("decoded_successfully") is True
        and video.get("real_simulation_frames") is True
        and video.get("two_dimensional_safety_overlay") is False,
        "video contract differs",
    )
    if target_link_v4:
        historical_outcome = case.get("historical_outcome")
        historical_metrics = historical_value.get("metrics")
        _require(
            isinstance(historical_outcome, Mapping)
            and isinstance(historical_metrics, Mapping)
            and historical_outcome.get("paper_car_failure") is True
            and historical_metrics.get("paper_collision") is True,
            "archived control CAR failure differs",
        )
        constraint_exposure_exact = bool(
            int(metrics.get("shield_decision_count", -1)) == len(physics)
            and all(
                int(row.get("constraint_trace", {}).get("sample_count", -1))
                == int(shield_binding["sample_count"])
                and row.get("constraint_trace", {}).get("sample_ledger_sha256")
                == shield_binding["sample_ledger_sha256"]
                for row in physics
            )
        )
        broad_contact_monitor_exact = bool(
            registered_contact_audit["settled_contact"] is False
            and metrics.get(
                "every_executed_substep_registered_contact_scope_monitored"
            )
            is True
            and metrics.get("every_executed_substep_contact_monitored") is True
        )
        nominal_passthrough_exact = all(
            row.get("shield_status") != "nominal_safe_exact_clone"
            or row.get("command_byte_identical_to_nominal") is True
            for row in physics
        )
        predivergence_parity_exact = bool(
            len(parity) == expected_parity_count
            and all(row.get("historical_match") is True for row in parity)
        )
        cached_then_fresh_exact = bool(planner.get("hybrid_contract_valid") is True)
        first_divergence_registered = bool(
            first_divergence is not None
            and divergence == first_divergence_action
            and planner.get("divergence_physical_boundary") == first_divergence
        )
        video_complete = bool(
            video.get("decoded_successfully") is True
            and int(video.get("frame_count", -1)) >= len(actions) + 1
        )
        reconstructed_apparatus = {
            "allocation_numeric_prerequisite_verified": True,
            "allowed_case_registry_verified": True,
            "historical_control_verified": True,
            "historical_direct_target_link_contact_verified": direct_historical_contact,
            "historical_control_car_failure": True,
            "historical_control_task_success": replay.historical_task_success is True,
            "archived_baseline_not_rerun": (
                provenance.get("historical_control_rerun") is False
                and historical.get("rerun") is False
            ),
            "frozen_runtime_parameter_hash_verified": (
                runtime_hashes.parameter_block_sha256
                == derived["frozen_runtime_parameter_sha256"]
            ),
            "protected_link_shield_sampling_exact": (
                shield_binding["exact_protected_field_ledger"] is True
            ),
            "all_authoritative_robot_collision_surfaces_contact_monitored": (
                broad_contact_monitor_exact
            ),
            "literal_link56_external_nonrobot_contacts_monitored": (
                broad_contact_monitor_exact
            ),
            "protected_link_arm_qvel_scope_verified": (
                shield_qvel_indices == list(shield_binding["arm_qvel_indices"])
                and len(shield_qvel_indices) == 7
            ),
            "direct_unit_gain_hinge_torque_actuators_verified": (
                torque_units_verified
            ),
            "seven_arm_torque_decision_verified": len(torque_actuators) == 7,
            "nonarm_controls_unchanged": nonarm_controls_unchanged,
            "original_osc_controller_verified": True,
            "static_field_admissible": field_audit["static_admissible"] is True,
            "every_executed_substep_protected_link_shielded": (
                constraint_exposure_exact
            ),
            "every_executed_substep_contact_monitored": broad_contact_monitor_exact,
            "complete_exposure_verified": (
                constraint_exposure_exact and broad_contact_monitor_exact
            ),
            "nominal_pass_through_bitwise_exact": nominal_passthrough_exact,
            "no_policy_query_before_divergence": (
                planner.get("no_policy_query_before_divergence") is True
            ),
            "cached_current_chunk_then_fresh_own_observation_policy": (
                cached_then_fresh_exact
            ),
            "all_predivergence_torque_commands_byte_identical_nominal": (
                predivergence_exact
            ),
            "predivergence_archived_state_action_observation_reward_goal_parity_exact": (
                predivergence_parity_exact
            ),
            "historical_contact_same_source_action_exception_requires_exact_callback_action_0_0_and_exact_contact_free_prior_completed_endpoint": (
                v4_same_source_action_exception_valid
            ),
            "all_live_substeps_before_first_divergence_contact_free": (
                v4_predivergence_contact_free
            ),
            "all_shield_decisions_valid": True,
            "all_actual_cbf_residual_postchecks_pass": all(
                row.get("actual_cbf_residual_postcheck_pass") is True
                for row in physics
            ),
            "complete_terminal_condition_reached": complete_terminal,
            "video_complete": video_complete,
        }
        for name, expected in reconstructed_apparatus.items():
            _require(
                metrics.get(name) is expected,
                "v4 apparatus metric %s differs from independent reconstruction"
                % name,
            )

        material_correction_l2 = (
            0.0
            if not first_material_rows
            else float(first_material_rows[0]["torque_correction_l2_nm"])
        )
        residual_improvement = (
            0.0
            if first_material_residual_improvement is None
            else float(first_material_residual_improvement)
        )
        next_qvel_change = (
            0.0
            if first_material_next_qvel_change is None
            else float(first_material_next_qvel_change)
        )
        nonstopping_motion = bool(
            first_material is not None
            and motion["joint"]
            >= float(
                derived["acceptance"][
                    "minimum_post_correction_joint_motion_integral_rad"
                ]
            )
            and motion["eef"]
            >= float(
                derived["acceptance"][
                    "minimum_post_correction_eef_path_length_m"
                ]
            )
        )
        stalled_after_correction = bool(
            first_material is not None and not nonstopping_motion
        )
        reconstructed_outcomes = {
            "any_robot_selected_obstacle_contact": registered_contact_audit[
                "any_robot_selected_obstacle_contact"
            ],
            "any_link56_external_nonrobot_contact": registered_contact_audit[
                "any_link56_external_nonrobot_contact"
            ],
            "any_registered_forbidden_contact": registered_contact,
            "safety_method_stop_before_physics": method_stopped,
            "treatment_stalled_after_correction": stalled_after_correction,
            "material_correction_present": first_material is not None,
            "first_any_byte_different_torque_registered_as_divergence": (
                first_divergence_registered
            ),
            "first_material_correction_protected_link_attributed": (
                first_material_minimum_is_protected_link
            ),
            "first_material_nominal_exact_protected_link_cbf_residual_negative": (
                first_material_nominal_negative
                and first_material_minimum_is_protected_link
            ),
            # Historical-target participation is diagnostic only.  The
            # registered causal gate is any row in the shared protected set.
            "first_material_correction_target_link_attributed": (
                first_material_minimum_is_target_link
            ),
            "first_material_nominal_exact_target_link_cbf_residual_negative": (
                first_material_nominal_negative
                and first_material_minimum_is_target_link
            ),
            "first_material_correction_before_historical_target_link_contact": (
                expected_before
            ),
            "task_incomplete_at_first_material_correction": bool(
                expected_task_incomplete
            ),
            "treatment_paper_car_avoided": expected_car_safe,
            "nonstopping_motion_after_correction": nonstopping_motion,
            "native_task_success": task_success,
        }
        for name, expected in reconstructed_outcomes.items():
            _require(
                metrics.get(name) is expected,
                "v4 outcome metric %s differs from independent reconstruction"
                % name,
            )
        reconstructed_numbers = {
            "material_torque_correction_l2_nm": material_correction_l2,
            "first_material_exact_cbf_residual_improvement_m2_per_s": (
                residual_improvement
            ),
            "first_material_exact_next_qvel_change_l2_rad_s": next_qvel_change,
            "post_correction_joint_motion_integral_rad": motion["joint"],
            "post_correction_eef_path_length_m": motion["eef"],
        }
        for name, expected in reconstructed_numbers.items():
            _close(metrics.get(name), expected, "v4 outcome metric %s" % name)

    independent = classify_canary(dict(metrics), protocol)
    _require(
        not independent.get("feasible")
        or (
            (
                first_material is not None
                and first_material_minimum_is_protected_link
                and first_material_audit is not None
                if target_link_v4
                else first_divergence_is_material
            )
            and bool(independently_validated_qp_audits)
        ),
        "positive feasibility lacks a protected-link-attributed material QP audit",
    )
    _require(
        metrics.get("allocation_numeric_prerequisite_verified") is True,
        "producer did not record the independently verified numeric prerequisite",
    )
    _require(
        fast._canonical(independent) == fast._canonical(result.get("classification")),
        "producer classification differs from independent reconstruction",
    )
    return {
        "status": "validated",
        "producer_status": "complete",
        "case_id": result["case_id"],
        "protocol_relative_path": expected_protocol_relative_path,
        "protocol_file_sha256": fast._file_sha256(protocol_path),
        "run_id": result["run_id"],
        "result_file_sha256": fast._file_sha256(result_path),
        "classification": independent,
        "action_count": len(actions),
        "physics_substep_count": len(physics),
        "solved_qp_count": solved_qp_count,
        "independently_kkt_validated_solved_qp_count": len(
            independently_validated_qp_audits
        ),
        "independent_kkt_validation_scope": (
            "first_byte_divergence_and_first_material_correction_rows_deduplicated"
            if target_link_v4
            else "first_byte_different_torque_row_only"
        ),
        "first_divergence_constraint_attribution": divergence_attribution,
        "first_material_constraint_attribution": material_attribution,
        "historical_target_contact_audit": historical_target_contact_audit,
        "partial_action_audit": partial_action_audit,
        "field_sample_static_audit": field_audit,
        "first_material_correction_physical_boundary": first_material,
        "literal_contact_present": measured_contact,
        "native_task_success": task_success,
        "producer_source_commit": expected_producer_commit,
        "producer_slurm_job_id": expected_producer_job_id,
        "numeric_prerequisite": numeric_prerequisite,
        "numeric_slurm_job_id": expected_numeric_job_id,
        "consumer_source_commit": expected_consumer_commit,
        "consumer_slurm_job_id": expected_consumer_job_id,
        "partial_output_interpreted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-case-id", required=True)
    parser.add_argument(
        "--expected-protocol-relative-path",
        default=Path("configs/vlsa_poisson_osc_arm_link_canary.v3.json"),
    )
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-validation-result", type=Path, required=True)
    parser.add_argument("--expected-numeric-job-id", required=True)
    parser.add_argument("--expected-producer-job-id", required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-consumer-job-id", required=True)
    parser.add_argument("--expected-consumer-commit", required=True)
    arguments = parser.parse_args()
    expected_protocol_relative_path = str(
        arguments.expected_protocol_relative_path
    )
    from main.poisson_fullbody.contracts import publish_hashed_json
    from main.poisson_fullbody import osc_arm_link_canary as canary_contract

    protocol_header = fast._json(
        arguments.protocol.resolve(), "post-OSC canary protocol"
    )
    validation_schema = (
        canary_contract.TARGET_LINK_VALIDATION_SCHEMA
        if protocol_header.get("schema_version")
        == "vlsa_poisson_osc_target_link_canary_protocol.v4"
        else canary_contract.VALIDATION_SCHEMA
    )

    candidate: Dict[str, Any] = {
        "schema_version": validation_schema,
        "status": "rejected",
        "expected_case_id": arguments.expected_case_id,
        "expected_protocol_relative_path": expected_protocol_relative_path,
        "partial_output_interpreted": False,
    }
    try:
        candidate.update(
            validate(
                arguments.result.resolve(),
                arguments.protocol.resolve(),
                arguments.historical_result_root.resolve(),
                arguments.numeric_validation_result,
                expected_case_id=arguments.expected_case_id,
                expected_protocol_relative_path=expected_protocol_relative_path,
                expected_numeric_job_id=arguments.expected_numeric_job_id,
                expected_producer_job_id=arguments.expected_producer_job_id,
                expected_producer_commit=arguments.expected_producer_commit,
                expected_consumer_job_id=arguments.expected_consumer_job_id,
                expected_consumer_commit=arguments.expected_consumer_commit,
            )
        )
    except Exception as error:
        candidate["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    publish_hashed_json(arguments.output.resolve(), candidate)
    print(json.dumps({"status": candidate["status"], "output": str(arguments.output)}, sort_keys=True))
    return 0 if candidate["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
