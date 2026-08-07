#!/usr/bin/env python3
"""Independently validate a distal three-ellipsoid exact-action replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARCHIVED_FILE_SHA256 = "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
ARCHIVED_PAYLOAD_SHA256 = "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
EXPECTED_ACTION_COUNT = 237


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("%s is missing or symlinked" % label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("%s must be one JSON object" % label)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    _require(commit == expected_commit, "validator source commit differs")
    _require(not status, "validator source tree is dirty")
    return {"commit": commit, "dirty": False, "branch": run("branch", "--show-current")}


def _timing_stats(values: Any) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and len(array) == EXPECTED_ACTION_COUNT, "timing count differs")
    _require(np.all(np.isfinite(array)) and np.all(array >= 0.0), "timing is invalid")
    return {
        "mean_seconds": float(np.mean(array)),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
        "maximum_seconds": float(np.max(array)),
    }


def validate(
    *,
    candidate_path: Path,
    archived_path: Path,
    config_path: Path,
    repo_root: Path,
    producer_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    candidate = _load(candidate_path, "candidate replay")
    archived = _load(archived_path, "archived Table-1 result")
    config = load_shadow_config(config_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "archived file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "archived payload differs")
    payload = {key: value for key, value in candidate.items() if key != "result_payload_sha256"}
    _require(candidate.get("result_payload_sha256") == _sha256(_canonical(payload)), "candidate payload hash is invalid")
    _require(candidate.get("schema_version") == "vlsa_distal_three_ellipsoid_replay.v1", "candidate schema differs")
    _require(candidate.get("status") == "validated", "candidate status differs")
    _require(candidate.get("case_id") == CASE_ID, "candidate case differs")
    _require(candidate["source"]["commit"] == producer_commit, "producer commit differs")
    _require(candidate["source"]["dirty"] is False, "producer source was dirty")
    _require(candidate["config"]["config_file_sha256"] == config["config_file_sha256"], "config identity differs")
    _require(candidate["archived_table1"]["file_sha256"] == ARCHIVED_FILE_SHA256, "candidate archived binding differs")
    _require(candidate["archived_table1"]["executed_sequence_sha256"] == archived["action_invariance_ledger"]["executed_sequence_sha256"], "executed ledger binding differs")
    for key in (
        "manifest_row_sha256",
        "initial_state_sha256",
        "initial_observation_sha256",
        "settled_simulator_state_sha256",
        "settled_active_obstacle_position_sha256",
        "policy_noise_schedule_sha256",
    ):
        _require(candidate["pairing"][key] == archived["pairing"][key], "pairing field differs: %s" % key)

    geometry = candidate["geometry"]
    rows = geometry["link_ellipsoids"]
    expected_bodies = ["robot0_link5", "robot0_link6", "robot0_link7"]
    _require(geometry["link_ellipsoid_count"] == 3 and len(rows) == 3, "geometry count differs")
    _require([row["body_name"] for row in rows] == expected_bodies, "geometry bodies differ")
    geometry_receipt: list[dict[str, Any]] = []
    for row in rows:
        certificate = row["enclosure_certificate"]
        points = np.asarray(certificate["source_vertices_world_m"], dtype=np.float64)
        center = np.asarray(row["center_m"], dtype=np.float64)
        rotation = np.asarray(row["rotation"], dtype=np.float64)
        semiaxes = np.asarray(row["semiaxes_m"], dtype=np.float64)
        _require(row["source_body_names"] == [row["body_name"]], "bound crosses rigid links")
        _require(row["bound_source"] == "compiled_mesh_vertex_mvee_enclosure", "bound source differs")
        _require(certificate.get("khachiyan_converged") is True, "MVEE did not converge")
        observed_hash = _sha256(np.ascontiguousarray(points, dtype="<f8").tobytes(order="C"))
        _require(observed_hash == certificate["source_vertices_float64_sha256"], "source vertex hash differs")
        normalized = np.sum((((points - center) @ rotation) / semiaxes) ** 2, axis=1)
        maximum = float(np.max(normalized))
        _require(maximum <= 1.0 + 1.0e-12 and maximum >= 0.999999, "MVEE containment/surface check failed")
        _require(abs(maximum - float(certificate["maximum_normalized_quadratic"])) <= 1.0e-12, "MVEE certificate differs")
        geometry_receipt.append(
            {
                "body_name": row["body_name"],
                "geom_name": row["geom_name"],
                "source_vertex_count": int(len(points)),
                "semiaxes_m": semiaxes.tolist(),
                "maximum_normalized_vertex_quadratic": maximum,
                "near_surface_vertex_count": int(certificate["near_surface_vertex_count"]),
                "khachiyan_iterations": int(certificate["khachiyan_iterations"]),
            }
        )

    actions = candidate["actions"]
    archived_actions = archived["actions"]
    _require(len(actions) == len(archived_actions) == EXPECTED_ACTION_COUNT, "action count differs")
    direct_geoms: set[str] = set()
    qp_total: list[float] = []
    qp_solve: list[float] = []
    shadow_total: list[float] = []
    qp_reasons: dict[str, int] = {}
    for index, (action, archived_action) in enumerate(zip(actions, archived_actions)):
        _require(action["step"] == archived_action["step"] == index, "action indexes differ")
        _require(action["archived_env_step_input_sha256"] == _sha256(_canonical(archived_action["env_step_input"])), "archived action hash differs")
        _require(action["done"] is archived_action["done"], "done flag differs")
        _require(action["goal_progress"]["values"] == archived_action["goal_progress"]["values"], "goal vector differs")
        _require(np.allclose(action["eef_position_m"], archived_action["post_step_controller_proxy"]["eef_position"], rtol=0.0, atol=1.0e-12), "end-effector replay differs")
        for event in action["robot_contact_events"]:
            name = event["other"].get("geom_name")
            if isinstance(name, str):
                direct_geoms.add(name)
        shadow = action["shadow"]
        _require(shadow["step"] == index and shadow["constraint_count"] == 3, "shadow constraint count differs")
        _require([item["body_name"] for item in shadow["constraints"]] == expected_bodies, "shadow body order differs")
        _require(all(len(item["cbf_row_m_per_rad"]) == 7 for item in shadow["constraints"]), "CBF row dimension differs")
        _require(shadow["qp"]["valid"] is True, "QP is invalid")
        _require(shadow["qp"]["diagnostics"]["input_constraint_count"] == 3, "OSQP input count differs")
        _require(shadow["D_sim"]["available"] is True, "D_sim is unavailable")
        reason = str(shadow["qp"]["reason"])
        qp_reasons[reason] = qp_reasons.get(reason, 0) + 1
        timing = shadow["qp"]["diagnostics"]["timing"]
        qp_total.append(float(timing["total_wall_seconds"]))
        qp_solve.append(float(timing["solve_wall_seconds"]))
        shadow_total.append(float(shadow["timing"]["total_shadow_wall_seconds"]))

    _require({"robot0_link5_collision", "robot0_link6_collision"}.issubset(direct_geoms), "direct L5/L6 contact evidence differs")
    raw = candidate["raw_simulation_evidence"]
    _require(raw["first_robot_contact_step"] == 187, "first robot contact differs")
    _require(raw["native_task_success"] is True and raw["native_task_success_step"] == 236, "native task success differs")
    total_stats = _timing_stats(qp_total)
    solve_stats = _timing_stats(qp_solve)
    observer_stats = _timing_stats(shadow_total)
    for expected, observed, label in (
        (candidate["shadow_summary"]["qp_total_wall_timing"], total_stats, "QP total"),
        (candidate["shadow_summary"]["qp_solve_wall_timing"], solve_stats, "QP solve"),
        (candidate["shadow_summary"]["total_shadow_wall_timing"], observer_stats, "shadow total"),
    ):
        _require(all(abs(float(expected[key]) - observed[key]) <= 1.0e-15 for key in observed), "%s timing summary differs" % label)
    validator_allocation = allocation_record()
    return {
        "schema_version": "vlsa_distal_three_ellipsoid_replay_validation.v1",
        "status": "validated",
        "case_id": CASE_ID,
        "source": _git_identity(repo_root, validator_commit),
        "validator_allocation": validator_allocation,
        "producer_allocation": candidate["allocation"],
        "candidate": {
            "path": str(candidate_path),
            "file_sha256": _file_sha256(candidate_path),
            "payload_sha256": candidate["result_payload_sha256"],
            "producer_commit": producer_commit,
        },
        "archived_table1": {
            "file_sha256": ARCHIVED_FILE_SHA256,
            "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
            "read_only": True,
        },
        "geometry": geometry_receipt,
        "raw_simulation_evidence": raw,
        "qp": {
            "constraint_count": 3,
            "step_count": EXPECTED_ACTION_COUNT,
            "reason_counts": dict(sorted(qp_reasons.items())),
            "all_valid": True,
            "first_nominal_violation": candidate["shadow_summary"]["first_nominal_violation"],
            "minimum_h_opt_m": candidate["shadow_summary"]["minimum_h_opt_m"],
            "total_wall_timing": total_stats,
            "solve_wall_timing": solve_stats,
            "observer_wall_timing": observer_stats,
        },
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        candidate_path=args.candidate.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        repo_root=args.repo_root.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve()), "receipt_payload_sha256": receipt["receipt_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
