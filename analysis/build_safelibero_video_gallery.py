#!/usr/bin/env python3
"""Build a static paired-video index for a strict SafeLIBERO aggregate.

The gallery is read-only: it does not render videos or repair result records.
By default the exact complete paired population declared by the aggregate is
required. ``--allow-partial`` exists only for debugging incomplete runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import quote

try:
    from analysis.aggregate_safelibero_aegis import (
        AggregationError as _AggregationError,
        iter_result_records as _iter_aggregate_result_records,
    )
except ModuleNotFoundError:
    from aggregate_safelibero_aegis import (  # type: ignore[no-redef]
        AggregationError as _AggregationError,
        iter_result_records as _iter_aggregate_result_records,
    )


SUMMARY_SCHEMA = "vlsa_table1_population_summary.v1"
GALLERY_SOURCE_KEY = "__gallery_source__"
GALLERY_TAXONOMY_KEY = "__gallery_taxonomy__"
EVALUATOR_ARM_DIRECTORIES = {"pi05", "aegis"}

TAXONOMY: tuple[tuple[str, str], ...] = (
    ("semantic_selector", "Semantic selector"),
    ("groundingdino", "GroundingDINO"),
    ("point_filtering_mvee", "Point filtering / MVEE"),
    ("qp_method_failure", "QP / method failure"),
    (
        "active_obstacle_displacement",
        "Active-obstacle displacement",
    ),
    (
        "mujoco_robot_obstacle_contact",
        "MuJoCo robot–obstacle contact",
    ),
    ("selector_active_mismatch", "Selector / active mismatch"),
    (
        "safe_but_task_failed_ood_proxy",
        "Safe but task failed (OOD proxy)",
    ),
    ("apparatus", "Apparatus"),
)

EVIDENCE_STATES = {"present", "absent", "unknown", "not_applicable"}
SCIENTIFIC_STATUSES = {
    "complete",
    "method_failure",
    "method_failure_passthrough",
}


class GalleryError(RuntimeError):
    """Raised when gallery input is incomplete, ambiguous, or unsafe."""


class CompactGalleryRecord(dict):
    """Internal marker for taxonomy evidence derived during streaming."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _case_value(result: Mapping[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    return _mapping(result.get("case")).get(key)


def _explicit_bool(*values: Any) -> bool | None:
    observed = [value for value in values if isinstance(value, bool)]
    if not observed:
        return None
    if any(value != observed[0] for value in observed[1:]):
        raise GalleryError("conflicting Boolean evidence in result")
    return observed[0]


def _explicit_int(*values: Any) -> int | None:
    observed = [
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if not observed:
        return None
    if any(value != observed[0] for value in observed[1:]):
        raise GalleryError("conflicting integer evidence in result")
    return observed[0]


def episode_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read only explicit episode evidence; never fill a missing metric."""

    metrics = _mapping(result.get("metrics"))
    collision = _explicit_bool(
        metrics.get("public_collision"),
        metrics.get("paper_collision"),
        result.get("public_collision"),
        result.get("paper_collision"),
    )
    task_success = _explicit_bool(
        metrics.get("task_success"),
        result.get("task_success"),
    )
    return {
        "collision": collision,
        "safe": None if collision is None else not collision,
        "task_success": task_success,
        "legacy_ets_steps": _explicit_int(
            metrics.get("legacy_ets_steps"),
            result.get("legacy_ets_steps"),
        ),
        "executed_action_count": _explicit_int(
            metrics.get("executed_action_count"),
            result.get("executed_action_count"),
        ),
    }


def _error_text(result: Mapping[str, Any]) -> str:
    values: list[str] = []
    for field in ("apparatus_error", "method_failure"):
        error = result.get(field)
        if isinstance(error, Mapping):
            values.extend(
                str(error.get(key, ""))
                for key in (
                    "component",
                    "type",
                    "message",
                    "traceback",
                )
            )
        elif error:
            values.append(str(error))
    return " ".join(values).lower()


def _tri_state(value: bool | None) -> str:
    if value is True:
        return "present"
    if value is False:
        return "absent"
    return "unknown"


def classify_failure_taxonomy(
    result: Mapping[str, Any],
    *,
    aegis_arm: bool,
) -> dict[str, str]:
    """Return explicit tri-state evidence for every registered failure class."""

    cached = (
        _mapping(result.get(GALLERY_TAXONOMY_KEY))
        if isinstance(result, CompactGalleryRecord)
        else {}
    )
    cached_arm = cached.get("aegis" if aegis_arm else "baseline")
    if isinstance(cached_arm, Mapping):
        taxonomy = dict(cached_arm)
        if set(taxonomy) != {key for key, _ in TAXONOMY}:
            raise GalleryError("cached taxonomy coverage error")
        if any(
            value not in EVIDENCE_STATES for value in taxonomy.values()
        ):
            raise GalleryError("cached taxonomy state error")
        return taxonomy

    status = result.get("status")
    metrics = episode_metrics(result)
    perception = _mapping(result.get("perception"))
    obstacle = _mapping(result.get("obstacle"))
    settled = _mapping(result.get("settled_observation"))
    contact = _mapping(result.get("contact_telemetry"))
    error_text = _error_text(result)

    taxonomy: dict[str, str] = {}
    if not aegis_arm:
        for key in (
            "semantic_selector",
            "groundingdino",
            "point_filtering_mvee",
            "qp_method_failure",
            "selector_active_mismatch",
        ):
            taxonomy[key] = "not_applicable"
    else:
        selector_error = any(
            token in error_text
            for token in (
                "codex label",
                "label case binding",
                "label image hash",
                "obstacle label",
                "semantic selector",
            )
        )
        if selector_error:
            taxonomy["semantic_selector"] = "present"
        elif (
            isinstance(settled.get("obstacle_label"), str)
            and bool(settled.get("obstacle_label"))
            and isinstance(settled.get("label_record"), Mapping)
        ):
            taxonomy["semantic_selector"] = "absent"
        else:
            taxonomy["semantic_selector"] = "unknown"

        method_failure = str(perception.get("method_failure", "")).lower()
        perception_status = str(perception.get("status", "")).lower()
        retained_failure = _mapping(result.get("method_failure"))
        retained_component = str(
            retained_failure.get("component", "")
        ).lower()
        retained_phase = str(retained_failure.get("phase", "")).lower()
        precontrol_geometry_failure = (
            retained_component == "aegis_geometry"
            and retained_phase == "precontrol"
        )
        method_error_text = " ".join(
            (
                method_failure,
                str(perception.get("error", "")).lower(),
                str(perception.get("reason", "")).lower(),
                str(retained_failure.get("message", "")).lower(),
                error_text,
            )
        )
        grounding_error = any(
            token in method_error_text
            for token in ("groundingdino", "grounding dino")
        )
        if grounding_error or method_failure == "no_grounded_points":
            taxonomy["groundingdino"] = "present"
        elif method_failure in {
            "point_filter_removed_all_points",
            "invalid_mvee",
            "degenerate_initial_direction",
        } or perception_status == "ready":
            taxonomy["groundingdino"] = "absent"
        else:
            taxonomy["groundingdino"] = "unknown"

        point_failure = method_failure in {
            "point_filter_removed_all_points",
            "invalid_mvee",
            "degenerate_initial_direction",
        } or (
            perception_status in {"mvee_failure"}
        )
        if any(
            token in method_error_text
            for token in (
                "mvee",
                "ellipse",
                "filtering_points",
                "convexhull",
                "convex hull",
                "qhull",
            )
        ):
            point_failure = True
        if point_failure:
            taxonomy["point_filtering_mvee"] = "present"
        elif perception_status == "ready":
            taxonomy["point_filtering_mvee"] = "absent"
        else:
            taxonomy["point_filtering_mvee"] = "unknown"

        qp_error = any(
            token in error_text for token in ("osqp", "qp ", "qp_")
        )
        actions = result.get("actions")
        qp_records = []
        if isinstance(actions, list):
            qp_records = [
                action.get("qp")
                for action in actions
                if isinstance(action, Mapping)
                and isinstance(action.get("qp"), Mapping)
            ]
        qp_bad = any(
            str(record.get("solver_status", "")).lower()
            not in {"optimal", "optimal_inaccurate"}
            for record in qp_records
        )
        if (
            precontrol_geometry_failure
            or method_failure
            in {
                "no_grounded_points",
                "point_filter_removed_all_points",
                "invalid_mvee",
                "degenerate_initial_direction",
            }
        ):
            taxonomy["qp_method_failure"] = "not_applicable"
        elif (
            qp_error
            or qp_bad
        ):
            taxonomy["qp_method_failure"] = "present"
        elif status == "complete" and qp_records:
            taxonomy["qp_method_failure"] = "absent"
        else:
            taxonomy["qp_method_failure"] = "unknown"

        mismatch = obstacle.get("selector_mismatch")
        taxonomy["selector_active_mismatch"] = _tri_state(
            mismatch if isinstance(mismatch, bool) else None
        )

    taxonomy["active_obstacle_displacement"] = _tri_state(
        metrics["collision"]
    )

    contact_value = contact.get("robot_active_obstacle_contact")
    contact_status = contact.get("status")
    if contact_value is True:
        taxonomy["mujoco_robot_obstacle_contact"] = "present"
    elif contact_status == "available" and contact_value is False:
        taxonomy["mujoco_robot_obstacle_contact"] = "absent"
    else:
        taxonomy["mujoco_robot_obstacle_contact"] = "unknown"

    if (
        isinstance(metrics["safe"], bool)
        and isinstance(metrics["task_success"], bool)
    ):
        taxonomy["safe_but_task_failed_ood_proxy"] = _tri_state(
            metrics["safe"] and not metrics["task_success"]
        )
    else:
        taxonomy["safe_but_task_failed_ood_proxy"] = "unknown"

    if status == "apparatus_failure" or result.get("apparatus_error"):
        taxonomy["apparatus"] = "present"
    elif status in SCIENTIFIC_STATUSES:
        taxonomy["apparatus"] = "absent"
    else:
        taxonomy["apparatus"] = "unknown"

    if set(taxonomy) != {key for key, _ in TAXONOMY}:
        raise GalleryError("internal taxonomy coverage error")
    if any(value not in EVIDENCE_STATES for value in taxonomy.values()):
        raise GalleryError("internal taxonomy state error")
    return taxonomy


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GalleryError("aggregate summary must be a JSON object")
    if (
        value.get("schema_version") != SUMMARY_SCHEMA
        or value.get("status") != "complete_population_validated"
    ):
        raise GalleryError("input is not a strict validated aggregate")
    population = _mapping(value.get("population"))
    if population.get("no_results_dropped") is not True:
        raise GalleryError("aggregate does not guarantee no result dropping")
    for key in ("cases", "arms", "results"):
        number = population.get(key)
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number < 1
        ):
            raise GalleryError(f"aggregate population.{key} is invalid")
    average = value.get("average")
    if not isinstance(average, Mapping) or not average:
        raise GalleryError("aggregate has no arm summaries")
    if len(average) != population["arms"]:
        raise GalleryError("aggregate arm count is inconsistent")
    return value


def _evaluator_artifact_root(path: Path) -> Path | None:
    """Return the output root for an evaluator-layout ``result.json``."""

    if (
        path.name != "result.json"
        or path.parent.parent.name not in EVALUATOR_ARM_DIRECTORIES
    ):
        return None
    return path.parent.parent.parent.resolve()


def _compact_result_record(
    result: Mapping[str, Any],
    *,
    path: Path,
    artifact_root: Path | None,
) -> CompactGalleryRecord:
    """Retain only the fixed-size evidence consumed by the gallery."""

    for reserved_key in (GALLERY_SOURCE_KEY, GALLERY_TAXONOMY_KEY):
        if reserved_key in result:
            raise GalleryError(
                f"{path}: result uses reserved field {reserved_key}"
            )
    metrics = episode_metrics(result)
    video = _mapping(result.get("video"))
    compact_video = {
        key: video.get(key)
        for key in ("path", "sha256")
        if key in video
    }
    taxonomy = {
        "baseline": classify_failure_taxonomy(
            result,
            aegis_arm=False,
        ),
        "aegis": classify_failure_taxonomy(
            result,
            aegis_arm=True,
        ),
    }
    return CompactGalleryRecord({
        "protocol_id": result.get("protocol_id"),
        "case_id": result.get("case_id"),
        "arm": result.get("arm"),
        "status": result.get("status"),
        "suite": _case_value(result, "suite"),
        "safety_level": _case_value(result, "safety_level"),
        "logical_task_index": _case_value(
            result, "logical_task_index"
        ),
        "resolved_task_index": _case_value(
            result, "resolved_task_index"
        ),
        "task_name": _case_value(result, "task_name"),
        "episode_index": _case_value(result, "episode_index"),
        "metrics": {
            "public_collision": metrics["collision"],
            "task_success": metrics["task_success"],
            "legacy_ets_steps": metrics["legacy_ets_steps"],
            "executed_action_count": metrics["executed_action_count"],
        },
        "video": compact_video,
        GALLERY_TAXONOMY_KEY: taxonomy,
        GALLERY_SOURCE_KEY: {
            "result_path": str(path),
            "artifact_root": (
                None if artifact_root is None else str(artifact_root)
            ),
        },
    })


def _iter_strict_result_records(path: Path) -> Iterable[dict[str, Any]]:
    """Translate the shared strict streaming decoder into gallery errors."""

    try:
        yield from _iter_aggregate_result_records(path)
    except _AggregationError as error:
        raise GalleryError(str(error)) from error


def load_result_records(
    paths: Iterable[Path],
    *,
    streaming_stats: MutableMapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Stream full inputs and retain only compact gallery records.

    Each object is decoded, classified, compacted, and released before the
    next object is decoded. Input syntax is content-driven: one object, a
    top-level array, and whitespace-separated JSON objects are accepted
    regardless of the filename suffix.
    """

    files: list[Path] = []
    seen_files: set[Path] = set()
    for supplied in paths:
        path = supplied.resolve()
        if path.is_dir():
            discovered = sorted(path.rglob("result.json"))
        elif path.is_file():
            discovered = [path]
        else:
            raise GalleryError(f"result input does not exist: {supplied}")
        for item in discovered:
            resolved = item.resolve()
            if resolved not in seen_files:
                files.append(resolved)
                seen_files.add(resolved)
    if not files:
        raise GalleryError("no per-episode result JSONs were found")
    records: list[dict[str, Any]] = []
    live_full_results = 0
    max_live_full_results = 0
    for path in files:
        artifact_root = _evaluator_artifact_root(path)
        for row in _iter_strict_result_records(path):
            live_full_results += 1
            max_live_full_results = max(
                max_live_full_results,
                live_full_results,
            )
            try:
                records.append(
                    _compact_result_record(
                        row,
                        path=path,
                        artifact_root=artifact_root,
                    )
                )
            finally:
                live_full_results -= 1
            del row
    if streaming_stats is not None:
        streaming_stats.clear()
        streaming_stats.update(
            {
                "result_files": len(files),
                "result_records": len(records),
                "max_live_full_results": max_live_full_results,
                "retained_compact_results": len(records),
            }
        )
    return records


def _arm_names(summary: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in _mapping(summary["average"]).keys()]


def index_results(
    summary: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    allow_partial: bool,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    arms = _arm_names(summary)
    expected_arm_set = set(arms)
    protocol_id = summary.get("protocol_id")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    cases: defaultdict[str, set[str]] = defaultdict(set)
    warnings: list[str] = []

    for result in records:
        case_id = result.get("case_id")
        arm = result.get("arm")
        if not isinstance(case_id, str) or not case_id:
            raise GalleryError("result has no valid case_id")
        if arm not in expected_arm_set:
            raise GalleryError(f"{case_id}: unexpected arm {arm!r}")
        if result.get("protocol_id") != protocol_id:
            raise GalleryError(f"{case_id}/{arm}: protocol mismatch")
        key = (case_id, str(arm))
        if key in indexed:
            raise GalleryError(f"duplicate per-episode result: {key}")
        indexed[key] = result
        cases[case_id].add(str(arm))

    population = _mapping(summary["population"])
    expected_cases = int(population["cases"])
    expected_results = int(population["results"])
    incomplete_cases = {
        case_id: sorted(expected_arm_set - observed)
        for case_id, observed in cases.items()
        if observed != expected_arm_set
    }
    if (
        len(indexed) != expected_results
        or len(cases) != expected_cases
        or incomplete_cases
    ):
        message = (
            "incomplete indexed result set: "
            f"results={len(indexed)}/{expected_results}, "
            f"cases={len(cases)}/{expected_cases}, "
            f"incomplete_pairs={len(incomplete_cases)}"
        )
        if not allow_partial:
            raise GalleryError(message)
        warnings.append(message)

    for case_id, observed_arms in cases.items():
        pair = [indexed[(case_id, arm)] for arm in observed_arms]
        reference = pair[0]
        for result in pair[1:]:
            for field in (
                "suite",
                "safety_level",
                "logical_task_index",
                "resolved_task_index",
                "task_name",
                "episode_index",
            ):
                if _case_value(result, field) != _case_value(reference, field):
                    raise GalleryError(
                        f"{case_id}: paired metadata differs for {field}"
                    )

    return indexed, warnings


def _require_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise GalleryError(f"{label} is not finite")
    return float(value)


def _assert_close(observed: float, expected: Any, label: str) -> None:
    target = _require_number(expected, label)
    if not math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-9):
        raise GalleryError(
            f"{label} differs: results={observed}, aggregate={target}"
        )


def verify_summary_alignment(
    summary: Mapping[str, Any],
    indexed: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    allow_partial: bool,
) -> list[str]:
    """Recompute displayed metrics and bind them to the aggregate."""

    if allow_partial:
        return [
            "Partial debugging mode: aggregate metric alignment was not "
            "asserted."
        ]
    warnings: list[str] = []
    arms = _arm_names(summary)
    suites_by_arm = _mapping(summary.get("suites"))
    for arm in arms:
        arm_results = [
            result
            for (_, result_arm), result in indexed.items()
            if result_arm == arm
        ]
        suite_summary = _mapping(suites_by_arm.get(arm))
        if not suite_summary:
            raise GalleryError(f"aggregate has no suite summaries for {arm}")
        suite_observed: list[dict[str, Any]] = []
        for suite, expected in suite_summary.items():
            rows = [
                result
                for result in arm_results
                if _case_value(result, "suite") == suite
            ]
            if not rows:
                raise GalleryError(f"{arm}/{suite}: no episode results")
            metrics = [episode_metrics(result) for result in rows]
            if any(
                value is None
                for metric in metrics
                for value in metric.values()
            ):
                raise GalleryError(
                    f"{arm}/{suite}: explicit episode metrics are incomplete"
                )
            observed = {
                "episodes": len(rows),
                "car_percent": 100.0
                * sum(bool(metric["safe"]) for metric in metrics)
                / len(rows),
                "tsr_percent": 100.0
                * sum(bool(metric["task_success"]) for metric in metrics)
                / len(rows),
                "legacy_ets_steps_mean": sum(
                    int(metric["legacy_ets_steps"]) for metric in metrics
                )
                / len(rows),
                "executed_action_count_mean": sum(
                    int(metric["executed_action_count"])
                    for metric in metrics
                )
                / len(rows),
                "status_counts": dict(
                    sorted(
                        Counter(
                            str(result.get("status")) for result in rows
                        ).items()
                    )
                ),
            }
            if observed["episodes"] != expected.get("episodes"):
                raise GalleryError(f"{arm}/{suite}: episode count differs")
            for key in (
                "car_percent",
                "tsr_percent",
                "legacy_ets_steps_mean",
                "executed_action_count_mean",
            ):
                _assert_close(
                    float(observed[key]),
                    expected.get(key),
                    f"{arm}/{suite}/{key}",
                )
            if observed["status_counts"] != expected.get("status_counts"):
                raise GalleryError(f"{arm}/{suite}: status counts differ")
            suite_observed.append(observed)

        overall = _mapping(summary["average"]).get(arm)
        if not isinstance(overall, Mapping):
            raise GalleryError(f"aggregate has no average for {arm}")
        if int(overall.get("episodes", -1)) != sum(
            value["episodes"] for value in suite_observed
        ):
            raise GalleryError(f"{arm}: overall episode count differs")
        for key in (
            "car_percent",
            "tsr_percent",
            "legacy_ets_steps_mean",
            "executed_action_count_mean",
        ):
            observed_mean = sum(
                float(value[key]) for value in suite_observed
            ) / len(suite_observed)
            _assert_close(
                observed_mean,
                overall.get(key),
                f"{arm}/average/{key}",
            )
        overall_status = Counter()
        for value in suite_observed:
            overall_status.update(value["status_counts"])
        if dict(sorted(overall_status.items())) != overall.get(
            "status_counts"
        ):
            raise GalleryError(f"{arm}: overall status counts differ")
    return warnings


def _video_record(
    result: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    video = _mapping(result.get("video"))
    raw_path = video.get("path")
    if raw_path is None:
        return {"href": None, "exists": False, "reason": "not recorded"}
    if not isinstance(raw_path, str) or not raw_path:
        raise GalleryError("video.path must be a non-empty relative string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise GalleryError(f"unsafe video path: {raw_path!r}")
    source = _mapping(result.get(GALLERY_SOURCE_KEY))
    registered_root = source.get("artifact_root")
    if registered_root is None:
        root = output_root.resolve()
    elif isinstance(registered_root, str) and registered_root:
        root = Path(registered_root).resolve()
    else:
        raise GalleryError("invalid per-result artifact root")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise GalleryError(
            f"video escapes per-result artifact root: {raw_path!r}"
        ) from error
    exists = target.is_file()
    expected_sha256 = video.get("sha256")
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
        ):
            raise GalleryError("video.sha256 must be lowercase SHA-256 hex")
        if exists:
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise GalleryError(
                    f"video SHA-256 mismatch: {raw_path!r}"
                )
    gallery_root = output_root.resolve()
    href_path = Path(os.path.relpath(target, start=gallery_root))
    return {
        "href": quote(href_path.as_posix(), safe="/"),
        "exists": exists,
        "reason": None if exists else "file not present",
    }


def _display_metric(value: bool | None, true_text: str, false_text: str) -> str:
    if value is True:
        return true_text
    if value is False:
        return false_text
    return "Unknown"


def _attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _text(value: Any) -> str:
    return html.escape(str(value))


def _option(value: str, label: str | None = None) -> str:
    return (
        f'<option value="{_attr(value)}">'
        f"{_text(value if label is None else label)}</option>"
    )


def _render_arm(
    result: Mapping[str, Any] | None,
    *,
    arm: str,
    baseline_arm: str,
    output_root: Path,
) -> str:
    if result is None:
        taxonomy = {key: "unknown" for key, _ in TAXONOMY}
        metrics = {
            "safe": None,
            "task_success": None,
            "legacy_ets_steps": None,
            "executed_action_count": None,
        }
        status = "missing_result"
        video = {"href": None, "exists": False, "reason": "missing result"}
    else:
        taxonomy = classify_failure_taxonomy(
            result, aegis_arm=arm != baseline_arm
        )
        metrics = episode_metrics(result)
        status = str(result.get("status", "unknown"))
        video = _video_record(result, output_root=output_root)

    car_state = (
        "safe"
        if metrics["safe"] is True
        else "collision"
        if metrics["safe"] is False
        else "unknown"
    )
    tsr_state = (
        "success"
        if metrics["task_success"] is True
        else "failure"
        if metrics["task_success"] is False
        else "unknown"
    )
    present = [key for key, value in taxonomy.items() if value == "present"]
    unknown = [key for key, value in taxonomy.items() if value == "unknown"]
    taxonomy_rows = "".join(
        (
            f'<li class="evidence evidence-{_attr(taxonomy[key])}">'
            f"<span>{_text(label)}</span>"
            f"<strong>{_text(taxonomy[key].replace('_', ' '))}</strong>"
            "</li>"
        )
        for key, label in TAXONOMY
    )
    if video["href"] is None:
        video_html = (
            '<div class="video-missing">No video link — '
            f"{_text(video['reason'])}</div>"
        )
    else:
        missing = (
            ""
            if video["exists"]
            else '<span class="warning">file not present</span>'
        )
        video_html = (
            '<video controls preload="metadata">'
            f'<source src="{_attr(video["href"])}" type="video/mp4">'
            "</video>"
            '<div class="video-link">'
            f'<a href="{_attr(video["href"])}">Open MP4</a>{missing}</div>'
        )
    return f"""
      <section class="arm-panel"
        data-arm="{_attr(arm)}"
        data-car="{_attr(car_state)}"
        data-tsr="{_attr(tsr_state)}"
        data-status="{_attr(status)}"
        data-failures="{_attr("|".join(present))}"
        data-has-unknown="{_attr("yes" if unknown else "no")}">
        <h3>{_text(arm)}</h3>
        <div class="badges">
          <span class="badge car-{_attr(car_state)}">CAR:
            {_text(_display_metric(metrics["safe"], "safe", "collision"))}</span>
          <span class="badge tsr-{_attr(tsr_state)}">TSR:
            {_text(_display_metric(metrics["task_success"], "success", "failed"))}</span>
          <span class="badge">Status: {_text(status)}</span>
        </div>
        {video_html}
        <p class="steps">Legacy ETS: {_text(metrics["legacy_ets_steps"])}
          · Executed actions: {_text(metrics["executed_action_count"])}</p>
        <details>
          <summary>Failure evidence ({len(present)} present,
            {len(unknown)} unknown)</summary>
          <ul class="evidence-list">{taxonomy_rows}</ul>
        </details>
      </section>
    """


def render_gallery(
    summary: Mapping[str, Any],
    indexed: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    output_root: Path,
    warnings: Sequence[str],
) -> str:
    arms = _arm_names(summary)
    baseline_arm = arms[0]
    case_ids = sorted({case_id for case_id, _ in indexed})
    cards: list[str] = []
    suite_values: set[str] = set()
    level_values: set[str] = set()
    task_values: set[str] = set()
    status_values: set[str] = {"missing_result"}

    for case_id in case_ids:
        available = [
            indexed[(case_id, arm)]
            for arm in arms
            if (case_id, arm) in indexed
        ]
        reference = available[0]
        suite = str(_case_value(reference, "suite") or "unknown")
        level = str(_case_value(reference, "safety_level") or "unknown")
        task_index = _case_value(reference, "logical_task_index")
        task_name = str(_case_value(reference, "task_name") or "unknown")
        task_key = f"{task_index}: {task_name}"
        suite_values.add(suite)
        level_values.add(level)
        task_values.add(task_key)
        for result in available:
            status_values.add(str(result.get("status", "unknown")))
        panels = "".join(
            _render_arm(
                indexed.get((case_id, arm)),
                arm=arm,
                baseline_arm=baseline_arm,
                output_root=output_root,
            )
            for arm in arms
        )
        cards.append(
            f"""
    <article class="case-card"
      data-suite="{_attr(suite)}"
      data-level="{_attr(level)}"
      data-task="{_attr(task_key)}">
      <header>
        <h2>{_text(case_id)}</h2>
        <p>{_text(suite)} · level {_text(level)} ·
          task {_text(task_index)} — {_text(task_name)}</p>
      </header>
      <div class="paired-grid">{panels}</div>
    </article>
            """
        )

    warning_html = "".join(
        f'<li class="warning">{_text(value)}</li>' for value in warnings
    )
    taxonomy_options = "".join(
        _option(key, label) for key, label in TAXONOMY
    )
    filters = f"""
    <section class="filters" aria-label="Gallery filters">
      <label>Suite<select id="suite-filter">
        {_option("", "All suites")}
        {''.join(_option(value) for value in sorted(suite_values))}
      </select></label>
      <label>Level<select id="level-filter">
        {_option("", "All levels")}
        {''.join(_option(value) for value in sorted(level_values))}
      </select></label>
      <label>Task<select id="task-filter">
        {_option("", "All tasks")}
        {''.join(_option(value) for value in sorted(task_values))}
      </select></label>
      <label>Arm<select id="arm-filter">
        {_option("", "Either arm")}
        {''.join(_option(value) for value in arms)}
      </select></label>
      <label>CAR<select id="car-filter">
        {_option("", "Any CAR")}
        {_option("safe", "Collision avoided")}
        {_option("collision", "Collision")}
        {_option("unknown", "Unknown")}
      </select></label>
      <label>TSR<select id="tsr-filter">
        {_option("", "Any TSR")}
        {_option("success", "Task success")}
        {_option("failure", "Task failed")}
        {_option("unknown", "Unknown")}
      </select></label>
      <label>Status<select id="status-filter">
        {_option("", "Any status")}
        {''.join(_option(value) for value in sorted(status_values))}
      </select></label>
      <label>Failure category<select id="failure-filter">
        {_option("", "Any failure evidence")}
        {taxonomy_options}
        {_option("__unknown__", "Any unknown evidence")}
      </select></label>
      <button id="reset-filters" type="button">Reset</button>
      <output id="visible-count"></output>
    </section>
    """
    population = _mapping(summary["population"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SafeLIBERO paired video gallery</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif;
      --line:#d7dce3; --ink:#18202a; --muted:#647184; --panel:#f7f9fc; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; color:var(--ink); background:#eef2f7; }}
    main {{ max-width:1500px; margin:auto; padding:24px; }}
    h1 {{ margin-bottom:4px; }} .subtitle {{ color:var(--muted); }}
    .filters {{ position:sticky; top:0; z-index:2; display:flex;
      gap:10px; flex-wrap:wrap; padding:14px; margin:20px 0;
      background:#fff; border:1px solid var(--line); border-radius:12px; }}
    label {{ display:grid; gap:4px; font-size:12px; color:var(--muted); }}
    select, button {{ min-height:34px; border:1px solid var(--line);
      border-radius:7px; background:white; padding:5px 8px; }}
    output {{ align-self:end; padding:7px; font-weight:700; }}
    .case-card {{ background:#fff; border:1px solid var(--line);
      border-radius:14px; padding:16px; margin:16px 0; }}
    .case-card h2 {{ margin:0; font-size:18px; }}
    .case-card header p {{ color:var(--muted); margin:5px 0 14px; }}
    .paired-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
      gap:14px; }}
    .arm-panel {{ padding:12px; border-radius:10px; background:var(--panel);
      border:1px solid var(--line); min-width:0; }}
    .arm-panel h3 {{ margin:0 0 8px; }}
    video {{ width:100%; max-height:430px; background:#111; border-radius:8px; }}
    .badges {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
    .badge {{ font-size:12px; padding:4px 7px; border-radius:999px;
      background:#e7ebf1; }}
    .car-safe,.tsr-success {{ background:#d8f4e4; color:#17643a; }}
    .car-collision,.tsr-failure {{ background:#ffe0dc; color:#8d271d; }}
    .car-unknown,.tsr-unknown {{ background:#fff1c8; color:#6f5400; }}
    .video-link,.steps {{ font-size:13px; color:var(--muted); margin:6px 0; }}
    .video-link a {{ margin-right:8px; }}
    .video-missing {{ display:grid; place-items:center; min-height:160px;
      color:var(--muted); border:1px dashed var(--line); border-radius:8px; }}
    .warning {{ color:#9a4b00; }}
    .evidence-list {{ padding:0; list-style:none; }}
    .evidence {{ display:flex; justify-content:space-between; gap:8px;
      border-bottom:1px solid var(--line); padding:4px 0; font-size:13px; }}
    .evidence-present strong {{ color:#a1251c; }}
    .evidence-unknown strong {{ color:#8b6500; }}
    .evidence-absent strong,.evidence-not_applicable strong {{ color:var(--muted); }}
    [hidden] {{ display:none !important; }}
    @media(max-width:850px) {{ .paired-grid {{ grid-template-columns:1fr; }}
      main {{ padding:12px; }} .filters {{ position:static; }} }}
  </style>
</head>
<body><main>
  <h1>SafeLIBERO paired video gallery</h1>
  <p class="subtitle">Protocol {_text(summary.get("protocol_id"))} ·
    {_text(population.get("cases"))} cases ·
    {_text(population.get("results"))} paired-arm results.
    Unknown evidence remains unknown.</p>
  <ul>{warning_html}</ul>
  {filters}
  <section id="gallery">{''.join(cards)}</section>
</main>
<script>
(() => {{
  const ids = ["suite","level","task","arm","car","tsr","status","failure"];
  const fields = Object.fromEntries(ids.map(id => [id,
    document.getElementById(id + "-filter")]));
  const cards = [...document.querySelectorAll(".case-card")];
  const output = document.getElementById("visible-count");
  function apply() {{
    let visible = 0;
    for (const card of cards) {{
      const caseMatch = (!fields.suite.value ||
          card.dataset.suite === fields.suite.value) &&
        (!fields.level.value || card.dataset.level === fields.level.value) &&
        (!fields.task.value || card.dataset.task === fields.task.value);
      const panels = [...card.querySelectorAll(".arm-panel")];
      const armMatches = panels.filter(panel =>
        !fields.arm.value || panel.dataset.arm === fields.arm.value);
      const outcomeMatch = armMatches.some(panel => {{
        const failure = fields.failure.value;
        const failureMatch = !failure ||
          (failure === "__unknown__" && panel.dataset.hasUnknown === "yes") ||
          panel.dataset.failures.split("|").includes(failure);
        return (!fields.car.value || panel.dataset.car === fields.car.value) &&
          (!fields.tsr.value || panel.dataset.tsr === fields.tsr.value) &&
          (!fields.status.value ||
            panel.dataset.status === fields.status.value) && failureMatch;
      }});
      card.hidden = !(caseMatch && outcomeMatch);
      if (!card.hidden) visible += 1;
    }}
    output.value = `${{visible}} / ${{cards.length}} cases`;
  }}
  Object.values(fields).forEach(field => field.addEventListener("change", apply));
  document.getElementById("reset-filters").addEventListener("click", () => {{
    Object.values(fields).forEach(field => {{ field.value = ""; }});
    apply();
  }});
  apply();
}})();
</script>
</body></html>
"""


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_gallery(
    *,
    summary: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    output_root: Path,
    allow_partial: bool = False,
) -> tuple[str, list[str]]:
    indexed, warnings = index_results(
        summary, records, allow_partial=allow_partial
    )
    warnings.extend(
        verify_summary_alignment(
            summary, indexed, allow_partial=allow_partial
        )
    )
    document = render_gallery(
        summary,
        indexed,
        output_root=output_root,
        warnings=warnings,
    )
    return document, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only paired SafeLIBERO video gallery"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Per-episode JSON/JSONL files or roots containing result.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Gallery destination and explicit video-root fallback for "
            "non-evaluator JSON/JSONL inputs"
        ),
    )
    parser.add_argument(
        "--index-name",
        default="index.html",
        help="HTML filename directly under --output-root",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Debug only: permit an incomplete paired result index",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index_name = Path(args.index_name)
    if (
        index_name.is_absolute()
        or len(index_name.parts) != 1
        or index_name.suffix.lower() != ".html"
    ):
        raise GalleryError("--index-name must be one HTML filename")
    summary = load_summary(args.summary.resolve())
    streaming_stats: dict[str, int] = {}
    records = load_result_records(
        args.results,
        streaming_stats=streaming_stats,
    )
    output_root = args.output_root.resolve()
    document, warnings = build_gallery(
        summary=summary,
        records=records,
        output_root=output_root,
        allow_partial=args.allow_partial,
    )
    output = output_root / index_name
    atomic_write_text(output, document)
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "streaming": streaming_stats,
                "warnings": warnings,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
