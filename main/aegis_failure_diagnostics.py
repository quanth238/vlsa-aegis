#!/usr/bin/env python3
"""Opt-in, action-invariant diagnostics for the released AEGIS baseline.

This module is deliberately dependency-light at import time.  Heavy runtime
dependencies are imported only inside the observation helpers.  The helpers
wrap released functions without changing their inputs or return values and
publish compact evidence artifacts outside the control path.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


DIAGNOSTICS_SCHEMA = "vlsa_table1_aegis_failure_diagnostics.v1"
GEOMETRY_SCHEMA = "vlsa_table1_aegis_geometry_diagnostics.v1"
CONTACT_SCHEMA = "vlsa_table1_active_obstacle_contacts.v3"
CONTACT_MODEL_AUTHORITY_SCHEMA = "vlsa_table1_contact_model_authority.v2"
TERMINAL_FRAME_SCHEMA = "vlsa_table1_terminal_frame.v1"
ACTION_LEDGER_SCHEMA = "vlsa_table1_action_invariance_ledger.v1"
REFERENCE_SCHEMA = "vlsa_table1_canary_action_reference.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    # ``memoryview(...).cast("B")`` raises for valid zero-sized arrays whose
    # shape contains a zero (for example, GroundingDINO returning no boxes).
    # ``tobytes(order="C")`` preserves the same contiguous byte stream for
    # non-empty arrays and correctly contributes zero payload bytes for an
    # empty array; dtype and shape remain bound by the header above.
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def array_descriptor(array: Any) -> dict[str, Any]:
    import numpy as np

    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "array_sha256": array_sha256(value),
        "finite": (
            bool(np.all(np.isfinite(value)))
            if np.issubdtype(value.dtype, np.number)
            else None
        ),
    }


def _tensor_numpy(value: Any) -> Any:
    import numpy as np

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _finite_nested(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number):
        converted = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(converted)):
            raise ValueError("non-finite diagnostic numeric value")
        return converted.tolist()
    return array.tolist()


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded JSON-safe solver/runtime diagnostic value."""

    if depth > 5:
        return repr(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict(), depth=depth + 1)
    if hasattr(value, "__dict__"):
        fields = {
            key: item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
        return _json_safe(fields, depth=depth + 1)
    try:
        array = _tensor_numpy(value)
    except Exception:
        return repr(value)
    if array.ndim == 0:
        return _json_safe(array.item(), depth=depth + 1)
    if array.size <= 64:
        return _finite_nested(array)
    return {
        "array": array_descriptor(array),
        "summary": {
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "mean": float(array.mean()),
        },
    }


def new_geometry_state(
    *,
    case_id: str,
    suite_name: str,
    label: str,
) -> dict[str, Any]:
    return {
        "schema_version": GEOMETRY_SCHEMA,
        "case_id": case_id,
        "suite": suite_name,
        "obstacle_label": label,
        "status": "collecting",
        "views": {},
        "filtering": {},
        "mvee": {},
        "_arrays": {},
    }


def _add_array(state: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    state["_arrays"][key] = array
    return array_descriptor(array)


def _predict_request(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    names = (
        "model",
        "image",
        "caption",
        "box_threshold",
        "text_threshold",
        "device",
    )
    bound = {
        name: kwargs.get(name, args[index] if index < len(args) else None)
        for index, name in enumerate(names)
    }
    image = bound.pop("image")
    bound.pop("model")
    record = {
        "caption": None if bound["caption"] is None else str(bound["caption"]),
        "box_threshold": (
            None
            if bound["box_threshold"] is None
            else float(bound["box_threshold"])
        ),
        "text_threshold": (
            None
            if bound["text_threshold"] is None
            else float(bound["text_threshold"])
        ),
        "device": None if bound["device"] is None else str(bound["device"]),
    }
    if image is not None:
        record["transformed_image"] = array_descriptor(_tensor_numpy(image))
    return record


def run_released_point_cloud_with_diagnostics(
    *,
    released: Any,
    image: Any,
    depth: Any,
    env: Any,
    view: str,
    label: str,
    grounding_model: Any,
    perception_dir: Path,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Call the exact released point-cloud function and observe its outputs."""

    import numpy as np
    from groundingdino.util import box_ops
    from groundingdino.util import inference

    original_predict = inference.predict
    original_box_conversion = box_ops.box_cxcywh_to_xyxy
    original_real_depth = released.get_real_depth_map
    captured: dict[str, Any] = {
        "request": {
            "caption": label,
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "device": device,
        }
    }

    def predict_wrapper(*args: Any, **kwargs: Any) -> Any:
        captured["request"] = _predict_request(args, kwargs)
        boxes, logits, phrases = original_predict(*args, **kwargs)
        captured["boxes"] = _tensor_numpy(boxes).copy()
        captured["logits"] = _tensor_numpy(logits).copy()
        captured["phrases"] = [str(phrase) for phrase in phrases]
        return boxes, logits, phrases

    def box_conversion_wrapper(boxes: Any) -> Any:
        converted = original_box_conversion(boxes)
        captured["boxes_xyxy"] = _tensor_numpy(converted).copy()
        return converted

    def real_depth_wrapper(sim: Any, raw_depth: Any) -> Any:
        value = original_real_depth(sim, raw_depth)
        captured["metric_depth"] = np.asarray(value).copy()
        return value

    inference.predict = predict_wrapper
    box_ops.box_cxcywh_to_xyxy = box_conversion_wrapper
    released.get_real_depth_map = real_depth_wrapper
    points: Any = np.empty((0, 3), dtype=float)
    failure: Exception | None = None
    try:
        points = released.get_point_cloud(
            image,
            depth,
            env,
            view,
            label,
            grounding_model,
            perception_dir,
            device=device,
        )
    except Exception as error:
        failure = error
    finally:
        inference.predict = original_predict
        box_ops.box_cxcywh_to_xyxy = original_box_conversion
        released.get_real_depth_map = original_real_depth

    boxes = np.asarray(
        captured.get("boxes", np.empty((0, 4), dtype=np.float32))
    )
    logits = np.asarray(
        captured.get("logits", np.empty((0,), dtype=np.float32))
    )
    phrases = list(captured.get("phrases", []))
    boxes_xyxy = np.asarray(
        captured.get("boxes_xyxy", np.empty((0, 4), dtype=np.float32))
    )
    record: dict[str, Any] = {
        "view": view,
        "status": (
            "execution_failure"
            if failure is not None
            else ("no_detection" if len(boxes) == 0 else "detected")
        ),
        "request": captured["request"],
        "input_image": array_descriptor(image),
        "input_depth": array_descriptor(depth),
        "metric_depth": (
            None
            if "metric_depth" not in captured
            else array_descriptor(captured["metric_depth"])
        ),
        "detections": {
            "count": int(len(boxes)),
            "returned_order_boxes_cxcywh": _finite_nested(boxes),
            "returned_order_boxes_cxcywh_array": array_descriptor(boxes),
            "returned_order_boxes_xyxy": _finite_nested(boxes_xyxy),
            "returned_order_boxes_xyxy_array": array_descriptor(boxes_xyxy),
            "returned_order_logits": _finite_nested(logits),
            "returned_order_logits_array": array_descriptor(logits),
            "returned_order_phrases": phrases,
            "selected_index": 0 if len(boxes) else None,
            "selection_rule": "released_first_returned_box",
        },
    }
    png_path = perception_dir / f"{view}.png"
    annotated_path = perception_dir / f"annotated_ {view}_image.jpg"
    record["rendered_inputs"] = {
        "png_path": png_path.name if png_path.is_file() else None,
        "png_sha256": sha256_path(png_path) if png_path.is_file() else None,
        "annotated_path": (
            annotated_path.name if annotated_path.is_file() else None
        ),
        "annotated_sha256": (
            sha256_path(annotated_path) if annotated_path.is_file() else None
        ),
    }
    if len(boxes):
        height, width = np.asarray(image).shape[:2]
        # Preserve GroundingDINO's returned dtype here.  The released code
        # multiplies its float32 Torch boxes by a same-device size tensor
        # before truncating to integers; promoting to float64 can move a
        # boundary by one pixel and would make the diagnostic crop disagree
        # with the controller's actual crop.
        selected_xyxy = np.asarray(boxes_xyxy[0])
        coordinate_dtype = (
            selected_xyxy.dtype
            if np.issubdtype(selected_xyxy.dtype, np.floating)
            else np.dtype(np.float32)
        )
        pixel_xyxy = (
            selected_xyxy
            * np.asarray(
                [width, height, width, height],
                dtype=coordinate_dtype,
            )
        ).astype(int)
        x1, y1, x2, y2 = [int(value) for value in pixel_xyxy]
        crop = [
            max(0, width - 1 - x2),
            max(0, height - 1 - y2),
            min(width, width - 1 - x1),
            min(height, height - 1 - y1),
        ]
        xmin, ymin, xmax, ymax = crop
        released_image = np.asarray(image)[::-1, ::-1]
        released_depth = np.asarray(
            captured.get("metric_depth", depth)
        )[::-1, ::-1].squeeze()
        cropped_rgb = released_image[ymin:ymax, xmin:xmax]
        cropped_depth = released_depth[ymin:ymax, xmin:xmax]
        record["detections"]["selected_crop"] = {
            "pixel_xyxy_before_released_axis_flip": [
                x1,
                y1,
                x2,
                y2,
            ],
            "released_rgb_depth_crop_xyxy": crop,
            "image_height": int(height),
            "image_width": int(width),
            "nonempty": bool(crop[2] > crop[0] and crop[3] > crop[1]),
            "rgb_array": array_descriptor(cropped_rgb),
            "metric_depth_array": array_descriptor(cropped_depth),
        }
    else:
        record["detections"]["selected_crop"] = None
        record["no_detection"] = {
            "explicit": True,
            "released_return": "empty_point_cloud",
        }
    valid_points = np.asarray(points)
    record["returned_point_cloud"] = array_descriptor(valid_points)
    if failure is not None:
        record["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
        }
        try:
            setattr(failure, "aegis_view_diagnostics", record)
        except Exception:
            pass
        raise failure
    return points, record


def record_view_points(
    state: dict[str, Any],
    *,
    view: str,
    view_record: Mapping[str, Any],
    points: Any,
) -> None:
    state["views"][view] = {
        **dict(view_record),
        "raw_points_array": _add_array(
            state, f"{view}_raw_points", points
        ),
    }


def _suite_filter_bounds(suite_name: str) -> dict[str, list[float]]:
    if "spatial" in suite_name or "goal" in suite_name:
        return {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [0.92, 1.5]}
    if "object" in suite_name:
        return {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [0.05, 0.5]}
    if "long" in suite_name:
        return {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [0.43, 0.8]}
    raise ValueError(f"unsupported released filtering suite: {suite_name}")


def record_filtering(
    state: dict[str, Any],
    *,
    fused_points: Any,
    released_filtered_points: Any,
    suite_name: str,
) -> None:
    """Reconstruct filtering stages read-only and bind the released output."""

    import numpy as np
    import open3d as o3d

    fused = np.asarray(fused_points, dtype=float)
    released_filtered = np.asarray(released_filtered_points, dtype=float)
    bounds = _suite_filter_bounds(suite_name)
    mask = (
        (fused[:, 0] > bounds["x"][0])
        & (fused[:, 0] < bounds["x"][1])
        & (fused[:, 1] > bounds["y"][0])
        & (fused[:, 1] < bounds["y"][1])
        & (fused[:, 2] > bounds["z"][0])
        & (fused[:, 2] < bounds["z"][1])
    )
    range_filtered = fused[mask]
    centroid = (
        None
        if not len(range_filtered)
        else np.asarray(range_filtered.mean(axis=0), dtype=float)
    )
    if len(range_filtered):
        distances = np.linalg.norm(range_filtered - centroid, axis=1)
        keep_count = int(len(range_filtered) * 0.8)
        sorted_indices = np.argsort(distances)
        nearest = range_filtered[sorted_indices[:keep_count]]
    else:
        distances = np.empty((0,), dtype=float)
        sorted_indices = np.empty((0,), dtype=np.int64)
        keep_count = 0
        nearest = range_filtered

    if len(nearest):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nearest))
        labels = np.asarray(
            cloud.cluster_dbscan(
                eps=0.0001,
                min_points=50,
                print_progress=False,
            ),
            dtype=np.int64,
        )
        largest = (
            int(np.bincount(labels[labels >= 0]).argmax())
            if labels.max() >= 0
            else None
        )
        shadow_filtered = (
            np.asarray(cloud.points)[labels == largest]
            if largest is not None
            else nearest
        )
    else:
        labels = np.empty((0,), dtype=np.int64)
        largest = None
        shadow_filtered = nearest

    state["filtering"] = {
        "status": "captured",
        "suite_argument": suite_name,
        "range_filter": {
            "strict_inequalities": True,
            "bounds": bounds,
            "input_count": int(len(fused)),
            "kept_count": int(len(range_filtered)),
        },
        "centroid_trim": {
            "centroid": (
                None if centroid is None else _finite_nested(centroid)
            ),
            "fraction_kept": 0.8,
            "keep_count_rule": "int(n * 0.8)",
            "keep_count": int(keep_count),
            "distance_min": (
                None if not len(distances) else float(distances.min())
            ),
            "distance_max": (
                None if not len(distances) else float(distances.max())
            ),
        },
        "dbscan": {
            "eps": 0.0001,
            "min_points": 50,
            "print_progress": False,
            "label_counts": {
                str(int(label)): int((labels == label).sum())
                for label in sorted(set(labels.tolist()))
            },
            "largest_nonnoise_label": largest,
            "released_rule": (
                "largest_nonnegative_cluster_else_keep_all_nearest80"
            ),
        },
        "counts": {
            "fused": int(len(fused)),
            "range_filtered": int(len(range_filtered)),
            "nearest80": int(len(nearest)),
            "released_filtered": int(len(released_filtered)),
            "shadow_filtered": int(len(shadow_filtered)),
        },
        "shadow_matches_released": (
            array_sha256(shadow_filtered)
            == array_sha256(released_filtered)
        ),
        "arrays": {
            "fused_points": _add_array(state, "fused_points", fused),
            "range_mask": _add_array(state, "range_mask", mask),
            "range_filtered_points": _add_array(
                state, "range_filtered_points", range_filtered
            ),
            "centroid_distances": _add_array(
                state, "centroid_distances", distances
            ),
            "centroid_sorted_indices": _add_array(
                state, "centroid_sorted_indices", sorted_indices
            ),
            "nearest80_points": _add_array(
                state, "nearest80_points", nearest
            ),
            "dbscan_labels": _add_array(
                state, "dbscan_labels", labels
            ),
            "shadow_filtered_points": _add_array(
                state, "shadow_filtered_points", shadow_filtered
            ),
            "released_filtered_points": _add_array(
                state, "released_filtered_points", released_filtered
            ),
        },
    }


def _solver_stats(problem: Any) -> dict[str, Any]:
    stats = getattr(problem, "solver_stats", None)
    if stats is None:
        return {"available": False}
    return {
        "available": True,
        "solver_name": _json_safe(getattr(stats, "solver_name", None)),
        "solve_time": _json_safe(getattr(stats, "solve_time", None)),
        "setup_time": _json_safe(getattr(stats, "setup_time", None)),
        "num_iters": _json_safe(getattr(stats, "num_iters", None)),
        "extra_stats": _json_safe(getattr(stats, "extra_stats", None)),
    }


def run_released_fit_ellipse_with_diagnostics(
    *,
    released: Any,
    points: Any,
    plot: bool,
    save_path: Path,
    state: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Run released ConvexHull/MVEE and observe exact intermediate results."""

    import numpy as np
    import scipy.spatial

    original_hull = scipy.spatial.ConvexHull
    original_mvee = released.mvee_cvxpy
    captured: dict[str, Any] = {"solver_calls": []}

    def hull_wrapper(values: Any, *args: Any, **kwargs: Any) -> Any:
        hull = original_hull(values, *args, **kwargs)
        try:
            captured["hull_vertices"] = np.asarray(
                hull.vertices
            ).copy()
            captured["hull_simplices"] = np.asarray(
                hull.simplices
            ).copy()
            captured["hull_equations"] = np.asarray(
                hull.equations
            ).copy()
        except Exception as diagnostic_error:
            captured["hull_observer_failure"] = {
                "type": type(diagnostic_error).__name__,
                "message": str(diagnostic_error),
            }
        return hull

    def mvee_wrapper(values: Any) -> Any:
        problem_class = released.cp.Problem
        original_solve = problem_class.solve

        def solve_observer(
            problem: Any, *args: Any, **kwargs: Any
        ) -> Any:
            call: dict[str, Any] = {
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
            }
            try:
                output = original_solve(problem, *args, **kwargs)
                call["return"] = _json_safe(output)
                return output
            except Exception as error:
                call["failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                raise
            finally:
                try:
                    call["status"] = str(
                        getattr(problem, "status", None)
                    )
                    call["objective"] = _json_safe(
                        getattr(problem, "value", None)
                    )
                    call["solver_stats"] = _solver_stats(problem)
                    captured["solver_calls"].append(call)
                except Exception as diagnostic_error:
                    captured["solver_observer_failure"] = {
                        "type": type(diagnostic_error).__name__,
                        "message": str(diagnostic_error),
                    }

        problem_class.solve = solve_observer
        try:
            center, matrix = original_mvee(values)
        finally:
            problem_class.solve = original_solve
        try:
            captured["mvee_input"] = np.asarray(values).copy()
            captured["center"] = np.asarray(center).copy()
            captured["matrix_A"] = np.asarray(matrix).copy()
        except Exception as diagnostic_error:
            captured["mvee_observer_failure"] = {
                "type": type(diagnostic_error).__name__,
                "message": str(diagnostic_error),
            }
        return center, matrix

    scipy.spatial.ConvexHull = hull_wrapper
    released.mvee_cvxpy = mvee_wrapper
    failure: Exception | None = None
    output: tuple[Any, Any, Any] | None = None
    try:
        output = released.fit_ellipse(
            points,
            plot=plot,
            save_path=save_path,
        )
    except Exception as error:
        failure = error
    finally:
        scipy.spatial.ConvexHull = original_hull
        released.mvee_cvxpy = original_mvee

    mvee_record: dict[str, Any] = {
        "status": "failure" if failure is not None else "captured",
        "solver_calls": captured["solver_calls"],
        "input_count": int(len(points)),
    }
    arrays = mvee_record.setdefault("arrays", {})
    for key in (
        "hull_vertices",
        "hull_simplices",
        "hull_equations",
        "mvee_input",
        "center",
        "matrix_A",
    ):
        if key in captured:
            arrays[key] = _add_array(state, key, captured[key])
    if output is not None:
        try:
            center, rotation, semiaxes = [
                np.asarray(value, dtype=float) for value in output
            ]
            matrix_A = np.asarray(captured["matrix_A"], dtype=float)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix_A)
            hull_points = np.asarray(captured["mvee_input"], dtype=float)
            offsets = hull_points - center
            quadratic = np.einsum(
                "ni,ij,nj->n", offsets, matrix_A, offsets
            )
            arrays.update(
                {
                    "released_center": _add_array(
                        state, "released_center", center
                    ),
                    "released_rotation": _add_array(
                        state, "released_rotation", rotation
                    ),
                    "released_semiaxes": _add_array(
                        state, "released_semiaxes", semiaxes
                    ),
                    "matrix_A_eigenvalues": _add_array(
                        state, "matrix_A_eigenvalues", eigenvalues
                    ),
                    "matrix_A_eigenvectors": _add_array(
                        state, "matrix_A_eigenvectors", eigenvectors
                    ),
                    "hull_quadratic_values": _add_array(
                        state, "hull_quadratic_values", quadratic
                    ),
                }
            )
            mvee_record.update(
                {
                    "center": _finite_nested(center),
                    "matrix_A": _finite_nested(matrix_A),
                    "rotation": _finite_nested(rotation),
                    "semiaxes": _finite_nested(semiaxes),
                    "eigenvalues": _finite_nested(eigenvalues),
                    "maximum_hull_quadratic_value": (
                        None
                        if not len(quadratic)
                        else float(quadratic.max())
                    ),
                    "maximum_enclosure_violation": (
                        None
                        if not len(quadratic)
                        else float(max(0.0, quadratic.max() - 1.0))
                    ),
                }
            )
        except Exception as diagnostic_error:
            mvee_record["status"] = "diagnostic_failure"
            mvee_record["observer_failure"] = {
                "type": type(diagnostic_error).__name__,
                "message": str(diagnostic_error),
            }
    if failure is not None:
        mvee_record["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
        }
    state["mvee"] = mvee_record
    if failure is not None:
        raise failure
    assert output is not None
    return output


def record_initial_geometry_direction(
    state: dict[str, Any],
    *,
    stale_proxy_center: Any,
    stale_proxy_rotation: Any,
    obstacle_center: Any,
    direction: Any,
) -> None:
    state["initial_direction"] = {
        "formula": "normalize(mvee_center - stale_pre_settle_eef_proxy_center)",
        "stale_proxy_center": _finite_nested(stale_proxy_center),
        "stale_proxy_rotation": _finite_nested(stale_proxy_rotation),
        "obstacle_center": _finite_nested(obstacle_center),
        "z_initial": _finite_nested(direction),
        "arrays": {
            "stale_proxy_center": _add_array(
                state, "stale_proxy_center", stale_proxy_center
            ),
            "stale_proxy_rotation": _add_array(
                state, "stale_proxy_rotation", stale_proxy_rotation
            ),
            "z_initial": _add_array(state, "z_initial", direction),
        },
    }


def record_geometry_failure(
    state: dict[str, Any],
    *,
    component: str,
    error: Exception,
) -> None:
    state["status"] = "failure"
    state["failure"] = {
        "component": component,
        "type": type(error).__name__,
        "message": str(error),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish_geometry_artifact(
    state: dict[str, Any],
    *,
    case_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    import numpy as np

    arrays = dict(state.get("_arrays", {}))
    path = case_dir / "aegis_geometry_diagnostics.npz"
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.partial.npz"
    )
    np.savez_compressed(
        str(temporary),
        **{key: arrays[key] for key in sorted(arrays)},
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    metadata = {
        key: array_descriptor(arrays[key]) for key in sorted(arrays)
    }
    public_record = {
        key: value for key, value in state.items() if key != "_arrays"
    }
    if public_record.get("status") == "collecting":
        raise ValueError(
            "geometry diagnostics cannot be published while still collecting"
        )
    return {
        "schema_version": GEOMETRY_SCHEMA,
        "path": str(path.relative_to(output_root)),
        "sha256": sha256_path(path),
        "format": "numpy_savez_compressed",
        "keys": sorted(arrays),
        "arrays": metadata,
        "record": public_record,
    }


def publish_terminal_frame_artifact(
    *,
    frame: Any,
    case_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    import numpy as np

    value = np.ascontiguousarray(frame)
    path = case_dir / "terminal_agentview.npy"
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.partial.npy"
    )
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "schema_version": TERMINAL_FRAME_SCHEMA,
        "path": str(path.relative_to(output_root)),
        "sha256": sha256_path(path),
        "format": "numpy_npy",
        "array": array_descriptor(value),
    }


def publish_contact_artifact(
    *,
    case_id: str,
    active_obstacle_name: str,
    model_authority: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    case_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    if not isinstance(active_obstacle_name, str) or not active_obstacle_name:
        raise ValueError("contact artifact requires an active obstacle name")
    authority = dict(model_authority)
    expected_authority_hash = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in authority.items()
                if key != "authority_sha256"
            }
        )
    )
    if (
        authority.get("schema_version") != CONTACT_MODEL_AUTHORITY_SCHEMA
        or authority.get("active_obstacle_name") != active_obstacle_name
        or authority.get("authority_sha256") != expected_authority_hash
    ):
        raise ValueError("contact model authority binding changed")
    for snapshot in snapshots:
        role_authority = snapshot.get("role_authority", {})
        raw_contact_ledger = snapshot.get("raw_contact_ledger")
        if (
            snapshot.get("active_obstacle_name") != active_obstacle_name
            or role_authority.get("model_authority_sha256")
            != expected_authority_hash
            or role_authority.get("task_context_sha256")
            != authority.get("task_context_sha256")
            or role_authority.get("active_obstacle_root_body_id")
            != authority.get("active_obstacle_root_body_id")
            or role_authority.get("task_context")
            != authority.get("task_context")
            or not isinstance(raw_contact_ledger, list)
            or snapshot.get("raw_contact_ledger_sha256")
            != sha256_bytes(canonical_json_bytes(raw_contact_ledger))
        ):
            raise ValueError(
                "contact snapshot differs from model/task authority"
            )
    payload = {
        "schema_version": CONTACT_SCHEMA,
        "case_id": case_id,
        "active_obstacle_name": active_obstacle_name,
        "model_authority": authority,
        "snapshots": list(snapshots),
    }
    raw = canonical_json_bytes(payload)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    path = case_dir / "active_obstacle_contacts.json.gz"
    _atomic_write_bytes(path, compressed)
    events = [
        event
        for snapshot in snapshots
        for event in snapshot.get("events", [])
    ]
    role_classes = (
        "robot",
        "static_support",
        "dynamic_task_object",
        "dynamic_other",
        "unknown",
    )
    events_by_role = {
        role: [
            event
            for event in events
            if event.get("other", {}).get("classification") == role
        ]
        for role in role_classes
    }
    unknown_roles = events_by_role["unknown"]
    role_authority_complete = (
        not unknown_roles
        and all(
            snapshot.get("role_authority", {}).get("status") == "complete"
            for snapshot in snapshots
        )
    )
    robot_events = events_by_role["robot"]
    nonrobot_events = [
        event
        for role in role_classes
        if role != "robot"
        for event in events_by_role[role]
    ]
    return {
        "schema_version": CONTACT_SCHEMA,
        "active_obstacle_name": active_obstacle_name,
        "model_authority_sha256": expected_authority_hash,
        "active_obstacle_root_body_id": authority.get(
            "active_obstacle_root_body_id"
        ),
        "task_context_sha256": authority.get("task_context_sha256"),
        "path": str(path.relative_to(output_root)),
        "sha256": sha256_path(path),
        "uncompressed_payload_sha256": sha256_bytes(raw),
        "format": "canonical_json_gzip_mtime_zero",
        "snapshot_count": len(snapshots),
        "event_count": len(events),
        "role_taxonomy": list(role_classes),
        "role_authority_complete": role_authority_complete,
        "event_counts_by_role": {
            role: len(events_by_role[role]) for role in role_classes
        },
        "steps_with_contact_by_role": {
            role: sorted(
                {int(event["step"]) for event in events_by_role[role]}
            )
            for role in role_classes
        },
        "first_contact_step_by_role": {
            role: (
                None
                if not events_by_role[role]
                else min(
                    int(event["step"]) for event in events_by_role[role]
                )
            )
            for role in role_classes
        },
        # Retained for receipt compatibility. Scientific analysis must use the
        # disjoint role counts above, never generic ``nonrobot`` as a failure
        # mechanism.
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


def action_invariance_ledger(
    *,
    actions: Sequence[Mapping[str, Any]],
    policy_queries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "nominal_raw": [action.get("nominal_raw") for action in actions],
        "nominal_translational": [
            action.get("nominal_translational") for action in actions
        ],
        "executed": [action.get("executed") for action in actions],
        "env_step_input": [
            action.get("env_step_input", action.get("executed"))
            for action in actions
        ],
        "control_path": [action.get("control_path") for action in actions],
        "z_before": [
            (
                None
                if not isinstance(action.get("qp"), Mapping)
                else action["qp"].get("z_before")
            )
            for action in actions
        ],
        "z_after": [
            (
                None
                if not isinstance(action.get("qp"), Mapping)
                else action["qp"].get("z_after")
            )
            for action in actions
        ],
    }
    policy_schedule = [
        {
            "query_index": query.get("query_index"),
            "rng_seed": query.get("rng_seed"),
            "returned_action_shape": query.get("returned_action_shape"),
            "returned_actions_sha256": query.get(
                "returned_actions_sha256"
            ),
        }
        for query in policy_queries
    ]
    hashes = {
        f"{name}_sequence_sha256": sha256_bytes(canonical_json_bytes(values))
        for name, values in fields.items()
    }
    hashes.update(
        {
            "policy_query_returned_action_hash_sequence_sha256": (
                sha256_bytes(
                    canonical_json_bytes(
                        [
                            row["returned_actions_sha256"]
                            for row in policy_schedule
                        ]
                    )
                )
            ),
            "policy_query_rng_seed_sequence_sha256": sha256_bytes(
                canonical_json_bytes(
                    [row["rng_seed"] for row in policy_schedule]
                )
            ),
            "policy_query_schedule_sha256": sha256_bytes(
                canonical_json_bytes(policy_schedule)
            ),
        }
    )
    combined = {
        **fields,
        "policy_queries": policy_schedule,
    }
    return {
        "schema_version": ACTION_LEDGER_SCHEMA,
        "action_count": len(actions),
        "policy_query_count": len(policy_queries),
        **hashes,
        "combined_ledger_sha256": sha256_bytes(
            canonical_json_bytes(combined)
        ),
    }


def compare_action_ledgers(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    include_diagnostic_only_fields: bool = True,
) -> list[str]:
    keys = {
        "action_count",
        "policy_query_count",
        "nominal_raw_sequence_sha256",
        "nominal_translational_sequence_sha256",
        "executed_sequence_sha256",
        "env_step_input_sequence_sha256",
        "control_path_sequence_sha256",
        "z_after_sequence_sha256",
        "policy_query_returned_action_hash_sequence_sha256",
        "policy_query_rng_seed_sequence_sha256",
        "policy_query_schedule_sha256",
    }
    if include_diagnostic_only_fields:
        keys.update(
            {
                "z_before_sequence_sha256",
                "combined_ledger_sha256",
            }
        )
    return sorted(
        key for key in keys if first.get(key) != second.get(key)
    )


def validate_artifact_descriptor(
    descriptor: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    path_value = descriptor.get("path")
    if not isinstance(path_value, str):
        raise ValueError("diagnostic artifact path is missing")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("diagnostic artifact path is unsafe")
    path = output_root / relative
    if not path.is_file():
        raise ValueError(f"diagnostic artifact is missing: {relative}")
    if descriptor.get("sha256") != sha256_path(path):
        raise ValueError(f"diagnostic artifact hash mismatch: {relative}")
