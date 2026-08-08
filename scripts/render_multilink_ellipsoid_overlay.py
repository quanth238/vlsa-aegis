#!/usr/bin/env python3
"""Render live SafeLIBERO arm ellipsoid bounds into the agent-view camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
COLORS = (
    (244, 67, 54, 245),
    (255, 152, 0, 245),
    (205, 220, 57, 245),
)
EE_COLOR = (0, 188, 212, 245)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    status = run("status", "--short")
    if status:
        raise ValueError("visualization source tree must be clean")
    return {"commit": run("rev-parse", "HEAD"), "dirty": False}


def _wire_loops(ellipsoid: Any, samples: int = 144) -> Iterable[Any]:
    import numpy as np

    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=True)
    for latitude in (-60.0, -30.0, 0.0, 30.0, 60.0):
        phi = math.radians(latitude)
        local = np.stack(
            (
                ellipsoid.semiaxes_m[0] * math.cos(phi) * np.cos(angles),
                ellipsoid.semiaxes_m[1] * math.cos(phi) * np.sin(angles),
                np.full_like(angles, ellipsoid.semiaxes_m[2] * math.sin(phi)),
            ),
            axis=1,
        )
        yield ellipsoid.center + local @ ellipsoid.rotation.T
    for longitude in range(0, 180, 30):
        theta = math.radians(longitude)
        local = np.stack(
            (
                ellipsoid.semiaxes_m[0] * np.cos(angles) * math.cos(theta),
                ellipsoid.semiaxes_m[1] * np.cos(angles) * math.sin(theta),
                ellipsoid.semiaxes_m[2] * np.sin(angles),
            ),
            axis=1,
        )
        yield ellipsoid.center + local @ ellipsoid.rotation.T


def _project(points_world: Any, world_to_camera: Any, intrinsic: Any, height: int) -> list[tuple[float, float] | None]:
    import numpy as np

    points = np.asarray(points_world, dtype=np.float64)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    pixels = (intrinsic @ camera.T).T
    output: list[tuple[float, float] | None] = []
    for point_camera, pixel in zip(camera, pixels):
        if point_camera[2] <= 1.0e-8 or not np.all(np.isfinite(pixel)):
            output.append(None)
            continue
        u = float(pixel[0] / pixel[2])
        v_bottom = float(pixel[1] / pixel[2])
        output.append((u, float(height - 1) - v_bottom))
    return output


def _segments(points: Sequence[tuple[float, float] | None]) -> Iterable[list[tuple[float, float]]]:
    current: list[tuple[float, float]] = []
    for point in points:
        if point is None:
            if len(current) > 1:
                yield current
            current = []
        else:
            current.append(point)
    if len(current) > 1:
        yield current


def _rotate_projection_180(
    points: Sequence[tuple[float, float] | None], width: int, height: int
) -> list[tuple[float, float] | None]:
    return [
        None
        if point is None
        else (float(width - 1) - point[0], float(height - 1) - point[1])
        for point in points
    ]


def _frame_arm_camera(env: Any, ellipsoids: Sequence[Any], camera_name: str) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    centers = np.stack([item.center for item in ellipsoids], axis=0)
    target = np.mean(centers, axis=0)
    span = max(
        float(np.linalg.norm(item.center - target) + np.max(item.semiaxes_m))
        for item in ellipsoids
    )
    camera_id = int(env.sim.model.camera_name2id(camera_name))
    fovy = float(env.sim.model.cam_fovy[camera_id])
    distance = 1.45 * span / math.tan(math.radians(fovy) * 0.5)
    view_direction = np.asarray([1.0, -1.0, 0.55], dtype=np.float64)
    view_direction /= np.linalg.norm(view_direction)
    position = target + distance * view_direction
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.column_stack((right, camera_up, -forward))
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    env.sim.model.cam_pos[camera_id] = position
    env.sim.model.cam_quat[camera_id] = quaternion_wxyz
    env.sim.forward()
    env.env._update_observables(force=True)
    return {
        "source_camera_name": camera_name,
        "semantics": "simulation_camera_repositioned_to_frame_all_configured_bounds",
        "position_world_m": position.tolist(),
        "target_world_m": target.tolist(),
        "quaternion_wxyz": quaternion_wxyz.tolist(),
        "fovy_degrees": fovy,
        "bounding_span_m": span,
    }


def render(
    repo_root: Path,
    manifest: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS,
        _build_environment,
        _runtime_imports,
        _settle,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import (
        DISTAL_ELLIPSOID_SCHEMA,
        DISTAL_PARTITIONED_SCHEMA,
        _link_ellipsoids,
        _mesh_link_ellipsoids,
        _mesh_link_partition_ellipsoids,
        _released_aegis_end_effector_ellipsoid,
        allocation_record,
        load_shadow_config,
    )

    rows = read_jsonl(manifest)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    if len(matches) != 1:
        raise ValueError("primary visualization case is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, _, observation, _ = _build_environment(
            runtime,
            case,
            render_resolution=768,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        config = load_shadow_config(config_path)
        protected = config["protected_body_names"]
        if config["schema_version"] == DISTAL_PARTITIONED_SCHEMA:
            geometry = config["robot_geometry"]
            ellipsoids = _mesh_link_partition_ellipsoids(
                env,
                protected,
                part_counts=geometry["part_counts"],
                relative_padding=float(geometry["relative_numerical_padding"]),
                tolerance=float(geometry["khachiyan_tolerance"]),
                max_iterations=int(geometry["khachiyan_max_iterations"]),
            )
            ellipsoids.append(_released_aegis_end_effector_ellipsoid(env))
        elif config["schema_version"] == DISTAL_ELLIPSOID_SCHEMA:
            ellipsoids = _mesh_link_ellipsoids(
                env,
                protected,
                relative_padding=float(
                    config["robot_geometry"]["relative_numerical_padding"]
                ),
                tolerance=float(config["robot_geometry"]["khachiyan_tolerance"]),
                max_iterations=int(
                    config["robot_geometry"]["khachiyan_max_iterations"]
                ),
                include_source_points=True,
            )
        else:
            ellipsoids = _link_ellipsoids(env, protected)
        expected_count = 8 if config["schema_version"] == DISTAL_PARTITIONED_SCHEMA else 3
        if len(ellipsoids) != expected_count:
            raise ValueError("live ellipsoid count differs from geometry contract")

        camera_name = "backview"
        camera_record = _frame_arm_camera(env, ellipsoids, camera_name)
        observation = env.env._get_observations()
        image = np.ascontiguousarray(
            np.asarray(observation["backview_image"], dtype=np.uint8)[
                ::-1, ::-1
            ]
        )
        if image.shape != (768, 768, 3):
            raise ValueError("arm-view simulator image shape differs")

        intrinsic = get_camera_intrinsic_matrix(
            env.sim, camera_name, image.shape[0], image.shape[1]
        )
        camera_to_world = get_camera_extrinsic_matrix(env.sim, camera_name)
        world_to_camera = np.linalg.inv(camera_to_world)
        output_dir.mkdir(parents=True, exist_ok=False)
        base_path = output_dir / "libero-armview.jpg"
        Image.fromarray(image).save(base_path, quality=88, optimize=True)

        link_records: list[dict[str, Any]] = []
        combined = Image.new("RGBA", (image.shape[1], image.shape[0]), (0, 0, 0, 0))
        font = ImageFont.load_default()
        for ordinal, ellipsoid in enumerate(ellipsoids):
            is_end_effector = ellipsoid.body_name == "robot0_end_effector"
            if is_end_effector:
                index = None
                partition_index = None
                color = EE_COLOR
                label = "EE"
                file_stem = "end-effector"
            else:
                index = int(ellipsoid.body_name.rsplit("link", 1)[1])
                certificate = dict(ellipsoid.enclosure_certificate or {})
                partition_index = certificate.get("partition_index")
                color = COLORS[index - 5] if 5 <= index <= 7 else COLORS[ordinal % len(COLORS)]
                label = "L%d" % index
                if partition_index is not None:
                    label += ".%d" % (int(partition_index) + 1)
                file_stem = "link-%d" % index
                if partition_index is not None:
                    file_stem += "-part-%d" % (int(partition_index) + 1)
            layer = Image.new("RGBA", combined.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            visible_points = 0
            for loop in _wire_loops(ellipsoid):
                projected = _project(loop, world_to_camera, intrinsic, image.shape[0])
                projected = _rotate_projection_180(
                    projected, image.shape[1], image.shape[0]
                )
                visible_points += sum(point is not None for point in projected)
                for segment in _segments(projected):
                    draw.line(segment, fill=color, width=2, joint="curve")
            center = _project(
                np.asarray([ellipsoid.center]),
                world_to_camera,
                intrinsic,
                image.shape[0],
            )
            center = _rotate_projection_180(
                center, image.shape[1], image.shape[0]
            )[0]
            if center is None:
                raise ValueError("link ellipsoid center is behind the camera")
            x, y = center
            if not (0.0 <= x < image.shape[1] and 0.0 <= y < image.shape[0]):
                raise ValueError("%s center is outside the framed camera" % ellipsoid.body_name)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.text(
                (x + 7, y - 7),
                label,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 230),
            )
            if visible_points < 100:
                raise ValueError("too few projected points for %s" % ellipsoid.body_name)
            layer_path = output_dir / (file_stem + ".png")
            layer.save(layer_path, optimize=True)
            combined = Image.alpha_composite(combined, layer)
            record = ellipsoid.to_record()
            record.update(
                {
                    "link_index": index,
                    "partition_index": partition_index,
                    "color_rgba": list(color),
                    "projected_center_px": [float(x), float(y)],
                    "visible_wire_points": visible_points,
                    "overlay_path": layer_path.name,
                    "overlay_sha256": _sha256_path(layer_path),
                }
            )
            link_records.append(record)

        combined_path = output_dir / "ellipsoid-overlay.png"
        combined.save(combined_path, optimize=True)
        preview = Image.alpha_composite(
            Image.fromarray(image).convert("RGBA"), combined
        ).convert("RGB")
        preview_path = output_dir / "libero-arm-ellipsoids.jpg"
        preview.save(preview_path, quality=90, optimize=True)
        record = {
            "schema_version": (
                "vlsa_distal_partitioned_ellipsoid_visualization.v1"
                if config["schema_version"] == DISTAL_PARTITIONED_SCHEMA
                else "vlsa_multilink_ellipsoid_visualization.v1"
            ),
            "case_id": CASE_ID,
            "config": {
                "path": str(config_path),
                "schema_version": config["schema_version"],
                "payload_sha256": config["config_payload_sha256"],
            },
            "source": _git_identity(repo_root),
            "allocation": allocation_record(),
            "simulator": "SafeLIBERO MuJoCo settled primary initial state",
            "settle_actions": TABLE_SETTLE_ACTIONS,
            "camera": camera_record,
            "display_transform": "rotate_180_matching_released_aegis_camera_preprocessing",
            "image_size": [image.shape[1], image.shape[0]],
            "base_image": {
                "path": base_path.name,
                "sha256": _sha256_path(base_path),
            },
            "combined_preview": {
                "path": preview_path.name,
                "sha256": _sha256_path(preview_path),
            },
        }
        if config["schema_version"] == DISTAL_PARTITIONED_SCHEMA:
            record.update(
                {
                    "ellipsoids": link_records,
                    "distal_partition_count": len(link_records) - 1,
                    "released_aegis_end_effector_proxy_unchanged": True,
                    "constraint_geometry_count": len(link_records),
                }
            )
        else:
            record["links"] = link_records
        record["payload_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
        metadata_path = output_dir / "visualization.json"
        temporary = output_dir / (".visualization.%d.tmp" % os.getpid())
        temporary.write_bytes(json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
        os.replace(str(temporary), str(metadata_path))
        return record
    finally:
        if env is not None:
            env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "manifests/vlsa_table1_population.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/vlsa_distal_three_ellipsoid_shadow_e05.v2.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = render(
        args.repo_root.resolve(),
        args.manifest.resolve(),
        args.config.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "rendered",
                "payload_sha256": record["payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # The legacy OSMesa stack can double-free its already-closed context during
    # interpreter teardown. All simulator and artifact cleanup has completed;
    # bypass only that process-global C-extension destructor path.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
