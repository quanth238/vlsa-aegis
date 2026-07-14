#!/usr/bin/env python3
"""Render the H03 minimum sphere/box pair as a dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import itertools
import json
from pathlib import Path


def _matvec(matrix: list[float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row * 3 + col] * vector[col] for col in range(3)) for row in range(3))


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def render(audit_path: Path, output_path: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or not audit.get("runs"):
        raise ValueError("Visualization requires a passed measurement audit")
    rollout = min(audit["runs"], key=lambda value: float(value["clearance_m"]))
    measurement = rollout["measurement"]
    center = measurement.get("conservative_obstacle_center_m")
    rotation = measurement.get("conservative_obstacle_rotation_world")
    half_size = measurement.get("conservative_obstacle_half_size_m")
    eef_center = measurement.get("conservative_eef_center_m")
    radius = measurement.get("conservative_eef_radius_m")
    if any(value is None for value in (center, rotation, half_size, eef_center, radius)):
        raise ValueError("Audit predates the H03 visualization metadata")
    corners = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        local = tuple(signs[index] * half_size[index] for index in range(3))
        offset = _matvec(rotation, local)
        corners.append(tuple(center[index] + offset[index] for index in range(3)))

    width, height = 920, 460
    panels = [((0, 1), 30, "Top view (world x/y)"), ((0, 2), 475, "Side view (world x/z)")]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#172033}.title{font-size:17px;font-weight:600}.label{font-size:13px}.note{font-size:12px;fill:#4b5563}</style>',
    ]
    for axes, left, title in panels:
        projected = [(corner[axes[0]], corner[axes[1]]) for corner in corners]
        hull = _convex_hull(projected)
        sphere_point = (eef_center[axes[0]], eef_center[axes[1]])
        xs = [point[0] for point in projected] + [sphere_point[0] - radius, sphere_point[0] + radius]
        ys = [point[1] for point in projected] + [sphere_point[1] - radius, sphere_point[1] + radius]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6) * 1.35
        origin_x = (max(xs) + min(xs)) / 2
        origin_y = (max(ys) + min(ys)) / 2
        scale = 330 / span

        def screen(point):
            return left + 205 + (point[0] - origin_x) * scale, 220 - (point[1] - origin_y) * scale

        polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(screen, hull))
        sphere_x, sphere_y = screen(sphere_point)
        elements.extend(
            [
                f'<text x="{left}" y="28" class="title">{html.escape(title)}</text>',
                f'<rect x="{left}" y="45" width="410" height="350" rx="8" fill="white" stroke="#d8dee9"/>',
                f'<polygon points="{polygon}" fill="#f59e0b55" stroke="#b45309" stroke-width="2"/>',
                f'<circle cx="{sphere_x:.2f}" cy="{sphere_y:.2f}" r="{radius * scale:.2f}" fill="#2563eb44" stroke="#1d4ed8" stroke-width="2"/>',
                f'<circle cx="{sphere_x:.2f}" cy="{sphere_y:.2f}" r="3" fill="#1d4ed8"/>',
                f'<text x="{left + 12}" y="420" class="label">orange: {html.escape(str(measurement["conservative_obstacle_geom"]))}</text>',
                f'<text x="{left + 12}" y="440" class="label">blue: CRFS EEF sphere (r={radius:.3f} m)</text>',
            ]
        )
    clearance = float(rollout["clearance_m"])
    elements.append(
        f'<text x="30" y="454" class="note">case={html.escape(str(audit["case_id"]))}; minimum clearance={clearance:.6f} m; substep={measurement["conservative_min_substep_index"]}</text>'
    )
    elements.append(f"<metadata>{html.escape(json.dumps({'audit': str(audit_path), 'clearance_m': clearance}))}</metadata>")
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(elements) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(Path(args.audit), Path(args.output))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
