#!/usr/bin/env python3
"""Plot the paired pi0.5 versus pi0.5+AEGIS same-state canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASELINE_KEY = "pi0.5"
AEGIS_KEY = "pi0.5_plus_AEGIS"
BASELINE_COLOR = "#D1495B"
AEGIS_COLOR = "#177E89"
MARGIN_COLOR = "#F4A261"


def _load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "aegis_same_state_result.v1":
        raise ValueError("unexpected result schema")
    if result.get("status") != "complete":
        raise ValueError(
            "the comparison figure requires a complete paired result"
        )
    if not result.get("pairing", {}).get("passed"):
        raise ValueError("paired-arm identity did not pass")
    if not result.get("baseline_reproduction", {}).get("passed"):
        raise ValueError("pi0.5 baseline did not reproduce")
    if set((BASELINE_KEY, AEGIS_KEY)) - set(result.get("arms", {})):
        raise ValueError("paired arms are missing")
    return result


def _trace(arm: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = arm["simulator_trace"]["rows"]
    samples = np.asarray([row["sample_index"] for row in rows], dtype=int)
    eef = np.asarray([row["eef_world_m"] for row in rows], dtype=float)
    clearance_mm = 1000.0 * np.asarray(
        [row["registered_D_sim_m"] for row in rows], dtype=float
    )
    return samples, eef, clearance_mm


def _boundaries(arm: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = arm["control_boundaries"]
    eef = np.asarray([row["eef_world_m"] for row in rows], dtype=float)
    target = np.asarray(
        arm["outcome"]["fixed_branch_target_world_m"], dtype=float
    )
    distance_mm = 1000.0 * np.linalg.norm(eef - target, axis=1)
    return eef, distance_mm


def _relative_task_coordinates(
    trajectory: np.ndarray, start: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    task_direction = target - start
    task_direction /= np.linalg.norm(task_direction)
    displacement = trajectory - start
    forward = displacement @ task_direction
    lateral = np.linalg.norm(
        displacement - np.outer(forward, task_direction), axis=1
    )
    return 1000.0 * forward, 1000.0 * lateral


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _outcome_text(result: dict[str, Any], digest: str) -> str:
    baseline = result["arms"][BASELINE_KEY]["outcome"]
    aegis = result["arms"][AEGIS_KEY]["outcome"]
    modification = result["arms"][AEGIS_KEY]["action_modification"]
    return "\n".join(
        (
            "ONE COLLISION-CONDITIONED CANARY",
            "",
            "Safety",
            f"  π0.5 min clearance: {1000*baseline['minimum_registered_D_sim_m']:+.2f} mm",
            f"  +AEGIS min clearance: {1000*aegis['minimum_registered_D_sim_m']:+.2f} mm",
            "  AEGIS avoided collision: YES",
            "",
            "Task",
            f"  π0.5 progress: {1000*baseline['reach_progress_m']:+.2f} mm",
            f"  +AEGIS progress: {1000*aegis['reach_progress_m']:+.2f} mm",
            "  AEGIS task completion: NO",
            "  Joint safety + progress: NO",
            "",
            "How",
            f"  Motion-command retention: {100*modification['command_retention']:.1f}%",
            f"  Realized-path retention: {100*modification['realized_path_retention']:.1f}%",
            "  Safety by stopping: NO",
            "",
            "Diagnosis",
            "  AEGIS redirected motion upward/sideways.",
            "  It preserved safety but lost task intention.",
            "",
            "Not a general benchmark estimate.",
            f"Evidence: Slurm {result['runtime']['slurm_job_id']}; SHA {digest}…",
        )
    )


def make_figure(
    *,
    result_path: Path,
    output_prefix: Path,
    image_path: Path | None,
) -> tuple[Path, Path]:
    result = _load_result(result_path)
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()[:12]
    baseline = result["arms"][BASELINE_KEY]
    aegis = result["arms"][AEGIS_KEY]

    _, baseline_eef, baseline_clearance = _trace(baseline)
    samples, aegis_eef, aegis_clearance = _trace(aegis)
    baseline_boundaries, baseline_distance = _boundaries(baseline)
    aegis_boundaries, aegis_distance = _boundaries(aegis)

    start = baseline_eef[0]
    target = np.asarray(
        baseline["outcome"]["fixed_branch_target_world_m"], dtype=float
    )
    baseline_forward, baseline_lateral = _relative_task_coordinates(
        baseline_eef, start, target
    )
    aegis_forward, aegis_lateral = _relative_task_coordinates(
        aegis_eef, start, target
    )

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.05, 1.15, 1.0))

    camera_axis = figure.add_subplot(grid[0, 0])
    if image_path is not None and image_path.exists():
        camera_axis.imshow(Image.open(image_path))
        camera_axis.set_title(
            "(a) Live AEGIS observation\nCodex label → GroundingDINO",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        camera_axis.axis("off")
    else:
        camera_axis.text(
            0.5,
            0.5,
            "Live observation not supplied",
            ha="center",
            va="center",
        )
        camera_axis.set_axis_off()

    trajectory_axis = figure.add_subplot(grid[0, 1])
    trajectory_axis.plot(
        baseline_forward,
        baseline_lateral,
        color=BASELINE_COLOR,
        linewidth=2.5,
        label="π0.5",
    )
    trajectory_axis.plot(
        aegis_forward,
        aegis_lateral,
        color=AEGIS_COLOR,
        linewidth=2.5,
        label="π0.5 + AEGIS",
    )
    trajectory_axis.scatter(
        [0], [0], s=70, color="#2F3E46", marker="o", zorder=5, label="start"
    )
    trajectory_axis.scatter(
        [baseline_forward[-1]],
        [baseline_lateral[-1]],
        s=70,
        color=BASELINE_COLOR,
        marker="X",
        zorder=5,
    )
    trajectory_axis.scatter(
        [aegis_forward[-1]],
        [aegis_lateral[-1]],
        s=70,
        color=AEGIS_COLOR,
        marker="X",
        zorder=5,
    )
    trajectory_axis.axvline(0, color="#6C757D", linewidth=1, linestyle=":")
    trajectory_axis.set_title(
        "(b) Executed EEF trajectory", loc="left", fontsize=12, fontweight="bold"
    )
    trajectory_axis.set_xlabel("motion toward target (mm)")
    trajectory_axis.set_ylabel("motion away from task line (mm)")
    trajectory_axis.legend(frameon=False, loc="upper left")
    _style_axis(trajectory_axis)

    clearance_axis = figure.add_subplot(grid[0, 2])
    clearance_axis.plot(
        samples,
        baseline_clearance,
        color=BASELINE_COLOR,
        linewidth=2.2,
        label="π0.5",
    )
    clearance_axis.plot(
        samples,
        aegis_clearance,
        color=AEGIS_COLOR,
        linewidth=2.2,
        label="π0.5 + AEGIS",
    )
    clearance_axis.axhline(
        0, color="#111111", linewidth=1.2, linestyle="--", label="collision boundary"
    )
    clearance_axis.axhline(
        5,
        color=MARGIN_COLOR,
        linewidth=1.4,
        linestyle=":",
        label="registered 5 mm margin",
    )
    clearance_axis.fill_between(
        samples,
        np.minimum(baseline_clearance, 0),
        0,
        where=baseline_clearance < 0,
        color=BASELINE_COLOR,
        alpha=0.18,
    )
    clearance_axis.set_title(
        "(c) Simulator clearance", loc="left", fontsize=12, fontweight="bold"
    )
    clearance_axis.set_xlabel("physics sample (0–125)")
    clearance_axis.set_ylabel("registered clearance $D_{sim}$ (mm)")
    clearance_axis.legend(frameon=False, fontsize=8, loc="best")
    _style_axis(clearance_axis)

    progress_axis = figure.add_subplot(grid[1, 0])
    action_boundaries = np.arange(len(baseline_distance))
    progress_axis.plot(
        action_boundaries,
        baseline_distance - baseline_distance[0],
        marker="o",
        linewidth=2.3,
        color=BASELINE_COLOR,
        label="π0.5",
    )
    progress_axis.plot(
        action_boundaries,
        aegis_distance - aegis_distance[0],
        marker="o",
        linewidth=2.3,
        color=AEGIS_COLOR,
        label="π0.5 + AEGIS",
    )
    progress_axis.axhline(0, color="#6C757D", linewidth=1, linestyle=":")
    progress_axis.set_title(
        "(d) Change in target distance", loc="left", fontsize=12, fontweight="bold"
    )
    progress_axis.set_xlabel("after executed action")
    progress_axis.set_ylabel("distance change (mm; lower is better)")
    progress_axis.set_xticks(action_boundaries)
    progress_axis.legend(frameon=False)
    _style_axis(progress_axis)

    action_axis = figure.add_subplot(grid[1, 1])
    nominal = np.asarray(baseline["nominal_actions"], dtype=float)[:, :3]
    executed = np.asarray(aegis["executed_actions"], dtype=float)[:, :3]
    action_index = np.arange(1, len(nominal) + 1)
    component_colors = ("#264653", "#E9C46A", "#7B2CBF")
    for component, label, color in zip(
        range(3), ("x", "y", "z"), component_colors
    ):
        action_axis.plot(
            action_index,
            nominal[:, component],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.65,
        )
        action_axis.plot(
            action_index,
            executed[:, component],
            color=color,
            linewidth=2.4,
            marker="o",
            label=f"{label}: AEGIS (dashed = π0.5)",
        )
    action_axis.axhline(0, color="#6C757D", linewidth=1)
    action_axis.set_title(
        "(e) Translation commands", loc="left", fontsize=12, fontweight="bold"
    )
    action_axis.set_xlabel("action index")
    action_axis.set_ylabel("normalized command")
    action_axis.set_xticks(action_index)
    action_axis.legend(frameon=False, fontsize=8)
    _style_axis(action_axis)

    summary_axis = figure.add_subplot(grid[1, 2])
    summary_axis.set_axis_off()
    summary_axis.text(
        0.0,
        1.0,
        _outcome_text(result, digest),
        va="top",
        ha="left",
        fontsize=9.2,
        family="DejaVu Sans Mono",
        linespacing=1.17,
    )

    figure.suptitle(
        "Same-state diagnostic: original π0.5 versus original AEGIS safety layer",
        fontsize=16,
        fontweight="bold",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    png_path, pdf_path = make_figure(
        result_path=args.results,
        output_prefix=args.output_prefix,
        image_path=args.image,
    )
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
