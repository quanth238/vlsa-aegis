#!/usr/bin/env python3
"""Calibrate the controlled-pilot sphere/box metric against MuJoCo contacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from crfs_harness.artifacts import atomic_write_json
from crfs_oracle.measurement import signed_distance_point_to_oriented_box


XML = r"""
<mujoco model="crfs_sphere_box_calibration">
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="box_body" pos="0 0 0" quat="0.965925826 0 0 0.258819045">
      <geom name="obstacle_box" type="box" size="0.04 0.07 0.12"
            contype="1" conaffinity="1" margin="0" gap="0"/>
    </body>
    <body name="sphere_body">
      <freejoint/>
      <geom name="eef_sphere" type="sphere" size="0.06"
            mass="1" contype="1" conaffinity="1" margin="0" gap="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def run_calibration(samples: int) -> dict:
    if samples < 50:
        raise ValueError("At least 50 calibration states are required")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    sphere_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "eef_sphere"))
    box_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_box"))
    positions = np.linspace(0.24, -0.02, samples, dtype=np.float64)
    records = []
    max_error = 0.0
    false_contacts = 0
    missed_contacts = 0
    first_contact_clearance = None

    for x_position in positions:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = (x_position, 0.01, 0.015)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        sphere_center = np.asarray(data.geom_xpos[sphere_id], dtype=np.float64)
        box_center = np.asarray(data.geom_xpos[box_id], dtype=np.float64)
        box_rotation = np.asarray(data.geom_xmat[box_id], dtype=np.float64).reshape(3, 3)
        analytic = signed_distance_point_to_oriented_box(
            sphere_center,
            box_center,
            box_rotation,
            model.geom_size[box_id],
        ) - float(model.geom_size[sphere_id, 0])
        fromto = np.empty(6, dtype=np.float64)
        raw = float(mujoco.mj_geomDistance(model, data, sphere_id, box_id, 1.0, fromto))
        paired_contacts = [
            contact
            for contact in data.contact[: data.ncon]
            if {int(contact.geom1), int(contact.geom2)} == {sphere_id, box_id}
        ]
        contact = bool(paired_contacts)
        contact_distance = min((float(value.dist) for value in paired_contacts), default=None)
        error = abs(raw - analytic)
        max_error = max(max_error, error)
        if contact and analytic > 1e-9:
            false_contacts += 1
        if not contact and analytic < -1e-9:
            missed_contacts += 1
        if contact and first_contact_clearance is None:
            first_contact_clearance = analytic
        records.append(
            {
                "sphere_center_m": sphere_center.tolist(),
                "analytic_clearance_m": analytic,
                "mujoco_clearance_m": raw,
                "absolute_error_m": error,
                "contact": contact,
                "contact_distance_m": contact_distance,
                "fromto_m": fromto.tolist(),
            }
        )

    spacing = float(abs(positions[1] - positions[0]))
    passed = bool(
        max_error <= 1e-8
        and false_contacts == 0
        and missed_contacts == 0
        and first_contact_clearance is not None
        and -spacing - 1e-8 <= first_contact_clearance <= 1e-8
    )
    return {
        "schema_version": "1.0",
        "gate": "H03-primitive-calibration",
        "status": "passed" if passed else "failed",
        "mujoco_version": mujoco.__version__,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "samples": samples,
        "sample_spacing_m": spacing,
        "max_analytic_vs_mujoco_error_m": max_error,
        "false_contact_count": false_contacts,
        "missed_contact_count": missed_contacts,
        "first_contact_clearance_m": first_contact_clearance,
        "pass_thresholds": {
            "max_distance_error_m": 1e-8,
            "false_contact_count": 0,
            "missed_contact_count": 0,
            "first_contact_interval_m": [-spacing - 1e-8, 1e-8],
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=261)
    args = parser.parse_args()
    result = run_calibration(args.samples)
    output = Path(args.output)
    atomic_write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
