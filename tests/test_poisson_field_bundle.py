"""Focused synthetic tests for the fail-closed static field constructor."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None
    for name in ("mujoco", "numpy", "scipy")
)


@unittest.skipUnless(
    DEPENDENCIES_PRESENT, "MuJoCo/NumPy/SciPy numerical dependencies unavailable"
)
class StaticFieldBundleTests(unittest.TestCase):
    XML = r"""
<mujoco model="static_field_bundle_test">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root">
      <body name="{link5_name}" pos="-0.313 0.087 0.613">
        <geom name="link5_collision" type="{link5_type}" size="0.03 0.03 0.03"
              mass="0.2" contype="1" conaffinity="1"/>
        <body name="{link6_name}" pos="0.017 0.011 0.248">
          <geom name="link6_collision" type="box" size="0.03 0.03 0.03"
                mass="0.2" contype="1" conaffinity="1"/>
        </body>
      </body>
    </body>
    <body name="selected_obstacle" pos="0.503 -0.307 0.701">
      <geom name="selected_obstacle_collision" type="{obstacle_type}"
            size="0.04 0.05 0.06" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""

    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        import numpy as np

        from main.poisson_fullbody.feasibility_protocol import (
            load_feasibility_protocol,
        )

        cls.mujoco = mujoco
        cls.np = np
        cls.protocol, cls.protocol_hashes = load_feasibility_protocol(
            ROOT / "configs" / "vlsa_poisson_runtime_protocol.canary.v5.json"
        )
        cls.previous_protocol, cls.previous_protocol_hashes = (
            load_feasibility_protocol(
                ROOT / "configs" / "vlsa_poisson_runtime_protocol.canary.v4.json"
            )
        )
        cls.legacy_protocol, cls.legacy_protocol_hashes = (
            load_feasibility_protocol(
                ROOT / "configs" / "vlsa_poisson_runtime_protocol.canary.v3.json"
            )
        )

    def _id(self, model, object_type, name):
        result = int(self.mujoco.mj_name2id(model, object_type, name))
        self.assertGreaterEqual(result, 0, name)
        return result

    def _scene(
        self,
        *,
        link5_name="robot0_link5",
        link6_name="robot0_link6",
        link5_type="box",
        obstacle_type="box",
    ):
        from main.poisson_fullbody.measurement import resolve_collision_geom_sets

        xml = self.XML.format(
            link5_name=link5_name,
            link6_name=link6_name,
            link5_type=link5_type,
            obstacle_type=obstacle_type,
        )
        model = self.mujoco.MjModel.from_xml_string(xml)
        data = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, data)
        body = self.mujoco.mjtObj.mjOBJ_BODY
        ids = {
            "robot_root": self._id(model, body, "robot_root"),
            "link5": self._id(model, body, link5_name),
            "link6": self._id(model, body, link6_name),
            "obstacle": self._id(model, body, "selected_obstacle"),
        }
        resolved = resolve_collision_geom_sets(
            model,
            robot_root_body_ids=(ids["robot_root"],),
            obstacle_root_body_ids=(ids["obstacle"],),
            link56_body_ids=(ids["link5"], ids["link6"]),
        )
        return model, data, ids, resolved

    def _sparse_free_occupancy(self, grid, boxes, epsilon):
        """Keep the same small world-space robot corridor free on either grid."""

        from main.poisson_fullbody.voxel_grid import OccupancyResult

        np = self.np
        raw = np.zeros(grid.cell_shape, dtype=bool)
        center = boxes[0].center
        index = np.floor((center - grid.lower) / grid.spacing).astype(int)
        index = np.minimum(np.maximum(index, 0), np.asarray(grid.cell_shape) - 1)
        raw[tuple(int(value) for value in index)] = True
        buffered = np.ones(grid.cell_shape, dtype=bool)
        # Covers both protected box surfaces with several free-cell layers.
        corridor_lower = np.asarray([-0.44, -0.04, 0.46], dtype=np.float64)
        corridor_upper = np.asarray([-0.14, 0.24, 0.96], dtype=np.float64)
        start = np.floor((corridor_lower - grid.lower) / grid.spacing).astype(int)
        stop = np.ceil((corridor_upper - grid.lower) / grid.spacing).astype(int)
        start = np.maximum(start, 0)
        stop = np.minimum(stop, np.asarray(grid.cell_shape))
        buffered[
            int(start[0]) : int(stop[0]),
            int(start[1]) : int(stop[1]),
            int(start[2]) : int(stop[2]),
        ] = False
        return OccupancyResult(
            grid=grid,
            raw_cells=raw,
            buffered_cells=buffered,
            epsilon=epsilon,
            numerical_tolerance=0.0,
        )

    def test_constructs_exact_immutable_audited_bundle(self) -> None:
        from main.poisson_fullbody.field_bundle import (
            build_static_field_bundle,
            solve_poisson_sor,
        )

        np = self.np
        model, data, ids, resolved = self._scene()
        state_before = {
            "time": float(data.time),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "geom_xpos": data.geom_xpos.copy(),
            "geom_xmat": data.geom_xmat.copy(),
        }
        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_occupancy",
            side_effect=self._sparse_free_occupancy,
        ) as occupancy_call, mock.patch(
            "main.poisson_fullbody.field_bundle.solve_poisson_sor",
            wraps=solve_poisson_sor,
        ) as solve_call:
            bundle = build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=self.protocol,
                protocol_hashes=self.protocol_hashes,
            )

        self.assertEqual(bundle.grid.vertex_shape, (116, 101, 111))
        self.assertEqual(bundle.protected_body_ids, (ids["link5"], ids["link6"]))
        self.assertEqual(
            bundle.protected_body_names, ("robot0_link5", "robot0_link6")
        )
        self.assertEqual(
            {component.geom_id for component in bundle.protected_samples.components},
            set(resolved.link56_geom_ids),
        )
        self.assertLess(
            bundle.protected_samples.maximum_surface_cover_radius_m,
            self.protocol["coverage"]["epsilon_m"],
        )
        self.assertGreater(bundle.diagnostics.minimum_initial_h_m2, 0.0)
        self.assertGreaterEqual(
            bundle.diagnostics.minimum_outer_boundary_clearance_m,
            bundle.diagnostics.required_outer_boundary_clearance_m,
        )
        self.assertTrue(bundle.diagnostics.poisson.passed)
        self.assertEqual(bundle.diagnostics.poisson_method, "red_black_sor")

        self.assertEqual(len(bundle.obstacle_boxes), 1)
        obstacle_geom = resolved.obstacle_geom_ids[0]
        np.testing.assert_allclose(
            bundle.obstacle_boxes[0].center, data.geom_xpos[obstacle_geom]
        )
        np.testing.assert_allclose(
            bundle.obstacle_boxes[0].R,
            data.geom_xmat[obstacle_geom].reshape(3, 3),
        )
        np.testing.assert_allclose(
            bundle.obstacle_boxes[0].half_extents,
            model.geom_size[obstacle_geom, :3],
        )
        self.assertEqual(occupancy_call.call_count, 1)
        _, occupancy_kwargs = occupancy_call.call_args
        self.assertEqual(
            occupancy_kwargs["epsilon"],
            self.protocol["occupancy"]["obstacle_clearance_m"],
        )
        self.assertEqual(solve_call.call_count, 1)
        _, solve_kwargs = solve_call.call_args
        self.assertEqual(solve_kwargs["omega"], 1.7)
        self.assertEqual(solve_kwargs["max_iterations"], 50000)
        self.assertEqual(solve_kwargs["check_every"], 20)
        self.assertEqual(solve_kwargs["tolerance"], 1e-8)

        self.assertEqual(float(data.time), state_before["time"])
        np.testing.assert_array_equal(data.qpos, state_before["qpos"])
        np.testing.assert_array_equal(data.qvel, state_before["qvel"])
        np.testing.assert_array_equal(data.geom_xpos, state_before["geom_xpos"])
        np.testing.assert_array_equal(data.geom_xmat, state_before["geom_xmat"])
        for value in vars(bundle.hashes).values():
            self.assertRegex(value, r"^[0-9a-f]{64}$")
        with self.assertRaises(AttributeError):
            bundle.field.values = np.zeros((1,))
        with self.assertRaises(ValueError):
            bundle.field.values[0, 0, 0] = 1.0
        for array in (
            bundle.grid.lower,
            bundle.grid.spacing,
            bundle.occupancy.raw_cells,
            bundle.occupancy.buffered_cells,
            bundle.domain.component_cells,
            bundle.domain.active_vertices,
            bundle.domain.boundary_vertices,
            bundle.domain.interior_vertices,
            bundle.obstacle_boxes[0].center,
            bundle.obstacle_boxes[0].R,
            bundle.obstacle_boxes[0].half_extents,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array.flat[0] = array.flat[0]
        with self.assertRaises(FrozenInstanceError):
            bundle.protected_samples.samples[0].body_name = "changed"

    def test_legacy_v3_exact_101_cubed_bundle_remains_supported(self) -> None:
        from main.poisson_fullbody.field_bundle import build_static_field_bundle

        model, data, _, resolved = self._scene()
        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_occupancy",
            side_effect=self._sparse_free_occupancy,
        ):
            bundle = build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=self.legacy_protocol,
                protocol_hashes=self.legacy_protocol_hashes,
            )

        self.assertEqual(bundle.grid.vertex_shape, (101, 101, 101))
        self.np.testing.assert_array_equal(
            bundle.grid.lower, self.np.asarray([-1.0, -1.0, 0.0])
        )
        self.np.testing.assert_array_equal(
            bundle.grid.spacing, self.np.full(3, 0.02)
        )

    def test_immutable_v4_full_subtree_workspace_remains_supported(self) -> None:
        from main.poisson_fullbody.field_bundle import build_static_field_bundle

        model, data, _, resolved = self._scene()
        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_occupancy",
            side_effect=self._sparse_free_occupancy,
        ):
            bundle = build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=self.previous_protocol,
                protocol_hashes=self.previous_protocol_hashes,
            )

        self.assertEqual(bundle.grid.vertex_shape, (116, 101, 111))
        self.np.testing.assert_array_equal(
            bundle.grid.lower, self.np.asarray([-1.3, -1.0, -0.2])
        )

    def test_schema_workspace_mismatch_is_rejected_before_geometry(self) -> None:
        from main.poisson_fullbody.feasibility_protocol import (
            bind_parameter_block,
            validate_feasibility_protocol,
        )
        from main.poisson_fullbody.field_bundle import build_static_field_bundle

        changed = copy.deepcopy(self.protocol)
        changed["workspace"]["grid_shape_vertices"][0] = 115
        changed["workspace"]["maximum_m"][0] = 0.98
        changed = bind_parameter_block(changed)
        changed_hashes = validate_feasibility_protocol(changed)
        model, data, _, resolved = self._scene()
        with mock.patch(
            "main.poisson_fullbody.field_bundle.clone_forwarded_state"
        ) as clone:
            with self.assertRaisesRegex(
                ValueError, "workspace does not match its registered schema"
            ):
                build_static_field_bundle(
                    model,
                    data,
                    resolved=resolved,
                    protocol=changed,
                    protocol_hashes=changed_hashes,
                )
        clone.assert_not_called()

    def test_protocol_identity_mismatch_fails_before_geometry(self) -> None:
        from main.poisson_fullbody.feasibility_protocol import ProtocolHashes
        from main.poisson_fullbody.field_bundle import build_static_field_bundle

        model, data, _, resolved = self._scene()
        wrong_hashes = ProtocolHashes(
            protocol_sha256="0" * 64,
            parameter_block_sha256=self.protocol_hashes.parameter_block_sha256,
        )
        with mock.patch(
            "main.poisson_fullbody.field_bundle.clone_forwarded_state"
        ) as clone:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_static_field_bundle(
                    model,
                    data,
                    resolved=resolved,
                    protocol=self.protocol,
                    protocol_hashes=wrong_hashes,
                )
        clone.assert_not_called()

        changed = copy.deepcopy(self.protocol)
        changed["poisson"]["max_iterations"] += 1
        with self.assertRaisesRegex(ValueError, "parameter_block_sha256"):
            build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=changed,
                protocol_hashes=self.protocol_hashes,
            )

    def test_wrong_protected_names_and_nonbox_obstacle_fail_closed(self) -> None:
        from main.poisson_fullbody.field_bundle import build_static_field_bundle

        model, data, _, resolved = self._scene(link5_name="link5")
        with self.assertRaisesRegex(ValueError, "robot0_link5/robot0_link6"):
            build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=self.protocol,
                protocol_hashes=self.protocol_hashes,
            )

        model, data, _, resolved = self._scene(obstacle_type="sphere")
        with self.assertRaisesRegex(ValueError, "not a supported MuJoCo box"):
            build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=self.protocol,
                protocol_hashes=self.protocol_hashes,
            )

    def test_empty_or_missing_protected_surface_components_fail_closed(self) -> None:
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.surface_sampling import (
            RobotSampleSet,
            build_robot_collision_samples,
        )

        model, data, _, resolved = self._scene()
        epsilon = float(self.protocol["coverage"]["epsilon_m"])
        empty = RobotSampleSet(
            samples=(),
            epsilon_m=epsilon,
            maximum_surface_cover_radius_m=0.0,
            sample_ledger_sha256="0" * 64,
            geom_records=(),
            coverage_semantics="synthetic_empty_failure",
        )
        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_robot_collision_samples",
            return_value=empty,
        ):
            with self.assertRaisesRegex(ValueError, "samples are empty"):
                build_static_field_bundle(
                    model,
                    data,
                    resolved=resolved,
                    protocol=self.protocol,
                    protocol_hashes=self.protocol_hashes,
                )

        complete = build_robot_collision_samples(
            model,
            data,
            geom_ids=resolved.link56_geom_ids,
            epsilon_m=epsilon,
        )
        retained_geom = complete.geom_records[0]["geom_id"]
        incomplete = RobotSampleSet(
            samples=tuple(
                sample for sample in complete.samples if sample.geom_id == retained_geom
            ),
            epsilon_m=complete.epsilon_m,
            maximum_surface_cover_radius_m=(
                complete.geom_records[0]["certified_surface_cover_radius_m"]
            ),
            sample_ledger_sha256=complete.sample_ledger_sha256,
            geom_records=(complete.geom_records[0],),
            coverage_semantics=complete.coverage_semantics,
        )
        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_robot_collision_samples",
            return_value=incomplete,
        ):
            with self.assertRaisesRegex(ValueError, "coverage mismatch; missing="):
                build_static_field_bundle(
                    model,
                    data,
                    resolved=resolved,
                    protocol=self.protocol,
                    protocol_hashes=self.protocol_hashes,
                )

        wrong_record = copy.deepcopy(complete.geom_records[0])
        wrong_record["certificate_kind"] = (
            "analytic_cylinder_parameter_grid_covering_bound"
        )
        wrong_certificate = RobotSampleSet(
            samples=complete.samples,
            epsilon_m=complete.epsilon_m,
            maximum_surface_cover_radius_m=(
                complete.maximum_surface_cover_radius_m
            ),
            sample_ledger_sha256=complete.sample_ledger_sha256,
            geom_records=(wrong_record,) + complete.geom_records[1:],
            coverage_semantics=complete.coverage_semantics,
        )
        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_robot_collision_samples",
            return_value=wrong_certificate,
        ):
            with self.assertRaisesRegex(ValueError, "differs from protocol"):
                build_static_field_bundle(
                    model,
                    data,
                    resolved=resolved,
                    protocol=self.protocol,
                    protocol_hashes=self.protocol_hashes,
                )

    def test_unsupported_protected_geom_and_nonconvergence_are_not_returned(self) -> None:
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.poisson_field import (
            PoissonSolveResult,
            poisson_diagnostics,
        )

        model, data, _, resolved = self._scene(link5_type="sphere")
        with self.assertRaisesRegex(ValueError, "unsupported robot collision geom"):
            build_static_field_bundle(
                model,
                data,
                resolved=resolved,
                protocol=self.protocol,
                protocol_hashes=self.protocol_hashes,
            )

        model, data, _, resolved = self._scene()

        def did_not_converge(system, **kwargs):
            values = self.np.zeros(system.shape, dtype=self.np.float64)
            tolerance = float(kwargs["tolerance"])
            diagnostics = poisson_diagnostics(
                system, values, residual_tolerance=tolerance
            )
            return PoissonSolveResult(
                values=values,
                converged=False,
                method="red_black_sor",
                iterations=int(kwargs["max_iterations"]),
                backward_error_target=tolerance,
                residual_target=tolerance,
                diagnostics=diagnostics,
                message="synthetic iteration limit",
            )

        with mock.patch(
            "main.poisson_fullbody.field_bundle.build_occupancy",
            side_effect=self._sparse_free_occupancy,
        ), mock.patch(
            "main.poisson_fullbody.field_bundle.solve_poisson_sor",
            side_effect=did_not_converge,
        ):
            with self.assertRaisesRegex(ValueError, "did not converge"):
                build_static_field_bundle(
                    model,
                    data,
                    resolved=resolved,
                    protocol=self.protocol,
                    protocol_hashes=self.protocol_hashes,
                )


if __name__ == "__main__":
    unittest.main()
