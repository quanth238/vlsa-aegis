from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from main.poisson_fullbody import registered_contact_monitor as monitor_module


class _Contact:
    def __init__(self, geom1: int, geom2: int, distance: float = 0.0) -> None:
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = distance


class RegisteredContactMonitorTests(unittest.TestCase):
    def test_suffix_monitor_preserves_absolute_physical_boundaries(self):
        resolved = SimpleNamespace(
            robot_geom_ids=(1,),
            obstacle_geom_ids=(2,),
            link56_geom_ids=(1,),
        )
        with mock.patch.object(
            monitor_module, "registered_contact_scope", return_value={"identity_sha256": "a" * 64}
        ), mock.patch.object(
            monitor_module, "clone_forwarded_state", return_value=object()
        ), mock.patch.object(
            monitor_module, "registered_contact_records", return_value=[]
        ):
            sim = SimpleNamespace(
                model=SimpleNamespace(_model=object()),
                data=SimpleNamespace(_data=object()),
            )
            monitor = monitor_module.RegisteredContactMonitor(
                sim,
                resolved,
                start_physical_boundary=4500,
                controller_updates_per_action=5,
                physics_substeps_per_controller_update=5,
            )
            snapshot = monitor.observe_post_integration(
                sim,
                source_action_index=180,
                physics_substep_index=0,
            )
            self.assertEqual(snapshot["physical_boundary"], 4500)
            self.assertEqual(
                monitor.result()["callback_cadence"]["start_physical_boundary"],
                4500,
            )

    def setUp(self) -> None:
        self.model = SimpleNamespace(
            ngeom=6,
            # Geom 5 is robot-owned but deliberately absent from the resolved
            # collision-geom set.  It must not be treated as external scene.
            geom_bodyid=[1, 2, 3, 4, 5, 1],
        )
        self.resolved = SimpleNamespace(
            robot_geom_ids=(0, 1),
            robot_body_ids=(1, 2),
            obstacle_geom_ids=(2,),
            link56_geom_ids=(1,),
        )
        fake_mujoco = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_GEOM=1, mjOBJ_BODY=2),
            mj_id2name=lambda _model, kind, object_id: (
                ("geom_%d" if kind == 1 else "body_%d") % object_id
            ),
        )
        self.mujoco_patch = mock.patch.dict(sys.modules, {"mujoco": fake_mujoco})
        self.mujoco_patch.start()

    def tearDown(self) -> None:
        self.mujoco_patch.stop()

    def test_scope_excludes_all_robot_owned_geometry_from_external_scene(self) -> None:
        scope = monitor_module.registered_contact_scope(self.model, self.resolved)
        self.assertEqual(scope["robot_geom_ids"], [0, 1])
        self.assertEqual(scope["robot_owned_geom_ids"], [0, 1, 5])
        self.assertEqual(scope["external_nonrobot_geom_ids"], [2, 3, 4])
        self.assertNotIn(5, scope["external_nonrobot_geom_ids"])
        self.assertEqual(len(scope["identity_sha256"]), 64)

    def test_records_derive_categories_from_registered_id_sets(self) -> None:
        scope = monitor_module.registered_contact_scope(self.model, self.resolved)
        data = SimpleNamespace(
            ncon=5,
            contact=[
                _Contact(0, 2),       # any robot vs selected obstacle
                _Contact(3, 1),       # link 5/6 vs another scene geom
                _Contact(0, 3),       # other robot link vs other scene: out of scope
                _Contact(1, 5),       # self/robot-owned: out of external scope
                _Contact(1, 4, 1e-3), # separated pair: not literal contact
            ],
        )
        records = monitor_module.registered_contact_records(
            self.model,
            data,
            scope,
            source_phase="post_integration_recomputed",
            physical_boundary=17,
            source_action_index=0,
            physics_substep_index=17,
        )
        self.assertEqual(len(records), 2)
        selected, shifted = records
        self.assertEqual(
            selected["contact_categories"],
            ["any_robot_vs_selected_obstacle"],
        )
        self.assertFalse(selected["link56_nonselected_external_contact"])
        self.assertEqual(
            shifted["contact_categories"],
            ["link56_vs_external_nonrobot"],
        )
        self.assertTrue(shifted["link56_nonselected_external_contact"])
        self.assertEqual(selected["executed_transition_start_boundary"], 17)
        self.assertEqual(selected["observed_state_boundary"], 18)
        self.assertTrue(all(len(row["record_sha256"]) == 64 for row in records))

    def test_live_and_forwarded_contact_phases_have_distinct_boundaries(self) -> None:
        data = SimpleNamespace(ncon=1, contact=[_Contact(1, 3)])
        sim = SimpleNamespace(model=self.model, data=data)
        with mock.patch.object(
            monitor_module, "clone_forwarded_state", side_effect=lambda _m, d: d
        ):
            monitor = monitor_module.RegisteredContactMonitor(
                sim,
                self.resolved,
                physics_substeps_per_action=2,
                controller_updates_per_action=1,
                physics_substeps_per_controller_update=2,
            )
            # Remove settled contact so this test isolates rollout cadence.
            data.ncon = 0
            first = monitor.observe_post_integration(
                sim, source_action_index=0, physics_substep_index=0
            )
            self.assertFalse(first["literal_contact"])
            data.ncon = 1
            second = monitor.observe_post_integration(
                sim, source_action_index=0, physics_substep_index=1
            )
            self.assertTrue(second["literal_contact"])
            self.assertEqual(second["executed_transition_start_boundary"], 1)
            self.assertEqual(second["observed_state_boundary"], 2)
            self.assertEqual(len(second["contacts"]), 2)
            phases = {row["source_phase"] for row in second["contacts"]}
            self.assertEqual(
                phases,
                {
                    "live_solver_phase_preintegration_geometry",
                    "post_integration_recomputed",
                },
            )
            live = next(
                row
                for row in second["contacts"]
                if row["source_phase"]
                == "live_solver_phase_preintegration_geometry"
            )
            forwarded = next(
                row
                for row in second["contacts"]
                if row["source_phase"] == "post_integration_recomputed"
            )
            self.assertEqual(live["observed_state_boundary"], 1)
            self.assertEqual(forwarded["observed_state_boundary"], 2)
            result = monitor.result()
            self.assertEqual(result["observed_physics_substeps"], 2)
            self.assertTrue(result["any_link56_nonselected_external_contact"])

    def test_observation_gap_is_rejected(self) -> None:
        data = SimpleNamespace(ncon=0, contact=[])
        sim = SimpleNamespace(model=self.model, data=data)
        with mock.patch.object(
            monitor_module, "clone_forwarded_state", side_effect=lambda _m, d: d
        ):
            monitor = monitor_module.RegisteredContactMonitor(
                sim, self.resolved, physics_substeps_per_action=2
            )
            with self.assertRaises(monitor_module.RegisteredContactMonitorError):
                monitor.observe_post_integration(
                    sim, source_action_index=0, physics_substep_index=1
                )


if __name__ == "__main__":
    unittest.main()
