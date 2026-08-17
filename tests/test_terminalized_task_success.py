import json
from pathlib import Path
import tempfile
import unittest

from main.multilink_ellipsoid.terminalized_task_success import (
    ARM_NAMES,
    contact_group,
    load_config,
    scientific_view,
    selected_arm_actions,
)


ROOT = Path(__file__).resolve().parents[1]


class TerminalizedTaskSuccessTest(unittest.TestCase):
    def test_registered_config_is_strict_no_qp_live_replan(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_terminalized_late_flow_task_success.v1.json"
        )
        self.assertEqual(tuple(config["state_protocol"]["arms"]), ARM_NAMES)
        self.assertFalse(config["control"]["released_AEGIS_EE_QP_enabled"])
        self.assertEqual(
            config["control"]["post_intervention_policy"],
            "repeated_late_flow_terminalized_compact_selector_reobserve_requery",
        )
        self.assertTrue(config["control"]["additional_correction_after_first_chunk"])

    def test_config_rejects_qp(self):
        source = ROOT / "configs/vlsa_distal_terminalized_late_flow_task_success.v1.json"
        value = json.loads(source.read_text())
        value["control"]["released_AEGIS_EE_QP_enabled"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "forbidden control"):
                load_config(path)

    def test_selected_actions_keep_only_our_late_flow_arm(self):
        import hashlib

        candidates = []
        outcomes = []
        source_names = (
            "nominal", "terminal_compact_selector",
            "late_flow_terminalized_compact_selector",
        )
        for index, name in enumerate(source_names):
            actions = [[float(index)] * 7 for _ in range(5)]
            digest = hashlib.sha256(
                json.dumps(actions, separators=(",", ":")).encode()
            ).hexdigest()
            candidates.append({"name": name, "source_executed_actions": actions})
            outcomes.append({
                "arm": name,
                "executed_actions_sha256": digest,
                "selection": {"source_candidate": "candidate-%d" % index},
            })
        pilot = {
            "fresh_selected_arm_execution": {
                "exact_case": {"candidates": candidates},
            },
            "scientific_view": {"outcome": {"records": outcomes}},
        }
        selected = selected_arm_actions(pilot)
        self.assertEqual([row["arm"] for row in selected], list(ARM_NAMES))
        self.assertEqual(selected[0]["actions"][0], [2.0] * 7)

    def test_contact_group_keeps_unmodeled_robot_contacts_visible(self):
        groups = {"palm": ["hand"], "L5": ["link5"]}
        self.assertEqual(contact_group("hand", groups), "palm")
        self.assertEqual(contact_group("finger", groups), "other_robot")

    def test_evaluator_reapplies_our_method_instead_of_raw_continuation(self):
        source = (
            ROOT / "scripts/evaluate_terminalized_late_flow_task_success.py"
        ).read_text()
        self.assertIn("class LiveTerminalizedSelector", source)
        self.assertIn(
            "online_late_flow_terminalized_compact_selector_no_QP", source,
        )
        self.assertNotIn("fresh_raw_pi05_full_cartesian_no_QP", source)

    def test_scientific_view_keeps_task_and_safety_separate(self):
        rows = []
        for name in ARM_NAMES:
            rows.append({
                "arm": name,
                "source_candidate": "nominal",
                "intervention_actions_sha256": "a",
                "complete_executed_actions_sha256": "b",
                "action_count": 200,
                "live_policy_query_count": 3,
                "native_task_success": True,
                "native_task_success_step": 199,
                "timeout": False,
                "raw_robot_contact_pass": False,
                "first_raw_robot_contact": {"step": 190, "substep": 1},
                "robot_contact_sample_count": 1,
                "robot_contact_sample_count_by_group": {"palm": 1},
                "robot_contact_events_sha256": "c",
                "paper_CAR_pass": True,
                "first_paper_CAR_step": None,
                "maximum_active_obstacle_l1_displacement_m": 0.0,
                "collision_free_task_success": False,
                "terminal_dynamic_state_sha256": "d",
                "goal_progress_summary_sha256": "e",
                "online_selection_count": 3,
                "online_selections_sha256": "f",
                "terminal_reason": "raw_robot_contact",
            })
        view = scientific_view(
            case_id="case", source_snapshot_sha256="state", arm_records=rows,
        )
        self.assertEqual(view["task_success_count"], 1)
        self.assertEqual(view["collision_free_task_success_count"], 0)


if __name__ == "__main__":
    unittest.main()
