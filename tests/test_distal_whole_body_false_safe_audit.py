import unittest

from main.multilink_ellipsoid.whole_body_false_safe_audit import (
    attribute_selection, selected_false_safe_states, summarize,
)


def _case(state_id, candidate, risks):
    return {
        "case_id": state_id,
        "exact_case": {"candidates": [{
            "name": candidate,
            "exact_group_target": {
                "known_outcome": True,
                "group_future_violation": risks,
                "group_contact_events": {
                    group: [] for group in risks
                },
            },
        }]},
    }


class WholeBodyFalseSafeAuditTest(unittest.TestCase):
    def test_selects_only_heldout_unsafe_selected_actions(self):
        prediction = {"arms": {"shared_constraint_135D": {"metrics": {
            "validation": {"global": {"states": [
                {"state_id": "v", "selected_candidate": "nominal",
                 "selected_actual_safe": False,
                 "selected_predicted_global_risk": -0.2},
            ]}},
            "test": {"global": {"states": [
                {"state_id": "t", "selected_candidate": "safe",
                 "selected_actual_safe": True,
                 "selected_predicted_global_risk": -0.1},
            ]}},
        }}}}
        rows = selected_false_safe_states(prediction)
        self.assertEqual([row["state_id"] for row in rows], ["v"])
        self.assertEqual(rows[0]["split"], "validation")

    def test_attributes_physical_and_diagnostic_witness_separately(self):
        selection = {
            "state_id": "v", "split": "validation",
            "selected_candidate": "nominal",
            "selected_predicted_global_risk": -0.2,
        }
        case = _case("v", "nominal", {
            "end_effector": 0.5, "palm": 0.2, "L5": -0.1,
            "L6": -0.3, "L7": -0.4,
        })
        row = attribute_selection(selection, case)
        self.assertEqual(row["active_physical_witness"], "palm")
        self.assertEqual(row["active_represented_witness"], "end_effector")
        self.assertEqual(summarize([row])["active_physical_witness_count"]["palm"], 1)

    def test_rejects_a_selection_that_is_physically_safe(self):
        selection = {
            "state_id": "v", "split": "test",
            "selected_candidate": "nominal",
            "selected_predicted_global_risk": -0.2,
        }
        case = _case("v", "nominal", {
            "end_effector": 0.5, "palm": -0.2, "L5": -0.1,
            "L6": -0.3, "L7": -0.4,
        })
        with self.assertRaisesRegex(ValueError, "not physical false-safe"):
            attribute_selection(selection, case)


if __name__ == "__main__":
    unittest.main()
