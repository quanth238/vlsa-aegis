import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_four_task_availability import (
    archived_result_path, load_config, load_population, summarize_records,
)


class FourTaskAvailabilityTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            Path("configs/vlsa_distal_l5_four_task_availability_audit.v1.json"),
            repo_root=Path("."),
        )

    def test_population_and_path(self):
        rows = load_population(
            Path("manifests/vlsa_table1_population.jsonl"), self.config,
        )
        self.assertEqual(len(rows), 400)
        row = next(item for item in rows if item["case_ordinal"] == 700)
        path = archived_result_path(Path("/tmp/table1"), row)
        self.assertEqual(
            str(path),
            "/tmp/table1/tasks/task-14/results/aegis/"
            "vlsa-t1-goal-ii-t2-e00/result.json",
        )

    def test_fold_gate_requires_both_levels(self):
        records = []
        for task in range(4):
            for level in ("I", "II"):
                records.append({
                    "case_id": f"t{task}-{level}",
                    "logical_task_index": task,
                    "safety_level": level,
                    "classification": "ELIGIBLE_TASK_SUCCESS_L5",
                    "eligible": not (task == 1 and level == "I"),
                })
        summary = summarize_records(records, self.config)
        folds = {
            item["held_out_logical_task_index"]: item
            for item in summary["folds"]
        }
        self.assertFalse(summary["all_four_task_folds_feasible"])
        self.assertFalse(folds[1]["fold_feasible"])
        self.assertTrue(folds[0]["fold_feasible"])


if __name__ == "__main__":
    unittest.main()
