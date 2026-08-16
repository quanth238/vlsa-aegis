import unittest

from main.multilink_ellipsoid.pi05_palm_l6_cohort import choose_records


def _record(case_id, group, target, *, success=True):
    return {
        "case_id": case_id,
        "task_level_group_id": group,
        "task_success": success,
        "eligible_target_groups": [target],
    }


class Pi05PalmL6CohortTest(unittest.TestCase):
    def test_abundant_palm_selection_reserves_multi_target_l6_source(self):
        config = {
            "target_order": ["palm", "L6"],
            "split_task_level_groups": {
                "train": ["g0", "g1"], "validation": [], "test": [],
            },
            "target_split_counts": {
                "palm": {"train": 1, "validation": 0, "test": 0},
                "L6": {"train": 1, "validation": 0, "test": 0},
            },
        }
        exclusive = _record("p0", "g0", "palm")
        shared = _record("shared", "g1", "palm")
        shared["eligible_target_groups"] = ["palm", "L6"]
        chosen = choose_records([exclusive, shared], config)
        self.assertEqual(
            [(row["case_id"], row["target_group"]) for row in chosen],
            [("p0", "palm"), ("shared", "L6")],
        )

    def test_selection_respects_strict_groups_and_unique_cases(self):
        config = {
            "target_order": ["palm", "L6"],
            "split_task_level_groups": {
                "train": ["g0"], "validation": ["g1"], "test": ["g2"],
            },
            "target_split_counts": {
                "palm": {"train": 1, "validation": 1, "test": 1},
                "L6": {"train": 1, "validation": 1, "test": 1},
            },
        }
        rows = []
        for index, group in enumerate(("g0", "g1", "g2")):
            rows.append(_record(f"p{index}", group, "palm"))
            rows.append(_record(f"l{index}", group, "L6"))
        chosen = choose_records(rows, config)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({row["case_id"] for row in chosen}), 6)
        split_by_group = {"g0": "train", "g1": "validation", "g2": "test"}
        self.assertTrue(all(
            row["split"] == split_by_group[row["task_level_group_id"]]
            for row in chosen
        ))

    def test_selection_fails_instead_of_replacing_missing_sources(self):
        config = {
            "target_order": ["palm", "L6"],
            "split_task_level_groups": {
                "train": ["g0"], "validation": ["g1"], "test": ["g2"],
            },
            "target_split_counts": {
                "palm": {"train": 1, "validation": 0, "test": 0},
                "L6": {"train": 1, "validation": 0, "test": 0},
            },
        }
        with self.assertRaisesRegex(ValueError, "insufficient eligible L6 train"):
            choose_records([_record("p0", "g0", "palm")], config)


if __name__ == "__main__":
    unittest.main()
