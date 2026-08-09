import unittest


class ContactDirectionalRankerTest(unittest.TestCase):
    def test_empirical_burden_value_uses_safety_count_then_penetration(self):
        from main.multilink_ellipsoid.contact_directional_ranker import (
            empirical_burden_p,
        )

        learned = {"raw_safe": False, "contact_count": 2, "summed_penetration_m": 0.2}
        random = [
            {"raw_safe": True, "contact_count": 0, "summed_penetration_m": 0.0},
            {"raw_safe": False, "contact_count": 2, "summed_penetration_m": 0.1},
            {"raw_safe": False, "contact_count": 2, "summed_penetration_m": 0.3},
            {"raw_safe": False, "contact_count": 3, "summed_penetration_m": 0.0},
        ]
        self.assertAlmostEqual(empirical_burden_p(learned, random), 3.0 / 5.0)

    def test_symmetric_pairs_preserve_groups_and_exclude_count_ties(self):
        try:
            import numpy  # noqa: F401
            from main.multilink_ellipsoid.contact_directional_ranker import (
                symmetric_directional_pairs,
            )
        except ImportError:
            self.skipTest("NumPy is available in the allocation environment")

        def record(index, x, contacts):
            return {
                "record_index": index,
                "state_step": 10,
                "split": "train",
                "candidate_source": "central_d0_%s" % ("plus" if x > 0 else "minus"),
                "nominal_xyz": [0.0, 0.0, 0.0],
                "candidate_xyz": [x, 0.0, 0.0],
                "raw_protected_contact_count": contacts,
            }

        pairs = symmetric_directional_pairs(
            [record(0, -0.1, 1), record(1, 0.1, 3)],
            allowed_source_prefixes=["central_"],
            maximum_radius_action=0.2,
            matching_tolerance=1e-12,
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["informative"])
        self.assertEqual(pairs[0]["better_record_index"], 0)
        self.assertEqual(pairs[0]["worse_record_index"], 1)


if __name__ == "__main__":
    unittest.main()
