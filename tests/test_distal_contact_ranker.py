import unittest


class DistalContactRankerTest(unittest.TestCase):
    def test_link_labels_keep_three_bodies_separate(self):
        from main.multilink_ellipsoid.contact_ranker import link_contact_labels

        self.assertEqual(
            link_contact_labels([
                {"protected_body_name": "robot0_link5"},
                {"protected_body_name": "robot0_link7"},
            ]),
            [1, 0, 1],
        )

    def test_unknown_body_is_rejected(self):
        from main.multilink_ellipsoid.contact_ranker import link_contact_labels

        with self.assertRaises(ValueError):
            link_contact_labels([{"protected_body_name": "robot0_link4"}])

    def test_calibration_is_strictly_below_all_unsafe(self):
        from main.multilink_ellipsoid.contact_ranker import (
            calibrate_zero_false_safe_threshold,
        )

        threshold = calibrate_zero_false_safe_threshold(
            [0.1, 0.4, 0.2, 0.8], [False, True, False, True]
        )
        self.assertLess(threshold, 0.4)
        self.assertGreater(threshold, 0.2)

    def test_selection_prefers_minimum_correction(self):
        from main.multilink_ellipsoid.contact_ranker import (
            select_minimal_predicted_safe,
        )

        records = [
            {"nominal_xyz": [0, 0, 0], "candidate_xyz": [0.5, 0, 0]},
            {"nominal_xyz": [0, 0, 0], "candidate_xyz": [0.1, 0, 0]},
            {"nominal_xyz": [0, 0, 0], "candidate_xyz": [0.2, 0, 0]},
        ]
        self.assertEqual(
            select_minimal_predicted_safe(records, [0.1, 0.2, 0.9], 0.5), 1
        )


if __name__ == "__main__":
    unittest.main()
