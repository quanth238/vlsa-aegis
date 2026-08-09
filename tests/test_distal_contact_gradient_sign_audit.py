import unittest


class ContactGradientSignAuditTest(unittest.TestCase):
    def test_paired_candidates_have_expected_sign_norm_and_no_clipping(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is available in the allocation environment")
        from main.multilink_ellipsoid.contact_gradient_sign_audit import (
            paired_gradient_candidates,
        )

        nominal = np.asarray([0.1, -0.2, 0.3])
        gradient = np.asarray([3.0, 0.0, 4.0])
        minus, plus, direction = paired_gradient_candidates(
            nominal, gradient, 0.1, 1.0
        )
        np.testing.assert_allclose(direction, [0.6, 0.0, 0.8])
        np.testing.assert_allclose(minus, nominal - 0.1 * direction)
        np.testing.assert_allclose(plus, nominal + 0.1 * direction)
        self.assertAlmostEqual(float(np.linalg.norm(minus - nominal)), 0.1)
        self.assertAlmostEqual(float(np.linalg.norm(plus - nominal)), 0.1)

    def test_contact_burden_ordering(self):
        from main.multilink_ellipsoid.contact_gradient_sign_audit import (
            compare_contact_burden,
            contact_burden,
        )

        safe = contact_burden({"raw_protected_contact_events": []})
        shallow = contact_burden({"raw_protected_contact_events": [
            {"distance_m": -0.001, "substep_index": 4},
        ]})
        deep = contact_burden({"raw_protected_contact_events": [
            {"distance_m": -0.003, "substep_index": 3},
        ]})
        self.assertEqual(compare_contact_burden(safe, shallow, 1e-12), "minus")
        self.assertEqual(compare_contact_burden(deep, shallow, 1e-12), "plus")
        self.assertEqual(compare_contact_burden(safe, safe, 1e-12), "tie")


if __name__ == "__main__":
    unittest.main()
