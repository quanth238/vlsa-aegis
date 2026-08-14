import unittest

from scripts.validate_distal_l5_aegis_consistent_risk import (
    _validate_proxy_validity,
)


class ProxyValidityTests(unittest.TestCase):
    def test_grouped_collection_retains_proxy_invalid_state(self):
        _validate_proxy_validity(
            grouped_collection=True,
            reported_count=20,
            observed_count=20,
            producer_zero_gate=False,
        )

    def test_grouped_collection_still_rejects_inconsistent_gate(self):
        with self.assertRaises(ValueError):
            _validate_proxy_validity(
                grouped_collection=True,
                reported_count=20,
                observed_count=20,
                producer_zero_gate=True,
            )

    def test_single_state_gate_rejects_proxy_invalid_state(self):
        with self.assertRaises(ValueError):
            _validate_proxy_validity(
                grouped_collection=False,
                reported_count=20,
                observed_count=20,
                producer_zero_gate=False,
            )


if __name__ == "__main__":
    unittest.main()
