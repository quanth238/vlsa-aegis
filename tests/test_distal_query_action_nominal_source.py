import unittest

from scripts.evaluate_distal_query_action_risk_e05 import (
    ARCHIVED_POST_AEGIS_NOMINAL,
    RAW_PI05_NOMINAL,
    resolve_nominal_action_source,
)


class QueryActionNominalSourceTest(unittest.TestCase):
    def test_new_raw_pi05_mode_is_explicit_and_legacy_default_is_preserved(self):
        self.assertEqual(
            resolve_nominal_action_source(None, released_aegis_enabled=False),
            ARCHIVED_POST_AEGIS_NOMINAL,
        )
        self.assertEqual(
            resolve_nominal_action_source(
                RAW_PI05_NOMINAL, released_aegis_enabled=False,
            ),
            RAW_PI05_NOMINAL,
        )
        self.assertEqual(
            resolve_nominal_action_source(None, released_aegis_enabled=True),
            RAW_PI05_NOMINAL,
        )
        with self.assertRaisesRegex(ValueError, "raw pi0.5"):
            resolve_nominal_action_source(
                ARCHIVED_POST_AEGIS_NOMINAL, released_aegis_enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
