import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.exact_group_boundary import payload_sha256
from main.multilink_ellipsoid.whole_body_combined_audit import (
    RESULT_SCHEMA as COVERAGE_SCHEMA,
    payload_sha256 as coverage_payload_sha256,
)
from main.multilink_ellipsoid.whole_body_final_bindings import (
    bind_combined_audit,
    bind_q_prediction,
)
from main.multilink_ellipsoid.whole_body_q_only_prediction import load_binding


ROOT = Path(__file__).resolve().parents[1]


class WholeBodyFinalBindingsTest(unittest.TestCase):
    def test_combined_binding_adds_validated_targeted_source(self):
        targeted = json.loads((
            ROOT / "configs/vlsa_distal_pi05_palm_l6_targeted_extension.v1.json"
        ).read_text())
        validation = {
            "schema_version": "vlsa_distal_exact_group_boundary_validation.v1",
            "independent_replay": {"exact_scientific_reproduction": True},
        }
        validation["validation_payload_sha256"] = payload_sha256(
            validation, key="validation_payload_sha256",
        )
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            targeted_path = tmp / "targeted.json"
            validation_path = tmp / "validation.json"
            targeted_path.write_text(json.dumps(targeted))
            validation_path.write_text(json.dumps(validation))
            value = bind_combined_audit(
                base_path=ROOT / "configs/vlsa_distal_whole_body_combined_audit.v1.json",
                targeted_cohort_path=targeted_path,
                targeted_artifact_root=tmp / "artifacts",
                targeted_artifact_commit="commit",
                targeted_validation_path=validation_path,
            )
        self.assertEqual(
            value["required_split_case_count"],
            {"train": 21, "validation": 7, "test": 10},
        )
        self.assertEqual(len(value["sources"]), 3)
        self.assertIn("initially_unsafe_policy", value)

    def test_prediction_binding_accepts_undercovered_audit(self):
        coverage = {
            "schema_version": COVERAGE_SCHEMA,
            "training_authorized": False,
            "combined_config": {"sources": [{}, {}, {}]},
        }
        coverage["audit_payload_sha256"] = coverage_payload_sha256(
            coverage, key="audit_payload_sha256",
        )
        protocol_path = (
            ROOT / "configs/vlsa_distal_whole_body_q_only_prediction_protocol.v1.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            coverage_path = tmp / "coverage.json"
            binding_path = tmp / "binding.json"
            coverage_path.write_text(json.dumps(coverage))
            value = bind_q_prediction(
                protocol_path=protocol_path,
                coverage_audit_path=coverage_path,
            )
            binding_path.write_text(json.dumps(value))
            from main.multilink_ellipsoid.whole_body_q_only_prediction import (
                load_protocol,
            )
            loaded = load_binding(binding_path, load_protocol(protocol_path))
        self.assertEqual(len(loaded["sources"]), 3)
        self.assertTrue(loaded["test_opened_once"])


if __name__ == "__main__":
    unittest.main()
