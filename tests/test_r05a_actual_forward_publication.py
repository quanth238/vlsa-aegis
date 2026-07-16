from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "r05a_actual_forward_canary.json"
ENVELOPE_PATH = ROOT / "schemas" / "r05a-actual-forward-canary-envelope.schema.json"
RECEIPT_PATH = (
    ROOT / "schemas" / "r05a-actual-forward-canary-publication-receipt.schema.json"
)

try:
    from publish_crfs_r05a_actual_forward_canary import (
        CHECKPOINT_CONFIG,
        CHECKPOINT_MODEL,
        NORMALIZATION_ASSET,
        _allocation_test_summary,
        _gpu_telemetry_summary,
    )

    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


class ActualForwardPublicationContractTest(unittest.TestCase):
    def test_both_publication_schema_hashes_are_frozen(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        artifact = config["artifact_contract"]
        self.assertEqual(artifact["publication_receipt_name"], "cpu-afterany-validation.json")
        for path_key, digest_key in (
            ("envelope_schema_path", "envelope_schema_sha256"),
            ("publication_receipt_schema_path", "publication_receipt_schema_sha256"),
        ):
            path = ROOT / artifact[path_key]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact[digest_key])

    def test_result_and_receipt_require_complete_raw_evidence_and_telemetry(self) -> None:
        expected_evidence = {
            "source_contract_path",
            "source_contract_sha256",
            "held_gpu_submission_path",
            "held_gpu_submission_sha256",
            "submission_path",
            "submission_sha256",
            "release_fingerprint_path",
            "release_fingerprint_sha256",
            "payload_path",
            "payload_sha256",
            "tensor_path",
            "tensor_sha256",
            "query_ledger_path",
            "query_ledger_sha256",
            "allocation_tests_log_path",
            "allocation_tests_log_sha256",
            "host_telemetry_path",
            "host_telemetry_sha256",
            "gpu_telemetry_path",
            "gpu_telemetry_sha256",
            "live_preflight_path",
            "live_preflight_sha256",
            "actual_config_path",
            "actual_config_sha256",
            "legacy_config_path",
            "legacy_config_sha256",
            "manifest_path",
            "manifest_sha256",
            "source_r02_path",
            "source_r02_sha256",
            "source_r02_config_path",
            "source_r02_config_sha256",
            "checkpoint_model_path",
            "checkpoint_model_sha256",
            "checkpoint_config_path",
            "checkpoint_config_sha256",
            "normalization_asset_path",
            "normalization_asset_sha256",
        }
        envelope = json.loads(ENVELOPE_PATH.read_text(encoding="utf-8"))
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(envelope["properties"]["raw_evidence"]["required"]), expected_evidence)
        self.assertEqual(set(receipt["$defs"]["evidence"]["required"]), expected_evidence)
        self.assertIn("execution_identity", envelope["required"])
        self.assertIn("measured_telemetry", envelope["required"])
        self.assertIn("measured_telemetry", receipt["required"])
        for schema in (envelope, receipt):
            host = schema["$defs"]["measuredTelemetry"]["properties"]["host"]
            gpu = schema["$defs"]["measuredTelemetry"]["properties"]["gpu"]
            self.assertIn("host_cgroup_sampled_current_high_water_bytes", host["required"])
            self.assertIn("memory_max_job_scope_hard_limit_bytes", host["required"])
            self.assertIn("gpu_uuid", gpu["required"])
            self.assertIn("maximum_compute_mib", gpu["required"])
            self.assertIn("maximum_device_mib", gpu["required"])

    def test_exact_runtime_checkpoint_assets_are_registered(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        bindings = config["frozen_source_bindings"]
        self.assertEqual(
            bindings["checkpoint_model"],
            {
                "path": "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors",
                "sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
            },
        )
        self.assertEqual(
            bindings["checkpoint_config"],
            {
                "path": "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/config.json",
                "sha256": "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
            },
        )
        self.assertEqual(
            bindings["normalization_asset"],
            {
                "path": "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json",
                "sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
            },
        )


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "AF-00A publication helpers require NumPy")
class ActualForwardPublicationHelperTest(unittest.TestCase):
    def test_publisher_runtime_paths_equal_frozen_assets(self) -> None:
        self.assertEqual(
            str(CHECKPOINT_MODEL),
            "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors",
        )
        self.assertEqual(
            str(CHECKPOINT_CONFIG),
            "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/config.json",
        )
        self.assertEqual(
            str(NORMALIZATION_ASSET),
            "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json",
        )

    def test_gpu_summary_is_measured_from_every_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text(
                "timestamp_ns,gpu_uuid,compute_mib,device_mib\n"
                "100,GPU-exact,0,2\n"
                "200,GPU-exact,1234,2345\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _gpu_telemetry_summary(path),
                {
                    "gpu_uuid": "GPU-exact",
                    "sample_count": 2,
                    "first_timestamp_ns": 100,
                    "last_timestamp_ns": 200,
                    "maximum_compute_mib": 1234,
                    "maximum_device_mib": 2345,
                },
            )

    def test_gpu_summary_rejects_nonincreasing_time_or_mixed_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text(
                "timestamp_ns,gpu_uuid,compute_mib,device_mib\n"
                "100,GPU-a,1,2\n"
                "100,GPU-b,3,4\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _gpu_telemetry_summary(path)

    def test_allocation_test_count_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests.log"
            path.write_text(".\n----------------------------------------------------------------------\nRan 27 tests in 1.000s\n\nOK\n", encoding="utf-8")
            self.assertEqual(
                _allocation_test_summary(path),
                {"passed": True, "test_count": 27, "skipped_test_count": 0},
            )


if __name__ == "__main__":
    unittest.main()
