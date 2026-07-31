from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from main.poisson_fullbody.contracts import (
    ArtifactContractError,
    final_is_resumable,
    publish_hashed_json,
    sha256_file,
)
from tests.test_poisson_result_schema import valid_payload


class AtomicResultContractTest(unittest.TestCase):
    def test_publish_is_atomic_and_byte_identical_republish_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = {"case_id": "case-a", "arm": "adapter_only"}
            self.assertEqual(publish_hashed_json(path, payload), "published")
            first_bytes = path.read_bytes()
            self.assertEqual(
                publish_hashed_json(path, payload), "identical_existing"
            )
            self.assertEqual(path.read_bytes(), first_bytes)
            self.assertFalse(any(path.parent.glob("*.partial")))

    def test_existing_different_final_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            publish_hashed_json(path, {"case_id": "case-a"})
            with self.assertRaisesRegex(ArtifactContractError, "overwrite"):
                publish_hashed_json(path, {"case_id": "case-b"})

    def test_truncated_or_self_hash_mismatched_final_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text('{"case_id":', encoding="utf-8")
            resumable, reason = final_is_resumable(
                path, {"case_id": "case-a"}, identity_fields=("case_id",)
            )
            self.assertFalse(resumable)
            self.assertIn("invalid", reason)

    def test_resume_requires_exact_identity_and_referenced_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.bin"
            trace.write_bytes(b"exact trace")
            reference = {
                "relative_path": "trace.bin",
                "bytes": trace.stat().st_size,
                "sha256": sha256_file(trace),
                "artifact_type": "physics_trace",
                "media_type": "application/octet-stream",
            }
            final = root / "result.json"
            payload = valid_payload()
            payload["artifact_references"] = [reference]
            publish_hashed_json(final, payload)
            fields = ("case_id", "arm", "run_id", "protocol_id")
            resumable, reason = final_is_resumable(
                final, payload, identity_fields=fields
            )
            self.assertFalse(resumable)
            self.assertIn("artifact_root", reason)

            resumable, reason = final_is_resumable(
                final, payload, identity_fields=fields, artifact_root=root
            )
            self.assertTrue(resumable, reason)

            wrong = dict(payload)
            wrong["arm"] = "joint_velocity_adapter_only"
            resumable, reason = final_is_resumable(
                final, wrong, identity_fields=fields, artifact_root=root
            )
            self.assertFalse(resumable)
            self.assertIn("arm", reason)

            trace.write_bytes(b"changed")
            resumable, reason = final_is_resumable(
                final, payload, identity_fields=fields, artifact_root=root
            )
            self.assertFalse(resumable)
            self.assertTrue("byte count" in reason or "SHA-256" in reason)

    def test_self_hashed_but_schema_invalid_final_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = {"case_id": "case-a", "arm": "adapter_only"}
            publish_hashed_json(path, payload)
            resumable, reason = final_is_resumable(
                path, payload, identity_fields=("case_id", "arm")
            )
            self.assertFalse(resumable)
            self.assertIn("schema_version", reason)

    def test_resume_identity_supports_exact_nested_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = valid_payload()
            publish_hashed_json(path, payload)
            expected = {
                "run_id": payload["run_id"],
                "provenance": {
                    "code_commit": payload["provenance"]["code_commit"],
                    "run_contract_sha256": payload["provenance"][
                        "run_contract_sha256"
                    ],
                },
                "runtime": {
                    "model": {
                        "checkpoint_sha256": payload["runtime"]["model"][
                            "checkpoint_sha256"
                        ]
                    }
                },
            }
            fields = (
                "run_id",
                "provenance.code_commit",
                "provenance.run_contract_sha256",
                "runtime.model.checkpoint_sha256",
            )
            resumable, reason = final_is_resumable(
                path, expected, identity_fields=fields
            )
            self.assertTrue(resumable, reason)

            expected["runtime"]["model"]["checkpoint_sha256"] = "f" * 64
            resumable, reason = final_is_resumable(
                path, expected, identity_fields=fields
            )
            self.assertFalse(resumable)
            self.assertIn("runtime.model.checkpoint_sha256", reason)

    def test_referenced_artifact_rejects_direct_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_directory = root / "real"
            real_directory.mkdir()
            trace = real_directory / "trace.bin"
            trace.write_bytes(b"exact trace")
            reference = {
                "relative_path": "trace-link.bin",
                "bytes": trace.stat().st_size,
                "sha256": sha256_file(trace),
                "artifact_type": "physics_trace",
                "media_type": "application/octet-stream",
            }
            (root / "trace-link.bin").symlink_to(trace)

            payload = valid_payload()
            payload["artifact_references"] = [reference]
            final = root / "result.json"
            publish_hashed_json(final, payload)
            resumable, reason = final_is_resumable(
                final,
                payload,
                identity_fields=("case_id", "arm"),
                artifact_root=root,
            )
            self.assertFalse(resumable)
            self.assertIn("symlinked", reason)

            final.unlink()
            (root / "trace-link.bin").unlink()
            (root / "directory-link").symlink_to(real_directory, target_is_directory=True)
            reference["relative_path"] = "directory-link/trace.bin"
            payload["artifact_references"] = [reference]
            publish_hashed_json(final, payload)
            resumable, reason = final_is_resumable(
                final,
                payload,
                identity_fields=("case_id", "arm"),
                artifact_root=root,
            )
            self.assertFalse(resumable)
            self.assertIn("symlinked", reason)

    def test_nonfinite_payload_cannot_be_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                publish_hashed_json(
                    Path(directory) / "result.json", {"metric": float("nan")}
                )


if __name__ == "__main__":
    unittest.main()
