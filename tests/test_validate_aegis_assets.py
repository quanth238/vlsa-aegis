from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import validate_aegis_assets as preflight


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


class ValidateAegisAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.name", "Asset Test")
        _git(self.repo, "config", "user.email", "assets@example.invalid")

        self.bddl = self.repo / "frozen/task.bddl"
        self.initial_states = self.repo / "frozen/task.pruned_init"
        self.bddl.parent.mkdir()
        self.bddl.write_text("task\n", encoding="utf-8")
        self.initial_states.write_text("state\n", encoding="utf-8")
        _git(self.repo, "add", "frozen")
        _git(self.repo, "commit", "-qm", "upstream")
        self.upstream_commit = _git(self.repo, "rev-parse", "HEAD")

        config = {
            "schema_version": preflight.CONFIG_SCHEMA,
            "protocol_id": "test-protocol",
            "source": {"upstream_commit": self.upstream_commit},
            "population": {"expected_cases": 1600},
        }
        self.config_path = self.repo / "config.json"
        config_raw = (
            json.dumps(config, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        self.config_path.write_bytes(config_raw)
        config_sha256 = _sha256(config_raw)

        bddl_sha256 = preflight.sha256_path(self.bddl)
        state_sha256 = preflight.sha256_path(self.initial_states)
        rows = []
        for index in range(1600):
            rows.append(
                {
                    "schema_version": preflight.MANIFEST_SCHEMA,
                    "protocol_id": "test-protocol",
                    "protocol_config_sha256": config_sha256,
                    "case_id": f"case-{index:04d}",
                    "bddl_path": "frozen/task.bddl",
                    "bddl_sha256": bddl_sha256,
                    "initial_states_path": "frozen/task.pruned_init",
                    "initial_states_sha256": state_sha256,
                }
            )
        manifest_raw = b"".join(
            preflight.canonical_json_bytes(row) + b"\n" for row in rows
        )
        self.manifest_path = self.repo / "manifest.jsonl"
        self.manifest_path.write_bytes(manifest_raw)
        receipt = {
            "schema_version": preflight.RECEIPT_SCHEMA,
            "protocol_config_sha256": config_sha256,
            "manifest_sha256": _sha256(manifest_raw),
            "manifest_rows": 1600,
        }
        self.receipt_path = self.repo / "receipt.json"
        self.receipt_path.write_text(
            json.dumps(receipt, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _git(
            self.repo,
            "add",
            "config.json",
            "manifest.jsonl",
            "receipt.json",
        )
        _git(self.repo, "commit", "-qm", "apparatus")
        self.expected_commit = _git(self.repo, "rev-parse", "HEAD")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _arguments(self, output: Path) -> list[str]:
        return [
            "--profile",
            "capture",
            "--repo-root",
            str(self.repo),
            "--expected-commit",
            self.expected_commit,
            "--config",
            str(self.config_path),
            "--manifest",
            str(self.manifest_path),
            "--manifest-receipt",
            str(self.receipt_path),
            "--output",
            str(output),
        ]

    def test_capture_preflight_binds_clean_source_and_all_sources(self) -> None:
        output = self.root / "receipts/preflight.json"
        output.parent.mkdir()
        self.assertEqual(preflight.main(self._arguments(output)), 0)
        value = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["profile"], "capture")
        self.assertEqual(value["case_count"], 1600)
        self.assertFalse(value["scientific_result"])
        self.assertFalse(value["source"]["git_dirty"])
        self.assertEqual(value["source"]["git_commit"], self.expected_commit)
        self.assertEqual(
            len(value["protocol"]["frozen_safelibero_sources"]), 2
        )
        self.assertFalse(value["assets"]["policy_model_executed"])
        self.assertFalse(value["assets"]["groundingdino_executed"])
        self.assertFalse(value["assets"]["qp_executed"])

    def test_receipt_is_immutable(self) -> None:
        output = self.root / "receipts/preflight.json"
        output.parent.mkdir()
        self.assertEqual(preflight.main(self._arguments(output)), 0)
        original = output.read_bytes()
        self.assertEqual(preflight.main(self._arguments(output)), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_dirty_source_is_rejected(self) -> None:
        (self.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        output = self.root / "receipts/preflight.json"
        output.parent.mkdir()
        self.assertEqual(preflight.main(self._arguments(output)), 2)
        self.assertFalse(output.exists())

    def test_file_hash_and_size_are_both_enforced(self) -> None:
        value = self.root / "asset.bin"
        value.write_bytes(b"abc")
        record = preflight.validate_file(
            value,
            expected_size=3,
            expected_sha256=_sha256(b"abc"),
            label="fixture",
        )
        self.assertEqual(record["bytes"], 3)
        with self.assertRaises(preflight.PreflightError):
            preflight.validate_file(
                value,
                expected_size=4,
                expected_sha256=_sha256(b"abc"),
                label="fixture",
            )
        with self.assertRaises(preflight.PreflightError):
            preflight.validate_file(
                value,
                expected_size=3,
                expected_sha256=_sha256(b"abd"),
                label="fixture",
            )

    def test_optional_full_jax_tree_hash_detects_same_size_mutation(self) -> None:
        checkpoint = self.root / "pi05"
        (checkpoint / "params").mkdir(parents=True)
        metadata = checkpoint / "params/meta"
        data = checkpoint / "params/data"
        metadata.write_bytes(b"metadata")
        data.write_bytes(b"weights")
        metadata_spec = {
            "params/meta": (
                len(b"metadata"),
                _sha256(b"metadata"),
            )
        }
        data_spec = {"params/data": len(b"weights")}
        with (
            mock.patch.dict(
                preflight.PI05_METADATA_FILES,
                metadata_spec,
                clear=True,
            ),
            mock.patch.dict(
                preflight.PI05_DATA_FILES,
                data_spec,
                clear=True,
            ),
        ):
            expected = preflight._tree_content_sha256(
                checkpoint, ("params/meta", "params/data")
            )
            record = preflight.validate_pi05_checkpoint(
                checkpoint, expected_tree_sha256=expected
            )
            self.assertTrue(record["full_content_hash_verified"])
            self.assertEqual(record["full_content_tree_sha256"], expected)

            data.write_bytes(b"WEIGHTS")
            with self.assertRaises(preflight.PreflightError):
                preflight.validate_pi05_checkpoint(
                    checkpoint, expected_tree_sha256=expected
                )

    def test_canary_accepts_one_frozen_label_but_population_requires_all(self) -> None:
        checkpoint = self.root / "pi05"
        checkpoint.mkdir()
        (checkpoint / "meta").write_bytes(b"meta")
        (checkpoint / "data").write_bytes(b"data")
        dino_config = self.root / "dino.py"
        dino_checkpoint = self.root / "dino.pth"
        dino_config.write_bytes(b"config")
        dino_checkpoint.write_bytes(b"checkpoint")
        label_manifest = self.root / "labels.jsonl"
        label_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "vlsa_table1_codex_label.v1",
                    "case_id": "case-0",
                    "settled_agentview_array_sha256": "a" * 64,
                    "obstacle_label": "red mug",
                    "reviewer": "codex",
                    "reviewed_at": "2026-07-17T00:00:00Z",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        cases = [
            {"case_id": "case-0", "suite": "safelibero_spatial"},
            {"case_id": "case-1", "suite": "safelibero_spatial"},
        ]
        with (
            mock.patch.dict(
                preflight.PI05_METADATA_FILES,
                {"meta": (4, _sha256(b"meta"))},
                clear=True,
            ),
            mock.patch.dict(
                preflight.PI05_DATA_FILES,
                {"data": 4},
                clear=True,
            ),
            mock.patch.dict(
                preflight.DINO_FILES,
                {
                    "config": (6, _sha256(b"config")),
                    "checkpoint": (10, _sha256(b"checkpoint")),
                },
                clear=True,
            ),
        ):
            record = preflight.validate_evaluation_assets(
                cases=cases,
                required_case_ordinals=[0],
                require_complete_label_population=False,
                pi05_checkpoint=checkpoint,
                expected_pi05_tree_sha256=None,
                dino_config=dino_config,
                dino_checkpoint=dino_checkpoint,
                label_manifest=label_manifest,
                expected_label_manifest_sha256=preflight.sha256_path(
                    label_manifest
                ),
            )
            self.assertEqual(
                record["frozen_codex_labels"]["required_cases"], 1
            )
            self.assertFalse(
                record["frozen_codex_labels"][
                    "all_population_cases_bound"
                ]
            )
            with self.assertRaises(preflight.PreflightError):
                preflight.validate_evaluation_assets(
                    cases=cases,
                    required_case_ordinals=[],
                    require_complete_label_population=True,
                    pi05_checkpoint=checkpoint,
                    expected_pi05_tree_sha256=None,
                    dino_config=dino_config,
                    dino_checkpoint=dino_checkpoint,
                    label_manifest=label_manifest,
                    expected_label_manifest_sha256=preflight.sha256_path(
                        label_manifest
                    ),
                )


if __name__ == "__main__":
    unittest.main()
