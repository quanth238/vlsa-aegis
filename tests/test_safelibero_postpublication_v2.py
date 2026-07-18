"""Tests for immutable source binding of post-publication analysis v2."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/build_safelibero_postpublication_v2.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


postpublication = load_module("postpublication_v2", SCRIPT)
aggregate = postpublication.aggregate


def publish(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def bind_payload(value: dict, field: str) -> dict:
    output = copy.deepcopy(value)
    output[field] = aggregate.sha256_bytes(
        aggregate.canonical_json_bytes(output)
    )
    return output


class PostpublicationV2Test(unittest.TestCase):
    def source_files(
        self, root: Path
    ) -> tuple[Path, Path, Path, dict]:
        summary = {
            "schema_version": postpublication.SUMMARY_SCHEMA_V1,
            "status": "complete_population_validated",
            "population": {"no_results_dropped": True},
            "accepted_result_payloads_sha256": "a" * 64,
        }
        summary_path = root / "summary.json"
        publish(summary_path, summary)
        prepublish = bind_payload(
            {
                "schema_version": (
                    postpublication.PREPUBLISH_SCHEMA_V1
                ),
                "status": "validated",
                "complete_paired_population": True,
                "no_results_dropped": True,
            },
            "receipt_payload_sha256",
        )
        prepublish_path = root / "prepublish.json"
        publish(prepublish_path, prepublish)
        publication = bind_payload(
            {
                "schema_version": (
                    postpublication.PUBLICATION_SCHEMA_V1
                ),
                "status": "published",
                "complete_paired_population": True,
                "no_results_dropped": True,
                "source_git_commit": "b" * 40,
                "run_id": "immutable-run",
                "summary": {
                    "path": "/remote/summary.json",
                    "sha256": aggregate.sha256_path(summary_path),
                },
                "prepublish_receipt": {
                    "path": "/remote/prepublish.json",
                    "sha256": aggregate.sha256_path(prepublish_path),
                    "receipt_payload_sha256": prepublish[
                        "receipt_payload_sha256"
                    ],
                },
                "gallery": {
                    "path": "/remote/gallery/index.html",
                    "sha256": "c" * 64,
                },
                "failure_analysis": {
                    "cases": {
                        "path": "/remote/failure/cases.jsonl",
                        "sha256": "d" * 64,
                    },
                    "report": {
                        "path": "/remote/failure/report.json",
                        "sha256": "e" * 64,
                    },
                    "markdown": {
                        "path": "/remote/failure/report.md",
                        "sha256": "f" * 64,
                    },
                },
            },
            "receipt_payload_sha256",
        )
        publication_path = root / "publication.json"
        publish(publication_path, publication)
        return publication_path, summary_path, prepublish_path, publication

    def test_source_binding_accepts_exact_local_mirrors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication, summary, prepublish, receipt = (
                self.source_files(root)
            )
            binding, loaded_summary = (
                postpublication.validate_v1_source_binding(
                    publication_receipt_path=publication,
                    summary_path=summary,
                    prepublish_receipt_path=prepublish,
                )
            )
            self.assertEqual(
                binding["v1_publication_receipt_sha256"],
                aggregate.sha256_path(publication),
            )
            self.assertEqual(
                binding["v1_accepted_result_payloads_sha256"],
                "a" * 64,
            )
            self.assertEqual(binding["v1_source_git_commit"], "b" * 40)
            self.assertEqual(binding["v1_run_id"], "immutable-run")
            self.assertEqual(binding["v1_gallery_sha256"], "c" * 64)
            self.assertEqual(
                binding["v1_failure_report_sha256"], "e" * 64
            )
            self.assertEqual(
                binding["v1_publication_receipt_payload_sha256"],
                receipt["receipt_payload_sha256"],
            )
            self.assertEqual(
                loaded_summary["schema_version"],
                postpublication.SUMMARY_SCHEMA_V1,
            )

    def test_source_binding_rejects_summary_changed_after_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication, summary, prepublish, _ = self.source_files(root)
            changed = json.loads(summary.read_text(encoding="utf-8"))
            changed["population"]["no_results_dropped"] = False
            publish(summary, changed)
            with self.assertRaisesRegex(
                postpublication.PostpublicationError,
                "summary is not complete",
            ):
                postpublication.validate_v1_source_binding(
                    publication_receipt_path=publication,
                    summary_path=summary,
                    prepublish_receipt_path=prepublish,
                )

    def test_source_binding_rejects_wrong_runtime_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication, summary, prepublish, _ = self.source_files(root)
            with self.assertRaisesRegex(
                postpublication.PostpublicationError,
                "differs from the accepted runtime",
            ):
                postpublication.validate_v1_source_binding(
                    publication_receipt_path=publication,
                    summary_path=summary,
                    prepublish_receipt_path=prepublish,
                    expected_source_commit="1" * 40,
                )

    def test_source_binding_rejects_receipt_payload_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication, summary, prepublish, _ = self.source_files(root)
            changed = json.loads(
                publication.read_text(encoding="utf-8")
            )
            changed["run_id"] = "different"
            publish(publication, changed)
            with self.assertRaisesRegex(
                postpublication.PostpublicationError,
                "payload hash differs",
            ):
                postpublication.validate_v1_source_binding(
                    publication_receipt_path=publication,
                    summary_path=summary,
                    prepublish_receipt_path=prepublish,
                )


if __name__ == "__main__":
    unittest.main()
