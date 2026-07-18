from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from analysis import aggregate_safelibero_aegis as aggregate
from analysis import build_aegis_failure_report_v3 as v3
from analysis import build_safelibero_postpublication_v3 as builder
from analysis import render_safelibero_table1_comparison as table_report


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def payload_hash(value: dict) -> str:
    return aggregate.sha256_bytes(aggregate.canonical_json_bytes(value))


class PostpublicationV3BuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.config = self.root / "config.json"
        self.manifest = self.root / "manifest.jsonl"
        self.manifest_receipt = self.root / "manifest-receipt.json"
        for path in (self.config, self.manifest, self.manifest_receipt):
            path.write_text("{}\n", encoding="utf-8")
        self.v1_publication = self.root / "population-publication.json"
        self.prepublish = self.root / "prepublish.json"
        self.v1_publication.write_text('{"status":"published"}\n')
        self.prepublish.write_text('{"status":"validated"}\n')
        self.v2_summary = self.root / "population-summary-v2.json"
        source_v1 = {
            "v1_source_git_commit": builder.EXPECTED_RUNTIME_COMMIT,
            "v1_run_id": builder.EXPECTED_RUN_ID,
            "v1_publication_receipt_sha256": sha(self.v1_publication),
            "v1_prepublish_receipt_sha256": sha(self.prepublish),
            "v1_population_summary_sha256": "1" * 64,
            "v1_accepted_result_payloads_sha256": "2" * 64,
        }
        self.claim_scope = {
            "method_label": (
                "pi0.5 + AEGIS translational conditioned on frozen "
                "per-case Codex obstacle labels"
            )
        }
        self.summary = {
            "schema_version": v3.SOURCE_SUMMARY_SCHEMA,
            "status": v3.SOURCE_SUMMARY_STATUS,
            "population": {
                "cases": 1600,
                "arms": 2,
                "results": 3200,
                "task_level_groups": 32,
                "no_results_dropped": True,
            },
            "source_v1": source_v1,
            "claim_scope": self.claim_scope,
            "accepted_result_payloads_sha256": "2" * 64,
        }
        write_json(self.v2_summary, self.summary)
        self.v2_receipt = self.root / "analysis-v2-receipt.json"
        receipt = {
            "schema_version": table_report.ANALYSIS_RECEIPT_SCHEMA,
            "status": table_report.ANALYSIS_RECEIPT_STATUS,
            "scientific_result": False,
            "source_v1": source_v1,
            "population": {
                "cases": 1600,
                "results": 3200,
                "no_cases_dropped": True,
                "accepted_result_payloads_sha256": "2" * 64,
            },
            "claim_scope": self.claim_scope,
            "artifacts": {
                "summary": {
                    "path": str(
                        table_report.ANALYSIS_PUBLICATION_ROOT
                        / (
                            f"{builder.EXPECTED_RUN_ID}-publisher-"
                            f"{builder.EXPECTED_PUBLISHER_JOB_ID}"
                        )
                        / "population-summary-v2.json"
                    ),
                    "sha256": sha(self.v2_summary),
                    "bytes": self.v2_summary.stat().st_size,
                }
            },
        }
        receipt["receipt_payload_sha256"] = payload_hash(receipt)
        write_json(self.v2_receipt, receipt)
        self.v2_submission = self.root / "submission-receipt.tsv"
        self.v2_release = self.root / "release-receipt.tsv"
        self.v2_job_id = "29999"
        self.v2_git_commit = "3" * 40
        self.v3_job_id = "30001"
        self.v3_git_commit = "4" * 40
        self.output = self.root / "analysis-v3"
        repository_root = Path(builder.__file__).resolve().parents[1]
        self.runner_sha = sha(
            repository_root
            / "slurm/run_vlsa_postpublication_analysis_v3.sh"
        )
        self.sbatch_sha = sha(
            repository_root
            / "slurm/vlsa_postpublication_analysis_v3.sbatch"
        )
        self.submit_helper_sha = sha(
            repository_root
            / "scripts/submit_vlsa_postpublication_analysis_v3.sh"
        )
        self.v2_submission.write_text(
            "\n".join(
                (
                    "schema_version\t"
                    "vlsa_postpublication_analysis_v2_submission_receipt.v1",
                    "status\theld_validated",
                    f"job_id\t{self.v2_job_id}",
                    f"job_name\t{builder.EXPECTED_V2_JOB_NAME}",
                    "population_array_job_id\t28609",
                    "publisher_job_id\t28610",
                    "dependency\tafterok:28610",
                    f"source_git_commit\t{builder.EXPECTED_RUNTIME_COMMIT}",
                    f"analysis_git_commit\t{self.v2_git_commit}",
                    "cpu_only\ttrue",
                    "held_before_release\ttrue",
                    "output_unused\ttrue",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.v2_release.write_text(
            "\n".join(
                (
                    "schema_version\t"
                    "vlsa_postpublication_analysis_v2_release_receipt.v1",
                    "status\treleased",
                    f"job_id\t{self.v2_job_id}",
                    f"job_name\t{builder.EXPECTED_V2_JOB_NAME}",
                    "population_array_job_id\t28609",
                    "publisher_job_id\t28610",
                    "dependency\tafterok:28610",
                    f"submission_receipt_sha256\t{sha(self.v2_submission)}",
                    "output_unused_before_release\ttrue",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.v3_submission = self.root / "v3-submission-receipt.tsv"
        self.v3_release = self.root / "v3-release-receipt.tsv"
        self.v3_submission.write_text(
            "\n".join(
                (
                    "schema_version\t"
                    "vlsa_postpublication_analysis_v3_submission_receipt.v1",
                    "status\theld_validated",
                    f"run_id\t{builder.EXPECTED_RUN_ID}",
                    f"job_id\t{self.v3_job_id}",
                    f"job_name\t{builder.EXPECTED_V3_JOB_NAME}",
                    "population_array_job_id\t28609",
                    "publisher_job_id\t28610",
                    f"analysis_v2_job_id\t{self.v2_job_id}",
                    f"dependency\tafterok:{self.v2_job_id}",
                    f"source_git_commit\t{builder.EXPECTED_RUNTIME_COMMIT}",
                    f"analysis_v2_git_commit\t{self.v2_git_commit}",
                    f"analysis_v3_git_commit\t{self.v3_git_commit}",
                    f"summary_v2_sha256\t{sha(self.v2_summary)}",
                    f"analysis_v2_receipt_sha256\t{sha(self.v2_receipt)}",
                    f"v2_submission_receipt_sha256\t{sha(self.v2_submission)}",
                    f"v2_release_receipt_sha256\t{sha(self.v2_release)}",
                    "v1_publication_receipt_sha256\t"
                    f"{sha(self.v1_publication)}",
                    f"prepublish_receipt_sha256\t{sha(self.prepublish)}",
                    f"config_sha256\t{sha(self.config)}",
                    f"manifest_sha256\t{sha(self.manifest)}",
                    "manifest_receipt_sha256\t"
                    f"{sha(self.manifest_receipt)}",
                    "v3_module_sha256\t"
                    f"{sha(Path(v3.__file__).resolve())}",
                    "builder_sha256\t"
                    f"{sha(Path(builder.__file__).resolve())}",
                    f"runner_sha256\t{self.runner_sha}",
                    f"sbatch_sha256\t{self.sbatch_sha}",
                    f"submit_helper_sha256\t{self.submit_helper_sha}",
                    f"output_dir\t{self.output}",
                    "partition\tmain",
                    "account\tnormal",
                    "qos\tnormal",
                    "nodes\t1",
                    "ntasks\t1",
                    "cpus_per_task\t4",
                    "memory\t32G",
                    "time_limit\t04:00:00",
                    "gpus\t0",
                    "exclude\tworker-3",
                    "output_unused\ttrue",
                    "cpu_only\ttrue",
                    "held_before_release\ttrue",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.v3_release.write_text(
            "\n".join(
                (
                    "schema_version\t"
                    "vlsa_postpublication_analysis_v3_release_receipt.v1",
                    "status\treleased",
                    f"run_id\t{builder.EXPECTED_RUN_ID}",
                    f"job_id\t{self.v3_job_id}",
                    f"job_name\t{builder.EXPECTED_V3_JOB_NAME}",
                    "population_array_job_id\t28609",
                    "publisher_job_id\t28610",
                    f"analysis_v2_job_id\t{self.v2_job_id}",
                    f"dependency\tafterok:{self.v2_job_id}",
                    "release_action\treleased_exact_held_job",
                    "observed_state_before_action\tPENDING",
                    "observed_reason_before_action\tJobHeldUser",
                    "observed_state_after_action\tPENDING",
                    "observed_reason_after_action\tDependency",
                    "submission_receipt_sha256\t"
                    f"{sha(self.v3_submission)}",
                    "output_unused_before_release\ttrue",
                )
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self) -> dict:
        return {
            "config_path": self.config,
            "manifest_receipt_path": self.manifest_receipt,
            "manifest_path": self.manifest,
            "results_root": self.results,
            "v1_publication_receipt_path": self.v1_publication,
            "prepublish_receipt_path": self.prepublish,
            "summary_v2_path": self.v2_summary,
            "analysis_v2_receipt_path": self.v2_receipt,
            "analysis_v2_submission_receipt_path": self.v2_submission,
            "analysis_v2_release_receipt_path": self.v2_release,
            "analysis_v3_submission_receipt_path": self.v3_submission,
            "analysis_v3_release_receipt_path": self.v3_release,
            "output_root": self.output,
            "analysis_v2_job_id": self.v2_job_id,
            "analysis_v2_git_commit": self.v2_git_commit,
            "analysis_v3_job_id": self.v3_job_id,
            "analysis_v3_git_commit": self.v3_git_commit,
            "expected_summary_v2_sha256": sha(self.v2_summary),
            "expected_analysis_v2_receipt_sha256": sha(self.v2_receipt),
            "expected_analysis_v2_submission_receipt_sha256": sha(
                self.v2_submission
            ),
            "expected_analysis_v2_release_receipt_sha256": sha(
                self.v2_release
            ),
            "expected_analysis_v3_submission_receipt_sha256": sha(
                self.v3_submission
            ),
            "expected_analysis_v3_release_receipt_sha256": sha(
                self.v3_release
            ),
            "expected_v1_publication_receipt_sha256": sha(
                self.v1_publication
            ),
            "expected_prepublish_receipt_sha256": sha(self.prepublish),
            "expected_config_sha256": sha(self.config),
            "expected_manifest_sha256": sha(self.manifest),
            "expected_manifest_receipt_sha256": sha(
                self.manifest_receipt
            ),
            "expected_v3_module_sha256": sha(Path(v3.__file__).resolve()),
            "expected_builder_sha256": sha(Path(builder.__file__).resolve()),
            "expected_runner_sha256": self.runner_sha,
            "expected_sbatch_sha256": self.sbatch_sha,
            "expected_submit_helper_sha256": self.submit_helper_sha,
        }

    @staticmethod
    def fake_v3_builder(**kwargs) -> dict:
        cases = kwargs["cases_output_path"]
        report = kwargs["report_output_path"]
        markdown = kwargs["markdown_output_path"]
        cases.write_bytes(
            b"".join(
                (
                    json.dumps(
                        {
                            "case_id": f"case-{index:04d}",
                            "schema_version": v3.CASE_SCHEMA,
                        },
                        sort_keys=True,
                    ).encode()
                    + b"\n"
                )
                for index in range(1600)
            )
        )
        value = {
            "schema_version": v3.REPORT_SCHEMA,
            "status": "complete_postpublication_failure_analysis_v3",
            "population": {
                "cases": 1600,
                "results": 3200,
                "no_cases_dropped": True,
                "exact_decision_partition": True,
            },
            "interpretation_scope": {"causality": "not claimed"},
        }
        value["report_payload_sha256"] = payload_hash(value)
        write_json(report, value)
        markdown.write_text("# v3\n", encoding="utf-8")
        return value

    def test_receipt_is_written_last_and_binds_v2_and_v3(self) -> None:
        with mock.patch.object(
            builder.v3,
            "build_population_failure_artifacts_v3",
            side_effect=self.fake_v3_builder,
        ):
            receipt = builder.build_postpublication_v3(**self.arguments())
        receipt_path = self.output / "analysis-v3-receipt.json"
        self.assertTrue(receipt_path.is_file())
        self.assertEqual(receipt["status"], builder.RECEIPT_STATUS)
        self.assertEqual(receipt["population"]["cases"], 1600)
        self.assertEqual(
            receipt["source_v2"]["analysis_job_id"], self.v2_job_id
        )
        self.assertEqual(
            receipt["source_v2"]["summary"]["sha256"],
            sha(self.v2_summary),
        )
        self.assertEqual(
            receipt["implementation"]["accepted_v3_implementation_commit"],
            "1c92370cb5b1278a4fcdda81332ec8eed9a49f8b",
        )
        self.assertEqual(
            receipt["launch"]["analysis_job_id"], self.v3_job_id
        )
        self.assertEqual(
            receipt["launch"]["submission_receipt"]["sha256"],
            sha(self.v3_submission),
        )
        self.assertEqual(receipt["launch"]["publisher_job_id"], "28610")
        self.assertEqual(receipt["launch"]["resources"]["gpus"], 0)
        self.assertEqual(
            receipt["launch"]["reviewed_source_sha256"][
                "allocation_runner"
            ],
            self.runner_sha,
        )
        self.assertEqual(
            receipt["protocol_inputs"]["manifest"]["sha256"],
            sha(self.manifest),
        )
        payload = dict(receipt)
        expected = payload.pop("receipt_payload_sha256")
        self.assertEqual(expected, payload_hash(payload))

    def test_builder_binds_canonical_physical_gate_commit(self) -> None:
        self.assertEqual(
            builder.EXPECTED_V3_IMPLEMENTATION_COMMIT,
            "1c92370cb5b1278a4fcdda81332ec8eed9a49f8b",
        )

    def test_wrong_summary_hash_rejects_before_output(self) -> None:
        arguments = self.arguments()
        arguments["expected_summary_v2_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "analysis-v2 summary SHA-256 differs",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_wrong_v2_job_identity_rejects_before_output(self) -> None:
        arguments = self.arguments()
        arguments["analysis_v2_job_id"] = "30000"
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "submission receipt job_id differs",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_release_must_bind_exact_submission(self) -> None:
        self.v2_release.write_text(
            self.v2_release.read_text().replace(
                sha(self.v2_submission), "f" * 64
            )
        )
        arguments = self.arguments()
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "release does not bind its submission",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_changed_accepted_v3_module_rejects(self) -> None:
        arguments = self.arguments()
        arguments["expected_v3_module_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "accepted analysis-v3 module SHA-256 differs",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_v3_release_must_bind_exact_submission(self) -> None:
        self.v3_release.write_text(
            self.v3_release.read_text().replace(
                sha(self.v3_submission), "f" * 64
            ),
            encoding="utf-8",
        )
        arguments = self.arguments()
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "analysis-v3 release does not bind its submission",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_v3_job_identity_is_fail_closed(self) -> None:
        arguments = self.arguments()
        arguments["analysis_v3_job_id"] = "30002"
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "analysis-v3 submission receipt job_id differs",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_v3_release_semantics_are_fail_closed(self) -> None:
        self.v3_release.write_text(
            self.v3_release.read_text().replace(
                "observed_reason_after_action\tDependency",
                "observed_reason_after_action\tJobHeldUser",
            ),
            encoding="utf-8",
        )
        arguments = self.arguments()
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error,
            "release receipt retains a held state",
        ):
            builder.build_postpublication_v3(**arguments)
        self.assertFalse(self.output.exists())

    def test_mid_derivation_input_mutation_has_no_terminal_receipt(
        self,
    ) -> None:
        def mutate_config(**kwargs) -> dict:
            report = self.fake_v3_builder(**kwargs)
            self.config.write_text('{"mutated":true}\n', encoding="utf-8")
            return report

        with mock.patch.object(
            builder.v3,
            "build_population_failure_artifacts_v3",
            side_effect=mutate_config,
        ):
            with self.assertRaisesRegex(
                builder.PostpublicationV3Error,
                "protocol config SHA-256 differs",
            ):
                builder.build_postpublication_v3(**self.arguments())
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "analysis-v3-receipt.json").exists())

    def test_existing_output_is_never_resumed(self) -> None:
        self.output.mkdir()
        with self.assertRaisesRegex(
            builder.PostpublicationV3Error, "output root must be unused"
        ):
            builder.build_postpublication_v3(**self.arguments())

    def test_partial_failure_never_fabricates_terminal_receipt(self) -> None:
        with mock.patch.object(
            builder.v3,
            "build_population_failure_artifacts_v3",
            side_effect=RuntimeError("injected derivation failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "injected derivation failure"
            ):
                builder.build_postpublication_v3(**self.arguments())
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "analysis-v3-receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
