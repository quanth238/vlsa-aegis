#!/usr/bin/env python3
"""Adversarial tests for the current-v2 postpublication transfer verifier."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from urllib.parse import quote

from scripts import vlsa_postpublication_transfer_manifest as helper


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def add_payload_hash(value: dict, field: str) -> dict:
    payload = dict(value)
    payload[field] = helper.sha256_bytes(
        helper.canonical_json_bytes(payload)
    )
    return payload


class SyntheticV2Population:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.run_id = "synthetic-vlsa-v2-population"
        self.source_commit = "1" * 40
        self.publisher_job_id = "9001"
        self.population_job_id = "9000"
        self.protocol_id = "synthetic-vlsa-v2"
        self.run_root = base / self.run_id
        self.inputs = base / "inputs"
        self.output_parent = base / "transfer-output"
        self.run_root.mkdir()
        self.inputs.mkdir()
        self.output_parent.mkdir()
        self.config_path = self.inputs / "protocol.json"
        self.manifest_path = self.inputs / "manifest.jsonl"
        self.manifest_receipt_path = self.inputs / "manifest-receipt.json"
        self.labels_path = self.inputs / "labels.jsonl"
        self.verifier_repository = base / "verifier-repository"
        self.verifier_script = (
            self.verifier_repository
            / "scripts"
            / "vlsa_postpublication_transfer_manifest.py"
        )
        self.publication_path = (
            self.run_root / "population-publication-receipt.json"
        )
        self.video_paths = []
        self.result_paths = []
        self.manifests = []
        self.labels = {}
        self._build_verifier_identity()
        self._build()

    @property
    def allocation(self) -> dict:
        return {
            "job_id": "9002",
            "node": "worker-1",
            "partition": "main",
            "cpus_per_task": 4,
            "memory_mib": 32768,
            "dependency": "afterok:{}".format(self.publisher_job_id),
            "gpu_allocation": False,
        }

    @property
    def verifier_identity(self) -> dict:
        return dict(self._verifier_identity)

    def _build_verifier_identity(self) -> None:
        self.verifier_script.parent.mkdir(parents=True)
        self.verifier_script.write_text(
            "#!/usr/bin/env python3\nprint('synthetic verifier')\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "init", "-q", str(self.verifier_repository)],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.verifier_repository),
                "add",
                self.verifier_script.relative_to(
                    self.verifier_repository
                ).as_posix(),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.verifier_repository),
                "-c",
                "user.name=Verifier Fixture",
                "-c",
                "user.email=verifier-fixture@example.invalid",
                "commit",
                "-qm",
                "Freeze synthetic verifier",
            ],
            check=True,
        )
        commit = subprocess.run(
            [
                "git",
                "-C",
                str(self.verifier_repository),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        digest = helper.stable_file_sha256_and_size(
            self.verifier_script, "synthetic verifier"
        )[0]
        self._verifier_identity = helper.validate_verifier_identity(
            expected_sha256=digest,
            expected_git_commit=commit,
            script_path=self.verifier_script,
            repository_root=self.verifier_repository,
        )

    def _compact_diagnostics(
        self, *, case_id: str, mode: str, arm: str
    ) -> dict:
        value = {
            "case_id": case_id,
            "mode": mode,
            "arm": arm,
            "result_status": "complete",
            "diagnostic_validation_sha256": "4" * 64,
            "failure_diagnostics_record_sha256": "5" * 64,
            "frozen_label_record_sha256": "6" * 64,
            "action_invariance_ledger_sha256": "7" * 64,
            "compact_evidence_sha256": "8" * 64,
            "contact_schema_version": helper.official.CONTACT_SCHEMA_V3,
            "contact_model_authority_schema_version": (
                helper.official.CONTACT_MODEL_AUTHORITY_SCHEMA_V2
            ),
        }
        return value

    def _build(self) -> None:
        config = {
            "schema_version": "vlsa_table1_translational_protocol.v1",
            "protocol_id": self.protocol_id,
            "arms": list(helper.EXPECTED_ARMS),
        }
        write_json(self.config_path, config)
        self.manifest_receipt_path.write_text(
            '{"synthetic":"manifest-receipt"}\n', encoding="utf-8"
        )

        manifest_lines = []
        label_lines = []
        publisher_inventory = []
        diagnostic_inventory = []
        accepted_payloads = []
        status_counts = Counter()
        for ordinal in range(helper.EXPECTED_CASES):
            case_id = "case-{:04d}".format(ordinal)
            task_index = ordinal // helper.CASES_PER_TASK
            manifest = {
                "schema_version": "vlsa_table1_population_case.v1",
                "protocol_id": self.protocol_id,
                "case_id": case_id,
                "case_ordinal": ordinal,
                "task_level_group_id": "group-{:02d}".format(task_index),
                "required_arms": list(helper.EXPECTED_ARMS),
                "max_steps": 2,
            }
            label = {
                "case_id": case_id,
                "obstacle_label": "synthetic obstacle",
            }
            self.manifests.append(manifest)
            self.labels[case_id] = label
            manifest_lines.append(
                helper.canonical_json_bytes(manifest) + b"\n"
            )
            label_lines.append(helper.canonical_json_bytes(label) + b"\n")
            for mode, arm in zip(helper.EXPECTED_MODES, helper.EXPECTED_ARMS):
                status = "complete"
                executed = 1
                if mode == "aegis" and ordinal == 0:
                    status = "method_failure"
                    executed = 0
                elif mode == "aegis" and ordinal == 1:
                    status = "method_failure_passthrough"
                case_root = (
                    self.run_root
                    / "tasks"
                    / "task-{}".format(task_index)
                    / "results"
                    / mode
                    / case_id
                )
                case_root.mkdir(parents=True, exist_ok=True)
                video_path = case_root / "episode.mp4"
                video_payload = (
                    "synthetic-video:{}:{}\n".format(case_id, mode)
                ).encode("ascii")
                video_path.write_bytes(video_payload)
                video_sha256 = helper.sha256_bytes(video_payload)
                compact = self._compact_diagnostics(
                    case_id=case_id, mode=mode, arm=arm
                )
                compact["result_status"] = status
                result = {
                    "schema_version": helper.RESULT_SCHEMA,
                    "protocol_id": self.protocol_id,
                    "case_id": case_id,
                    "case_ordinal": ordinal,
                    "mode": mode,
                    "arm": arm,
                    "status": status,
                    "scientific_result": True,
                    "case": manifest,
                    "actions": [{} for _ in range(executed)],
                    "metrics": {"executed_action_count": executed},
                    "video": {
                        "path": "{}/{}/episode.mp4".format(mode, case_id),
                        "sha256": video_sha256,
                        "frames": executed + int(status == "method_failure"),
                        "fps": 30,
                        "complete_episode": True,
                    },
                    "synthetic_compact_diagnostics": compact,
                }
                result["result_payload_sha256"] = helper.sha256_bytes(
                    helper.canonical_json_bytes(result)
                )
                result_path = case_root / "result.json"
                write_json(result_path, result)
                result_sha256 = helper.stable_file_sha256_and_size(
                    result_path, "synthetic result"
                )[0]
                item = {
                    "relative_result_path": result_path.relative_to(
                        self.run_root
                    ).as_posix(),
                    "result_sha256": result_sha256,
                    "result_payload_sha256": result[
                        "result_payload_sha256"
                    ],
                    "video_sha256": video_sha256,
                    "case_id": case_id,
                    "arm": arm,
                    "status": status,
                    "case_ordinal": ordinal,
                    "task_index": task_index,
                    "policy_mode": mode,
                    "diagnostics": compact,
                }
                publisher_inventory.append(item)
                diagnostic_inventory.append(
                    {
                        "case_ordinal": ordinal,
                        "task_index": task_index,
                        **compact,
                    }
                )
                accepted_payloads.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "result_payload_sha256": result[
                            "result_payload_sha256"
                        ],
                    }
                )
                status_counts[status] += 1
                self.video_paths.append(video_path)
                self.result_paths.append(result_path)

        self.manifest_path.write_bytes(b"".join(manifest_lines))
        self.labels_path.write_bytes(b"".join(label_lines))
        self.publisher_inventory = publisher_inventory
        self.diagnostic_inventory = diagnostic_inventory
        accepted_payloads.sort(
            key=lambda row: (row["case_id"], row["arm"])
        )
        self.publisher_inventory_sha256 = helper.sha256_bytes(
            helper.canonical_json_bytes(publisher_inventory)
        )
        self.diagnostic_inventory_sha256 = helper.sha256_bytes(
            helper.canonical_json_bytes(diagnostic_inventory)
        )
        self.accepted_payloads_sha256 = helper.sha256_bytes(
            helper.canonical_json_bytes(accepted_payloads)
        )
        self.status_counts = dict(sorted(status_counts.items()))

        self.contract = {
            "schema_version": "vlsa_table1_run_contract.v1",
            "run_stage": "population",
            "run_id": self.run_id,
            "git_commit": self.source_commit,
            "config_sha256": helper.stable_file_sha256_and_size(
                self.config_path, "config"
            )[0],
            "manifest_sha256": helper.stable_file_sha256_and_size(
                self.manifest_path, "manifest"
            )[0],
            "manifest_receipt_sha256": (
                helper.stable_file_sha256_and_size(
                    self.manifest_receipt_path, "manifest receipt"
                )[0]
            ),
            "label_manifest_sha256": (
                helper.stable_file_sha256_and_size(
                    self.labels_path, "labels"
                )[0]
            ),
        }
        contract_path = self.run_root / "run-contract.tsv"
        contract_path.write_text(
            "".join(
                "{}\t{}\n".format(key, value)
                for key, value in self.contract.items()
            ),
            encoding="utf-8",
        )
        self.contract_record = {
            "path": str(contract_path.resolve()),
            "sha256": helper.stable_file_sha256_and_size(
                contract_path, "run contract"
            )[0],
        }

        publication_root = (
            self.run_root / "publication-attempts" / "job-9001"
        )
        failure_root = publication_root / "failure-analysis"
        gallery_root = publication_root / "gallery"
        failure_root.mkdir(parents=True)
        gallery_root.mkdir()
        prepublish_path = publication_root / "prepublish-validation.json"
        summary_path = publication_root / "population-summary.json"
        gallery_path = gallery_root / "index.html"

        failure_rows = []
        for manifest in self.manifests:
            row = {
                "schema_version": helper.FAILURE_CASE_SCHEMA,
                "case_id": manifest["case_id"],
                "case_ordinal": manifest["case_ordinal"],
            }
            failure_rows.append(
                add_payload_hash(row, "record_payload_sha256")
            )
        failure_cases_path = failure_root / "cases.jsonl"
        failure_cases_path.write_bytes(
            b"".join(
                helper.canonical_json_bytes(row) + b"\n"
                for row in failure_rows
            )
        )
        failure_report = add_payload_hash(
            {
                "schema_version": helper.FAILURE_REPORT_SCHEMA,
                "status": "complete_population_failure_analysis",
                "population": {
                    "cases": helper.EXPECTED_CASES,
                    "results": helper.EXPECTED_RESULTS,
                    "no_cases_dropped": True,
                    "all_car_failures_classified": True,
                    "causal_limits_preserved": True,
                },
                "counts": {
                    "case_count": helper.EXPECTED_CASES,
                    "video_count": helper.EXPECTED_RESULTS,
                    "unclassified_aegis_car_failures": 0,
                    "all_videos_hash_verified": True,
                },
                "case_record_ledger_sha256": helper.sha256_bytes(
                    helper.canonical_json_bytes(
                        [
                            {
                                "case_id": row["case_id"],
                                "record_payload_sha256": row[
                                    "record_payload_sha256"
                                ],
                            }
                            for row in failure_rows
                        ]
                    )
                ),
            },
            "report_payload_sha256",
        )
        failure_report_path = failure_root / "report.json"
        write_json(failure_report_path, failure_report)
        failure_markdown_path = failure_root / "report.md"
        failure_markdown_path.write_text(
            "# Synthetic exhaustive failure report\n", encoding="utf-8"
        )

        gallery_parts = ["<!doctype html>\n"]
        gallery_root_resolved = gallery_root.resolve()
        for manifest in self.manifests:
            gallery_parts.append(
                '<article class="case-card"><h2>{}</h2>\n'.format(
                    manifest["case_id"]
                )
            )
            ordinal = manifest["case_ordinal"]
            task_index = ordinal // helper.CASES_PER_TASK
            for mode in helper.EXPECTED_MODES:
                video_path = (
                    self.run_root
                    / "tasks"
                    / "task-{}".format(task_index)
                    / "results"
                    / mode
                    / manifest["case_id"]
                    / "episode.mp4"
                )
                href = quote(
                    Path(
                        os.path.relpath(
                            video_path.resolve(),
                            start=gallery_root_resolved,
                        )
                    ).as_posix(),
                    safe="/",
                )
                gallery_parts.append(
                    '<section class="arm-panel"><video controls>'
                    '<source src="{}" type="video/mp4"></video>'
                    '<a href="{}">Open MP4</a></section>\n'.format(
                        href, href
                    )
                )
            gallery_parts.append("</article>\n")
        gallery_path.write_text("".join(gallery_parts), encoding="utf-8")

        prepublish = add_payload_hash(
            {
                "schema_version": helper.PREPUBLISH_SCHEMA,
                "status": "validated",
                "scientific_result": False,
                "complete_paired_population": True,
                "no_results_dropped": True,
                "source_git_commit": self.source_commit,
                "run_id": self.run_id,
                "population_array_job_id": self.population_job_id,
                "run_contract": self.contract_record,
                "result_artifacts": {
                    "count": helper.EXPECTED_RESULTS,
                    "status_counts": self.status_counts,
                    "inventory_sha256": self.publisher_inventory_sha256,
                },
                "failure_diagnostics": {
                    "count": helper.EXPECTED_RESULTS,
                    "inventory_sha256": self.diagnostic_inventory_sha256,
                    "contact_schema_version": (
                        helper.official.CONTACT_SCHEMA_V3
                    ),
                    "contact_model_authority_schema_version": (
                        helper.official.CONTACT_MODEL_AUTHORITY_SCHEMA_V2
                    ),
                },
            },
            "receipt_payload_sha256",
        )
        write_json(prepublish_path, prepublish)
        summary = {
            "schema_version": helper.SUMMARY_SCHEMA,
            "status": "complete_population_validated",
            "protocol_id": self.protocol_id,
            "accepted_result_payloads_sha256": (
                self.accepted_payloads_sha256
            ),
            "population": {
                "cases": helper.EXPECTED_CASES,
                "arms": len(helper.EXPECTED_ARMS),
                "results": helper.EXPECTED_RESULTS,
                "task_level_groups": helper.EXPECTED_TASKS,
                "no_results_dropped": True,
            },
        }
        write_json(summary_path, summary)
        failure_report.pop("report_payload_sha256")
        failure_report["source"] = {
            "population_summary_sha256": (
                helper.stable_file_sha256_and_size(
                    summary_path, "synthetic summary"
                )[0]
            ),
            "population_validation_receipt_sha256": (
                helper.stable_file_sha256_and_size(
                    prepublish_path, "synthetic prepublish"
                )[0]
            ),
            "accepted_result_payloads_sha256": (
                self.accepted_payloads_sha256
            ),
        }
        failure_report = add_payload_hash(
            failure_report, "report_payload_sha256"
        )
        write_json(failure_report_path, failure_report)

        def artifact(path: Path) -> dict:
            return {
                "path": str(path.resolve()),
                "sha256": helper.stable_file_sha256_and_size(
                    path, "publication artifact"
                )[0],
            }

        prepublish_record = artifact(prepublish_path)
        prepublish_record["receipt_payload_sha256"] = prepublish[
            "receipt_payload_sha256"
        ]
        report_record = artifact(failure_report_path)
        report_record["report_payload_sha256"] = failure_report[
            "report_payload_sha256"
        ]
        cases_record = artifact(failure_cases_path)
        cases_record["count"] = helper.EXPECTED_CASES
        publication = add_payload_hash(
            {
                "schema_version": helper.PUBLICATION_SCHEMA,
                "status": "published",
                "scientific_result": False,
                "complete_paired_population": True,
                "no_results_dropped": True,
                "source_git_commit": self.source_commit,
                "run_id": self.run_id,
                "population_array_job_id": self.population_job_id,
                "publisher_slurm": {
                    "job_id": self.publisher_job_id,
                    "host": "worker-1",
                    "dependency": "afterany:{}".format(
                        self.population_job_id
                    ),
                },
                "prepublish_receipt": prepublish_record,
                "summary": artifact(summary_path),
                "gallery": artifact(gallery_path),
                "failure_analysis": {
                    "cases": cases_record,
                    "report": report_record,
                    "markdown": artifact(failure_markdown_path),
                },
            },
            "receipt_payload_sha256",
        )
        write_json(self.publication_path, publication)
        self.publication_sha256 = helper.stable_file_sha256_and_size(
            self.publication_path, "publication receipt"
        )[0]
        self.prepublish_path = prepublish_path
        self.summary_path = summary_path
        self.gallery_path = gallery_path
        self.failure_cases_path = failure_cases_path
        self.failure_report_path = failure_report_path
        self.failure_markdown_path = failure_markdown_path

    def _mock_validate_result(
        self,
        *,
        path,
        task_results_root,
        config,
        manifest,
        expected_commit,
    ):
        del task_results_root, config, expected_commit
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["case"] != manifest:
            raise RuntimeError("synthetic case mismatch")
        video_path = path.parent / "episode.mp4"
        video_sha256 = helper.stable_file_sha256_and_size(
            video_path, "synthetic video"
        )[0]
        if video_sha256 != result["video"]["sha256"]:
            raise RuntimeError("synthetic video SHA-256 mismatch")
        result_sha256 = helper.stable_file_sha256_and_size(
            path, "synthetic result"
        )[0]
        expected_payload = result.pop("result_payload_sha256")
        observed_payload = helper.sha256_bytes(
            helper.canonical_json_bytes(result)
        )
        result["result_payload_sha256"] = expected_payload
        if expected_payload != observed_payload:
            raise RuntimeError("synthetic result payload mismatch")
        return result, {
            "relative_result_path": path.as_posix(),
            "result_sha256": result_sha256,
            "result_payload_sha256": expected_payload,
            "video_sha256": video_sha256,
            "case_id": result["case_id"],
            "arm": result["arm"],
            "status": result["status"],
        }

    def patch_runtime(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                helper.official,
                "validate_run_contract",
                return_value=(self.contract, self.contract_record),
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.aggregate,
                "load_protocol",
                return_value=(
                    {
                        "protocol_id": self.protocol_id,
                        "arms": list(helper.EXPECTED_ARMS),
                    },
                    self.manifests,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.official,
                "_load_population_frozen_label_records",
                return_value=self.labels,
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.official,
                "_population_result_specs",
                return_value=tuple(
                    zip(helper.EXPECTED_MODES, helper.EXPECTED_ARMS)
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.official,
                "_validate_population_result",
                side_effect=self._mock_validate_result,
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.official,
                "_validate_population_result_binding",
                return_value=None,
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.failure_validation,
                "validate_diagnostic_result",
                side_effect=lambda result, **_: {
                    "case_id": result["case_id"],
                    "mode": result["mode"],
                },
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.official,
                "_compact_population_diagnostic_evidence",
                side_effect=lambda result, **_: result[
                    "synthetic_compact_diagnostics"
                ],
            )
        )
        stack.enter_context(
            mock.patch.object(
                helper.aggregate, "validate_pairs", return_value=None
            )
        )
        return stack

    def generate(self, name: str):
        with self.patch_runtime():
            return helper.generate_transfer_bundle(
                run_root=self.run_root,
                publication_receipt=self.publication_path,
                expected_publication_receipt_sha256=(
                    self.publication_sha256
                ),
                expected_run_id=self.run_id,
                expected_source_commit=self.source_commit,
                expected_population_array_job_id=self.population_job_id,
                expected_publisher_job_id=self.publisher_job_id,
                config_path=self.config_path,
                manifest_path=self.manifest_path,
                manifest_receipt_path=self.manifest_receipt_path,
                labels_path=self.labels_path,
                output_dir=self.output_parent / name,
                allocation=self.allocation,
                verifier_identity=self.verifier_identity,
                generated_utc="2026-07-18T00:00:00Z",
            )

    def validate_and_enumerate(self):
        with self.patch_runtime():
            chain = helper.validate_publication_chain(
                run_root=self.run_root,
                publication_receipt=self.publication_path,
                expected_publication_receipt_sha256=(
                    self.publication_sha256
                ),
                expected_run_id=self.run_id,
                expected_source_commit=self.source_commit,
                expected_population_array_job_id=self.population_job_id,
                expected_publisher_job_id=self.publisher_job_id,
                config_path=self.config_path,
                manifest_path=self.manifest_path,
                manifest_receipt_path=self.manifest_receipt_path,
                labels_path=self.labels_path,
            )
            enumeration = helper.reconstruct_v2_inventory(chain)
        return chain, enumeration


class TransferManifestV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="vlsa-postpublication-v2-tests-"
        )
        cls.fixture = SyntheticV2Population(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_00_exact_cpu_afterok_gate(self) -> None:
        environment = {
            "SLURM_JOB_ID": "9002",
            "SLURMD_NODENAME": "worker-1",
            "SLURM_JOB_PARTITION": "main",
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_MEM_PER_NODE": "32768",
            "SLURM_JOB_DEPENDENCY": "afterok:9001",
            "CUDA_VISIBLE_DEVICES": "NoDevFiles",
        }
        allocation = helper.validate_cpu_allocation(
            environment,
            expected_publisher_job_id="9001",
            actual_hostname="worker-1",
        )
        self.assertFalse(allocation["gpu_allocation"])
        environment["SLURM_JOB_DEPENDENCY"] = "afterany:9001"
        with self.assertRaisesRegex(
            helper.TransferVerificationError, "dependency"
        ):
            helper.validate_cpu_allocation(
                environment,
                expected_publisher_job_id="9001",
                actual_hostname="worker-1",
            )
        environment["SLURM_JOB_DEPENDENCY"] = "afterok:9001"
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(
            helper.TransferVerificationError, "CPU-only"
        ):
            helper.validate_cpu_allocation(
                environment,
                expected_publisher_job_id="9001",
                actual_hostname="worker-1",
            )

    def test_03_injected_allocation_cannot_bypass_cpu_contract(self) -> None:
        valid = self.fixture.allocation
        self.assertEqual(
            helper.validate_allocation_record(
                valid,
                expected_publisher_job_id=self.fixture.publisher_job_id,
            ),
            valid,
        )
        for field, changed in (
            ("dependency", "afterany:9001"),
            ("partition", "gpu"),
            ("cpus_per_task", 8),
            ("memory_mib", 65536),
            ("gpu_allocation", True),
            ("node", "worker-3"),
        ):
            with self.subTest(field=field):
                invalid = dict(valid)
                invalid[field] = changed
                with self.assertRaises(
                    helper.TransferVerificationError
                ):
                    helper.validate_allocation_record(
                        invalid,
                        expected_publisher_job_id=(
                            self.fixture.publisher_job_id
                        ),
                    )

    def test_05_verifier_sha_commit_and_clean_tree_are_bound(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vlsa-verifier-identity-"
        ) as directory:
            root = Path(directory)
            script = root / "scripts" / "verifier.py"
            script.parent.mkdir()
            script.write_text("print('verified')\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "add", "scripts/verifier.py"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Verifier Test",
                    "-c",
                    "user.email=verifier@example.invalid",
                    "commit",
                    "-qm",
                    "Freeze verifier",
                ],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            digest = helper.stable_file_sha256_and_size(
                script, "test verifier"
            )[0]
            identity = helper.validate_verifier_identity(
                expected_sha256=digest,
                expected_git_commit=commit,
                script_path=script,
                repository_root=root,
            )
            self.assertFalse(identity["repository_dirty"])
            script.write_text("print('changed')\n", encoding="utf-8")
            with self.assertRaisesRegex(
                helper.TransferVerificationError, "SHA-256"
            ):
                helper.validate_verifier_identity(
                    expected_sha256=digest,
                    expected_git_commit=commit,
                    script_path=script,
                    repository_root=root,
                )

    def test_07_verifier_mutation_before_publication_is_rejected(self) -> None:
        target = self.fixture.verifier_script
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"# mutation\n")
            with self.assertRaisesRegex(
                helper.TransferVerificationError, "SHA-256"
            ):
                self.fixture.generate("mutated-verifier")
        finally:
            target.write_bytes(original)

    def test_10_complete_current_v2_3200_row_bundle(self) -> None:
        outcome = self.fixture.generate("complete")
        rows = [
            json.loads(line)
            for line in Path(outcome["manifest_path"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(rows), helper.EXPECTED_RESULTS)
        expected_row_keys = {
            "schema_version",
            "case_ordinal",
            "task_index",
            "task_level_group_id",
            "case_id",
            "arm",
            "mode",
            "status",
            "result",
            "video",
            "publisher_v2_inventory_item",
        }
        self.assertTrue(
            all(set(row) == expected_row_keys for row in rows)
        )
        self.assertTrue(
            all(
                isinstance(row["publisher_v2_inventory_item"], dict)
                for row in rows
            )
        )
        self.assertEqual(
            len({row["video"]["path"] for row in rows}),
            helper.EXPECTED_RESULTS,
        )
        self.assertTrue(
            all(
                rows[index : index + 2][0]["mode"] == "pi05"
                and rows[index : index + 2][1]["mode"] == "aegis"
                for index in range(0, len(rows), 2)
            )
        )
        self.assertTrue(
            all(
                row["schema_version"] == helper.ROW_SCHEMA
                and not Path(row["video"]["path"]).is_absolute()
                and ".." not in Path(row["video"]["path"]).parts
                for row in rows
            )
        )
        receipt = json.loads(
            Path(outcome["receipt_path"]).read_text(encoding="utf-8")
        )
        helper.verify_payload_hash(
            receipt,
            "receipt_payload_sha256",
            "transfer receipt",
        )
        self.assertEqual(receipt["schema_version"], helper.RECEIPT_SCHEMA)
        self.assertEqual(
            receipt["cross_checks"][
                "publisher_v2_result_inventory_sha256"
            ],
            self.fixture.publisher_inventory_sha256,
        )
        self.assertEqual(
            receipt["cross_checks"][
                "publisher_v2_diagnostic_inventory_sha256"
            ],
            self.fixture.diagnostic_inventory_sha256,
        )
        self.assertTrue(
            receipt["cross_checks"]["all_failure_artifacts_rehashed"]
        )
        self.assertEqual(
            receipt["verifier"], self.fixture.verifier_identity
        )
        self.assertFalse(receipt["transport"]["videos_copied"])

    def test_12_wrong_population_array_job_is_rejected(self) -> None:
        with self.fixture.patch_runtime():
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "population array job ID",
            ):
                helper.validate_publication_chain(
                    run_root=self.fixture.run_root,
                    publication_receipt=self.fixture.publication_path,
                    expected_publication_receipt_sha256=(
                        self.fixture.publication_sha256
                    ),
                    expected_run_id=self.fixture.run_id,
                    expected_source_commit=self.fixture.source_commit,
                    expected_population_array_job_id="wrong-job",
                    expected_publisher_job_id=(
                        self.fixture.publisher_job_id
                    ),
                    config_path=self.fixture.config_path,
                    manifest_path=self.fixture.manifest_path,
                    manifest_receipt_path=(
                        self.fixture.manifest_receipt_path
                    ),
                    labels_path=self.fixture.labels_path,
                )

    def test_15_timeout_finalizer_binds_distinct_artifact_publisher(
        self,
    ) -> None:
        recovery_job_id = "9003"
        authority_path = (
            self.fixture.run_root
            / "publication-attempts"
            / "job-{}".format(recovery_job_id)
            / "publisher-timeout-recovery-authority.json"
        )
        authority = add_payload_hash(
            {
                "schema_version": helper.TIMEOUT_RECOVERY_SCHEMA,
                "status": "validated",
                "scientific_result": False,
                "run_id": self.fixture.run_id,
                "population_array_job_id": self.fixture.population_job_id,
                "preserves_immutable_result_tree": True,
                "permits_inference_or_simulation": False,
                "recovery_scope": "population_finalize_only",
                "timed_out_publisher": {
                    "publisher_slurm": {
                        "job_id": self.fixture.publisher_job_id,
                        "host": "worker-1",
                        "dependency": "afterany:{}".format(
                            self.fixture.population_job_id
                        ),
                        "state": "TIMEOUT",
                        "exit_code": "0:0",
                    },
                    "attempt_root": str(
                        (
                            self.fixture.run_root
                            / "publication-attempts"
                            / "job-{}".format(
                                self.fixture.publisher_job_id
                            )
                        ).resolve()
                    ),
                    "final_receipt_missing": True,
                    "failure_receipt_missing": True,
                },
                "recovery_publisher_slurm": {
                    "job_id": recovery_job_id,
                    "host": "worker-2",
                    "dependency": "afterany:{}".format(
                        self.fixture.publisher_job_id
                    ),
                },
            },
            "receipt_payload_sha256",
        )
        original_publication = self.fixture.publication_path.read_bytes()
        try:
            write_json(authority_path, authority)
            publication = json.loads(original_publication)
            publication.pop("receipt_payload_sha256")
            publication["publisher_slurm"] = dict(
                authority["recovery_publisher_slurm"]
            )
            publication["publisher_timeout_recovery_authority"] = {
                "path": str(authority_path.resolve()),
                "sha256": helper.stable_file_sha256_and_size(
                    authority_path,
                    "synthetic timeout recovery authority",
                )[0],
                "receipt_payload_sha256": authority[
                    "receipt_payload_sha256"
                ],
            }
            publication = add_payload_hash(
                publication, "receipt_payload_sha256"
            )
            write_json(self.fixture.publication_path, publication)
            publication_sha256 = helper.stable_file_sha256_and_size(
                self.fixture.publication_path,
                "synthetic recovered publication",
            )[0]
            arguments = {
                "run_root": self.fixture.run_root,
                "publication_receipt": self.fixture.publication_path,
                "expected_publication_receipt_sha256": (
                    publication_sha256
                ),
                "expected_run_id": self.fixture.run_id,
                "expected_source_commit": self.fixture.source_commit,
                "expected_population_array_job_id": (
                    self.fixture.population_job_id
                ),
                "expected_publisher_job_id": recovery_job_id,
                "config_path": self.fixture.config_path,
                "manifest_path": self.fixture.manifest_path,
                "manifest_receipt_path": (
                    self.fixture.manifest_receipt_path
                ),
                "labels_path": self.fixture.labels_path,
            }
            with self.fixture.patch_runtime():
                chain = helper.validate_publication_chain(
                    **arguments,
                    expected_artifact_publisher_job_id=(
                        self.fixture.publisher_job_id
                    ),
                )
            self.assertEqual(
                chain["timeout_recovery_path"], authority_path.resolve()
            )
            self.assertEqual(
                chain["prepublish_path"],
                self.fixture.prepublish_path.resolve(),
            )
            with self.fixture.patch_runtime(), self.assertRaisesRegex(
                helper.TransferVerificationError,
                "timed-out publisher/job_id",
            ):
                helper.validate_publication_chain(**arguments)
        finally:
            self.fixture.publication_path.write_bytes(original_publication)
            try:
                authority_path.unlink()
                authority_path.parent.rmdir()
            except FileNotFoundError:
                pass

    def test_20_tampered_video_is_rejected(self) -> None:
        target = self.fixture.video_paths[0]
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(
                helper.TransferVerificationError, "video SHA-256"
            ):
                self.fixture.generate("tampered-video")
        finally:
            target.write_bytes(original)

    def test_25_missing_result_is_rejected(self) -> None:
        target = self.fixture.result_paths[-1]
        backup = target.with_suffix(".backup")
        target.rename(backup)
        try:
            with self.assertRaises(
                (helper.TransferVerificationError, FileNotFoundError)
            ):
                self.fixture.generate("missing-result")
        finally:
            backup.rename(target)

    def test_27_missing_video_is_rejected(self) -> None:
        target = self.fixture.video_paths[-1]
        backup = target.with_suffix(".backup")
        target.rename(backup)
        try:
            with self.assertRaises(
                (helper.TransferVerificationError, FileNotFoundError)
            ):
                self.fixture.generate("missing-video")
        finally:
            backup.rename(target)

    def test_30_extra_video_is_rejected(self) -> None:
        extra = (
            self.fixture.run_root
            / "tasks"
            / "task-0"
            / "results"
            / "pi05"
            / "extra-case"
            / "episode.mp4"
        )
        extra.parent.mkdir()
        extra.write_bytes(b"extra")
        try:
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "final video path set",
            ):
                self.fixture.generate("extra-video")
        finally:
            extra.unlink()
            extra.parent.rmdir()

    def test_32_extra_result_is_rejected(self) -> None:
        extra = (
            self.fixture.run_root
            / "tasks"
            / "task-0"
            / "results"
            / "pi05"
            / "extra-case"
            / "result.json"
        )
        extra.parent.mkdir()
        extra.write_text("{}\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "final result path set",
            ):
                self.fixture.generate("extra-result")
        finally:
            extra.unlink()
            extra.parent.rmdir()

    def test_35_symlink_under_tasks_is_rejected(self) -> None:
        link = (
            self.fixture.run_root
            / "tasks"
            / "task-0"
            / "results"
            / "forbidden-link"
        )
        link.symlink_to(self.fixture.video_paths[0])
        try:
            with self.assertRaisesRegex(
                helper.TransferVerificationError, "symlink"
            ):
                self.fixture.generate("task-symlink")
        finally:
            link.unlink()

    def test_40_failure_artifact_tamper_is_rejected(self) -> None:
        targets = (
            (
                self.fixture.failure_cases_path,
                "failure case ledger file SHA-256",
            ),
            (
                self.fixture.failure_report_path,
                "failure report file SHA-256",
            ),
            (
                self.fixture.failure_markdown_path,
                "failure report Markdown file SHA-256",
            ),
        )
        for target, message in targets:
            with self.subTest(target=target.name):
                original = target.read_bytes()
                try:
                    target.write_bytes(original + b"tamper")
                    with self.assertRaisesRegex(
                        helper.TransferVerificationError,
                        message,
                    ):
                        self.fixture.generate(
                            "tampered-failure-{}".format(target.name)
                        )
                finally:
                    target.write_bytes(original)

    def test_41_extra_failure_artifact_is_rejected(self) -> None:
        extra = self.fixture.failure_cases_path.parent / "unbound.txt"
        extra.write_text("extra\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "failure-analysis entry set",
            ):
                self.fixture.generate("extra-failure-artifact")
        finally:
            extra.unlink()

    def test_42_exact_cpu_afterok_28610_launch_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        batch_path = (
            root
            / "slurm"
            / "vlsa_postpublication_transfer_manifest.sbatch"
        )
        runner_path = (
            root
            / "slurm"
            / "run_vlsa_postpublication_transfer_manifest.sh"
        )
        submit_path = (
            root
            / "scripts"
            / "submit_vlsa_postpublication_transfer_manifest_28610.sh"
        )
        for path in (batch_path, runner_path, submit_path):
            subprocess.run(
                ["bash", "-n", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        batch = batch_path.read_text(encoding="utf-8")
        runner = runner_path.read_text(encoding="utf-8")
        submit = submit_path.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=32G",
            "#SBATCH --time=04:00:00",
            "#SBATCH --exclude=worker-3",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(directive, batch)
        self.assertNotIn("#SBATCH --gres", batch)
        self.assertNotIn("#SBATCH --gpus", batch)
        self.assertNotIn("#SBATCH --array", batch)
        self.assertIn("readonly PUBLISHER_JOB_ID=28610", runner)
        self.assertIn("readonly POPULATION_ARRAY_JOB_ID=28609", runner)
        self.assertIn(
            "1592aa59361f431ba96c6ddcbebcb596f6c20853",
            runner,
        )
        self.assertIn("afterok:$PUBLISHER_JOB_ID", runner)
        self.assertIn("EXPECTED_VERIFIER_SHA256", runner)
        self.assertIn("EXPECTED_VERIFIER_GIT_COMMIT", runner)
        self.assertIn("EXPECTED_RUNNER_SHA256", runner)
        self.assertIn("EXPECTED_SBATCH_SHA256", runner)
        self.assertIn(
            "vlsa_postpublication_transfer_manifest.py", runner
        )
        self.assertIn(
            "--expected-population-array-job-id", runner
        )
        self.assertIn("--expected-verifier-sha256", runner)
        self.assertIn("--expected-verifier-git-commit", runner)
        self.assertIn(
            "readonly POPULATION_ARRAY_JOB_ID=28609", submit
        )
        self.assertIn("readonly PUBLISHER_JOB_ID=28610", submit)
        self.assertIn("sbatch --parsable --hold", submit)
        self.assertIn(
            '--dependency="afterok:$PUBLISHER_JOB_ID"', submit
        )
        self.assertIn("scontrol show job -dd -o", submit)
        self.assertIn("Reason JobHeldUser", submit)
        self.assertIn("scontrol release \"$job_id\"", submit)
        self.assertIn("submission-receipt.tsv", submit)
        self.assertIn("release-receipt.tsv", submit)
        self.assertIn("JobState PENDING", submit)
        self.assertIn("MinMemoryNode 32G", submit)
        self.assertIn("TimeLimit 04:00:00", submit)
        self.assertIn("ExcNodeList worker-3", submit)
        self.assertIn("cpu_only\\ttrue", submit)
        self.assertIn("output_unused\\ttrue", submit)
        self.assertNotIn("--export=ALL", submit)
        self.assertNotIn("--dependency=afterany:", submit)
        self.assertIn("EXPECTED_RUNNER_SHA256", submit)
        self.assertIn("EXPECTED_SBATCH_SHA256", submit)
        self.assertIn("sacct -n -X -j", submit)
        self.assertLess(
            submit.index("sbatch --parsable --hold"),
            submit.index("scontrol release \"$job_id\""),
        )
        self.assertLess(
            submit.index("submission-receipt.tsv"),
            submit.index("scontrol release \"$job_id\""),
        )

    def test_45_gallery_link_mismatch_is_rejected(self) -> None:
        target = self.fixture.gallery_path
        original = target.read_text(encoding="utf-8")
        changed = original.replace("episode.mp4", "wrong.mp4", 1)
        target.write_text(changed, encoding="utf-8")
        publication = json.loads(
            self.fixture.publication_path.read_text(encoding="utf-8")
        )
        original_publication = self.fixture.publication_path.read_bytes()
        old_sha = self.fixture.publication_sha256
        try:
            publication["gallery"]["sha256"] = (
                helper.stable_file_sha256_and_size(target, "changed gallery")[
                    0
                ]
            )
            publication.pop("receipt_payload_sha256")
            publication = add_payload_hash(
                publication, "receipt_payload_sha256"
            )
            write_json(self.fixture.publication_path, publication)
            changed_publication_sha = (
                helper.stable_file_sha256_and_size(
                    self.fixture.publication_path, "changed publication"
                )[0]
            )
            self.fixture.publication_sha256 = changed_publication_sha
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "gallery video source inventory",
            ):
                self.fixture.generate("wrong-gallery-link")
        finally:
            self.fixture.publication_sha256 = old_sha
            target.write_text(original, encoding="utf-8")
            self.fixture.publication_path.write_bytes(original_publication)

    def test_50_v1_prepublish_is_rejected(self) -> None:
        original_prepublish = self.fixture.prepublish_path.read_bytes()
        original_publication = self.fixture.publication_path.read_bytes()
        prepublish = json.loads(original_prepublish)
        publication = json.loads(original_publication)
        old_sha = self.fixture.publication_sha256
        try:
            prepublish["schema_version"] = (
                "vlsa_table1_population_prepublish_validation.v1"
            )
            prepublish.pop("receipt_payload_sha256")
            prepublish = add_payload_hash(
                prepublish, "receipt_payload_sha256"
            )
            write_json(self.fixture.prepublish_path, prepublish)
            publication["prepublish_receipt"]["sha256"] = (
                helper.stable_file_sha256_and_size(
                    self.fixture.prepublish_path, "changed prepublish"
                )[0]
            )
            publication["prepublish_receipt"][
                "receipt_payload_sha256"
            ] = prepublish["receipt_payload_sha256"]
            publication.pop("receipt_payload_sha256")
            publication = add_payload_hash(
                publication, "receipt_payload_sha256"
            )
            write_json(self.fixture.publication_path, publication)
            self.fixture.publication_sha256 = (
                helper.stable_file_sha256_and_size(
                    self.fixture.publication_path, "changed publication"
                )[0]
            )
            with self.assertRaisesRegex(
                helper.TransferVerificationError, "prepublish/schema_version"
            ):
                self.fixture.generate("v1-prepublish")
        finally:
            self.fixture.publication_sha256 = old_sha
            self.fixture.prepublish_path.write_bytes(original_prepublish)
            self.fixture.publication_path.write_bytes(original_publication)

    def test_55_wrong_v2_inventory_is_rejected(self) -> None:
        original_prepublish = self.fixture.prepublish_path.read_bytes()
        original_publication = self.fixture.publication_path.read_bytes()
        original_failure_report = (
            self.fixture.failure_report_path.read_bytes()
        )
        prepublish = json.loads(original_prepublish)
        publication = json.loads(original_publication)
        failure_report = json.loads(original_failure_report)
        old_sha = self.fixture.publication_sha256
        try:
            prepublish["result_artifacts"]["inventory_sha256"] = "f" * 64
            prepublish.pop("receipt_payload_sha256")
            prepublish = add_payload_hash(
                prepublish, "receipt_payload_sha256"
            )
            write_json(self.fixture.prepublish_path, prepublish)
            changed_prepublish_sha = (
                helper.stable_file_sha256_and_size(
                    self.fixture.prepublish_path, "changed prepublish"
                )[0]
            )
            failure_report["source"][
                "population_validation_receipt_sha256"
            ] = changed_prepublish_sha
            failure_report.pop("report_payload_sha256")
            failure_report = add_payload_hash(
                failure_report, "report_payload_sha256"
            )
            write_json(
                self.fixture.failure_report_path, failure_report
            )
            publication["prepublish_receipt"]["sha256"] = (
                changed_prepublish_sha
            )
            publication["prepublish_receipt"][
                "receipt_payload_sha256"
            ] = prepublish["receipt_payload_sha256"]
            publication["failure_analysis"]["report"]["sha256"] = (
                helper.stable_file_sha256_and_size(
                    self.fixture.failure_report_path,
                    "changed failure report",
                )[0]
            )
            publication["failure_analysis"]["report"][
                "report_payload_sha256"
            ] = failure_report["report_payload_sha256"]
            publication.pop("receipt_payload_sha256")
            publication = add_payload_hash(
                publication, "receipt_payload_sha256"
            )
            write_json(self.fixture.publication_path, publication)
            self.fixture.publication_sha256 = (
                helper.stable_file_sha256_and_size(
                    self.fixture.publication_path, "changed publication"
                )[0]
            )
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "v2 publisher result inventory",
            ):
                self.fixture.generate("wrong-v2-inventory")
        finally:
            self.fixture.publication_sha256 = old_sha
            self.fixture.prepublish_path.write_bytes(original_prepublish)
            self.fixture.failure_report_path.write_bytes(
                original_failure_report
            )
            self.fixture.publication_path.write_bytes(original_publication)

    def test_60_post_enumeration_mutation_is_rejected(self) -> None:
        chain, enumeration = self.fixture.validate_and_enumerate()
        target = self.fixture.video_paths[-1]
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"late mutation")
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "changed after enumeration",
            ):
                helper.publish_bundle(
                    output_dir=self.fixture.output_parent / "late-mutation",
                    chain=chain,
                    enumeration=enumeration,
                    allocation=self.fixture.allocation,
                    verifier_identity=self.fixture.verifier_identity,
                    generated_utc="2026-07-18T00:00:00Z",
                )
        finally:
            target.write_bytes(original)

    def test_62_post_enumeration_extra_artifact_is_rejected(self) -> None:
        chain, enumeration = self.fixture.validate_and_enumerate()
        extra = (
            self.fixture.run_root
            / "tasks"
            / "task-0"
            / "results"
            / "aegis"
            / "late-extra"
            / "episode.mp4"
        )
        extra.parent.mkdir()
        extra.write_bytes(b"late extra")
        try:
            with self.assertRaisesRegex(
                helper.TransferVerificationError,
                "path set changed after enumeration",
            ):
                helper.publish_bundle(
                    output_dir=(
                        self.fixture.output_parent
                        / "late-extra-artifact"
                    ),
                    chain=chain,
                    enumeration=enumeration,
                    allocation=self.fixture.allocation,
                    verifier_identity=self.fixture.verifier_identity,
                    generated_utc="2026-07-18T00:00:00Z",
                )
        finally:
            extra.unlink()
            extra.parent.rmdir()

    def test_64_mutation_during_staging_is_rejected(self) -> None:
        chain, enumeration = self.fixture.validate_and_enumerate()
        target = self.fixture.video_paths[0]
        original = target.read_bytes()
        original_write = helper.write_exclusive
        mutated = False

        def write_then_mutate(path, payload):
            nonlocal mutated
            original_write(path, payload)
            if not mutated:
                target.write_bytes(original + b"staging mutation")
                mutated = True

        try:
            with mock.patch.object(
                helper,
                "write_exclusive",
                side_effect=write_then_mutate,
            ):
                with self.assertRaisesRegex(
                    helper.TransferVerificationError,
                    "changed after enumeration",
                ):
                    helper.publish_bundle(
                        output_dir=(
                            self.fixture.output_parent
                            / "staging-mutation"
                        ),
                        chain=chain,
                        enumeration=enumeration,
                        allocation=self.fixture.allocation,
                        verifier_identity=self.fixture.verifier_identity,
                        generated_utc="2026-07-18T00:00:00Z",
                    )
        finally:
            target.write_bytes(original)

    def test_66_verifier_mutation_during_staging_is_rejected(self) -> None:
        chain, enumeration = self.fixture.validate_and_enumerate()
        target = self.fixture.verifier_script
        original = target.read_bytes()
        original_write = helper.write_exclusive
        mutated = False

        def write_then_mutate(path, payload):
            nonlocal mutated
            original_write(path, payload)
            if not mutated:
                target.write_bytes(original + b"# staging mutation\n")
                mutated = True

        try:
            with mock.patch.object(
                helper,
                "write_exclusive",
                side_effect=write_then_mutate,
            ):
                with self.assertRaisesRegex(
                    helper.TransferVerificationError,
                    "SHA-256",
                ):
                    helper.publish_bundle(
                        output_dir=(
                            self.fixture.output_parent
                            / "verifier-staging-mutation"
                        ),
                        chain=chain,
                        enumeration=enumeration,
                        allocation=self.fixture.allocation,
                        verifier_identity=self.fixture.verifier_identity,
                        generated_utc="2026-07-18T00:00:00Z",
                    )
        finally:
            target.write_bytes(original)

    def test_70_no_replace_output_publication(self) -> None:
        existing = self.fixture.output_parent / "already-exists"
        existing.mkdir()
        with self.assertRaisesRegex(
            helper.TransferVerificationError, "already exists"
        ):
            self.fixture.generate("already-exists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
