from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    sys.path.insert(0, str(ROOT / "main"))
    from crfs_oracle import r05a_constrained_flow_publication as publication
    from crfs_oracle.r05a_constrained_flow_canary import (
        constrained_flow_scientific_config_hash,
    )
except (ImportError, ModuleNotFoundError) as exc:  # dependency-free local gate.
    IMPORT_ERROR = exc


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"constrained-flow publication checks require allocation dependencies: {IMPORT_ERROR}",
)
class ConstrainedFlowPublicationTest(unittest.TestCase):
    maxDiff = None

    def test_draft_apparatus_cannot_be_used_as_execution_release(self) -> None:
        apparatus = json.loads(
            (ROOT / publication.APPARATUS_CONFIG_PATH).read_text(encoding="utf-8")
        )
        apparatus["ready_to_run"] = False
        apparatus["blocked_on"] = ["fixture_not_released"]
        apparatus["execution_release"] = None
        with self.assertRaisesRegex(ValueError, "not released"):
            publication._validate_execution_release(apparatus, run_id="fixture")

    def test_released_execution_identity_and_resources_are_exact(self) -> None:
        apparatus = json.loads(
            (ROOT / publication.APPARATUS_CONFIG_PATH).read_text(encoding="utf-8")
        )
        apparatus["ready_to_run"] = True
        apparatus["blocked_on"] = []
        apparatus["execution_release"] = {
            "schema_version": "1.0",
            "artifact_role": "r05a_constrained_flow_canary_execution_release",
            "decision_artifact": publication.RELEASE_DECISION_PATH,
            "accepted_implementation_commit": "a" * 40,
            "run_id": "fixture",
            "single_submission": True,
            "source_host": "worker-1",
            "resources": dict(publication.SOURCE_RESOURCE_CONTRACT),
            "release_only_parent_required": True,
            "allowed_release_diff_paths": list(publication.RELEASE_ONLY_PATHS),
            "automatic_resubmission_allowed": False,
            "automatic_next_experiment_allowed": False,
        }
        observed = publication._validate_execution_release(apparatus, run_id="fixture")
        self.assertEqual(observed, apparatus["execution_release"])
        self.assertEqual(
            publication.RELEASE_DECISION_PATH,
            "docs/decisions/0048-preserve-fd-diagnostic-and-normalize-terminal-transport.md",
        )
        self.assertEqual(len(publication.BOUND_REPOSITORY_PATHS), 66)
        for relative in (
            publication.FD_DIAGNOSTIC_A_EVIDENCE_PATH,
            publication.FD_DIAGNOSTIC_A_EVIDENCE_TEST_PATH,
            publication.WEBSOCKET_POLICY_SERVER_PATH,
            publication.MSGPACK_NUMPY_PATH,
            publication.WEBSOCKET_CLIENT_POLICY_PATH,
        ):
            self.assertIn(relative, publication.BOUND_REPOSITORY_PATHS)
        apparatus["fd_diagnostic_a_evidence_binding"]["evidence_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "diagnostic-A evidence binding changed"):
            publication._validate_execution_release(apparatus, run_id="fixture")
        apparatus["fd_diagnostic_a_evidence_binding"] = dict(
            publication.FD_DIAGNOSTIC_A_EVIDENCE_BINDING
        )
        apparatus["vinuni_h100_guide_contract"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "VinUni H100 guide contract changed"):
            publication._validate_execution_release(apparatus, run_id="fixture")
        apparatus["vinuni_h100_guide_contract"] = dict(
            publication.VINUNI_H100_GUIDE_CONTRACT
        )
        apparatus["execution_release"]["resources"]["host_memory_mib"] = 32768
        with self.assertRaisesRegex(ValueError, "resources changed"):
            publication._validate_execution_release(apparatus, run_id="fixture")

    def test_allocation_log_requires_every_registered_suite_once_and_zero_skips(self) -> None:
        counts = publication._load_allocation_test_counts(
            ROOT / publication.ALLOCATION_TEST_REGISTRY_PATH
        )
        self.assertEqual(sum(counts.values()), 99)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "allocation-focused-tests.log"
            lines: list[str] = []
            for suite, count in counts.items():
                lines.extend(
                    (
                        f"[{suite}] Ran {count} tests in 1.0s",
                        f"[{suite}] OK",
                        f"verified_test_suite={suite} expected={count} observed={count} skips=0 status=passed",
                    )
                )
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            digest, observed = publication._parse_allocation_test_log(
                log, expected_counts=counts
            )
            self.assertEqual(digest, _sha(log))
            self.assertEqual(observed, counts)
            log.write_text(
                log.read_text(encoding="utf-8")
                + "verified_test_suite=unregistered.py expected=1 observed=1 skips=0 status=passed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unregistered suite"):
                publication._parse_allocation_test_log(log, expected_counts=counts)

    def test_terminal_variants_may_omit_legacy_only_when_record_and_filesystem_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_dir = (Path(directory) / publication.CASE_ID).resolve()
            case_dir.mkdir()
            cfs = case_dir / "constrained-flow-payload.json"
            legacy = case_dir / "canary-payload.json"
            for variant in (
                "terminal_apparatus_failure",
                "terminal_finite_difference_rejection",
            ):
                value = {
                    "payload_variant": variant,
                    "legacy_payload": {
                        "path": str(legacy),
                        "exists": False,
                        "sha256": None,
                    },
                }
                _write_json(cfs, value)
                loaded, digest, legacy_loaded, legacy_digest = (
                    publication._load_raw_payload_pair(
                        constrained_path=cfs,
                        legacy_path=legacy,
                        case_dir=case_dir,
                    )
                )
                self.assertEqual(loaded, value)
                self.assertEqual(digest, _sha(cfs))
                self.assertIsNone(legacy_loaded)
                self.assertIsNone(legacy_digest)

            oom_value = dict(value)
            oom_value["failure"] = {
                "error_type": "RuntimeError",
                "message": "CUDA out of memory",
            }
            _write_json(cfs, oom_value)
            with self.assertRaisesRegex(ValueError, "out-of-memory"):
                publication._load_raw_payload_pair(
                    constrained_path=cfs, legacy_path=legacy, case_dir=case_dir
                )
            _write_json(cfs, value)

            _write_json(legacy, {"payload_type": "unexpected"})
            with self.assertRaisesRegex(ValueError, "absence is inconsistent"):
                publication._load_raw_payload_pair(
                    constrained_path=cfs, legacy_path=legacy, case_dir=case_dir
                )
            legacy.unlink()
            legacy.symlink_to(case_dir / "missing-legacy-target.json")
            with self.assertRaisesRegex(ValueError, "absence is inconsistent"):
                publication._load_raw_payload_pair(
                    constrained_path=cfs, legacy_path=legacy, case_dir=case_dir
                )

    def test_complete_comparison_requires_exact_legacy_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_dir = (Path(directory) / publication.CASE_ID).resolve()
            case_dir.mkdir()
            cfs = case_dir / "constrained-flow-payload.json"
            legacy = case_dir / "canary-payload.json"
            _write_json(legacy, {"payload_type": "legacy"})
            _write_json(
                cfs,
                {
                    "payload_variant": "complete_comparison",
                    "legacy_payload_sha256": "0" * 64,
                },
            )
            with self.assertRaisesRegex(ValueError, "legacy-byte binding"):
                publication._load_raw_payload_pair(
                    constrained_path=cfs, legacy_path=legacy, case_dir=case_dir
                )
            value = {
                "payload_variant": "complete_comparison",
                "legacy_payload_sha256": _sha(legacy),
            }
            _write_json(cfs, value)
            _, _, loaded, digest = publication._load_raw_payload_pair(
                constrained_path=cfs, legacy_path=legacy, case_dir=case_dir
            )
            self.assertEqual(loaded, {"payload_type": "legacy"})
            self.assertEqual(digest, _sha(legacy))

    def test_source_contract_binds_exact_paths_hashes_and_held_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            repository = base / "repo"
            repository.mkdir()
            bound = repository / "bound.txt"
            bound.write_text("bound\n", encoding="utf-8")
            experiment_root = base / "experiments"
            run_root = experiment_root / "fixture-run"
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            source_job = "123"
            commit = "a" * 40
            held = {
                "schema_version": "1.0",
                "artifact_role": "r05a_constrained_flow_canary_held_gpu_submission",
                "status": "sbatch_returned_held_gpu_id",
                "run_id": run_root.name,
                "git_commit": commit,
                "gpu_slurm_array_job_id": source_job,
                "gpu_slurm_array_task_id": 0,
                "exact_gpu_task_id": "123_0",
                "source_node": "worker-1",
                "resources": dict(publication.SOURCE_RESOURCE_CONTRACT),
                "released_at_receipt_time": False,
                "scientific_claim_allowed": False,
                "infeasibility_claim_allowed": False,
                "simulator_efficacy_claim_allowed": False,
                "probe_training_authorized": False,
                "timestamp_utc": "2026-07-16T00:00:00Z",
            }
            held_path = run_root / "held-gpu-submission.json"
            _write_json(held_path, held)
            live_preflight = run_root / "vinuni-preflight.txt"
            live_preflight.write_text("fresh VinUni preflight\n", encoding="utf-8")
            paths = {
                "run_root": str(run_root),
                "case_dir": str(case_dir),
                "legacy_payload": str(case_dir / "canary-payload.json"),
                "constrained_flow_payload": str(case_dir / "constrained-flow-payload.json"),
                "host_telemetry": str(case_dir / "host-cgroup-sampled-current.tsv"),
                "gpu_samples": str(case_dir / "gpu-memory-samples.csv"),
                "allocation_tests_log": str(case_dir / "allocation-focused-tests.log"),
                "hidden_candidate": str(case_dir / ".results.candidate.json"),
                "result": str(case_dir / "results.json"),
                "validation_receipt": str(run_root / "cpu-afterany-validation.json"),
            }
            contract = {
                "schema_version": "1.0",
                "artifact_role": "r05a_constrained_flow_canary_source_contract",
                "status": "gpu_held_sources_bound_before_cpu_submission",
                "run_id": run_root.name,
                "git_commit": commit,
                "git_dirty": False,
                "source_node": "worker-1",
                "gpu_slurm_array_job_id": source_job,
                "gpu_slurm_array_task_id": 0,
                "exact_gpu_task_id": "123_0",
                "resources": dict(publication.SOURCE_RESOURCE_CONTRACT),
                "artifact_paths": paths,
                "frozen_bindings": {"fixture": "bound"},
                "repository_file_sha256": {"bound.txt": _sha(bound)},
                "held_gpu_submission_path": str(held_path),
                "held_gpu_submission_sha256": _sha(held_path),
                "live_preflight_path": str(live_preflight),
                "live_preflight_sha256": _sha(live_preflight),
                "scientific_claim_allowed": False,
                "infeasibility_claim_allowed": False,
                "simulator_efficacy_claim_allowed": False,
                "probe_training_authorized": False,
                "timestamp_utc": "2026-07-16T00:00:01Z",
            }
            contract_path = run_root / "source-contract.json"
            _write_json(contract_path, contract)
            with (
                mock.patch.object(publication, "EXPERIMENT_ROOT", experiment_root),
                mock.patch.object(
                    publication, "BOUND_REPOSITORY_PATHS", frozenset({"bound.txt"})
                ),
                mock.patch.object(
                    publication,
                    "_expected_frozen_bindings",
                    return_value={"fixture": "bound"},
                ),
            ):
                observed, observed_run, observed_case = publication._validate_source_contract(
                    contract_path,
                    expected_sha256=_sha(contract_path),
                    repository=repository,
                    source_job_id=source_job,
                    expected_git_commit=commit,
                )
                self.assertEqual(observed, contract)
                self.assertEqual(observed_run, run_root)
                self.assertEqual(observed_case, case_dir)
                live_preflight.write_text("mutated preflight\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "fresh VinUni preflight binding changed"
                ):
                    publication._validate_source_contract(
                        contract_path,
                        expected_sha256=_sha(contract_path),
                        repository=repository,
                        source_job_id=source_job,
                        expected_git_commit=commit,
                    )
                live_preflight.write_text("fresh VinUni preflight\n", encoding="utf-8")
                contract["frozen_bindings"]["fixture"] = "runtime-identity-changed"
                _write_json(contract_path, contract)
                with self.assertRaisesRegex(ValueError, "frozen source bindings changed"):
                    publication._validate_source_contract(
                        contract_path,
                        expected_sha256=_sha(contract_path),
                        repository=repository,
                        source_job_id=source_job,
                        expected_git_commit=commit,
                    )
                contract["frozen_bindings"]["fixture"] = "bound"
                contract["repository_file_sha256"]["bound.txt"] = "0" * 64
                _write_json(contract_path, contract)
                with self.assertRaisesRegex(ValueError, "repository binding changed"):
                    publication._validate_source_contract(
                        contract_path,
                        expected_sha256=_sha(contract_path),
                        repository=repository,
                        source_job_id=source_job,
                        expected_git_commit=commit,
                    )

    def test_terminal_raw_payload_builds_only_apparatus_inconclusive_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run_root = base / "terminal-fixture"
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            legacy = case_dir / "canary-payload.json"
            cfs = case_dir / "constrained-flow-payload.json"
            host = case_dir / "host-cgroup-sampled-current.tsv"
            gpu = case_dir / "gpu-memory-samples.csv"
            tests_log = case_dir / "allocation-focused-tests.log"
            source = run_root / "source-contract.json"
            submission = run_root / "submission.json"
            result = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            commit = "a" * 40
            job = "123"
            scientific_path = ROOT / publication.SCIENTIFIC_CONFIG_PATH
            legacy_config_path = ROOT / publication.LEGACY_CONFIG_PATH
            scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
            terminal = {
                "schema_version": "1.0",
                "payload_type": "r05a_constrained_flow_transport_canary_payload",
                "payload_variant": "terminal_apparatus_failure",
                "status": "apparatus_inconclusive",
                "case_id": publication.CASE_ID,
                "run_id": run_root.name,
                "config": {
                    "constrained_flow_path": str(scientific_path),
                    "constrained_flow_sha256": _sha(scientific_path),
                    "constrained_flow_scientific_hash": constrained_flow_scientific_config_hash(
                        scientific
                    ),
                    "legacy_path": str(legacy_config_path),
                    "legacy_sha256": _sha(legacy_config_path),
                },
                "failure": {
                    "stage": "paired_transport_execution",
                    "error_type": "RuntimeError",
                    "message": "registered apparatus failure",
                    "finite_failure_is_infeasibility": False,
                    "numeric_failure_is_method_negative": False,
                },
                "legacy_payload": {
                    "path": str(legacy),
                    "exists": False,
                    "sha256": None,
                },
                "provenance": {
                    "git_commit": commit,
                    "source_node": "worker-1",
                    "slurm_job_id": job,
                    "slurm_array_job_id": job,
                    "slurm_array_task_id": "0",
                    "timestamp_utc": "2026-07-16T00:00:00+00:00",
                },
                "outcome": {
                    "status": "apparatus_inconclusive",
                    "scientific_claim_allowed": False,
                    "infeasibility_claim_allowed": False,
                    "collision_or_progress_claim_allowed": False,
                    "probe_training_authorized": False,
                    "automatic_next_gate_authorized": False,
                },
                "simulator_use": {
                    "policy_generated_action_steps_executed": 0,
                    "teacher_generated_action_steps_executed": 0,
                    "efficacy_rollouts_executed": 0,
                    "simulator_efficacy_evaluated": False,
                },
            }
            _write_json(cfs, terminal)
            host.write_text("host\n", encoding="utf-8")
            gpu.write_text(
                "timestamp_ns,gpu_uuid,compute_used_mib,device_used_mib\n"
                "1,GPU-fixture,1,2\n",
                encoding="utf-8",
            )
            counts = publication._load_allocation_test_counts(
                ROOT / publication.ALLOCATION_TEST_REGISTRY_PATH
            )
            lines: list[str] = []
            for suite, count in counts.items():
                lines.extend(
                    (
                        f"[{suite}] Ran {count} tests in 1.0s",
                        f"[{suite}] OK",
                        f"verified_test_suite={suite} expected={count} observed={count} skips=0 status=passed",
                    )
                )
            tests_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _write_json(source, {"fixture": True})
            _write_json(submission, {"fixture": True})
            apparatus_sha = _sha(ROOT / publication.APPARATUS_CONFIG_PATH)
            schema_sha = _sha(ROOT / publication.ENVELOPE_SCHEMA_PATH)
            registry_sha = _sha(ROOT / publication.ALLOCATION_TEST_REGISTRY_PATH)
            contract = {
                "run_id": run_root.name,
                "repository_file_sha256": {
                    publication.APPARATUS_CONFIG_PATH: apparatus_sha,
                    publication.ENVELOPE_SCHEMA_PATH: schema_sha,
                },
            }

            def git_output(_repository, *args):
                if args == ("rev-parse", "HEAD"):
                    return commit
                if args == ("status", "--porcelain"):
                    return ""
                raise AssertionError(args)

            host_envelope = {
                "raw_trace_path": str(host),
                "raw_trace_sha256": _sha(host),
            }
            with (
                mock.patch.object(publication, "_git_output", side_effect=git_output),
                mock.patch.object(
                    publication, "_validate_execution_release", return_value={}
                ),
                mock.patch.object(publication, "_validate_release_commit"),
                mock.patch.object(
                    publication,
                    "_validate_apparatus_bindings",
                    return_value=(schema_sha, registry_sha),
                ),
                mock.patch.object(
                    publication,
                    "_validate_source_contract",
                    return_value=(contract, run_root, case_dir),
                ),
                mock.patch.object(
                    publication, "_validate_submission", return_value=_sha(submission)
                ),
                mock.patch.object(
                    publication,
                    "parse_full_lifetime_telemetry",
                    return_value={"raw_trace_sha256": _sha(host)},
                ),
                mock.patch.object(
                    publication, "_host_envelope", return_value=host_envelope
                ),
                mock.patch.object(publication, "_validate_schema", return_value=[]),
            ):
                envelope = publication.build_constrained_flow_envelope(
                    legacy_payload_path=legacy,
                    constrained_flow_payload_path=cfs,
                    host_telemetry_path=host,
                    gpu_samples_path=gpu,
                    allocation_tests_log=tests_log,
                    source_contract_path=source,
                    expected_source_contract_sha256=_sha(source),
                    submission_path=submission,
                    source_job_id=job,
                    source_job_state="COMPLETED",
                    source_exit_code="0:0",
                    publisher_job_id="124",
                    expected_git_commit=commit,
                    result_path=result,
                    receipt_path=receipt,
                    repo_root=ROOT,
                )
            self.assertEqual(envelope["status"], "apparatus_inconclusive")
            self.assertFalse(envelope["raw_payloads"]["legacy"]["present"])
            self.assertFalse(envelope["telemetry"]["gpu"]["process_peak_available"])
            self.assertEqual(
                envelope["telemetry"]["gpu"]["process_peak_owner"],
                "unavailable_no_legacy_trace",
            )
            self.assertIsNone(
                envelope["telemetry"]["gpu"]["process_peak_allocated_bytes"]
            )
            self.assertIsNone(
                envelope["telemetry"]["gpu"]["process_peak_reserved_bytes"]
            )
            self.assertEqual(
                envelope["telemetry"]["gpu"]["sampled_device_high_water_scope"],
                "full_cfs_policy_lifecycle_periodic_lower_bound",
            )
            self.assertTrue(
                envelope["raw_payloads"]["constrained_flow"][
                    "semantically_validated"
                ]
            )
            self.assertFalse(
                envelope["interpretation"][
                    "one_case_transport_mechanism_claim_allowed"
                ]
            )
            self.assertEqual(sum(envelope["allocation_tests"]["expected_counts"].values()), 99)

    def _publication_fixture(self, base: Path) -> tuple[dict, dict]:
        run_root = base / "run"
        case_dir = run_root / publication.CASE_ID
        case_dir.mkdir(parents=True)
        files = {
            "cfs": case_dir / "constrained-flow-payload.json",
            "tests": case_dir / "allocation-focused-tests.log",
            "host": case_dir / "host-cgroup-sampled-current.tsv",
            "gpu": case_dir / "gpu-memory-samples.csv",
            "contract": run_root / "source-contract.json",
            "submission": run_root / "submission.json",
        }
        for name, path in files.items():
            path.write_text(name + "\n", encoding="utf-8")
        result = case_dir / "results.json"
        receipt = run_root / "cpu-afterany-validation.json"
        envelope = {
            "raw_payloads": {
                "legacy": {"present": False, "path": str(case_dir / "canary-payload.json"), "sha256": None},
                "constrained_flow": {"path": str(files["cfs"]), "sha256": _sha(files["cfs"])},
            },
            "allocation_tests": {"log_path": str(files["tests"]), "log_sha256": _sha(files["tests"])},
            "telemetry": {
                "host": {"raw_trace_path": str(files["host"]), "raw_trace_sha256": _sha(files["host"])},
                "gpu": {"raw_samples_path": str(files["gpu"]), "raw_samples_sha256": _sha(files["gpu"])},
            },
            "publication": {
                "source_contract_receipt_path": str(files["contract"]),
                "source_contract_receipt_sha256": _sha(files["contract"]),
                "atomic_submission_receipt_path": str(files["submission"]),
                "atomic_submission_receipt_sha256": _sha(files["submission"]),
            },
            "interpretation": {"one_case_transport_mechanism_claim_allowed": False},
        }
        kwargs = {
            "result_path": result,
            "receipt_path": receipt,
            "repo_root": ROOT,
            "source_job_id": "123",
            "publisher_job_id": "124",
            "source_contract_path": files["contract"],
            "expected_source_contract_sha256": _sha(files["contract"]),
        }
        return envelope, kwargs

    def test_cpu_publisher_is_atomic_and_receipt_binds_published_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            envelope, kwargs = self._publication_fixture(Path(directory))
            with (
                mock.patch.object(
                    publication,
                    "build_constrained_flow_envelope",
                    return_value=envelope,
                ) as builder,
                mock.patch.object(publication, "_validate_schema", return_value=[]),
            ):
                result, receipt = publication.publish_constrained_flow_envelope(
                    **kwargs
                )
            self.assertEqual(builder.call_count, 2)
            self.assertTrue(result.is_file())
            self.assertTrue(receipt.is_file())
            receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt_value["result_sha256"], _sha(result))
            self.assertTrue(receipt_value["published"])

    def test_raw_mutation_between_candidate_builds_rolls_back_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            envelope, kwargs = self._publication_fixture(Path(directory))
            cfs_path = Path(envelope["raw_payloads"]["constrained_flow"]["path"])
            calls = 0

            def build(**_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    cfs_path.write_text("mutated\n", encoding="utf-8")
                return envelope

            with (
                mock.patch.object(
                    publication, "build_constrained_flow_envelope", side_effect=build
                ),
                mock.patch.object(publication, "_validate_schema", return_value=[]),
            ):
                with self.assertRaisesRegex(ValueError, "changed after candidate"):
                    publication.publish_constrained_flow_envelope(**kwargs)
            self.assertFalse(Path(kwargs["result_path"]).exists())
            self.assertFalse(Path(kwargs["receipt_path"]).exists())
            self.assertFalse(
                (Path(kwargs["result_path"]).parent / ".results.candidate.json").exists()
            )

    def test_cli_exposes_exact_cpu_afterany_contract(self) -> None:
        source = (
            ROOT / "main/publish_crfs_r05a_constrained_flow_canary.py"
        ).read_text(encoding="utf-8")
        for flag in (
            "--legacy-payload",
            "--constrained-flow-payload",
            "--host-telemetry",
            "--gpu-samples",
            "--allocation-tests-log",
            "--source-contract",
            "--expected-source-contract-sha256",
            "--submission",
            "--source-job-id",
            "--source-job-state",
            "--source-exit-code",
            "--publisher-job-id",
            "--expected-git-commit",
            "--output",
            "--validation-receipt",
        ):
            self.assertIn(f'parser.add_argument("{flag}", required=True)', source)


if __name__ == "__main__":
    unittest.main()
