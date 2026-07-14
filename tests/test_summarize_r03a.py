from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from crfs_harness.artifacts import atomic_write_json, content_hash, file_sha256
from main import summarize_r03a as SUMMARY
from tests import test_r03a_validation as CASE_FIXTURES


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/r03a_analytic_kill_test_eligible.jsonl"
CONFIG = ROOT / "configs/experiments/r03a_analytic_kill_test.json"
RUN_ID = "r03a-population-fixture"
CHECKPOINT_ID = "/fixture/checkpoint"
SOURCE_ARRAY_JOB_IDS_BY_HOST = {"worker-1": "7001", "worker-2": "7002"}
SUMMARY_JOB_ID = "7003"


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source(
    case: dict, *, oracle_passed: bool, delta_record: dict, source_host: str
) -> dict:
    base = CASE_FIXTURES.fixture()
    source_arms = {
        name: {"gate": {"passed": False}, "repeats": []}
        for name in SUMMARY.SOURCE_ARM_KEYS
    }
    source_arms["direct_witness"]["gate"]["passed"] = True
    source_arms["oracle_residual"]["gate"]["passed"] = oracle_passed
    source_arms["frozen"]["repeats"] = [
        {
            "start_eef_center_m": [0.1, 0.2, 0.3],
            "branch_obstacle_boxes": [
                {
                    "name": "fixture-obstacle",
                    "center_m": [0.4, 0.5, 0.6],
                    "rotation_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "half_size_m": [0.01, 0.02, 0.03],
                }
            ],
        }
    ]
    return {
        "case_id": case["case_id"],
        "status": "completed",
        "provenance": {
            "case_record": dict(case),
            "host": source_host,
            "noise": copy.deepcopy(base["provenance"]["noise"]),
        },
        "pairing": {
            "policy_observation": copy.deepcopy(
                base["pairing"]["policy_observation"]["source"]
            ),
            "branch_snapshot": copy.deepcopy(
                base["pairing"]["branch_snapshot"]["source"]
            ),
            "eager_actions": copy.deepcopy(base["arms"]["frozen"]["full_actions"]),
            "eager_trace": copy.deepcopy(base["arms"]["frozen"]["trace"]),
        },
        "directions": {
            "arrays": {"delta_star_model": copy.deepcopy(delta_record)},
            "l2_norms": {"delta_star_model": 1.0},
        },
        "arms": source_arms,
        "outcome": {
            "r01_feasible_conditioned": True,
            "nominal_collision_reproduced": True,
            "direct_witness_reconfirmed": True,
            "arm_gate_pass": {
                name: source_arms[name]["gate"]["passed"]
                for name in SUMMARY.SOURCE_ARM_KEYS
            },
        },
    }


def _source_pairing_records(source: dict) -> dict[str, dict]:
    repeat = source["arms"]["frozen"]["repeats"][0]
    return {
        "policy_observation": source["pairing"]["policy_observation"],
        "branch_snapshot": source["pairing"]["branch_snapshot"],
        "branch_geometry": SUMMARY._canonical_rollout_geometry(repeat),
        "noise": source["provenance"]["noise"],
        "source_frozen_actions": source["pairing"]["eager_actions"],
        "source_frozen_trace": source["pairing"]["eager_trace"],
    }


def _source_pairing_hashes(source: dict) -> dict[str, str]:
    records = _source_pairing_records(source)
    return {
        name: (
            record["sha256"]
            if name
            in {
                "policy_observation",
                "noise",
                "source_frozen_actions",
                "source_frozen_trace",
            }
            else content_hash(record)
        )
        for name, record in records.items()
    }


def _result(
    case: dict,
    source: dict,
    source_path: Path,
    *,
    case_index: int,
    config_hash: str,
    config_file_sha256: str,
    r03_summary_path: Path,
    mid_passed: bool,
    early_passed: bool,
    terminal: bool,
) -> dict:
    value = CASE_FIXTURES.fixture(terminal=terminal)
    value["case_id"] = case["case_id"]
    value["run_id"] = RUN_ID
    value["config_hash"] = config_hash
    value["provenance"]["case_record"] = dict(case)
    value["provenance"]["config_file_sha256"] = config_file_sha256
    value["provenance"]["checkpoint_id"] = CHECKPOINT_ID
    source_host = source["provenance"]["host"]
    value["provenance"]["host"] = source_host
    value["provenance"]["partition"] = "main"
    value["provenance"]["slurm_array_task_id"] = str(case_index)
    value["provenance"]["slurm_array_job_id"] = SOURCE_ARRAY_JOB_IDS_BY_HOST[
        source_host
    ]
    value["source_evidence"]["r03_summary_path"] = str(r03_summary_path)
    value["source_evidence"]["r02_case_path"] = str(source_path)
    value["source_evidence"]["r02_case_sha256"] = file_sha256(source_path)

    source_records = _source_pairing_records(source)
    for name, digest in _source_pairing_hashes(source).items():
        value["pairing"][name] = {
            "source": copy.deepcopy(source_records[name]),
            "fresh": copy.deepcopy(source_records[name]),
            "source_sha256": digest,
            "fresh_sha256": digest,
        }
    value["provenance"]["noise"] = copy.deepcopy(source["provenance"]["noise"])
    value["outcome"]["source_arm_gate_pass"] = {
        name: source["arms"][name]["gate"]["passed"]
        for name in SUMMARY.SOURCE_ARM_KEYS
    }
    if not terminal:
        value["arms"] = {
            "frozen": CASE_FIXTURES.arm("frozen", passed=False),
            "analytic_trajectory_mid": CASE_FIXTURES.arm(
                "analytic_trajectory_mid", passed=mid_passed
            ),
            "analytic_trajectory_early": CASE_FIXTURES.arm(
                "analytic_trajectory_early", passed=early_passed
            ),
        }
        value["arms"]["frozen"]["full_actions"] = copy.deepcopy(
            source["pairing"]["eager_actions"]
        )
        value["arms"]["frozen"]["trace"] = copy.deepcopy(
            source["pairing"]["eager_trace"]
        )
        value["arms"]["frozen"]["trace_sha256"] = source["pairing"][
            "eager_trace"
        ]["sha256"]
        value["outcome"]["arm_gate_pass"] = {
            name: value["arms"][name]["gate"]["passed"]
            for name in SUMMARY.EXPECTED_ARMS
        }
        value["outcome"]["arm_status"] = {
            name: value["arms"][name]["status"] for name in SUMMARY.EXPECTED_ARMS
        }
    return value


class PopulationFixture:
    def __init__(
        self,
        root: Path,
        *,
        mid_successes: int = 0,
        early_successes: int = 0,
        terminal_index: int | None = None,
        policy_failure_index: int | None = None,
        policy_failure_arm: str = "analytic_trajectory_mid",
    ) -> None:
        self.root = root
        self.source_root = root / "r02"
        self.results_root = root / "r03a"
        self.r03_summary_path = root / "r03-summary.json"
        atomic_write_json(self.r03_summary_path, {"fixture": True})
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config_hash = SUMMARY.VALIDATION.r03a_config_hash(config)
        config_file_sha256 = file_sha256(CONFIG)
        source_hashes: dict[str, str] = {}
        cases = _cases()
        for index, case in enumerate(cases):
            source_host = "worker-2" if index == 5 else "worker-1"
            delta_record = CASE_FIXTURES.fixture()["budget"]["source_delta_star_model"]
            source = _source(
                case,
                oracle_passed=index < SUMMARY.VALIDATION.PRIVILEGED_REFERENCE_SPS_COUNT,
                delta_record=delta_record,
                source_host=source_host,
            )
            source_path = self.source_root / case["case_id"] / "r02-paired.json"
            atomic_write_json(source_path, source)
            source_hashes[case["case_id"]] = file_sha256(source_path)
            result = _result(
                case,
                source,
                source_path,
                case_index=index,
                config_hash=config_hash,
                config_file_sha256=config_file_sha256,
                r03_summary_path=self.r03_summary_path,
                mid_passed=index < mid_successes,
                early_passed=index < early_successes,
                terminal=index == terminal_index,
            )
            if index == policy_failure_index:
                CASE_FIXTURES.bind_arm(
                    result,
                    policy_failure_arm,
                    CASE_FIXTURES.unrun_failure_arm(
                        policy_failure_arm, "policy_failure"
                    ),
                )
            result_path = (
                self.results_root / case["case_id"] / SUMMARY.RESULT_FILENAME
            )
            atomic_write_json(result_path, result)
        self.contract = SUMMARY.SummaryContract(
            manifest_cases=tuple(cases),
            manifest_sha256=file_sha256(MANIFEST),
            config_path=str(CONFIG),
            config_file_sha256=config_file_sha256,
            config_hash=config_hash,
            r03_summary_path=str(self.r03_summary_path),
            r03_summary_sha256=SUMMARY.VALIDATION.R03_SUMMARY_SHA256,
            r03_ordered_result_set_digest=(
                SUMMARY.VALIDATION.R03_ORDERED_RESULT_SET_DIGEST
            ),
            r02_result_hashes=source_hashes,
            r02_results_root=str(self.source_root),
            run_id=RUN_ID,
            checkpoint_id=CHECKPOINT_ID,
            checkpoint_sha256=SUMMARY.VALIDATION.CHECKPOINT_SHA256,
            expected_source_slurm_array_job_ids_by_host=(
                SOURCE_ARRAY_JOB_IDS_BY_HOST
            ),
            expected_git_commit="3" * 40,
            expected_summary_slurm_job_id=SUMMARY_JOB_ID,
        )
        fixed_launch = {
            "schema_version": "1.0",
            "scientific_claim_allowed": False,
            "run_id": RUN_ID,
            "git_commit": "3" * 40,
            "manifest_sha256": file_sha256(MANIFEST),
            "config_sha256": config_file_sha256,
            "schema_sha256": (
                "2f0f9db28ae9d62582e88bc70afe40ef4e37245a60190863e2f3419a135d70c3"
            ),
            "decision_sha256": SUMMARY.VALIDATION.DECISION_SHA256,
            "r03_summary_sha256": SUMMARY.VALIDATION.R03_SUMMARY_SHA256,
            "checkpoint_sha256": SUMMARY.VALIDATION.CHECKPOINT_SHA256,
        }
        reservation = {
            **fixed_launch,
            "artifact_role": "r03a_source_node_grouped_launch_reservation",
            "status": "reserved",
            "groups": {
                "worker-1": {"indices": "0-4,6-16", "concurrency": 1},
                "worker-2": {"indices": "5", "concurrency": 1},
            },
        }
        reservation_path = self.results_root / "launch-reservation.json"
        atomic_write_json(reservation_path, reservation)
        launch = {
            **fixed_launch,
            "artifact_role": "r03a_source_node_grouped_launch",
            "status": "held_validated",
            "launch_reservation_sha256": file_sha256(reservation_path),
            "groups": {
                host: {
                    "indices": "0-4,6-16" if host == "worker-1" else "5",
                    "concurrency": 1,
                    "slurm_array_job_id": job_id,
                }
                for host, job_id in SOURCE_ARRAY_JOB_IDS_BY_HOST.items()
            },
        }
        launch_path = self.results_root / "grouped-launch.json"
        atomic_write_json(launch_path, launch)
        summary_submission = {
            **fixed_launch,
            "artifact_role": "r03a_population_summary_submission",
            "status": "dependency_registered",
            "grouped_launch_sha256": file_sha256(launch_path),
            "source_slurm_array_job_ids_by_host": dict(
                SOURCE_ARRAY_JOB_IDS_BY_HOST
            ),
            "slurm_summary_job_id": SUMMARY_JOB_ID,
            "dependency": "afterany:7001:7002",
        }
        atomic_write_json(
            self.results_root / "summary-submission.json", summary_submission
        )

    def mark_as_real_allocation_evidence(self) -> None:
        for case in self.contract.manifest_cases:
            path = self.results_root / case["case_id"] / SUMMARY.RESULT_FILENAME
            result = json.loads(path.read_text(encoding="utf-8"))
            result["provenance"]["evidence_tier"] = SUMMARY.REAL_EVIDENCE_TIER
            result["provenance"]["git_dirty"] = False
            atomic_write_json(path, result)

    def summarize(self, *, allow_synthetic: bool = True) -> dict:
        return SUMMARY.summarize_r03a_population(
            MANIFEST,
            CONFIG,
            self.source_root,
            self.r03_summary_path,
            self.results_root,
            run_id=RUN_ID,
            checkpoint_id=CHECKPOINT_ID,
            checkpoint_sha256=SUMMARY.VALIDATION.CHECKPOINT_SHA256,
            expected_git_commit="3" * 40,
            expected_source_slurm_array_job_ids_by_host=(
                SOURCE_ARRAY_JOB_IDS_BY_HOST
            ),
            source_validator=lambda _value: [],
            contract_override=self.contract,
            allow_synthetic_implementation_evidence=allow_synthetic,
            require_schema_dependency=False,
        )


class SummarizeR03ATest(unittest.TestCase):
    def test_source_job_map_requires_two_distinct_numeric_ids(self) -> None:
        for value in (
            {"worker-1": "7001"},
            {"worker-1": "7001", "worker-2": "7001"},
            {"worker-1": "7001", "worker-2": "not-a-job"},
            {"worker-1": "7001", "worker-2": None},
        ):
            with self.subTest(value=value), self.assertRaises(
                SUMMARY.SummaryContractError
            ):
                SUMMARY._source_job_ids_by_host(value)  # type: ignore[arg-type]

    def test_source_geometry_hash_uses_sorted_padded_identity_not_raw_box_order(self) -> None:
        delta = CASE_FIXTURES.fixture()["budget"]["source_delta_star_model"]
        source = _source(
            _cases()[0],
            oracle_passed=True,
            delta_record=delta,
            source_host="worker-1",
        )
        boxes = source["arms"]["frozen"]["repeats"][0]["branch_obstacle_boxes"]
        boxes.append(
            {
                "name": "aaa-first-after-sort",
                "center_m": [0.7, 0.8, 0.9],
                "rotation_world": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "half_size_m": [0.02, 0.02, 0.02],
            }
        )
        forward = SUMMARY._source_pairing_hashes(source)["branch_geometry"]
        boxes.reverse()
        reversed_order = SUMMARY._source_pairing_hashes(source)["branch_geometry"]
        self.assertEqual(forward, reversed_order)

    def test_below_nine_makes_strong_analytic_mandatory_future_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(
                Path(directory), mid_successes=8, early_successes=8
            )
            summary = population.summarize()
        self.assertEqual(
            summary["kill_test"]["decision"],
            "strong_analytic_below_privileged_reference_mandatory_future_baseline",
        )
        self.assertIs(
            summary["kill_test"]["pure_learned_clearance_probe_necessity_rejected"],
            False,
        )
        self.assertEqual(summary["arms"]["analytic_trajectory_mid"]["successes"], 8)
        self.assertEqual(summary["arms"]["analytic_trajectory_early"]["successes"], 8)
        self.assertEqual(summary["arms"]["analytic_trajectory_mid"]["denominator"], 17)

    def test_either_arm_at_nine_rejects_pure_learned_clearance_necessity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(
                Path(directory), mid_successes=9, early_successes=3
            )
            summary = population.summarize()
        self.assertEqual(
            summary["kill_test"]["decision"],
            "pure_learned_clearance_probe_necessity_rejected",
        )
        self.assertIs(
            summary["kill_test"]["pure_learned_clearance_probe_necessity_rejected"],
            True,
        )
        self.assertEqual(summary["source_r03_reference"]["privileged_sps_count"], 9)
        self.assertIs(summary["probe_training_authorized"], False)
        self.assertIs(summary["confirmatory_r04_unblocked"], False)

    def test_all_cases_and_timing_samples_are_reported_without_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(
                Path(directory), mid_successes=4, early_successes=5
            )
            summary = population.summarize()
        self.assertEqual(summary["population"]["validated_cases"], 17)
        self.assertEqual(len(summary["population"]["case_ids"]), 17)
        source_reference = summary["source_r03_reference"]
        self.assertEqual(source_reference["privileged_arm"], "oracle_residual")
        self.assertEqual(
            {
                name: value["successes"]
                for name, value in source_reference["arms"].items()
            },
            {
                "frozen": 0,
                "direct_witness": 17,
                "random_residual": 0,
                "analytic_geometry_residual": 0,
                "oracle_residual": 9,
                "bridge_diagnostic": 0,
            },
        )
        self.assertEqual(
            source_reference["arms"]["direct_witness"]["success_case_ids"],
            summary["population"]["case_ids"],
        )
        self.assertEqual(
            len(source_reference["arms"]["oracle_residual"]["success_case_ids"]),
            9,
        )
        self.assertEqual(
            summary["identities"]["source_slurm_array_job_ids_by_host"],
            SOURCE_ARRAY_JOB_IDS_BY_HOST,
        )
        self.assertEqual(
            summary["identities"]["summary_slurm_job_id"], SUMMARY_JOB_ID
        )
        self.assertRegex(
            summary["identities"]["summary_submission_sha256"], r"^[0-9a-f]{64}$"
        )
        for arm_name in SUMMARY.ANALYTIC_ARMS:
            arm = summary["arms"][arm_name]
            self.assertEqual(len(arm["success_case_ids"]), arm["successes"])
            self.assertEqual(len(arm["failure_case_ids"]), 17 - arm["successes"])
            self.assertEqual(arm["policy_seconds"]["count"], 34)
        self.assertIs(summary["kill_test"]["both_arms_reported_without_selection"], True)

    def test_nominal_population_mismatch_blocks_the_kill_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(
                Path(directory), mid_successes=12, early_successes=12, terminal_index=0
            )
            population.mark_as_real_allocation_evidence()
            summary = population.summarize(allow_synthetic=False)
        self.assertEqual(summary["status"], "failed_population_integrity")
        self.assertIs(summary["scientific_evidence"], False)
        self.assertEqual(summary["kill_test"]["decision"], "not_evaluable_population_mismatch")
        self.assertIsNone(
            summary["kill_test"]["pure_learned_clearance_probe_necessity_rejected"]
        )
        self.assertEqual(
            summary["population"]["nominal_collision_not_reconfirmed_case_ids"],
            [summary["population"]["case_ids"][0]],
        )

    def test_policy_failure_keeps_denominator_but_makes_kill_test_not_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(
                Path(directory),
                mid_successes=8,
                early_successes=8,
                policy_failure_index=0,
            )
            population.mark_as_real_allocation_evidence()
            summary = population.summarize(allow_synthetic=False)
        case_id = summary["population"]["case_ids"][0]
        self.assertEqual(summary["status"], "failed_population_integrity")
        self.assertIs(summary["scientific_evidence"], False)
        self.assertEqual(summary["population"]["validated_cases"], 17)
        self.assertIs(summary["population"]["fixed_denominator_preserved"], True)
        self.assertEqual(summary["population"]["policy_failure_case_ids"], [case_id])
        self.assertEqual(summary["kill_test"]["decision"], "not_evaluable_policy_failure")
        self.assertIsNone(
            summary["kill_test"]["pure_learned_clearance_probe_necessity_rejected"]
        )

    def test_nonfinite_pre_reply_failure_remains_a_substantive_fixed_denominator_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(Path(directory), mid_successes=0, early_successes=0)
            case_id = population.contract.manifest_cases[0]["case_id"]
            result_path = population.results_root / case_id / SUMMARY.RESULT_FILENAME
            result = json.loads(result_path.read_text(encoding="utf-8"))
            CASE_FIXTURES.bind_arm(
                result,
                "analytic_trajectory_mid",
                CASE_FIXTURES.unrun_failure_arm(
                    "analytic_trajectory_mid", "nonfinite_failure"
                ),
            )
            atomic_write_json(result_path, result)
            summary = population.summarize()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["population"]["policy_failure_case_ids"], [])
        self.assertEqual(
            summary["arms"]["analytic_trajectory_mid"]["status_counts"],
            {"failed_gate": 16, "nonfinite_failure": 1},
        )
        self.assertEqual(
            summary["kill_test"]["decision"],
            "strong_analytic_below_privileged_reference_mandatory_future_baseline",
        )

    def test_synthetic_fixtures_cannot_be_misreported_as_scientific_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(Path(directory))
            with self.assertRaisesRegex(
                SUMMARY.SummaryContractError, "synthetic.*scientific summary"
            ):
                population.summarize(allow_synthetic=False)
            summary = population.summarize(allow_synthetic=True)
        self.assertIs(summary["scientific_evidence"], False)

    def test_summary_checkout_commit_must_equal_every_case_provenance_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            population = PopulationFixture(Path(directory))
            for case in population.contract.manifest_cases:
                path = population.results_root / case["case_id"] / SUMMARY.RESULT_FILENAME
                result = json.loads(path.read_text(encoding="utf-8"))
                result["provenance"]["git_commit"] = "4" * 40
                atomic_write_json(path, result)
            with self.assertRaisesRegex(
                SUMMARY.SummaryContractError,
                "provenance commit differs from the reviewed summary checkout",
            ):
                population.summarize()

    def test_grouped_launch_and_reservation_are_claim_bindings(self) -> None:
        for mutation in ("job", "commit", "reservation"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                population = PopulationFixture(Path(directory))
                launch_path = population.results_root / "grouped-launch.json"
                launch = json.loads(launch_path.read_text(encoding="utf-8"))
                if mutation == "job":
                    launch["groups"]["worker-2"]["slurm_array_job_id"] = "9999"
                elif mutation == "commit":
                    launch["git_commit"] = "4" * 40
                else:
                    launch["launch_reservation_sha256"] = "a" * 64
                atomic_write_json(launch_path, launch)
                with self.assertRaisesRegex(
                    SUMMARY.SummaryContractError, "grouped launch"
                ):
                    population.summarize()

    def test_summary_submission_is_a_claim_binding(self) -> None:
        for mutation in ("job", "dependency", "launch", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                population = PopulationFixture(Path(directory))
                path = population.results_root / "summary-submission.json"
                submission = json.loads(path.read_text(encoding="utf-8"))
                if mutation == "job":
                    submission["slurm_summary_job_id"] = "9999"
                elif mutation == "dependency":
                    submission["dependency"] = "afterany:7002:7001"
                elif mutation == "launch":
                    submission["grouped_launch_sha256"] = "a" * 64
                else:
                    path.unlink()
                if mutation != "missing":
                    atomic_write_json(path, submission)
                with self.assertRaisesRegex(
                    SUMMARY.SummaryContractError, "summary submission"
                ):
                    population.summarize()

    def test_missing_extra_and_cross_source_tampering_fail_closed(self) -> None:
        mutations = (
            "missing",
            "extra",
            "source_hash",
            "source_arm",
            "budget",
            "pairing",
            "source_job",
            "source_host",
            "source_task",
            "partition",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                population = PopulationFixture(Path(directory))
                case_id = population.contract.manifest_cases[0]["case_id"]
                result_path = population.results_root / case_id / SUMMARY.RESULT_FILENAME
                if mutation == "missing":
                    result_path.unlink()
                elif mutation == "extra":
                    extra = population.results_root / "extra-case" / SUMMARY.RESULT_FILENAME
                    atomic_write_json(extra, CASE_FIXTURES.fixture())
                else:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if mutation == "source_hash":
                        result["source_evidence"]["r02_case_sha256"] = "9" * 64
                    elif mutation == "source_arm":
                        result["outcome"]["source_arm_gate_pass"]["oracle_residual"] = False
                    elif mutation == "budget":
                        result["budget"]["source_reported_full_model_l2"] = 0.9
                    elif mutation == "pairing":
                        digest = "e" * 64
                        result["pairing"]["branch_geometry"]["source_sha256"] = digest
                        result["pairing"]["branch_geometry"]["fresh_sha256"] = digest
                    elif mutation == "source_job":
                        result["provenance"]["slurm_array_job_id"] = (
                            SOURCE_ARRAY_JOB_IDS_BY_HOST["worker-2"]
                        )
                    elif mutation == "source_host":
                        result["provenance"]["host"] = "worker-2"
                        result["provenance"]["slurm_array_job_id"] = (
                            SOURCE_ARRAY_JOB_IDS_BY_HOST["worker-2"]
                        )
                    elif mutation == "source_task":
                        result["provenance"]["slurm_array_task_id"] = "16"
                    elif mutation == "partition":
                        result["provenance"]["partition"] = "mig"
                    atomic_write_json(result_path, result)
                with self.assertRaises(SUMMARY.SummaryContractError):
                    population.summarize()


if __name__ == "__main__":
    unittest.main()
