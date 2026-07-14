from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_endpoint_free", ROOT / "main/summarize_endpoint_free.py"
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)

CONFIG_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
R00_HASH = "c" * 64
RUN_ID = "r01-test"
MANIFEST = ROOT / "manifests/oracle_h05_colliding.jsonl"


def manifest_cases() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]


def artifact(
    case: dict,
    *,
    p_min_changed: bool,
    p_zero_changed: bool | None = None,
    nominal_reproduced: bool = True,
    p_min_miss: bool = False,
    p_zero_miss: bool = False,
    attempts: list[dict] | None = None,
) -> dict:
    if p_zero_changed is None:
        p_zero_changed = p_min_changed
    p_min_any = p_min_changed
    p_zero_any = p_zero_changed or p_min_changed
    return {
        "schema_version": "1.0",
        "gate": "R01",
        "case_id": case["case_id"],
        "run_id": RUN_ID,
        "status": (
            "nominal_collision_not_reproduced"
            if not nominal_reproduced
            else "verified_safe_progress"
            if p_min_changed
            else "no_verified_safe_progress"
        ),
        "config_hash": CONFIG_HASH,
        "provenance": {
            **{key: case[key] for key in (
                "task_suite",
                "safety_level",
                "task_index",
                "episode_index",
                "group_id",
                "environment_seed",
                "policy_seed",
                "random_control_seed",
            )},
            "case_record": dict(case),
            "git_commit": "d" * 40,
            "git_dirty": False,
            "baseline_commit": "e" * 40,
            "input_manifest_sha256": SUMMARY.file_sha256(MANIFEST),
            "checkpoint_sha256": CHECKPOINT_HASH,
            "r00_summary_sha256": R00_HASH,
            "slurm_job_id": "12345",
            "slurm_array_task_id": "0",
            "partition": "main",
            "host": "gpu01",
            "device": "0",
        },
        "calibration": {"summary_sha256": R00_HASH, "p_min_m": 0.01},
        "nominal": {"collision_reproduced": nominal_reproduced},
        "searches": {
            "p_min": {"outcome": "not_found_within_budget" if p_min_miss else "candidate_found"},
            "p_zero": {"outcome": "not_found_within_budget" if p_zero_miss else "candidate_found"},
        },
        "verification": {
            "p_min": {"attempts": list(attempts or [])},
            "p_zero": {"attempts": []},
        },
        "outcome": {
            "nominal_collision_reproduced": nominal_reproduced,
            "p_min_changed_witness_verified": p_min_changed,
            "p_zero_changed_witness_verified": p_zero_changed,
            "p_min_any_action_verified": p_min_any,
            "p_zero_any_action_verified": p_zero_any,
        },
    }


def write_population(root: Path, records: list[dict]) -> None:
    for value in records:
        path = root / value["case_id"] / SUMMARY.RESULT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


class EndpointFreeSummaryTest(unittest.TestCase):
    def summarize(self, root: Path, validator):
        return SUMMARY.summarize_endpoint_free_population(
            MANIFEST,
            root,
            expected_run_id=RUN_ID,
            expected_config_hash=CONFIG_HASH,
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            expected_r00_summary_sha256=R00_HASH,
            repo_root=ROOT,
            validator=validator,
        )

    def test_exact_population_with_twelve_changed_p_min_witnesses_passes(self) -> None:
        cases = manifest_cases()
        records = []
        for index, case in enumerate(cases):
            attempts = []
            if index < 12:
                attempts.append({"simulator_safety_pass": True})
            if index == 13:
                attempts.append(
                    {
                        "simulator_safety_pass": True,
                        "proxy_clearance_false_safe": True,
                        "proxy_clearance_false_negative": True,
                        "proxy_joint_false_negative_at_search_threshold": True,
                    }
                )
            records.append(
                artifact(
                    case,
                    p_min_changed=index < 12,
                    p_zero_changed=index < 13,
                    p_min_miss=index == 14,
                    p_zero_miss=index in {14, 15},
                    attempts=attempts,
                )
            )
        calls = []

        def validator(value):
            calls.append(value["case_id"])
            return []

        with tempfile.TemporaryDirectory() as directory:
            write_population(Path(directory), records)
            summary = self.summarize(Path(directory), validator)

        self.assertEqual(len(calls), 20)
        self.assertTrue(summary["population"]["valid"])
        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["counts"]["changed_action_p_min_gate_numerator"], 12)
        self.assertEqual(summary["counts"]["changed_action_p_zero_only_rescues"], 1)
        self.assertEqual(summary["counts"]["proxy_clearance_false_safe_attempts"], 1)
        self.assertEqual(summary["counts"]["proxy_clearance_false_negative_attempts"], 1)
        self.assertEqual(summary["counts"]["proxy_joint_false_negative_attempts"], 1)
        self.assertEqual(summary["counts"]["bounded_search_miss_p_min"], 1)
        self.assertEqual(summary["counts"]["bounded_search_miss_p_zero"], 2)
        self.assertEqual(summary["counts"]["bounded_search_miss_both"], 1)

    def test_population_mismatch_zeroes_gate_numerator(self) -> None:
        cases = manifest_cases()
        records = [artifact(case, p_min_changed=True) for case in cases[:-1]]
        extra = artifact(cases[0], p_min_changed=True)
        extra["case_id"] = "unexpected-case"
        extra["provenance"]["case_record"] = {**cases[0], "case_id": "unexpected-case"}
        records.append(extra)

        with tempfile.TemporaryDirectory() as directory:
            write_population(Path(directory), records)
            summary = self.summarize(Path(directory), lambda _value: [])

        self.assertFalse(summary["population"]["valid"])
        self.assertEqual(summary["status"], "invalid_population")
        self.assertEqual(summary["counts"]["changed_action_p_min_gate_numerator"], 0)
        self.assertEqual(summary["counts"]["changed_action_p_min_rescues_observed"], 19)
        self.assertEqual(len(summary["population"]["missing_case_ids"]), 1)
        self.assertEqual(summary["population"]["unexpected_case_ids"], ["unexpected-case"])

    def test_hash_or_dirty_provenance_mismatch_fails_closed(self) -> None:
        records = [artifact(case, p_min_changed=True) for case in manifest_cases()]
        records[0]["config_hash"] = "d" * 64
        records[1]["provenance"]["git_dirty"] = True

        with tempfile.TemporaryDirectory() as directory:
            write_population(Path(directory), records)
            summary = self.summarize(Path(directory), lambda _value: [])

        self.assertFalse(summary["gate_passed"])
        self.assertEqual(summary["counts"]["changed_action_p_min_gate_numerator"], 0)
        self.assertEqual(len(summary["population"]["invalid_artifacts"]), 2)

    def test_bounded_miss_language_avoids_nonexistence_claim(self) -> None:
        records = [
            artifact(case, p_min_changed=False, p_min_miss=True, p_zero_miss=True)
            for case in manifest_cases()
        ]
        with tempfile.TemporaryDirectory() as directory:
            write_population(Path(directory), records)
            summary = self.summarize(Path(directory), lambda _value: [])
        serialized = json.dumps(summary, sort_keys=True).lower()
        self.assertEqual(summary["status"], "failed_threshold")
        self.assertNotIn("infeasible", serialized)
        self.assertIn("not non-existence certificates", summary["decision"])

    def test_cpu_slurm_template_invokes_atomic_summarizer_inputs(self) -> None:
        source = (ROOT / "slurm/endpoint_free_summary.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=2", source)
        self.assertIn("#SBATCH --mem=16G", source)
        self.assertNotIn("--gres", source)
        for variable in ("CONFIG_HASH", "CHECKPOINT_SHA256", "R00_SUMMARY_SHA256"):
            self.assertIn(f'${{{variable}:?', source)
        self.assertIn("main/summarize_endpoint_free.py", source)
        self.assertIn('--config-hash "$CONFIG_HASH"', source)


if __name__ == "__main__":
    unittest.main()
