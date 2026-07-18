"""Adversarial tests for the final SafeLIBERO Table-1 renderer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "render_safelibero_table1_comparison.py"
TARGETS = ROOT / "configs" / "vlsa_table1_translational.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = load_module("safelibero_table1_renderer", SCRIPT)


JOINT_BASELINE = {
    "safe_task_success": 6,
    "safe_task_failure": 4,
    "collision_task_success": 19,
    "collision_task_failure": 21,
}
JOINT_AEGIS = {
    "safe_task_success": 30,
    "safe_task_failure": 10,
    "collision_task_success": 5,
    "collision_task_failure": 5,
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bind_receipt(value: dict) -> dict:
    output = copy.deepcopy(value)
    output.pop("receipt_payload_sha256", None)
    output["receipt_payload_sha256"] = renderer.sha256_bytes(
        renderer.canonical_json_bytes(output)
    )
    return output


def make_analysis_receipt(summary_path: Path, summary: dict) -> dict:
    provenance_path = (
        "/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v2/"
        "vlsa-table1-contact-authority-population-20260718a-"
        "publisher-28610/population-summary-v2.json"
    )
    return bind_receipt(
        {
            "schema_version": renderer.ANALYSIS_RECEIPT_SCHEMA,
            "status": renderer.ANALYSIS_RECEIPT_STATUS,
            "scientific_result": False,
            "source_v1": copy.deepcopy(summary["source_v1"]),
            "population": {
                "cases": 1600,
                "results": 3200,
                "no_cases_dropped": True,
                "accepted_result_payloads_sha256": summary[
                    "accepted_result_payloads_sha256"
                ],
            },
            "artifacts": {
                "summary": {
                    "path": provenance_path,
                    "sha256": renderer.sha256_path(summary_path),
                    "bytes": summary_path.stat().st_size,
                }
            },
            "claim_scope": copy.deepcopy(summary["claim_scope"]),
        }
    )


def group_ids() -> list[str]:
    output = []
    for suite in ("spatial", "goal", "object", "long"):
        for level in ("i", "ii"):
            for task in range(4):
                output.append(f"vlsa-t1-{suite}-{level}-t{task}")
    return output


def make_block(arm: str, *, scale: int, groups: int | None = None) -> dict:
    denominator = 50 * scale
    joint_base = JOINT_BASELINE if arm == renderer.ARMS[0] else JOINT_AEGIS
    joint = {key: value * scale for key, value in joint_base.items()}
    car_success = joint["safe_task_success"] + joint["safe_task_failure"]
    tsr_success = (
        joint["safe_task_success"] + joint["collision_task_success"]
    )
    ets_sum = (10000 if arm == renderer.ARMS[0] else 9000) * scale
    action_sum = (10050 if arm == renderer.ARMS[0] else 9050) * scale
    value = {
        "episodes": denominator,
        "status_counts": {"complete": denominator},
        "retained_method_failures": 0,
        "car_percent": 100.0 * car_success / denominator,
        "tsr_percent": 100.0 * tsr_success / denominator,
        "legacy_ets_steps_mean": ets_sum / denominator,
        "executed_action_count_mean": action_sum / denominator,
        "car": {
            "success_count": car_success,
            "failure_count": denominator - car_success,
            "denominator": denominator,
            "percent": 100.0 * car_success / denominator,
        },
        "tsr": {
            "success_count": tsr_success,
            "failure_count": denominator - tsr_success,
            "denominator": denominator,
            "percent": 100.0 * tsr_success / denominator,
        },
        "legacy_ets_steps": {
            "sum": ets_sum,
            "denominator": denominator,
            "mean": ets_sum / denominator,
        },
        "executed_action_count": {
            "sum": action_sum,
            "denominator": denominator,
            "mean": action_sum / denominator,
        },
        "joint_outcome_counts": joint,
    }
    if groups is not None:
        value["groups"] = groups
    return value


def joint_transport(scale: int) -> dict[str, int]:
    remaining = {
        key: value * scale for key, value in JOINT_AEGIS.items()
    }
    output: dict[str, int] = {}
    for baseline in renderer.JOINT_OUTCOMES:
        row_remaining = JOINT_BASELINE[baseline] * scale
        for aegis in renderer.JOINT_OUTCOMES:
            count = min(row_remaining, remaining[aegis])
            output[f"baseline_{baseline}_to_aegis_{aegis}"] = count
            row_remaining -= count
            remaining[aegis] -= count
        if row_remaining:
            raise AssertionError("invalid fixture transport")
    if any(remaining.values()):
        raise AssertionError("invalid fixture transport columns")
    return output


def project_joint(joint: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    car = {key: 0 for key in renderer.CAR_TRANSITIONS}
    task = {key: 0 for key in renderer.TASK_TRANSITIONS}
    for baseline in renderer.JOINT_OUTCOMES:
        for aegis in renderer.JOINT_OUTCOMES:
            count = joint[f"baseline_{baseline}_to_aegis_{aegis}"]
            baseline_safe = baseline.startswith("safe_")
            aegis_safe = aegis.startswith("safe_")
            baseline_success = baseline.endswith("_success")
            aegis_success = aegis.endswith("_success")
            car_key = (
                f"baseline_{'safe' if baseline_safe else 'collision'}"
                f"_to_aegis_{'safe' if aegis_safe else 'collision'}"
            )
            task_key = (
                f"baseline_{'success' if baseline_success else 'failure'}"
                f"_to_aegis_{'success' if aegis_success else 'failure'}"
            )
            car[car_key] += count
            task[task_key] += count
    return car, task


def transition_record(scale: int) -> dict:
    joint = joint_transport(scale)
    car, task = project_joint(joint)
    return {
        "denominator": 50 * scale,
        "arms": {"baseline": renderer.ARMS[0], "aegis": renderer.ARMS[1]},
        "car": car,
        "task": task,
        "joint": joint,
    }


def make_summary() -> dict:
    config = json.loads(TARGETS.read_text(encoding="utf-8"))
    targets = config["published_table1_targets"]
    groups = {
        arm: {
            group_id: make_block(arm, scale=1)
            for group_id in group_ids()
        }
        for arm in renderer.ARMS
    }
    suites = {
        arm: {
            suite: make_block(arm, scale=8, groups=8)
            for suite in renderer.SUITES
        }
        for arm in renderer.ARMS
    }
    average = {
        arm: make_block(arm, scale=32, groups=32)
        for arm in renderer.ARMS
    }
    published_differences = {}
    for arm in renderer.ARMS:
        arm_differences = {}
        for scope in renderer.SCOPES:
            block = average[arm] if scope == "average" else suites[arm][scope]
            target = targets[arm][scope]
            arm_differences[scope] = {
                "car_percent_delta": (
                    block["car_percent"] - target["car_percent"]
                ),
                "tsr_percent_delta": (
                    block["tsr_percent"] - target["tsr_percent"]
                ),
                "legacy_ets_steps_delta": (
                    block["legacy_ets_steps_mean"] - target["ets_steps"]
                ),
            }
        published_differences[arm] = arm_differences
    method_differences = {}
    for scope in renderer.SCOPES:
        baseline = (
            average[renderer.ARMS[0]]
            if scope == "average"
            else suites[renderer.ARMS[0]][scope]
        )
        aegis = (
            average[renderer.ARMS[1]]
            if scope == "average"
            else suites[renderer.ARMS[1]][scope]
        )
        method_differences[scope] = {
            "denominator": baseline["episodes"],
            "car_success_count_delta": (
                aegis["car"]["success_count"]
                - baseline["car"]["success_count"]
            ),
            "car_percentage_point_delta": (
                aegis["car_percent"] - baseline["car_percent"]
            ),
            "tsr_success_count_delta": (
                aegis["tsr"]["success_count"]
                - baseline["tsr"]["success_count"]
            ),
            "tsr_percentage_point_delta": (
                aegis["tsr_percent"] - baseline["tsr_percent"]
            ),
            "legacy_ets_steps_sum_delta": (
                aegis["legacy_ets_steps"]["sum"]
                - baseline["legacy_ets_steps"]["sum"]
            ),
            "legacy_ets_steps_mean_delta": (
                aegis["legacy_ets_steps_mean"]
                - baseline["legacy_ets_steps_mean"]
            ),
            "executed_action_count_sum_delta": (
                aegis["executed_action_count"]["sum"]
                - baseline["executed_action_count"]["sum"]
            ),
            "executed_action_count_mean_delta": (
                aegis["executed_action_count_mean"]
                - baseline["executed_action_count_mean"]
            ),
        }
    return {
        "schema_version": renderer.SUMMARY_SCHEMA,
        "status": renderer.SUMMARY_STATUS,
        "protocol_id": renderer.PROTOCOL_ID,
        "claim_scope": {
            "method_label": (
                "pi0.5 + AEGIS translational conditioned on frozen "
                "per-case Codex obstacle labels"
            ),
            "baseline_method_label": "pi0.5 translational",
            "aegis_method_label": (
                "pi0.5 + AEGIS translational conditioned on frozen "
                "per-case Codex obstacle labels"
            ),
            "table_scope": (
                "two-row translational Table-1 reproduction: "
                "pi0.5 and pi0.5+AEGIS only"
            ),
            "population_scope": (
                "complete frozen 1,600-case SafeLIBERO population for "
                "the two registered translational arms"
            ),
            "openvla_oft_included": False,
            "paper_semantic_selector_reproduced": False,
            "paper_exact_end_to_end_reproduction_claimed": False,
            "clearance_available": False,
            "minimum_clearance_claimed": False,
            "configured_scope": "Codex-conditioned AEGIS",
        },
        "source_v1": {
            "v1_publication_receipt_sha256": "a" * 64,
            "v1_population_summary_sha256": "b" * 64,
            "v1_prepublish_receipt_sha256": "d" * 64,
            "v1_accepted_result_payloads_sha256": "c" * 64,
            "v1_source_git_commit": renderer.EXPECTED_RUNTIME_COMMIT,
            "v1_run_id": (
                "vlsa-table1-contact-authority-population-20260718a"
            ),
        },
        "accepted_result_payloads_sha256": "c" * 64,
        "population": {
            "cases": 1600,
            "arms": 2,
            "results": 3200,
            "task_level_groups": 32,
            "no_results_dropped": True,
        },
        "metric_semantics": {
            "car": (
                "maximum active-obstacle L1 position displacement strictly "
                "greater than 0.001 m"
            ),
            "tsr": "SafeLIBERO simulator done",
            "table_ets": (
                "mean released zero-based control-loop counter "
                "legacy_ets_steps"
            ),
            "executed_actions_reported_separately": True,
        },
        "task_level_groups": groups,
        "suites": suites,
        "average": average,
        "paired_transitions": {
            "overall": transition_record(32),
            "suites": {
                suite: transition_record(8) for suite in renderer.SUITES
            },
        },
        "method_differences": method_differences,
        "published_table1_differences": published_differences,
    }


class Table1ComparisonRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="table1-renderer-tests-"
        )
        self.root = Path(self.temporary.name)
        self.summary = self.root / "population-summary-v2.json"
        summary = make_summary()
        write_json(self.summary, summary)
        self.analysis_receipt = self.root / "analysis-v2-receipt.json"
        write_json(
            self.analysis_receipt,
            make_analysis_receipt(self.summary, summary),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def render(self, output_name: str = "report") -> tuple[dict, Path]:
        output = self.root / output_name
        receipt = renderer.build_report(
            summary_path=self.summary,
            analysis_receipt_path=self.analysis_receipt,
            paper_targets_path=TARGETS,
            output_root=output,
        )
        return receipt, output

    def mutate(self, callback) -> None:
        value = json.loads(self.summary.read_text(encoding="utf-8"))
        callback(value)
        write_json(self.summary, value)

    def test_complete_population_renders_deterministically(self) -> None:
        first_receipt, first = self.render("first")
        second_receipt, second = self.render("second")
        self.assertEqual(first_receipt, second_receipt)
        self.assertEqual(
            (first / "table1-comparison.md").read_bytes(),
            (second / "table1-comparison.md").read_bytes(),
        )
        self.assertEqual(
            (first / "table1-comparison.tex").read_bytes(),
            (second / "table1-comparison.tex").read_bytes(),
        )
        markdown = (first / "table1-comparison.md").read_text(
            encoding="utf-8"
        )
        latex = (first / "table1-comparison.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn("1,600 cases / 3,200 results", markdown)
        self.assertIn("20.00% (80/400)", markdown)
        self.assertIn("Long-suite AEGIS CAR of 79.63%", markdown)
        self.assertIn("Direct paired method effect", markdown)
        self.assertIn("\\begin{table*}", latex)
        self.assertIn("79.63\\%", latex)
        receipt = json.loads(
            (first / "table1-report-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(receipt["validation"]["apparatus_failure_count"], 0)
        self.assertEqual(
            receipt["inputs"]["analysis_v2_receipt_sha256"],
            renderer.sha256_path(self.analysis_receipt),
        )
        self.assertNotEqual(
            receipt["inputs"]["analysis_v2_summary_provenance_path"],
            receipt["inputs"]["local_population_summary_path"],
        )
        self.assertEqual(
            receipt["inputs"]["local_population_summary_path"],
            str(self.summary.resolve()),
        )
        self.assertEqual(
            receipt["receipt_payload_sha256"],
            first_receipt["receipt_payload_sha256"],
        )

    def test_cli_prints_only_terminal_receipt_identity(self) -> None:
        output = self.root / "cli"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--summary",
                str(self.summary),
                "--analysis-receipt",
                str(self.analysis_receipt),
                "--paper-targets",
                str(TARGETS),
                "--output-root",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        status = json.loads(completed.stdout)
        self.assertEqual(
            status["status"], "rendered_from_complete_validated_population"
        )
        self.assertTrue((output / "table1-report-receipt.json").is_file())

    def test_rejects_partial_population_before_creating_output(self) -> None:
        self.mutate(lambda value: value["population"].update(cases=1599))
        output = self.root / "partial"
        with self.assertRaisesRegex(
            renderer.TableReportError, "partial"
        ):
            self.render("partial")
        self.assertFalse(output.exists())

    def test_rejects_apparatus_status(self) -> None:
        def change(value):
            group = next(
                iter(
                    value["task_level_groups"][renderer.ARMS[1]].values()
                )
            )
            group["status_counts"] = {"apparatus_failure": 50}

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "apparatus/non-scientific"
        ):
            self.render("apparatus")

    def test_rejects_inconsistent_exact_numerator(self) -> None:
        def change(value):
            group = next(
                iter(
                    value["task_level_groups"][renderer.ARMS[0]].values()
                )
            )
            group["car"]["success_count"] += 1

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "exact denominator is inconsistent"
        ):
            self.render("numerator")

    def test_rejects_changed_target_bytes(self) -> None:
        changed_targets = self.root / "targets.json"
        changed_targets.write_bytes(TARGETS.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            renderer.TableReportError, "config hash differs"
        ):
            renderer.build_report(
                summary_path=self.summary,
                analysis_receipt_path=self.analysis_receipt,
                paper_targets_path=changed_targets,
                output_root=self.root / "changed-target",
            )

    def test_rejects_paper_delta_not_bound_to_targets(self) -> None:
        def change(value):
            value["published_table1_differences"][renderer.ARMS[1]][
                "average"
            ]["car_percent_delta"] += 0.25

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "car_percent_delta differs"
        ):
            self.render("target-delta")

    def test_rejects_joint_outcome_marginal_tampering(self) -> None:
        def change(value):
            joint = value["paired_transitions"]["overall"]["joint"]
            first = renderer.JOINT_TRANSITIONS[0]
            second = renderer.JOINT_TRANSITIONS[1]
            joint[first] -= 1
            joint[second] += 1

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "not exact joint projections"
        ):
            self.render("joint")

    def test_rejects_car_transition_tampering(self) -> None:
        def change(value):
            car = value["paired_transitions"]["overall"]["car"]
            car["baseline_safe_to_aegis_safe"] -= 1
            car["baseline_safe_to_aegis_collision"] += 1

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "not exact joint projections"
        ):
            self.render("car-transition")

    def test_rejects_car_table_not_exact_joint_projection(self) -> None:
        def change(value):
            car = value["paired_transitions"]["overall"]["car"]
            car["baseline_safe_to_aegis_safe"] -= 1
            car["baseline_safe_to_aegis_collision"] += 1
            car["baseline_collision_to_aegis_safe"] += 1
            car["baseline_collision_to_aegis_collision"] -= 1

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "not exact joint projections"
        ):
            self.render("car-projection")

    def test_rejects_suite_transitions_not_summing_to_overall(self) -> None:
        def change(value):
            record = value["paired_transitions"]["suites"][
                renderer.SUITES[0]
            ]
            joint = record["joint"]
            bss_ass = (
                "baseline_safe_task_success_to_aegis_safe_task_success"
            )
            bss_acf = (
                "baseline_safe_task_success_to_aegis_collision_task_failure"
            )
            bcf_ass = (
                "baseline_collision_task_failure_to_aegis_safe_task_success"
            )
            bcf_acf = (
                "baseline_collision_task_failure_to_aegis_collision_task_failure"
            )
            joint[bss_ass] -= 1
            joint[bss_acf] += 1
            joint[bcf_ass] += 1
            joint[bcf_acf] -= 1
            record["car"], record["task"] = project_joint(joint)

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError,
            "suite paired transitions do not sum to overall",
        ):
            self.render("transition-hierarchy")

    def test_rejects_renamed_frozen_task_level_group(self) -> None:
        def change(value):
            old = "vlsa-t1-spatial-i-t0"
            new = "vlsa-t1-spatial-i-arbitrary"
            for arm in renderer.ARMS:
                groups = value["task_level_groups"][arm]
                groups[new] = groups.pop(old)

        self.mutate(change)
        with self.assertRaisesRegex(
            renderer.TableReportError, "identities differ"
        ):
            self.render("renamed-group")

    def test_rejects_incomplete_source_v1_binding(self) -> None:
        self.mutate(
            lambda value: value["source_v1"].pop(
                "v1_publication_receipt_sha256"
            )
        )
        with self.assertRaisesRegex(
            renderer.TableReportError, "source-v1 binding is incomplete"
        ):
            self.render("source-binding")

    def test_rejects_wrong_or_unsafe_source_run_id(self) -> None:
        for index, run_id in enumerate(
            ("different-run", "nested/attacker", "control\ncharacter")
        ):
            with self.subTest(run_id=run_id):
                summary = make_summary()
                summary["source_v1"]["v1_run_id"] = run_id
                write_json(self.summary, summary)
                write_json(
                    self.analysis_receipt,
                    make_analysis_receipt(self.summary, summary),
                )
                with self.assertRaisesRegex(
                    renderer.TableReportError, "source run ID differs"
                ):
                    self.render(f"wrong-run-{index}")

    def test_rejects_changed_metric_semantics(self) -> None:
        self.mutate(
            lambda value: value["metric_semantics"].update(
                car="direct contact"
            )
        )
        with self.assertRaisesRegex(
            renderer.TableReportError, "metric_semantics/car differs"
        ):
            self.render("semantics")

    def test_rejects_summary_bytes_not_bound_to_analysis_receipt(self) -> None:
        self.summary.write_bytes(self.summary.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            renderer.TableReportError, "summary bytes/SHA differ"
        ):
            self.render("summary-binding")

    def test_rejects_analysis_receipt_payload_tampering(self) -> None:
        value = json.loads(
            self.analysis_receipt.read_text(encoding="utf-8")
        )
        value["population"]["cases"] = 1599
        write_json(self.analysis_receipt, value)
        with self.assertRaisesRegex(
            renderer.TableReportError, "payload hash differs"
        ):
            self.render("receipt-payload")

    def test_rejects_rehashed_incomplete_analysis_receipt(self) -> None:
        value = json.loads(
            self.analysis_receipt.read_text(encoding="utf-8")
        )
        value["population"]["cases"] = 1599
        write_json(self.analysis_receipt, bind_receipt(value))
        with self.assertRaisesRegex(
            renderer.TableReportError, "population is incomplete"
        ):
            self.render("receipt-population")

    def test_rejects_rehashed_analysis_receipt_wrong_summary_path(
        self,
    ) -> None:
        value = json.loads(
            self.analysis_receipt.read_text(encoding="utf-8")
        )
        value["artifacts"]["summary"]["path"] = str(
            self.root / "different-summary.json"
        )
        write_json(self.analysis_receipt, bind_receipt(value))
        with self.assertRaisesRegex(
            renderer.TableReportError, "unsafe/unexpected topology"
        ):
            self.render("receipt-path")

    def test_rejects_rehashed_analysis_receipt_wrong_publisher_job(
        self,
    ) -> None:
        value = json.loads(
            self.analysis_receipt.read_text(encoding="utf-8")
        )
        descriptor = value["artifacts"]["summary"]
        descriptor["path"] = descriptor["path"].replace(
            "publisher-28610", "publisher-99999"
        )
        write_json(self.analysis_receipt, bind_receipt(value))
        with self.assertRaisesRegex(
            renderer.TableReportError, "unsafe/unexpected topology"
        ):
            self.render("receipt-publisher")

    def test_rejects_rehashed_analysis_receipt_source_mismatch(self) -> None:
        value = json.loads(
            self.analysis_receipt.read_text(encoding="utf-8")
        )
        value["source_v1"]["v1_run_id"] = "different-run"
        write_json(self.analysis_receipt, bind_receipt(value))
        with self.assertRaisesRegex(
            renderer.TableReportError, "source-v1 binding differs"
        ):
            self.render("receipt-source")

    def test_rejects_rehashed_analysis_receipt_claim_mismatch(self) -> None:
        value = json.loads(
            self.analysis_receipt.read_text(encoding="utf-8")
        )
        value["claim_scope"]["clearance_available"] = True
        write_json(self.analysis_receipt, bind_receipt(value))
        with self.assertRaisesRegex(
            renderer.TableReportError, "claim scope differs"
        ):
            self.render("receipt-claim")

    def test_rejects_symlinked_analysis_receipt(self) -> None:
        link = self.root / "analysis-receipt-link.json"
        link.symlink_to(self.analysis_receipt)
        with self.assertRaisesRegex(
            renderer.TableReportError, "missing or symlinked"
        ):
            renderer.build_report(
                summary_path=self.summary,
                analysis_receipt_path=link,
                paper_targets_path=TARGETS,
                output_root=self.root / "receipt-symlink",
            )

    def test_rejects_symlinked_summary(self) -> None:
        link = self.root / "summary-link.json"
        link.symlink_to(self.summary)
        with self.assertRaisesRegex(
            renderer.TableReportError, "missing or symlinked"
        ):
            renderer.build_report(
                summary_path=link,
                analysis_receipt_path=self.analysis_receipt,
                paper_targets_path=TARGETS,
                output_root=self.root / "symlink",
            )

    def test_rejects_existing_output_root(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(
            renderer.TableReportError, "must be unused"
        ):
            self.render("existing")

    def test_rejects_nonfinite_json(self) -> None:
        self.summary.write_text('{"value": NaN}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            renderer.TableReportError, "nonfinite"
        ):
            self.render("nonfinite")


if __name__ == "__main__":
    unittest.main()
